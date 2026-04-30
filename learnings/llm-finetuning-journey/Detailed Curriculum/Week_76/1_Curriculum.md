# Week 76 Curriculum — Multi-Turn Agentic SQL with Tool Use

## Learning Objectives

By the end of this week, you will be able to:

- Construct a multi-turn training dataset where the model generates partial SQL, receives an execution error, and refines its output.
- Implement a tool-calling loop in Python that connects your fine-tuned model to a live PostgreSQL database for agentic self-correction.
- Format tool-call messages using a structured schema (JSON function signatures) that the model is trained to emit and parse.
- Fine-tune your model on multi-turn agentic trajectories using SFT with correct loss masking across all assistant turns.
- Evaluate agentic SQL performance with metrics that capture correction ability, not just single-shot accuracy.

---

## Concepts

### What Is Agentic SQL?

Single-shot NL→SQL is the task you have been training for: given a schema and a question, produce SQL in one pass. Agentic SQL extends this into a loop. The model generates a candidate query, that query is executed against a real or simulated database, and the result — whether a table of rows, an error message, or empty results — is fed back to the model. The model then decides: accept the result, ask a clarifying question, or generate a revised query.

This is not a novel concept. It mirrors the ReAct pattern (Reason + Act) from the agent literature. The key insight for SQL is that execution is cheap and deterministic: you always know whether a query ran successfully, and you often know whether the result is semantically wrong (wrong row count, NULL where a value is expected, wrong column). This makes SQL an ideal domain for agentic self-correction because the feedback signal is unambiguous.

The practical motivation is significant. Your current model achieves 83.1% EM on Custom-200 in single-shot mode. Studies on agentic SQL (DIN-SQL, DAIL-SQL, and the CHESS system) show that allowing one correction round typically adds 3–6 pp EM on hard queries, precisely the queries your model currently fails on.

### Tool-Calling Format

Tool use in language models is formalized as structured function calls. The model learns to emit a JSON block — sometimes embedded in its response text, sometimes as a dedicated message type — that specifies a function name and arguments. An orchestrating runtime intercepts this block, executes the function, and returns the result as a new message in the conversation.

For SQL specifically, your tool schema looks like:

```json
{
  "name": "execute_sql",
  "description": "Execute a SQL query against the PostgreSQL database and return rows or an error string.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {"type": "string", "description": "Valid PostgreSQL SQL statement"}
    },
    "required": ["query"]
  }
}
```

The model learns that when it needs to execute SQL, it emits a structured call in a designated format. Your training data must contain examples where this call appears in the correct position in the conversation, the tool response appears as a `tool` role message, and the model's final answer follows that response.

The exact format depends on your model family. For Qwen2.5-Coder, tool calls use the `<tool_call>` and `<tool_response>` special tokens. For Llama 3.1, the format uses `<|python_tag|>` for code execution contexts or a JSON-in-text format. You must use `apply_chat_template` with `tools=` argument to get the canonical format for your model.

### Multi-Turn Dataset Construction

Building multi-turn agentic trajectories requires a different data pipeline than single-shot SFT.

**Trajectory format.** Each training example is a multi-turn conversation:

- Turn 1 (user): Schema + natural language question.
- Turn 2 (assistant): Reasoning (optional) + tool call with candidate SQL.
- Turn 3 (tool): Execution result (rows, error, empty).
- Turn 4 (assistant): Either final SQL if result is correct, or a revised tool call if not.
- Turn 5 (tool): Result of revised query.
- Turn 6 (assistant): Final answer.

**Sources for trajectories.** You have three options. First, generate trajectories programmatically: run your existing model or GPT-4o on Spider/BIRD questions, execute the generated SQL against a local database, log errors, and use GPT-4o to generate the correction turn. Second, use datasets that already contain execution traces (CHASE dataset, BIRD-SQL with execution). Third, use your existing single-shot dataset and inject synthetic errors: take correct SQL, apply rule-based mutations (drop a JOIN, use a wrong column), record the error, then include the correct SQL as the correction.

The third approach is fastest and most aligned with your TimescaleDB domain. For 500–2000 trajectories, this is sufficient to teach the self-correction behavior.

**Loss masking.** In multi-turn SFT, you must mask loss on user turns and tool-result turns. You only compute loss on assistant turns. The `DataCollatorForCompletionOnlyLM` must be adapted to handle multiple assistant turns per example. The simplest reliable approach: concatenate the full conversation, create a label array of the same length, and set labels to -100 everywhere except the token positions belonging to assistant turns.

### Agentic Loop Implementation

The runtime that executes your trained model in agentic mode is a simple Python loop:

```python
def agentic_sql_loop(model, tokenizer, question, schema, db_conn, max_rounds=3):
    messages = [{"role": "user", "content": f"{schema}\n\nQuestion: {question}"}]
    for _ in range(max_rounds):
        output = generate(model, tokenizer, messages)
        if is_tool_call(output):
            sql = extract_sql(output)
            result = execute_sql(db_conn, sql)
            messages.append({"role": "assistant", "content": output})
            messages.append({"role": "tool", "content": str(result)})
        else:
            return output  # Final answer
    return output  # Return last output if max rounds reached
```

The key engineering decisions are: how to detect a tool call (string matching, JSON parsing, or special token detection), how to handle database errors safely (catch exceptions, format as strings, never expose credentials), and when to stop (max rounds, or when model emits a non-tool response).

### Evaluation for Agentic SQL

Single-shot EM is not sufficient to evaluate an agentic system. You need:

- **First-attempt EM:** EM of the model's first SQL emission, before any correction. This tells you whether the model degrades on single-shot ability.
- **Final EM:** EM of the SQL in the model's last assistant turn. This is the headline metric.
- **Correction rate:** Fraction of initially-wrong queries that the model successfully corrects after one round.
- **Efficiency:** Mean number of tool calls per question. A model that takes 3 rounds to reach correct SQL is less efficient than one that takes 1.5 rounds on average.

On your Custom-200 benchmark, track all four metrics and compare to your single-shot baseline (83.1% EM).

---

## Connections

This week builds on your Week 48 (GRPO) and Week 50 (reward shaping) work, where you already implemented execution-based feedback. The difference is that this week the feedback happens at inference time, not training time. It also connects to Week 66 (deployment infrastructure), because the agentic loop requires a live database connection and a running model server. Next week's bilingual extension will add a language-routing layer on top of this agentic loop.

---

## Common Misconceptions / Pitfalls

Multi-turn SFT does not require reinforcement learning. You are training on fixed trajectories using standard cross-entropy loss, with correct masking. RL comes in if you want to optimize the policy online, which is out of scope this week.

Do not compute loss on tool-result turns. Tool results are observations, not model outputs. Computing loss on them teaches the model to predict database results, which is useless.

The self-correction loop at inference time is independent of how the model was trained. Even a single-shot-trained model can be put inside an agentic loop. The difference is that a model trained on multi-turn trajectories learns to use the error feedback more effectively.

Avoid training on trajectories where the correction is always one round. Real failures require different numbers of rounds. Balance your dataset across zero, one, and two correction examples.

---

## Time Allocation (6–8 hours)

- 1.5 hours: Read tool-calling format docs for your base model; implement `apply_chat_template` with tools and inspect output.
- 2 hours: Build trajectory generation pipeline; create 500 agentic training examples.
- 2 hours: Implement multi-turn SFT with correct loss masking; run 500 steps; verify loss curve.
- 1 hour: Implement the agentic inference loop; test against 20 Custom-200 examples; compare first-attempt vs final EM.
- 0.5 hours: Log all metrics to W&B project `week-76-agentic-sql`; write decision memo.
