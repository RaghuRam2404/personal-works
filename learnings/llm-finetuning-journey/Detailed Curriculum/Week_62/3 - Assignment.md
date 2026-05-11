# Week 62 Assignment — Head-to-Head Model Comparison

## Setup Checklist

- [ ] OpenAI API key (budget ~$15 for this week)
- [ ] Anthropic API key (budget ~$8 for this week)
- [ ] `openai`, `anthropic` Python packages installed
- [ ] SQLCoder-7B downloaded or accessible via HuggingFace
- [ ] DeepSeek-Coder-V2-Lite accessible (via Together AI or local)
- [ ] Week 61 eval harness (`eval_harness.py`) tested and working

---

## Task 1 — Extend Eval Harness for API Models

**Goal:** Add API model support to your harness without changing the evaluation logic.

**Requirements:**
Extend `eval_harness.py` with a `--model-type` flag:
- `--model-type local`: loads model from HuggingFace path (existing code)
- `--model-type openai`: calls OpenAI API with `client.chat.completions.create()`
- `--model-type anthropic`: calls Anthropic API

API model interface:
```python
def call_api_model(model_type, model_name, prompt, system_prompt):
    if model_type == "openai":
        response = openai_client.chat.completions.create(
            model=model_name,
            messages=[{"role":"system","content":system_prompt},
                      {"role":"user","content":prompt}],
            temperature=0,
            max_tokens=512,
        )
        return response.choices[0].message.content
    elif model_type == "anthropic":
        response = anthropic_client.messages.create(
            model=model_name,
            max_tokens=512,
            temperature=0,
            system=system_prompt,
            messages=[{"role":"user","content":prompt}],
        )
        return response.content[0].text
```

Cache all API responses to a local file (`api_cache.json`) — never re-call the API for an example you've already evaluated.

**Deliverable:** Extended `eval_harness.py` with API support; 10-example test on GPT-4o-mini (cheap test).

---

## Task 2 — Run All Model Evaluations

**Goal:** Collect scores for all 6 models on all benchmarks.

**Requirements:**
Run the harness for each model on your 200-example custom benchmark AND a 100-example BIRD-SQL sample:

```bash
# Your model
python eval_harness.py --model <your-handle>/postgres-sqlcoder-7b-final \
  --model-type local --benchmark custom --output results_final.json

# Base Qwen (lower bound)  
python eval_harness.py --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --model-type local --benchmark custom --output results_base_qwen.json

# SQLCoder
python eval_harness.py --model defog/sqlcoder-7b-2 \
  --model-type local --benchmark custom --output results_sqlcoder.json

# GPT-4o (costs ~$12)
python eval_harness.py --model gpt-4o --model-type openai \
  --benchmark custom --output results_gpt4o.json

# Claude 3.5 Sonnet (costs ~$8)
python eval_harness.py --model claude-3-5-sonnet-20241022 \
  --model-type anthropic --benchmark custom --output results_claude.json

# DeepSeek-Coder-V2-Lite (via Together AI or local)
python eval_harness.py --model deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct \
  --model-type local --benchmark custom --output results_deepseek.json
```

**Deliverable:** 6 × `results_*.json` files committed; total API spend ≤ $25.

---

## Task 3 — Compile the Comparison Table

**Goal:** One table that tells the full story.

**Requirements:**
Write `head_to_head_comparison.md` with:

```markdown
## Main Results

| Model | Params | Custom (all) | TimescaleDB | Standard PG | BIRD-100 | Cost/query |
|-------|--------|-------------|-------------|-------------|----------|-----------|
| Your model | 7B | X% ±Y% | X% | X% | X% | ~$0 |
| Base Qwen2.5-Coder | 7B | X% | X% | X% | X% | ~$0 |
| SQLCoder-7B | 7B | X% | X% | X% | X% | ~$0 |
| DeepSeek-Coder-V2-Lite | 16B | X% | X% | X% | X% | ~$0 |
| Claude 3.5 Sonnet | 200B+ | X% | X% | X% | X% | ~$0.02 |
| GPT-4o | 200B+ | X% | X% | X% | X% | ~$0.03 |
```

Add sections:
- "Where we win" — 5 TimescaleDB examples where your model is correct and GPT-4o is not
- "Where we lose" — 5 examples where GPT-4o is correct and your model is not, with diagnosis
- "Cost analysis" — cost-per-correct-query comparison
- "Statistical notes" — CI for your model's scores

**Deliverable:** `head_to_head_comparison.md` committed.

---

## Task 4 — Error Pattern Analysis

**Goal:** Understand WHY you lose, not just THAT you lose.

**Requirements:**
For your model's failures on the custom benchmark, classify each failure into:
- A: Wrong table/column name (schema hallucination)
- B: Correct schema but wrong SQL logic (wrong WHERE, wrong aggregation)
- C: Correct SQL logic but wrong TimescaleDB function (wrong hyperfunction syntax)
- D: Timeout or execution error (infrastructure issue)
- E: Other

Write `error_analysis.md` with the distribution by category and 2–3 examples per category.

**Deliverable:** `error_analysis.md` committed.

---

## Stretch Goals

- Run eval on Spider 1.0 dev (1,034 examples) for all 6 models — requires API budget management
- Implement "model ensemble": for each question, generate SQL with all 3 local models, execute all, pick the one that executes correctly; what is the ensemble accuracy?
- Profile inference latency: record time-to-first-token and total generation time per model per query; add to cost analysis (your model is faster on local hardware despite lower parameter count vs. API models)
