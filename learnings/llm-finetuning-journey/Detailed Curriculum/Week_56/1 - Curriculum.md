# Week 56 — Conversational Multi-Turn SQL (CoSQL/SParC Style)

## Learning Objectives

By the end of this week, you will be able to:

- Explain the difference between single-turn and multi-turn text-to-SQL and why it matters for real applications
- Understand the CoSQL and SParC dataset formats and what makes them challenging
- Convert single-turn SQL examples into coherent multi-turn conversations
- Design a chat template that correctly represents multi-turn SQL history for training
- Add 5,000 multi-turn examples to your v3 dataset in the correct format

## Why Multi-Turn SQL Matters

Every single-turn SQL example in your training data represents a pristine, self-contained question: "How many sensors have average temperature above 30°C?" In production, users don't work this way. They start with a vague question, see the result, then refine: "Show me sensors... now filter to only building A... now group by floor... now add the maintenance date."

This conversational dynamic requires the model to:
- Track context across turns (which tables are already mentioned, what schema was established)
- Understand implicit references ("now filter to only building A" requires knowing "sensors" from the first turn)
- Handle clarifications, corrections, and follow-up questions
- Know when a new question is a refinement of the previous one vs. a new independent query

A model trained only on single-turn examples will fail on this pattern. A model with 5K multi-turn examples will handle it reliably.

## Concepts

### CoSQL: Conversational Text-to-SQL

CoSQL (Yu et al., 2019) is a dataset of SQL queries grounded in multi-turn dialogues between a user and a system. Each dialogue has 3–7 turns. The user asks questions; the "system" (the SQL engine) generates SQL, runs it, shows results; the user responds to those results with a follow-up.

**Key characteristics of CoSQL:**
- 30K+ annotated conversation turns across 200 Spider databases
- Each turn has a NL utterance, the system SQL, and the database result
- Users ask clarifying questions, revise queries, and reference prior results implicitly

**What CoSQL teaches your model:**
- How to track the "query context" across a conversation
- How to handle references like "do the same for 2024" or "now include only X"
- How to correct a previous wrong interpretation

### SParC: Sequential Paraphrase Corpus

SParC (Yu et al., 2019) focuses on sequential paraphrases: users refine their questions in subsequent turns by restating or modifying them. SParC has ~4K conversations, 2–5 turns each, all grounded in the Spider database schemas.

**Difference from CoSQL:** SParC users don't see intermediate results — they express sequential intents purely based on their own goal decomposition. CoSQL users react to query results. Both patterns matter for production.

### Format: How to Represent Multi-Turn in a Chat Template

For decoder-only models (Qwen2.5-Coder-7B), multi-turn SQL must be formatted as a chat history. Each turn contains the user's question and the assistant's SQL. The full context (all prior turns) appears in the input for each new turn.

```
[SYSTEM] You are an expert PostgreSQL/TimescaleDB engineer.
Schema: {schema_ddl}

[USER] Show all sensors in building A.
[ASSISTANT] SELECT sensor_id, name, floor FROM sensors WHERE building = 'A';

[USER] Now filter to only sensors with average temperature above 30°C in the last 24 hours.
[ASSISTANT] SELECT s.sensor_id, s.name, s.floor
FROM sensors s
JOIN (
    SELECT sensor_id, AVG(temperature) as avg_temp
    FROM readings
    WHERE time > NOW() - INTERVAL '24 hours'
    GROUP BY sensor_id
) r ON s.sensor_id = r.sensor_id
WHERE s.building = 'A' AND r.avg_temp > 30;

[USER] Add the last maintenance date for each sensor.
[ASSISTANT] ...
```

This entire context (growing with each turn) is the input to the model. The model predicts only the final assistant turn. During training, you compute loss only on the assistant tokens (mask the user and system tokens).

### Converting Single-Turn to Multi-Turn

You likely do not have enough conversational data from CoSQL/SParC alone. The strategy:

1. **Direct CoSQL/SParC download:** Download and convert to your training format. This gives you ~5K conversations for free (after deduplication and quality check).

2. **Synthetic multi-turn generation:** Take an existing single-turn example. Ask the teacher model: "Given this question and SQL, generate a 3-turn conversation where the user progressively refines this query." The teacher generates turns 2 and 3 based on the original. You validate that each turn's SQL executes correctly.

3. **Decomposition:** Take a complex SQL query and decompose it: each sub-query or filtering step becomes a turn in the conversation. This requires some heuristic logic but is low-cost.

### Building TimescaleDB-Specific Multi-Turn Examples

The most valuable multi-turn examples for your domain involve time-series refinement patterns that are specific to TimescaleDB:

- "Show sensor readings for today" → "Now aggregate to hourly averages" → "Use gap-filling for missing hours"
- "Find anomalous readings" → "Define anomaly as > 2 std deviations from daily mean" → "Show only sensors that have been anomalous for 3+ consecutive hours"

These sequences are not in CoSQL or SParC (which use generic Spider databases). You must generate them specifically for your TimescaleDB schemas.

### The Conversation Loss Mask

A critical implementation detail: during training on multi-turn data, you must mask out all tokens except the final assistant turn's tokens when computing the loss. If you compute loss on user tokens, the model learns to predict user questions (not useful) and the loss scale is wrong.

In TRL/SFT format, this is handled by setting the role to `assistant` for response tokens and `user` for prompt tokens. Verify your collator is correctly masking.

### Common Misconceptions and Pitfalls

**"I can just concatenate all turns and train on the full sequence."** Without masking, the model trains on user tokens too, corrupting the gradient signal. Always use the `DataCollatorForSeq2Seq` with correct masking or TRL's built-in chat template handling.

**"CoSQL is directly usable for my domain."** CoSQL uses Spider databases (generic: cars, academic, music). You need to verify that the SQL in CoSQL is valid PostgreSQL (not SQLite, which CoSQL primarily targets). Run all CoSQL examples through Postgres; fix syntax issues with sqlglot.

**"More turns is better."** Very long conversations (7+ turns) are hard to learn from because the context window requirement grows and the signal-to-noise ratio drops. Prefer 3–4 turn conversations.

## Connections

This week completes Dataset v3. After this, you merge: filtered single-turn (Week 55 output) + 5K multi-turn (this week's output) = final `postgres-sql-v3`. Week 58 trains on this combined dataset.

## Time Allocation (6–8 hrs)

- 1h: Download CoSQL and SParC; run conversion scripts; count valid PostgreSQL examples
- 1h: Generate 2K synthetic multi-turn TimescaleDB examples using teacher model
- 1.5h: Build and run the multi-turn training format conversion pipeline
- 1h: Validate all multi-turn SQL executes correctly; apply quality filter
- 1h: Merge with single-turn filtered dataset; push final v3 to HuggingFace
- 0.5h: Commit, tag, and document
