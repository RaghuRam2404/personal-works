# Week 54 Assignment — Generate 30K Synthetic SQL Pairs

## Setup Checklist

- [ ] OpenAI API key (or Anthropic API key) set in environment variable `OPENAI_API_KEY`
- [ ] PostgreSQL running locally with test schemas loaded (from Week 53's schema list)
- [ ] `openai`, `asyncio`, `httpx`, `psycopg2`, `sqlglot`, `datasketch` installed
- [ ] Week 53 gap analysis (`gap_analysis.md`) open — this drives your skill targets
- [ ] W&B project `week-54-generation` created

---

## Task 1 — Teacher Prompt Library

**Goal:** Write and validate teacher prompts for every skill category in your gap analysis.

**Requirements:**
- Create `prompts/` directory in your repo
- Write one prompt template per skill category (from gap_analysis.md)
- Each template must include:
  - System role description (precise domain expert framing)
  - Schema DDL placeholder `{schema_ddl}`
  - Target skill specification (explicit SQL constructs required)
  - Difficulty specification (`{difficulty}`)
  - 2 few-shot examples from your hand-curated set
  - JSON output format specification
  - Diversity instruction (list recent generated examples to avoid)
- Test each prompt manually (one API call) — verify the output parses and the SQL executes
- Save verified prompts as `prompts/<skill_name>.txt`

**Deliverable:** `prompts/` directory with ≥ 12 prompt files committed to `week-54-generation` branch.

---

## Task 2 — Async Generation Pipeline

**Goal:** Build a production-quality pipeline that generates 30K examples without requiring babysitting.

**Requirements:**
Write `generate_v3.py` that:
- Loads `gap_analysis.md` to determine how many examples to generate per skill
- For each skill, selects a schema DDL from your schema library at random
- Calls the teacher API asynchronously (max 10 concurrent calls)
- Parses JSON output; on parse failure, logs the raw response and skips
- For each parsed (question, sql) pair:
  - Runs SQL against Postgres; marks `execution_status` as `pass/fail/timeout`
  - Computes MinHash; checks against LSH index; marks `is_duplicate` as True/False
  - Saves to JSONL with fields: `id`, `skill`, `difficulty`, `question`, `sql`, `schema_name`, `execution_status`, `is_duplicate`, `teacher_model`, `prompt_version`
- Saves checkpoint every 100 examples (as `checkpoint_<N>.jsonl`)
- Logs to W&B: examples per minute, execution rate per skill, parse rate, duplicate rate
- Implements exponential backoff on API rate limit errors (429)

```python
# Skeleton structure
import asyncio
import openai
import psycopg2
import wandb
from datasketch import MinHash, MinHashLSH

async def generate_batch(client, skill, schema_ddl, difficulty, n=5):
    prompt = load_prompt(skill).format(
        schema_ddl=schema_ddl,
        difficulty=difficulty,
        recent_examples=get_recent_examples(skill, n=3)
    )
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
    )
    return parse_examples(response.choices[0].message.content)

async def main():
    # Load gap targets, schema library, resume from checkpoint
    # Run generate_batch concurrently with asyncio.gather
    # Validate, deduplicate, save each example
    pass
```

**Requirements:**
- Pipeline must be resumable from any checkpoint
- Must not crash on API errors — catch all exceptions, log, and continue
- W&B must show real-time execution rate per skill

**Deliverable:** `generate_v3.py` committed. W&B run `week-54-generation` with at least 1,000 examples generated and metrics logged.

---

## Task 3 — First 5K Generation Run

**Goal:** Validate the full pipeline produces usable data before committing to the full 30K run.

**Requirements:**
- Run your pipeline targeting the top 5 skill gaps (highest priority from gap_analysis.md)
- Generate at least 1,000 examples per skill (5,000 total)
- Analyze the results:
  - What is your overall execution rate?
  - Which skill has the lowest execution rate? Why?
  - What is your parse rate?
  - What percentage are near-duplicates?
- Write `generation_report_5k.md` with these metrics and your diagnosis

**Acceptance criteria:**
- Overall execution rate ≥ 55% (lower means your prompts need revision)
- Parse rate ≥ 85%
- Duplicate rate ≤ 15%

If you miss any criterion, revise your prompts and re-run before moving to the full 30K.

**Deliverable:** `generation_report_5k.md` + `data/v3_5k_raw.jsonl` committed.

---

## Task 4 — Push Raw Generated Data to HuggingFace

**Goal:** Preserve your raw (unfiltered) generation output before Week 55's filtering removes examples.

**Requirements:**
- Upload `v3_5k_raw.jsonl` (or full 30K if complete) to HuggingFace as a private dataset
- Dataset name: `<your-handle>/postgres-sql-v3-raw`
- Include the data card from Week 53 as the dataset README

**Deliverable:** HuggingFace dataset URL.

---

## Stretch Goals

- Implement Magpie-style generation in addition to Genie-style: compare which produces more diverse questions
- Estimate SQL complexity for each generated example using `sqlglot` AST depth; log distribution to W&B
- Run your first 1,000 generated examples through your v2 model to check if it can already solve them; examples your model gets right are "too easy" and less valuable for training
