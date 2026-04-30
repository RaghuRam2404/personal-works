# Week 55 Assignment Solutions

## Task 2 — Judge Prompt Key Snippet

```python
JUDGE_PROMPT = """You are an expert PostgreSQL and TimescaleDB engineer evaluating training data quality.

Schema:
{schema_ddl}

Question: {question}

SQL Answer:
```sql
{sql}
```

Rate this example on a 1-5 scale using this rubric:
5 - SQL exactly answers the question, uses schema correctly, production-appropriate style and efficiency
4 - SQL correctly answers with minor style issues or mild inefficiency  
3 - SQL answers the question but has notable problems (wrong aggregate, missing WHERE, suboptimal join)
2 - SQL is syntactically valid but semantically wrong, or references non-existent schema elements
1 - SQL has syntax errors, wrong tables/columns, or fails entirely to address the question

Correctness dominates style. A concise wrong answer scores 1. A verbose but correct answer scores at most 4.

Return only JSON: {{"score": <1-5>, "reason": "<one sentence>"}}"""

async def judge_example(ex):
    prompt = JUDGE_PROMPT.format(
        schema_ddl=ex["schema_ddl"],
        question=ex["question"],
        sql=ex["sql"]
    )
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,  # deterministic judge
        max_tokens=100
    )
    try:
        result = json.loads(response.choices[0].message.content)
        return result["score"], result["reason"]
    except:
        return None, "parse_error"
```

## Calibration Target

```
Agreement rate target: > 80%
Cohen's kappa target: > 0.6

Typical result after prompt v1: 72% agreement, kappa 0.48
Typical result after adding correctness-dominates-style rule: 83% agreement, kappa 0.67
```

---

## Common Gotchas

- **Judge always returns 5.** Your rubric is too lenient. Add explicit penalties and require one concrete flaw to score < 5.
- **Empty result set filter false positives.** Queries with aggregation over an empty table return 0 rows — this is valid. Use EXPLAIN instead of COUNT(*) check, or only flag COUNT = 0 for non-aggregate queries.
- **Filtering removes too many TimescaleDB examples.** Apply skill-adaptive thresholds: `min_score = 3 if skill_count < 300 else 4`.
- **Judge is slow at scale.** Batch 5 examples per API call using a list format in the prompt. This cuts API calls by 5×.

---

## How to Verify You Did It Right

1. `judge_calibration_results.md` shows agreement ≥ 80% and kappa ≥ 0.6
2. `v3_filtered.jsonl` has ≥ 20,000 lines
3. Per-stage attrition in W&B shows no single stage removing > 60% (if it does, something is misconfigured)
4. `post_filter_gap_analysis.md` confirms no skill < 150 examples
5. `v3_rejected.jsonl` is populated — if it is empty, your filter is not running

**Expected stage-by-stage attrition (approximate):**

| Stage | Input | Output | Attrition |
|-------|-------|--------|-----------|
| Execution pass | 30,000 | 18,000 | 40% |
| Deduplication | 18,000 | 15,000 | 17% |
| Empty result / complexity mismatch | 15,000 | 13,500 | 10% |
| LLM judge score ≥ 4 | 13,500 | 9,000 | 33% |
| Rare skill relaxation (add back score-3) | — | +1,500 | — |
| **Final v3 filtered** | — | **~10,500** | — |

If your dataset is short of 20K after filtering, you need to: generate more (another generation pass in Week 54 gaps), augment Spider training data, or include hand-curated examples from earlier phases.
