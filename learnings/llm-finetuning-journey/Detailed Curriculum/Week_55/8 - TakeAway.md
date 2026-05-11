# Week 55 TakeAway — LLM-as-Judge Filtering

**One-liner:** Filter at temperature=0.0, calibrate against human labels first, apply skill-adaptive thresholds.

---

## Judge Prompt Structure

```
System: expert PostgreSQL engineer, evaluates training data
Body: schema_ddl + question + sql
Rubric: explicit 1-5 with correctness-dominates-style rule
Few-shot: one score-5 example + one score-2 example with reasoning
Output: {"score": N, "reason": "one sentence"}
```

## Filtering Pipeline Order (cheap → expensive)

```python
stages = [
    ("execution", lambda ex: ex["execution_status"] == "pass"),
    ("dedup",     lambda ex: not ex["is_duplicate"]),
    ("nonempty",  lambda ex: validate_result_not_empty(ex)),
    ("complexity",lambda ex: complexity_label_matches(ex)),
    ("judge",     lambda ex: judge_score(ex) >= threshold(ex["skill"])),
]

def threshold(skill):
    return 3 if skill_count[skill] < 300 else 4
```

## Calibration Check

```python
from sklearn.metrics import cohen_kappa_score

human = [1 if ex["human_label"] == "keep" else 0 for ex in cal_set]
judge = [1 if ex["judge_score"] >= 4 else 0 for ex in cal_set]
agreement = sum(h==j for h,j in zip(human,judge)) / len(human)
kappa = cohen_kappa_score(human, judge)
# Targets: agreement > 0.80, kappa > 0.60
```

---

## Decision Rules

- Always use temperature=0.0 for judging (deterministic, auditable)
- If agreement < 80%: add few-shot examples at score boundaries (2/3, 3/4) to prompt
- If TimescaleDB examples < 300 after filtering: relax threshold to score ≥ 3 for that skill
- Never remove more than 80% of any skill's examples — check per-skill counts after each filter stage
- Save all rejected examples with filter reason — you may need them for ablation studies

---

## Numbers to Remember

- Calibration set size: 100 examples minimum
- Agreement target: > 80%
- Cohen's kappa target: > 0.6
- Judge filtering cost: ~$2 per 15K examples at GPT-4o-mini pricing
- Typical retention rate through full pipeline: 35–50% of raw generated data

---

## Red Flags

- Judge agreement < 70%: rubric is too ambiguous; do not scale to full dataset yet
- TimescaleDB examples < 150 after filtering: generation prompts for that skill need revision before training
- Any skill → 0 examples after filtering: catastrophic — immediately relax threshold or hand-write examples
- All rejected examples at same filter stage: that stage's logic has a bug
