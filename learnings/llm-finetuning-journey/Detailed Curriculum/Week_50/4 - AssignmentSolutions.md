# Week 50 Assignment Solutions

## Task 1 — Diagnosis Template

Your `diagnosis.md` should follow this structure:

```markdown
## Failure Mode 1 (Rank: High Impact)
**Evidence:** Complex query execution accuracy (v3 vs v2): 49% vs 48% — near zero improvement.
**Hypothesis:** The GRPO training prompt set in Week 47–48 contained only 8% complex queries 
  (3+ JOINs, CTEs). Most gradient came from simple/medium prompts.
**Proposed fix:** Add 250 complex SQL prompts to the GRPO training set. These should have 
  v3 success rate 20–50% (verified by diagnostic test).
**Expected result:** Complex query execution accuracy increases to 57%+ (≥8pp over v3).

## Failure Mode 2 (Rank: Medium Impact)
**Evidence:** semantic_accuracy (v3) = 55% < semantic_accuracy (v2) = 57%. 
  GRPO improved execution but not semantic correctness.
**Hypothesis:** The reward function returned 0.2 (executes unverified) for 45% of training 
  completions because reference SQL was not available. Model optimized for "executes" not "correct".
**Proposed fix:** Add expected_rows to 80% of training prompts. Reduce reward level 0.2 to 0.05.
**Expected result:** Semantic accuracy increases to 60%+ (≥3pp over v3).
```

---

## Task 2 — Path B: Dataset Expansion Code

```python
# Generating complex SQL prompts synthetically
import json

complex_templates = [
    "Get the top 5 customers by total order value in the last 90 days, showing customer name and total",
    "Calculate the month-over-month revenue growth for each product category in 2024",
    "Find all products that were ordered more than 100 times but have had 0 orders in the last 30 days",
    "Show the running total of orders per customer ordered by order date",
    "List all customers who ordered in January but not in February 2024",
]

# Diagnostic test: measure v3 success rate on these templates
from reward_fn import sql_reward_fn, extract_sql
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("your-handle/postgres-sqlcoder-7b-v3-grpo")
tokenizer = AutoTokenizer.from_pretrained("your-handle/postgres-sqlcoder-7b-v3-grpo")

success_rates = {}
for prompt in complex_templates:
    completions = []
    for _ in range(8):  # K=8 samples
        inputs = tokenizer(prompt, return_tensors="pt")
        output = model.generate(**inputs, max_new_tokens=256, temperature=0.7, do_sample=True)
        completions.append(tokenizer.decode(output[0][inputs["input_ids"].shape[1]:]))
    rewards = sql_reward_fn(completions, [prompt]*8)
    success_rate = sum(1 for r in rewards if r >= 0.5) / len(rewards)
    success_rates[prompt] = success_rate

# Filter: keep prompts with 20-60% success rate (not too easy, not too hard)
usable_prompts = [p for p, rate in success_rates.items() if 0.2 <= rate <= 0.6]
```

**Common gotchas:**
- Do not add prompts where v3 already succeeds >80% — they will produce zero-gradient groups
- Do not add prompts where v3 succeeds <5% — they will produce all-zero groups on the other end
- The 20–60% range is the "zone of proximal development" for GRPO
- After adding prompts, re-run Week 47's diagnostic test to verify the reward distribution has not become degenerate

---

## Task 4 — Iteration Log Template

```markdown
# Week 50 Iteration Log

## Experiment 1: Complex Query Dataset Expansion

**Hypothesis:** v3 has poor complex query accuracy because the GRPO training distribution
  was biased toward simple queries. Adding 250 complex prompts will fix this.

**Experiment:** 
- Added 250 complex prompts (3+ JOINs, CTEs) to the GRPO training set
- Filtered to prompts with v3 success rate 20–55%
- Ran 300 GRPO steps from v3 checkpoint with the expanded dataset
- Config unchanged: K=8, LR=5e-7, β=0.05

**Result:**
| Metric | v3 | v3-iter1 | Change |
|---|---|---|---|
| Overall exec accuracy | 81% | 84% | +3pp |
| Complex query exec acc | 49% | 61% | +12pp |
| Semantic accuracy | 55% | 56% | +1pp |
| mean_reward (final step) | 0.38 | 0.44 | +0.06 |

**Analysis:** Hypothesis confirmed. Complex query accuracy improved by 12pp, exceeding the 
8pp target. The 300-step run was sufficient — reward_std remained healthy throughout.
Semantic accuracy barely moved — will address this in Week 51 with reward function fix.

## Next experiment (Week 51): Reward function fix for semantic accuracy
```

---

## How to Verify You Did It Right

1. Your diagnosis document has specific numerical evidence (not "v3 was bad on complex queries" but "v3 achieved 49% vs v3-iter1 targeting 57%+").
2. The iteration log has all 5 sections: Hypothesis, Experiment, Result, Analysis, Next.
3. v3-iter1 is better than v3 on the targeted metric by at least the expected amount.
4. You made only 1–2 changes simultaneously and can attribute improvement to specific changes.
