# Week 50 TakeAway — Iteration Framework

**One-liner:** Every iteration experiment must have a hypothesis; change one thing at a time; start from v3, not v1.

---

## Iteration Process (mandatory structure)

```
1. HYPOTHESIS: "I believe X is wrong because [evidence]."
2. EXPERIMENT: "I will change Y specifically."
3. METRIC: "The target metric should change from A to B."
4. RUN: 200–500 steps from v3 checkpoint (NOT from v1).
5. RESULT: actual metric values.
6. ANALYSIS: was the hypothesis correct? What next?
```

---

## Failure Mode → Fix Map

| Failure mode | Symptom | Fix |
|---|---|---|
| Too simple training data | Complex acc same as v2 | Add 200+ complex prompts, filter to 20-60% success rate |
| Execution without semantics | Exec acc up, sem acc down | Add reference SQL; demote level-0.2 reward to 0.05 |
| Mode collapse | reward_std → 0 | Increase temperature (0.7→0.9), switch to harder prompts |
| High KL (>10 nats) | Model generates refusals/gibberish | Increase β (0.05→0.15), restart from pre-KL-explosion checkpoint |
| Aggregation always SELECT * | Model uses broad queries | Add anti-hack guard for aggregation prompts |

---

## Diagnostic Code for Prompt Difficulty

```python
def measure_success_rate(prompt, model, reward_fn, K=8, temperature=0.7):
    completions = [generate(model, prompt, temperature) for _ in range(K)]
    rewards = reward_fn(completions, [prompt]*K)
    return sum(1 for r in rewards if r >= 0.5) / K

# Use only prompts with 0.20 <= rate <= 0.60 for GRPO
usable = [p for p in prompts if 0.2 <= measure_success_rate(p, ...) <= 0.6]
```

---

## Decision Rules

- Start iteration from v3 checkpoint, not v1 — you want to patch, not restart
- Change ≤ 2 things per experiment (ideally 1)
- Stop adding compute if reward_std < 0.05 for >50 steps — mode collapse; fix root cause first
- Target metric per experiment: pick ONE metric; verify it improves
- Budget: each 300-step GRPO run costs ~$6–10 on RunPod A100 80GB

---

## Numbers to Remember

- Usable prompt difficulty: 20–60% v3 success rate
- 300 steps: enough to verify a targeted hypothesis
- reward_std stop criterion: < 0.05 for 50+ consecutive steps = mode collapse
- One experiment: one hypothesis, one change, one primary metric
