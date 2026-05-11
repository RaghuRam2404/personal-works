# Week 71 TakeAway — Frontier Reading: Tulu 3, SmolLM2, OLMo 2

Three papers; one consensus: curate better data, verify rewards objectively, train in stages.

## The 2024–2025 Instruction Tuning Consensus

```
1. Data quality > quantity (all three papers)
2. Verifiable rewards (code/SQL execution) > learned reward models
3. Staged training (CPT → SFT → alignment) with domain specificity
4. On-policy data generation with external verification is safe
5. Open intermediate checkpoints enable capability attribution
```

## Paper Extraction Template (reuse for Weeks 72–74)

```markdown
## Paper: [Title]

### Top 3 contributions (1 line each)
1.
2.
3.

### Key result I cannot ignore
- Benchmark: X | Model: Y | Accuracy: Z

### What differs from my work
-

### What I can directly apply
- If I apply [X], I expect [Y] because [Z]
```

## Techniques to Apply (from these papers)

- On-policy generation (Tulu 3): use current model to generate SQL → verify → add to SFT data
- Preference pair quality filter (Tulu 3): filter DPO pairs to high-margin examples only
- Staged mid-training upweighting (OLMo 2): for bilingual model (Week 77), upweight Tamil SQL 3x in second CPT phase
- Tokenizer audit (SmolLM2): check SQL keyword tokenization for all base models in Week 75

## Decision Rules

- If a paper finding is on a different parameter scale: check if the mechanism still holds at 7B
- If a claim says "10K pairs outperform 100K": verify with your own task (SQL may differ from general IF)
- If model collapse is a concern in on-policy generation: add executability verification as the filter
- If intermediate checkpoints were not saved: document this in the next training run, not the next paper

## Numbers to Remember

- Tulu 3: 10K quality DPO pairs > 100K noisy pairs (DPO signal quality dominates quantity)
- SmolLM2: 1.7B model can match 3B+ if data quality is high enough
- OLMo 2: mid-training phase contributes 2–4 pp on downstream evals
- Active reading time per paper: 90 minutes (abstract + eval + data + training sections)
