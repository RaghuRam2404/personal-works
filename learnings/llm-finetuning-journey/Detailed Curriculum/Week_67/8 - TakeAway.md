# Week 67 TakeAway — Technical Report: Outline, Problem Statement, Dataset

A technical report's job is reproducibility, not novelty. Make every number traceable to a log.

## Key Document Structure

```
Abstract (150–200w): problem gap → method → key number → release
Introduction: problem → gap → contributions (bulleted) → overview
Dataset (600–900w): source → size → processing → example → table
Training: stages → hyperparameters → compute
Evaluation: benchmarks → table → analysis
Ablations: what you removed and what it cost
Limitations: honest gaps
Appendix: full hyperparameter table, compute budget, repro checklist
```

## Abstract Template

```
[Domain problem + gap]: [your approach]: [CPT+SFT+DPO+GRPO].
[Model] achieves [X%] on [your benchmark], outperforming [A] ([a%])
and [B] ([b%]). We release [training code, datasets, 3 quantized variants].
```

## Decision Rules

- If a number appears in the paper: it must trace to a log file or script output
- If a comparison model was not run by you: mark it "(published)" in the table
- If a dataset number changed after filtering: report the final post-filter count
- If hedging appears ("we believe", "we hope"): replace with measured evidence or delete
- Contributions = deliverables; Findings = measurements — never conflate them

## Numbers to Remember

- Abstract target: 150–200 words
- Dataset section target: 600–900 words
- Total report target: 3,000–5,000 words across all sections
- Contamination check: ROUGE-L overlap < 0.8 between train and eval
- LLM-judge acceptance rate to report: X% (your actual from Week 55)

## Red Flags

- Abstract contains "we believe" or "we hope": remove — state measurements only
- Dataset section says "we filtered" without acceptance rate: add the rate
- Comparison table mixes published scores with your scores on different benchmarks
- Model version not cited for closed-source comparisons (GPT-4o, Claude)
- Dataset counts in paper do not match counts in released HuggingFace dataset
