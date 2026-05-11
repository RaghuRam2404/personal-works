# Week 70 TakeAway — Technical Report: Limitations, Polish, Publish

A limitations section that is specific, testable, and honest is a sign of scientific maturity — not weakness.

## Limitations Section Formula

```
[Specific capability or scope] was [not tested / not characterized].
[Practical impact for a user who relies on this].
[The experiment that would quantify or fix it].
```

## Appendix Must-Haves

```
A. Hyperparameters: optimizer (name, β1, β2, ε), weight decay, clip, 
   LoRA (r, α, targets), LR, schedule, warmup, batch, steps — all 4 stages
B. Compute: stage | GPU type | #GPUs | hours | total GPU-h | cost
C. Data release: dataset name | HF link | license | split sizes
D. Model release: all 4 repos (BF16, GGUF, AWQ, GPTQ) | license
E. Eval prompt template: exact verbatim string used in every benchmark
F. Custom-200 description: how written, by whom, when, schema sources
```

## Final Consistency Pass Commands

```bash
# Check every key number appears consistently
grep -n "83.1\|25,500\|25500\|102M\|5,000\|1,534" report/final_report.md

# Check every citation has a matching reference
grep -o "\[.*et al.*\]" report/final_report.md | sort | uniq
```

## Decision Rules

- If a limitation is vague: add the specific thing that is not tested and the benchmark that would measure it
- If future work says "we plan to": add the mechanism, compute requirement, and success metric
- If a number appears differently in two places: fix it before PDF generation
- For arXiv without institutional email: find one endorser via your professional network
- Never paraphrase the eval prompt template in the appendix: copy-paste verbatim from the eval script

## Numbers to Remember

- Full report target: 3,500–5,500 words (main body, excluding appendix)
- Limitations: 5 specific, 50–80 words each
- Future work: 3 specific directions, each with mechanism + success metric
- arXiv processing time: 1–2 business days after submission
- HuggingFace Daily Papers threshold for curation: notable result on a standard benchmark

## Red Flags

- Limitations section is one paragraph: add specificity — each limitation needs its own paragraph
- Future work is a wish list: add mechanisms and success metrics
- Report published without consistency pass: numbers will disagree somewhere
- Appendix omits the evaluation prompt template: the most common reproducibility failure
- arXiv submission without checking PDF renders correctly: always download and read the generated PDF before declaring submission complete
