# Week 78 TakeAway — Final Phase 6 Gate and Course Wrap

**This week in 15 words:** Audit, publish, and reflect — an unshared model is an invisible model; own your results.

---

## Capstone Checklist (copy-paste into your audit)

```
[ ] postgres-sqlcoder-7b-final on HuggingFace Hub (public)
[ ] GGUF Q4_K_M + Q5_K_M uploaded
[ ] GPTQ INT4 uploaded
[ ] Model card with benchmark table (Custom-200, BIRD-SQL, Spider)
[ ] Usage snippet verified end-to-end
[ ] Technical report (PDF, arXiv or HF Papers)
[ ] Training code on GitHub (CPT, SFT, DPO, GRPO scripts)
[ ] Evaluation script (Custom-200 EM) published
[ ] Ollama Modelfile committed to GitHub
[ ] Blog post published (URL logged)
[ ] Final eval results JSON (loaded from HF Hub, not local)
[ ] Course retrospective written
```

---

## Key Numbers — Full Pipeline Summary

```
Base:        Qwen2.5-Coder-7B-Instruct
CPT:         100M tokens, PostgreSQL + TimescaleDB + SQL corpora
SFT-v3:      25K+ examples, LoRA r=64 alpha=128 lr=2e-4
DPO-v3:      5K pairs, beta=0.1
GRPO-final:  1.5K prompts, K=8, execution-based reward

Custom-200:  83.1% EM  (GPT-4o: 79.4%, Claude 3.5: 81.2%)
BIRD-SQL:    68.4% EX
Spider 1.0:  82.7% EM
```

---

## Decision Rules for Publishing

If benchmark number exists only in your W&B run → write it into the model card before the run expires.

If model card usage snippet has not been copy-paste tested → test it before publishing.

If technical report references "our evaluation" without a reproducible eval script → add the script link before publishing.

If your GitHub repo has no README → add one now; a README is the difference between a repo and a portfolio.

If you cannot explain your result in two sentences to a non-ML engineer → your blog post is not ready yet.

---

## What You Learned (18-Month Distillation)

- Attention is just weighted averaging; everything else is engineering.
- Scaling laws are real; they also have domain exceptions — that is your research contribution.
- SFT teaches format; DPO and GRPO teach preference; none of them create knowledge the base model does not have.
- Quantization at 4-bit costs ~1.5 pp EM; the latency and memory savings are worth it for deployment.
- The hardest part of any ML project is not the model — it is the evaluation.

---

## Red Flags (Final Gate)

Model card benchmark table shows numbers not in your eval logs → fabrication risk; verify every number against a run.

HuggingFace repo is private → your model does not exist publicly; make it public.

Course retrospective is one paragraph → you are rushing; the reflection is the most valuable part of week 78.

No public code → no reproducibility → your results cannot be trusted by the community; publish the training scripts.
