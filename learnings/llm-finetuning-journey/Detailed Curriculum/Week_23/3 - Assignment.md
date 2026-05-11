# Week 23 Assignment — Evaluation Harness

## Setup Checklist

- [ ] `pip install lm-eval` (version >= 0.4)
- [ ] Your 50M model checkpoint accessible (locally or on HuggingFace Hub)
- [ ] Colab Pro or RunPod A100 available for running eval (takes 1–2 hours total)
- [ ] `gpt2` model available (HuggingFace will auto-download)

---

## Task 1 — Run lm-evaluation-harness on GPT-2

**Goal:** Establish a baseline for a published model on the same benchmarks you will run on your model.

**Requirements:**

Run the following evaluation and save all results to `results/gpt2_eval.json`:

```bash
lm_eval \
  --model hf \
  --model_args pretrained=gpt2 \
  --tasks hellaswag,arc_easy,arc_challenge,mmlu \
  --num_fewshot 0 \
  --device cuda:0 \
  --batch_size 16 \
  --output_path results/gpt2_eval.json \
  --log_samples
```

Note: MMLU has 57 subtasks. If running all of them takes too long, use the MMLU 5-task subset:
```bash
--tasks hellaswag,arc_easy,arc_challenge,mmlu_abstract_algebra,mmlu_computer_security
```

Record all scores in a table.

**Deliverable:** `results/gpt2_eval.json` + a Markdown table in `week-23-eval-report.md` showing GPT-2 scores.

**Expected GPT-2 results (0-shot):**
- HellaSwag: 31.6%
- ARC-Easy: 43.2%
- ARC-Challenge: 25.9%
- MMLU (average): ~26% (near random)

If your results differ by more than 5 percentage points from these numbers, something is wrong with your setup.

---

## Task 2 — Run lm-evaluation-harness on Your 50M Model

**Goal:** Compare your trained model to GPT-2.

**Requirements:**

```bash
lm_eval \
  --model hf \
  --model_args pretrained=<your-handle>/fineweb-50m-pretrain \
  --tasks hellaswag,arc_easy,arc_challenge \
  --num_fewshot 0 \
  --device cuda:0 \
  --batch_size 16 \
  --output_path results/your_model_eval.json
```

If your model is not yet on HuggingFace, you can evaluate the local checkpoint:
```bash
--model_args pretrained=./pretrain-50m/checkpoint/,tokenizer=./pretrain-50m/tokenizer/
```

Note: Your model uses a custom tokenizer. You may need to register it or use the HuggingFace-compatible format. If evaluation fails due to tokenizer issues, use the `--model custom` path with a wrapper (see Resources).

**Deliverable:** `results/your_model_eval.json` + scores added to the comparison table in `week-23-eval-report.md`.

---

## Task 3 — Write the Evaluation Comparison Report

**Goal:** Produce a rigorous comparison report that a colleague could use to understand your model's capabilities.

**Requirements:**

Write `week-23-eval-report.md` with the following sections:

**Section 1: Methodology**
- What benchmarks were used and why
- How lm-evaluation-harness scores each benchmark (log-likelihood, not generation)
- What 0-shot means vs. 5-shot

**Section 2: Results Table**

| Benchmark | Random Baseline | GPT-2 (117M, 40B tok) | Your 50M (2B tok) |
|---|---|---|---|
| HellaSwag | 25% | ? | ? |
| ARC-Easy | 25% | ? | ? |
| ARC-Challenge | 25% | ? | ? |
| MMLU (if run) | 25% | ? | ? |
| Val Perplexity (FineWeb-Edu) | — | ~18 | ? |

**Section 3: Analysis**
- Where does your model beat/match/lose to GPT-2? Why?
- Why is your 50M model closer to or farther from random than GPT-2?
- What explains the gap between the two models (parameters, tokens, data quality)?

**Section 4: Domain Relevance**
- None of these benchmarks measure SQL generation. What would a SQL-specific evaluation look like?
- Propose a 3-step evaluation protocol for your Phase 6 model that directly measures PostgreSQL text-to-SQL quality.

**Acceptance criteria:**
- Report is at least 600 words
- All 4 sections present
- Real numbers from your eval runs (not placeholders)

GitHub commit: `week-23-eval-harness`

---

## Task 4 — Manual Log-Likelihood Evaluation

**Goal:** Understand how benchmark scoring works by implementing it yourself.

**Requirements:**

Write `manual_eval.py` that manually scores 5 HellaSwag examples:

```python
from lm_eval.tasks import get_task_dict
from lm_eval.api.task import Task

# Or use the raw dataset:
from datasets import load_dataset
ds = load_dataset("Rowan/hellaswag", split="validation")
```

For 5 examples:
1. Load the question + 4 options
2. For each option, compute `log P(option | question)` using your model
3. Select the option with highest log P
4. Compare to the ground truth label

Report accuracy on these 5 examples.

**Deliverable:** `pretrain-50m/manual_eval.py`

---

## Stretch Goals

- Run GSM8K (0-shot) on your model to measure basic math ability. Expected: ~0% — useful to see the contrast with larger models
- Add a 5-shot evaluation for ARC-Easy (prepend 5 labeled examples before each question) and compare to 0-shot
- Measure your model's performance on a simple SQL classification task: given 10 pairs of (correct SQL, wrong SQL), does your model assign higher log-probability to the correct one?
