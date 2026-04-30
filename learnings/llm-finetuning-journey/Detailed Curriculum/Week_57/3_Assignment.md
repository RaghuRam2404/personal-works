# Week 57 Assignment — Run Continued Pretraining on 100M-Token Domain Corpus

## Setup Checklist

- [ ] RunPod account with billing set up; H100 80GB instance available
- [ ] HuggingFace account with write access
- [ ] Sufficient storage on RunPod for: model weights (14GB) + corpus (JSONL ~2GB) + optimizer states (~28GB)
- [ ] Unsloth installed: `pip install unsloth` (RunPod Unsloth template preferred)
- [ ] W&B account; API key set in environment

---

## Task 1 — Build the 100M-Token Corpus

**Goal:** Assemble and preprocess a domain-specific text corpus.

**Requirements:**
Write `build_cpt_corpus.py` that:
- Downloads and processes each source:
  - PostgreSQL docs: use `wget --mirror https://www.postgresql.org/docs/current/` then extract text with `beautifulsoup4`
  - TimescaleDB docs: similar approach for `https://docs.timescale.com/`
  - Stack Overflow dump: download the `Posts.xml` from [archive.org/details/stackexchange](https://archive.org/details/stackexchange), filter for tags containing `postgresql` or `timescaledb`
  - GitHub SQL files: use HuggingFace `bigcode/the-stack-v2` filtered to `.sql` extension
- Applies quality filters:
  - Minimum 50 words per document
  - Maximum 40% non-ASCII characters per document
  - Remove exact-duplicate documents (URL-based or MD5 hash)
- Saves each document as a JSON line: `{"text": "...", "source": "...", "url": "..."}`
- Counts total tokens using Qwen2.5-Coder tokenizer
- Stops adding sources once 100M tokens are accumulated

**Acceptance criteria:** Final corpus file `cpt_corpus.jsonl` with ≥ 100M tokens verified.

**Deliverable:** `build_cpt_corpus.py` + token count logged to W&B tag `week-57-corpus`.

---

## Task 2 — Preprocess and Pack the Corpus

**Goal:** Format the corpus for efficient CPT training.

**Requirements:**
Write `pack_corpus.py` that:
- Loads `cpt_corpus.jsonl`
- Tokenizes each document using Qwen2.5-Coder tokenizer
- Packs documents into sequences of exactly 2048 tokens (insert EOS token between documents)
- Saves as a HuggingFace `datasets.Dataset` with feature `input_ids`
- Uploads to HuggingFace as private dataset `<your-handle>/postgres-cpt-corpus-packed`

**Deliverable:** Packed dataset on HuggingFace. Log packing efficiency (fraction of tokens that are actual content vs. padding) to W&B — target > 95% (good packing).

---

## Task 3 — Smoke Test Locally (Colab or Mac)

**Goal:** Verify the training script works before spending RunPod GPU time.

**Requirements:**
Write `train_cpt.py` using Unsloth + Trainer that:
- Loads Qwen2.5-Coder-7B-Instruct (use 4-bit quantized for smoke test)
- Configures LoRA: rank=16, target modules=`["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]`
- Uses `DataCollatorForLanguageModeling(mlm=False)` — no masking, causal LM
- Runs for 100 steps on a small 10K-token subset
- Logs loss to W&B project `week-57-cpt`

**Acceptance criteria:** Training completes 100 steps without error. Loss decreases from step 1 to step 100.

**Deliverable:** `train_cpt.py` committed. W&B run `week-57-smoke` visible.

---

## Task 4 — Full CPT Run on RunPod H100

**Goal:** Run the full 100M-token CPT training on RunPod H100.

**Requirements:**
- Spin up RunPod H100 80GB instance (use the Unsloth template)
- Upload `train_cpt.py`; modify for full run:
  - Remove 4-bit quantization (use bf16 full precision for better quality)
  - Increase batch size to 8 (per-device), gradient accumulation 8 (effective batch = 64)
  - Learning rate: 5e-5 with cosine schedule, 100-step warmup
  - Train for 1 epoch (≈ 763 steps)
  - Save checkpoint every 200 steps
- Monitor in real-time via W&B
- **Terminate the RunPod instance immediately after training completes**

**Log to W&B:**
- Training loss per step
- Perplexity on 1,000-token held-out PostgreSQL doc excerpt (compute every 100 steps)
- Perplexity on 1,000-token held-out Wikipedia text (should remain stable — forgetting monitor)

**Acceptance criteria:**
- Training completes all ~763 steps
- Final training loss ≤ 1.9
- PostgreSQL held-out perplexity decreases from baseline
- Wikipedia perplexity does not increase by more than 0.5 bits

**Deliverable:** CPT checkpoint pushed to HuggingFace as `<your-handle>/qwen2.5-coder-7b-postgres-cpt`.

---

## Stretch Goals

- Implement a domain vocabulary enrichment check: after CPT, compute the model's per-token probability for TimescaleDB-specific tokens ("time_bucket", "locf", "interpolate", etc.) and compare to the base model — CPT should increase these by 10–30%
- Run a before/after completion test: prompt "SELECT * FROM sensor_readings WHERE time >" and see if the CPT model completes with TimescaleDB-idiomatic patterns more often
- Profile your corpus: plot token count by source; verify no single source dominates > 70%
