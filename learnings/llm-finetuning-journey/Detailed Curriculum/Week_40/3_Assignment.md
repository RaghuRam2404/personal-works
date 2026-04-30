# Week 40 Assignment — Phase 4 Gate

## Setup Checklist

- [ ] GitHub repo with all Phase 4 code (Weeks 29–39)
- [ ] HuggingFace account and `huggingface-hub` CLI installed and logged in
- [ ] W&B account with Phase 4 runs logged
- [ ] `held_out_test.json` (100 examples), `train_15k.jsonl`, `val_500.jsonl` accessible
- [ ] Week 39 `eval_harness.py` working end-to-end

---

## Task 1 — Run the Gate Checklist

**Goal:** Verify every Phase 4 minimum requirement is met. Be honest. A gap you paper over now will surface as a blocker in Phase 5.

**Requirements:**

- Complete the checklist below. For each item, record: pass / fail / partial, and one sentence of evidence.

| Item | Status | Evidence |
|---|---|---|
| At least 4 fine-tuning runs (SFT, LoRA scratch, LoRA peft, QLoRA) | | |
| 7B model fine-tuned on Colab Pro without OOM | | |
| W&B logs exist for at least 3 runs | | |
| Execution-based eval harness runs end-to-end | | |
| Exec correctness ≥ 60% on held_out_test.json | | |
| 15K domain dataset in repo or HuggingFace Datasets | | |
| v1 adapter loadable via PeftModel.from_pretrained | | |

- For any item marked "fail" or "partial": write a one-paragraph remediation plan in `week40_gate_checklist.md` — what you will do to fix it and when.

**Deliverable:** `week40_gate_checklist.md` committed to your GitHub repo

---

## Task 2 — Publish the Model to HuggingFace Hub

**Goal:** Push your best adapter (`postgres-sqlcoder-7b-v1`) to the Hub with a complete model card.

**Requirements:**

- Push the adapter using `model.push_to_hub("<your-handle>/postgres-sqlcoder-7b-v1")` or `adapter_model.save_pretrained` + `upload_folder`
- The model card (`README.md` on the Hub) must include:
  - Base model name and version (`Qwen/Qwen2.5-Coder-7B`)
  - Training data description (15K PostgreSQL text-to-SQL examples, synthetic + curated)
  - Training configuration: LoRA rank, alpha, target modules, QLoRA settings, LR, epochs, batch size
  - Prompt format (exact format the model expects — schema + question → SQL)
  - Evaluation results table: exec success %, exec correctness %, exact match % on 100-example held-out test
  - Known limitations: which SQL patterns the model struggles with (from your Week 39 error analysis)
  - License: apache-2.0 (matching Qwen's license)
- Verify the adapter loads in a fresh Python session:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_use_double_quant=True)
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-7B",
                                             quantization_config=bnb, device_map="auto")
model = PeftModel.from_pretrained(base, "<your-handle>/postgres-sqlcoder-7b-v1")
# Generate one test SQL to confirm it works
```

**Deliverable:** `<your-handle>/postgres-sqlcoder-7b-v1` live on HuggingFace with a complete model card; paste the Hub URL into `week40_gate_checklist.md`

---

## Task 3 — Push the Domain Dataset

**Goal:** Make your 15K dataset permanently accessible for Phase 5 training reruns.

**Requirements:**

- Upload `train_15k.jsonl` and `val_500.jsonl` as a HuggingFace Dataset: `<your-handle>/postgres-text2sql-15k`
- Keep `held_out_test.json` in your GitHub repo only — do NOT publish it on the Hub (it is your private benchmark)
- Add a dataset card describing: source of examples, generation method (Week 37 synthetic pipeline), schema coverage, split sizes
- Verify the dataset loads:

```python
from datasets import load_dataset
ds = load_dataset("<your-handle>/postgres-text2sql-15k")
print(ds)  # Should show train: 14500, validation: 500
```

**Deliverable:** Dataset live on HuggingFace Hub; URL in `week40_gate_checklist.md`

---

## Task 4 — Write a Phase 4 Retrospective

**Goal:** Consolidate what you learned. Writing forces clarity in a way that passive review does not.

**Requirements:**

- Write `week40_retrospective.md` (400–600 words) covering:
  - The 3 most important things you learned in Phase 4 (be specific — not "I learned about LoRA" but "I learned that alpha controls the effective learning rate of LoRA updates and setting alpha = 2 × rank is a reliable default")
  - The 3 things that surprised you most (unexpected results, debugging surprises, concepts that were harder or easier than expected)
  - The 3 areas where you still feel uncertain and want to reinforce in Phase 5
  - Your current exec correctness number and your target by end of Phase 5

**Deliverable:** `week40_retrospective.md` committed to your GitHub repo

---

## Stretch Goals

- Run inference-time majority voting: for each held-out test example, generate 5 SQL candidates with temperature=0.7, execute all 5, and pick the majority result set. Report how much this improves exec correctness over greedy decoding — this is a preview of Phase 5 inference-time search.
- Create a simple Gradio demo that takes a schema + question and returns the model's SQL with exec correctness feedback from a live Postgres instance.
- Compare your model against a GPT-4o zero-shot baseline on the 100 held-out examples (requires OpenAI API key).
