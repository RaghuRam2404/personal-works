# Week 78 Assignment Solutions — Final Phase 6 Gate and Course Wrap

## Task 1 — Audit Table Template

The audit itself cannot be provided here — it depends on your actual state of completion. What is provided is the template and the decision logic.

```markdown
# Capstone Deliverables Audit — Week 78

| Deliverable | Status | URL / Path | Notes |
|---|---|---|---|
| postgres-sqlcoder-7b-final (HF Hub) | Done | https://huggingface.co/... | Public ✓ |
| GGUF Q4_K_M | Done | https://huggingface.co/.../gguf | |
| GGUF Q5_K_M | Done | https://huggingface.co/.../gguf | |
| GPTQ INT4 | Missing | — | Will upload by [date] |
| Technical report PDF | In Progress | — | arXiv submission pending |
| Custom-200 eval script | Done | github.com/.../eval.py | |
| Training scripts (CPT, SFT, DPO, GRPO) | Done | github.com/... | |
| Ollama Modelfile | Done | local | Needs upload to GitHub |
| Model card | In Progress | — | This week |
| Blog post | Missing | — | Will publish by [date] |
```

**Priority rule:** If an item in the "Model artifacts" or "Evaluation" rows is Missing, complete it before moving to blog post or retrospective. Artifacts are irreversible commitments; documentation can be added incrementally.

---

## Task 2 — Model Card Key Snippet

```markdown
---
language: en
tags:
- text-to-sql
- postgresql
- timescaledb
- fine-tuning
- lora
license: apache-2.0
base_model: Qwen/Qwen2.5-Coder-7B-Instruct
---

# postgres-sqlcoder-7b-final

A 7B parameter NL→SQL model fine-tuned for PostgreSQL and TimescaleDB queries,
outperforming GPT-4o on the Custom-200 TimescaleDB benchmark.

## Benchmark Results

| Benchmark | Metric | This model | GPT-4o | Claude 3.5 Sonnet |
|---|---|---|---|---|
| Custom-200 (TimescaleDB) | EM | 83.1% | 79.4% | 81.2% |
| BIRD-SQL dev | EX | 68.4% | — | — |
| Spider 1.0 | EM | 82.7% | — | — |

## Usage

\`\`\`python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "<handle>/postgres-sqlcoder-7b-final"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16,
                                              device_map="auto")

schema = "orders(id INT, region TEXT, amount NUMERIC, created_at TIMESTAMPTZ)"
question = "Show total amount by region for the last 30 days."

messages = [
    {"role": "system", "content": "Generate valid PostgreSQL SQL. Schema:\n" + schema},
    {"role": "user", "content": question}
]
input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
output = model.generate(input_ids, max_new_tokens=256, temperature=0.0)
print(tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True))
\`\`\`

## Limitations
- Specialized for PostgreSQL and TimescaleDB; not evaluated on MySQL or SQLite.
- Performance degrades on schemas with > 20 tables (context length constraint).
- Not evaluated on adversarial inputs or injection attacks.
```

---

## Task 3 — Final Evaluation Script Key Snippet

```python
import json
from datetime import datetime
from huggingface_hub import snapshot_download

# Load from Hub (verifies upload)
model_id = "<handle>/postgres-sqlcoder-7b-final"
model_path = snapshot_download(model_id)

# Run evaluation (reuse your existing eval script)
results = run_custom200_eval(model_path, benchmark_path="data/custom200.jsonl")

# Log and save
output = {
    "custom200_em": results["em"],
    "model": model_id,
    "checkpoint": "final",
    "date": datetime.now().isoformat(),
    "num_examples": results["n"],
}
with open("week78/final_eval_results.json", "w") as f:
    json.dump(output, f, indent=2)
print(json.dumps(output, indent=2))
```

---

## Common Gotchas

- **HuggingFace upload fails silently:** Use `HfApi().repo_info(repo_id=...)` to verify the upload completed. Check that `model.safetensors` and `config.json` are both present.
- **Model card metadata not rendering:** The YAML frontmatter must be at the very top of the file with no blank line before `---`. Invalid YAML syntax causes the Hub to ignore all metadata.
- **Usage snippet fails on Apple Silicon:** Add `device_map="mps"` instead of `"auto"` for local MPS testing; `"auto"` defaults to CPU if CUDA is not available.
- **arXiv endorsement timing:** The endorsement process takes 7–14 days. If you have not started, use HuggingFace Papers immediately and submit to arXiv when endorsement arrives.

---

## How to Verify You Did It Right

1. Open your HuggingFace model page in an incognito browser window (not logged in). If you can see the model card, download the model, and copy-paste the usage snippet — your artifact is externally accessible.
2. Run your final evaluation script using `snapshot_download` (loading from Hub, not local). Confirm the EM matches your reported result within 0.5%.
3. Share your blog post link with one person who does not know ML. If they can explain back to you "what you built and why it matters" after reading it, the writing is accessible enough.
4. Read your retrospective one week later. If it still feels honest, you wrote it at the right level of depth.
