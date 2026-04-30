# Week 75 Assignment Solutions

## Task 1: Chat Template Verification

The template verification script catches the most common error — training with the wrong stop token or the wrong turn structure:

```python
from transformers import AutoTokenizer
import json

MODELS = {
    "r1distill": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "llama31":   "meta-llama/Llama-3.1-8B-Instruct",
    "gemma2":    "google/gemma-2-9b-it",
}

SYSTEM = "You are a PostgreSQL SQL expert. Output only valid SQL. No explanation."

def make_example(schema, question, sql, model_name):
    tok = AutoTokenizer.from_pretrained(model_name)
    msgs = [
        {"role": "system",    "content": SYSTEM},
        {"role": "user",      "content": f"### Schema\n{schema}\n### Question\n{question}\n### SQL Query"},
        {"role": "assistant", "content": sql},
    ]
    # tokenize=False gives the string; add_generation_prompt=False includes assistant turn
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)

ds = load_from_disk("data/sqlcoder_v3_train")
sample = ds[0]

for name, model_id in MODELS.items():
    text = make_example(sample["schema"], sample["question"], sample["sql"], model_id)
    print(f"\n=== {name} ===")
    print(text[:500])  # first 500 chars
    print("---")
```

What to look for in the output:
- R1-Distill-Qwen: `<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\nSELECT...<|im_end|>`
- Llama 3.1: `<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n...<|eot_id|>`
- Gemma 2: `<bos><start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\nSELECT...<end_of_turn>`

## Task 2: SFT with Unsloth — Template-Aware Data Preparation

```python
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    max_seq_length=2048,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model, r=64, lora_alpha=128,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
)

# Unsloth auto-detects the correct chat template for known models
# For R1-Distill-Qwen-7B, it uses the ChatML template (same as Qwen2.5)
# For Llama 3.1, it uses the Llama 3 template automatically
tokenizer = get_chat_template(tokenizer, chat_template="chatml")  # or "llama-3"
```

## Task 3: Evaluation Script

```python
import json, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def evaluate_model(model_path, benchmark_path, temperature=0.1):
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    benchmark = json.load(open(benchmark_path))

    correct = 0
    latencies = []
    for ex in benchmark:
        prompt = build_prompt(ex["schema"], ex["question"])  # your template
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=512, temperature=temperature, do_sample=False)
        latencies.append(time.perf_counter() - t0)
        generated = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        if normalize_sql(generated) == normalize_sql(ex["sql"]):
            correct += 1

    return {
        "accuracy": correct / len(benchmark),
        "avg_latency": sum(latencies) / len(latencies),
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
    }
```

## Task 4: Decision Memo Template

```markdown
## Model Selection Decision — Week 75

Winner: [Model name] with [X]% on Custom-200 vs [Y]% for existing Qwen2.5 model.

Margin: [X - Y] pp.

Decision: [Switch / Keep existing]

If switching (margin ≥ 2 pp):
- Quantizing to Q4_K_M GGUF and pushing as postgres-sqlcoder-v2
- Using as base for Weeks 76–77
- Biggest risk: Weeks 76–77 may introduce bugs attributed to the new model
  rather than the multi-turn/bilingual changes; mitigate by keeping a Qwen2.5
  baseline for regression testing

If keeping (margin < 2 pp):
- Within statistical uncertainty of the 200-example benchmark (±5.3 pp CI)
- Switching models is not justified; continue with existing postgres-sqlcoder-7b
```

## Common Gotchas

- Llama 3.1's chat template requires the system prompt to be inside `<|start_header_id|>system<|end_header_id|>` — do not use the Qwen ChatML format.
- R1-Distill-Qwen-7B was trained from Qwen2.5-Math (not Coder) — its initial SQL accuracy may be slightly lower than your Qwen2.5-Coder baseline, but it may catch up or exceed after SFT.
- Gemma 2's sliding window attention uses alternating local (4096-token) and global attention layers — long prompts that exceed 4096 tokens will only have global attention on alternate layers. Test with your largest schema before committing to Gemma 2.
- W&B hyperparam logging: if you use the same project name across all runs, use the `group` parameter to organize by model: `wandb.init(project="week75", group="llama31")`.

## How to Verify You Did It Right

After SFT on each candidate, generate one SQL query with the raw model using the same inference script as your eval harness. The output should be clean SQL with no chat formatting artifacts (no `<|im_end|>` or `<end_of_turn>` in the middle of the SQL). If you see stop tokens in the SQL output, the tokenizer's `skip_special_tokens=True` flag is missing or the model was not trained with the correct stop configuration.
