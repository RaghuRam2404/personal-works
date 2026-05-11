# Week 58 Assignment Solutions

## Task 2 — Completion-Only Collator Pattern

```python
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

# For Qwen2.5 ChatML format, the assistant response starts with:
response_template = "<|im_start|>assistant\n"

collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tokenizer,
    mlm=False,
)

# Apply chat template to dataset
def format_example(example):
    return {"text": tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )}

dataset = dataset.map(format_example)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    data_collator=collator,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=2048,
)
```

## Quick Domain Eval (Run Every 500 Steps via Callback)

```python
from transformers import TrainerCallback
import psycopg2

class DomainEvalCallback(TrainerCallback):
    def __init__(self, eval_examples, conn):
        self.eval_examples = eval_examples[:20]  # quick: 20 examples
        self.conn = conn
    
    def on_evaluate(self, args, state, control, model, **kwargs):
        correct = 0
        for ex in self.eval_examples:
            prompt = tokenizer.apply_chat_template(ex["messages"][:-1],
                tokenize=False, add_generation_prompt=True)
            ids = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=300, do_sample=False)
            pred_sql = tokenizer.decode(out[0][ids.input_ids.shape[1]:],
                                       skip_special_tokens=True)
            cur = self.conn.cursor()
            try:
                cur.execute(f"BEGIN; EXPLAIN {pred_sql}; ROLLBACK;")
                self.conn.rollback()
                correct += 1
            except:
                self.conn.rollback()
        wandb.log({"domain_exec_accuracy": correct / len(self.eval_examples)},
                  step=state.global_step)
```

---

## Common Gotchas

- **Loss stays high after warmup (> 2.0).** Usually means the chat template is misconfigured and the model is computing loss on user/system tokens too (DataCollatorForCompletionOnlyLM not applied). Verify by checking a batch: `collator.tokenize_row(sample)` should show -100 labels for non-assistant tokens.
- **CUDA OOM at batch size 4.** Reduce to batch size 2 + accumulation 16. If still OOM, use 4-bit loading (load_in_4bit=True), but bf16 is preferred for quality.
- **Training loss decreases but domain_exec_accuracy is flat.** The model is learning the instruction-following format but not the SQL correctness. Check: are your eval examples in the same format as training? Are they from the same domain?
- **Eval loss increases after 1 epoch.** Lower the learning rate to 1e-4 and set `load_best_model_at_end=True`. The best checkpoint is at epoch 1 boundary.

---

## How to Verify You Did It Right

1. `sft_eval_results.md` shows SFT-v3 outperforms Phase 5 GRPO on custom benchmark (or is close, with further improvement expected from DPO/GRPO in Weeks 59–60)
2. W&B run shows smooth training loss curve: starts ~1.5–2.0, ends ~0.8–1.2
3. Completion-only collator is active: training loss should be notably lower than if you trained on all tokens (all-token loss would be ~0.4–0.6 higher)
4. `domain_exec_accuracy` at end of training is at least 5 percentage points higher than the Phase 5 baseline
5. HuggingFace model page has a README with training details (dataset version, hyperparameters, eval results)
