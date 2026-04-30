# Week 57 TakeAway — Continued Pretraining

**One-liner:** CPT = causal LM on raw domain text, exactly 1 epoch, with EOS between documents.

---

## Corpus Packing

```python
# Pack documents with EOS separators
all_ids = []
for doc in corpus:
    ids = tokenizer.encode(doc["text"], add_special_tokens=False)
    all_ids.extend(ids + [tokenizer.eos_token_id])

# Cut into fixed-length sequences
sequences = [all_ids[i:i+2048] for i in range(0, len(all_ids)-2048, 2048)]
```

## CPT Training Config (key differences from SFT)

```python
TrainingArguments(
    learning_rate=5e-5,          # lower than SFT's 2e-4
    num_train_epochs=1,          # NEVER more than 1
    gradient_accumulation_steps=8,  # large effective batch
    warmup_steps=100,            # longer warmup than SFT
    weight_decay=0.1,            # stronger than SFT
)
DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
# NOT DataCollatorForSeq2Seq — no masking, predict all tokens
```

## Forgetting Monitor

```python
# Log every 100 steps
ppl_domain = compute_perplexity(model, pg_held_out)
ppl_general = compute_perplexity(model, wiki_held_out)
wandb.log({"ppl_domain": ppl_domain, "ppl_general": ppl_general})
# Stop if ppl_general increases > 0.5 bits
```

---

## Decision Rules

- 1 epoch only — catastrophic forgetting starts in epoch 2
- Stop if Wikipedia perplexity increases > 0.5 bits/token
- Target corpus composition: > 25% TimescaleDB-specific, < 60% StackOverflow
- Full bf16 on H100 better than 4-bit LoRA for CPT quality
- Always insert EOS between documents during packing
- TERMINATE RunPod instance immediately after training

---

## Numbers to Remember

- CPT learning rate: 5e-5 (3× lower than SFT LoRA learning rate)
- Effective batch size: 64 (8 per device × 8 gradient accumulation)
- 100M tokens at H100 ≈ 763 steps ≈ 1.5–2 hours
- RunPod H100 80GB ≈ $2.79/hr → CPT costs ~$4–6
- Domain perplexity target: ≥ 10% decrease vs. base model
- General perplexity guard: < 0.5 bits increase

---

## Red Flags

- Loss stuck above 2.5: corpus has too much boilerplate or broken encoding
- Wikipedia perplexity rises > 0.5 bits: stop immediately, revert to prior checkpoint
- Packing efficiency < 80%: too many very short documents — filter them out
- RunPod instance still running after training: terminate immediately
