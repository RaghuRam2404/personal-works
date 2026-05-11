# Week 28 TakeAway — Fine-Tuning Fundamentals

**One-liner:** Three ways to update a pretrained model; SFT is the first required stage for any task-specific deployment.

---

## The Three Methods at a Glance

| Method | Data | Loss | When |
|---|---|---|---|
| Continued pretraining | Raw text (unlabeled) | CLM (next-token) | Domain vocab missing |
| SFT | (input, output) pairs | CE on output tokens only | Task behavior needed |
| Instruction tuning | (instruction, response) | CE on response tokens only | General assistant |

---

## Post-Training Pipeline

```
Base → SFT → DPO → GRPO → Deploy
        ↑         ↑       ↑
     format    preference  verifiable
      + task    ranking     reward
```

---

## Key Numbers to Remember

- InstructGPT SFT stage: ~13K examples. Massive behavioral shift from tiny data.
- LoRA rank: weight updates are intrinsically low-rank (you will prove this Week 30)
- For SQL domain: Qwen2.5-Coder-7B already knows SQL; start with SFT, not continued pretraining
- 3 epochs on 5K examples with full SFT = overfitting risk — use 1–2 epochs

---

## Input Masking Pattern (SFT Loss)

```python
# In SFT, only response tokens contribute to loss
# Input tokens: forward pass context, but labels = -100 (ignored)
# SFTTrainer handles this automatically via dataset_text_field

labels = input_ids.clone()
labels[:, :prompt_length] = -100  # mask prompt tokens
loss = cross_entropy(logits, labels)
```

---

## Decision Rules

- If base model tokenizes your domain poorly → add continued pretraining first
- If model knows the domain but not the task format → SFT on 1K–10K examples
- If model generates correct format but wrong content → more data or DPO
- If you want binary reward optimization (SQL execution) → GRPO (Phase 5)

---

## Red Flags During Training

- Val loss rising while train loss falls → overfitting, stop early
- Model generates identical outputs for different inputs → mode collapse, reduce epochs
- Model loses general capability → catastrophic forgetting, lower LR or use LoRA
- Loss spikes mid-training → LR too high or corrupted batch
