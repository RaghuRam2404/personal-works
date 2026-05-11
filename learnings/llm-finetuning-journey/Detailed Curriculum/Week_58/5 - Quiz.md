# Week 58 Quiz — Full SFT on v3

## Multiple Choice

**Q1.** You start SFT from your CPT checkpoint. After 50 warmup steps, the training loss is 1.95. If you had started from the base model (no CPT), the starting loss would have been approximately 2.4. What does this difference tell you?

A) The CPT checkpoint has memorized the SFT training data (contamination).
B) CPT has shifted the model's distribution closer to your domain, so the model needs fewer steps to fit the SFT data.
C) The CPT checkpoint is overtrained and is actually worse than the base model.
D) The difference is random variation; CPT has no effect on SFT initialization.

---

**Q2.** After 2 epochs of SFT, your training loss is 0.82 and your validation loss is 1.24. The gap is 0.42. What does this indicate?

A) The model is severely overfit; stop training and use a much smaller LoRA rank.
B) The gap is within normal range for SFT; validate by checking domain execution accuracy, not just loss.
C) The learning rate is too high; reduce by 10× and restart training.
D) The validation set is contaminated with training examples; rebuild the split.

---

**Q3.** You use `DataCollatorForCompletionOnlyLM` but accidentally set the response_template to the user role delimiter instead of the assistant role delimiter. What happens during training?

A) Training fails with a KeyError — the wrong template causes an exception.
B) The model computes loss on user question tokens instead of assistant SQL tokens, training in the wrong direction.
C) The collator falls back to computing loss on all tokens, effectively running standard causal LM.
D) The model's performance is identical — loss masking has negligible effect.

---

**Q4.** Your SFT model achieves 71% execution accuracy on your custom benchmark. Your Phase 5 GRPO model achieved 65%. You still have DPO (Week 59) and GRPO (Week 60) ahead. What is a realistic target for the final model after the full pipeline?

A) ~71% — SFT typically represents the ceiling; DPO and GRPO rarely add more than 1–2%.
B) ~78–85% — DPO and GRPO with good data and tuning typically add 7–14 percentage points over SFT.
C) ~95%+ — DPO and GRPO are the most powerful alignment methods and should achieve near-perfection.
D) Unknown — GRPO can either significantly improve or significantly regress the model.

---

## Short Answer

**Q5.** Your training loss curve is smooth but you notice that every 200 steps there is a small spike (+0.1) followed by immediate recovery. What is the likely cause? Is this a problem?

---

**Q6.** You plan to train for 2 epochs. At epoch 1 (781 steps), your domain execution accuracy is 71%. At epoch 2 (1,562 steps), it is 70%. Should you continue to epoch 3? What information do you need to decide?

---

**Q7.** Your SFT model performs better on simple JOIN queries (85% accuracy) than on TimescaleDB hyperfunctions (52% accuracy). You have 3 weeks before the DPO run. What actions can you take NOW to improve TimescaleDB hyperfunction accuracy without re-running the full SFT?

---

## Deep Scenario

**Q8.** Your 7B model after SFT-v3 achieves 71% on your custom benchmark. GPT-4o achieves 83% on the same benchmark. Diagnose what is likely causing the 12-point gap. Consider:
- Data coverage (what types of queries might be in the gap?)
- Model capacity (is 7B enough?)
- Training procedure (is SFT the right final step?)
- Evaluation methodology (are there confounds?)

Propose 3 specific interventions ranked by estimated impact, with a brief justification for each.
