# Week 8 — Quiz (Phase 1 Gate Assessment)

This quiz covers all of Phase 1. It doubles as a diagnostic for the Phase Gate. If you cannot answer these without notes, identify which week to revisit.

---

**Q1.** You open a new Python file and write a training loop from scratch. Which of the following represents the correct order of operations within a single training step?

A) `forward → loss → backward → zero_grad → optimizer.step`
B) `zero_grad → forward → loss → backward → optimizer.step`
C) `forward → zero_grad → loss → backward → optimizer.step`
D) `zero_grad → forward → backward → loss → optimizer.step`

---

**Q2.** Your nanoGPT-style model has `n_layer=4, n_head=4, n_embd=128, block_size=128, vocab_size=80`. Approximately how many parameters does the model have? (You don't need an exact answer — identify the correct order of magnitude.)

A) ~80,000 (80K)
B) ~800,000 (800K)
C) ~8,000,000 (8M)
D) ~80,000,000 (80M)

---

**Q3.** You are looking at a W&B run. Train loss decreases from 3.5 to 1.8 over 5000 steps, but validation loss decreases from 3.5 to 1.9 for the first 2000 steps, then increases back to 2.4 by step 5000. What is happening, and what is the correct intervention?

A) The model is underfitting. Increase model size or train longer.
B) The model is overfitting after step 2000. Interventions: add dropout, increase weight decay, reduce model size, or get more training data.
C) The learning rate is too high. Reduce by 10× and restart.
D) The validation set is contaminated with training data. Check data pipeline.

---

**Q4.** Complete this statement: In AdamW, weight decay is applied _______, whereas in standard Adam with L2 regularization, weight decay is applied _______.

A) After the optimizer step / before the gradient computation.
B) Directly to the parameter (independent of gradient history) / to the gradient (before adaptive scaling divides it, which corrupts the regularization effect).
C) Only to bias parameters / to all parameters.
D) Per epoch / per step.

---

**Q5.** Your char-level transformer on Spider SQL generates: `SELECT t1.name FROM t2 WHERE t1.id = t2.id GROUP`. It stops after `GROUP`. Why might this happen, and how would you fix it?

A) `GROUP` is the most common final token. Fix: post-process outputs to add `BY`.
B) The model's context window is exhausted — the generated sequence reached `block_size` characters. Fix: increase `block_size` or use a smaller seed prompt.
C) The sampling temperature is too low. Fix: increase temperature.
D) The model has not learned to predict `BY` after `GROUP`. Fix: train for more steps.

---

**Q6.** In your Week 7 data pipeline, you set `labels[attention_mask == 0] = -100`. A colleague claims you also need to shift `labels` by one position to the right for causal LM training. Who is correct?

A) You are correct — HuggingFace's `AutoModelForCausalLM` handles the shift internally.
B) Your colleague is correct — you must manually shift `labels` by one position.
C) Both are required — you shift AND mask.
D) Neither is required — the model trains without any label manipulation.

---

**Q7.** Describe what BPE's byte-level foundation means for handling a SQL query containing a user-supplied Japanese comment: `-- ユーザー名`. Will this cause an error in a byte-level BPE tokenizer? Why or why not?

---

**Q8.** A junior engineer on your team writes:
```python
for step in range(1000):
    x, y = get_batch()
    loss = model(x, y)
    loss.backward()
    optimizer.step()
```
List every bug in this code and describe the consequence of each.

---

**Q9 (Capstone reflection — short answer).** After running your capstone project, you generate 10 SQL samples from your trained model. 8 of the 10 begin with `SELECT` and contain recognizable SQL keywords, but the table and column names are gibberish (e.g., `FROM t1_x3 WHERE col_g2 > 5`). Is this a sign that training failed? What would you need to fix this in a real SQL generation system?
