# Week 4 — Quiz

---

**Q1.** In a vanilla RNN processing a SQL query of 500 characters, the gradient of the loss at the final character with respect to the hidden state at character 1 involves a product of 499 Jacobian matrices. What is the most likely outcome and why?

A) The gradient is large — recurrent connections amplify signals over time.
B) The gradient is approximately 0 — repeated multiplication by the recurrent weight matrix (with spectral radius < 1) causes exponential decay.
C) The gradient is exactly 1 — tanh is a contracting map that stabilizes gradients.
D) The gradient oscillates — it is positive and negative alternately across time.

---

**Q2.** You forget to call `.detach()` on the hidden state between TBPTT chunks. What are the consequences?

A) The model trains correctly but more slowly.
B) PyTorch raises an error because the graph has loops.
C) The backward pass traverses back through all previous chunks, making memory usage proportional to the full sequence length and gradients stale (from already-updated parameters).
D) The hidden state is reset to zero, breaking temporal continuity.

---

**Q3.** In PyTorch, `nn.LSTM` returns `output, (h_n, c_n)`. You call `nn.LSTM(64, 256, num_layers=2, batch_first=True)` on input of shape `(32, 100, 64)`. What is the shape of `h_n`?

A) `(32, 256)` — batch × hidden.
B) `(100, 32, 256)` — seq × batch × hidden.
C) `(2, 32, 256)` — num_layers × batch × hidden.
D) `(2, 256)` — num_layers × hidden.

---

**Q4.** The forget gate in an LSTM outputs a value near 0 for every time step. What does this mean for the cell state, and what is the practical consequence?

A) The cell state is preserved perfectly — forgetting nothing.
B) The cell state is completely erased at every step — the LSTM has no persistent memory and behaves like a vanilla RNN.
C) The input gate becomes active to compensate, storing more information.
D) The output gate controls the forget behavior when `f_t ≈ 0`.

---

**Q5.** Your character-level LSTM generates: `SELECT SeLeCt SeLeCt FROM frOM FROM`. What does this symptom most likely indicate?

A) The model is overfitting to the training data.
B) The temperature during sampling is too high (near or above 1.5), causing the model to sample from a nearly uniform distribution and not exploit its learned patterns.
C) The embedding dimension is too small.
D) Teacher forcing is not being applied correctly.

---

**Q6.** Why can't you simply train a 2-layer LSTM on sequences of 50,000 characters without any form of TBPTT?

A) PyTorch's `nn.LSTM` has a built-in sequence length limit of 1,000.
B) The backward pass must store all intermediate hidden states and activations for all 50,000 steps, which would exceed available GPU memory and requires O(T) time to compute.
C) Teacher forcing becomes unstable beyond 10,000 characters.
D) The forget gate saturates and stops learning after 1,000 steps.

---

**Q7 (short answer).** Explain the train/inference distribution mismatch caused by teacher forcing, and describe how this would manifest specifically when generating a 20-token SQL clause like `SELECT t1.id, t2.name FROM table1`.

---

**Q8 (short answer).** Give two concrete reasons why transformers replaced LSTMs as the dominant architecture for sequence modeling, tied to specific architectural properties of each.
