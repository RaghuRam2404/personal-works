# Week 4 — Assignment

## Setup Checklist

- [ ] Spider SQL corpus from Week 2 available: `week_02/sql_corpus.txt`. If not, re-run `extract_sql_tokens.py`.
- [ ] For character-level LSTM, you need the raw SQL query strings (not token sequences). Extract them: load `train_spider.json`, take the `query` field, concatenate with newlines → `week_04/sql_queries.txt`.
- [ ] Colab Free tier available. Training time: ~30 minutes on T4.
- [ ] W&B project `week-04-char-lstm` created.

---

## Task 1 — RNN Cell from Scratch

**Goal:** Prove you understand the RNN update equation by implementing it without `nn.RNN`.

**Requirements:**
- Create `week_04/rnn_scratch.py`.
- Implement a `VanillaRNNCell` class with:
  - Parameters: `W_hh` (hidden→hidden), `W_xh` (input→hidden), `b_h` (bias).
  - `forward(x, h_prev)` → `h_new` using the formula `tanh(x @ W_xh + h_prev @ W_hh + b_h)`.
- Implement an `RNNModel` that uses your `VanillaRNNCell` to process a batch of sequences.
- Verify correctness by checking that your cell output matches `nn.RNNCell` for the same input and weights.
- Train this model on a tiny sequence prediction task (predict the next character in "hello world" repeated 100 times) for 200 steps. Print the loss curve.

**Deliverable:** `week_04/rnn_scratch.py` with the verification assertion and a printed loss curve.

---

## Task 2 — LSTM Cell from Scratch

**Goal:** Implement the full LSTM equations and verify against PyTorch.

**Requirements:**
- Create `week_04/lstm_scratch.py`.
- Implement an `LSTMCell` class with all four gate computations:
  - `f_t`, `i_t`, `g_t`, `o_t` as described in the curriculum.
  - Output both `h_t` and `c_t`.
- Verify: for a random input, your cell's `(h, c)` must match `nn.LSTMCell`'s output given the same weights (you will need to manually copy weights from `nn.LSTMCell` to your cell for comparison).
- In a comment block at the top of the file, write 3–4 sentences explaining: what does the forget gate do, and what happens to the model's behavior if `f_t` is always exactly 0? What if it's always exactly 1?

**Deliverable:** `week_04/lstm_scratch.py` with the verification assertion passing.

---

## Task 3 — Character-Level LSTM on SQL Queries

**Goal:** Train a character-level LSTM to generate SQL-like text.

**Requirements:**
- Work in `week_04/char_lstm_sql.py` (or a Colab notebook, saved to `week_04/char_lstm_sql.ipynb`).
- Dataset: use `week_04/sql_queries.txt` — raw SQL query strings from Spider.
- Build a character vocabulary from the corpus (expect ~60–80 unique characters including letters, digits, `*`, `(`, `)`, `,`, space, newline, etc.).
- Model architecture: 1 or 2-layer LSTM, hidden size 256, embedding size 64.
- Training:
  - Sequence length (TBPTT chunk size): 64 characters.
  - Batch size: 32.
  - Use teacher forcing during training.
  - Use `detach()` correctly between chunks.
  - Train for at least 5000 steps.
  - Log to W&B project `week-04-char-lstm`: `train/loss` every 100 steps.
- After training, generate 5 SQL samples of 200 characters each using temperature 0.8. Save them in `week_04/sql_samples.txt`.
- Achieve training loss < 1.5 nats by the end of training.

**Deliverable:** Training script, `week_04/sql_samples.txt` with 5 samples, W&B run link. Commit message: `week-04-char-lstm`.

**Hints:**
- `nn.LSTM` returns `output, (h_n, c_n)`. `output` has shape `(seq_len, batch, hidden_size)`. `h_n` and `c_n` have shape `(num_layers, batch, hidden_size)`.
- For truncated BPTT, detach the hidden state tuple: `(h.detach(), c.detach())`.
- Temperature sampling: `probs = F.softmax(logits / temperature, dim=-1)`, then `torch.multinomial(probs, 1)`.
- Expected samples: mostly garbage, but with recognizable SQL patterns (`SELECT`, `FROM`, `WHERE` appearing) and correct structure some of the time. See the solution file for example outputs.

---

## Task 4 — Written Reflection

**Goal:** Consolidate understanding of why we moved beyond RNNs.

**Requirements:**
- In `week_04/reflection.md`, write ~300 words addressing:
  1. What is the fundamental computational bottleneck in training an LSTM on sequences of length 10,000?
  2. How does attention (as you understand it theoretically from your pre-course knowledge) address this bottleneck?
  3. In the SQL generation task, why might a char-level LSTM struggle to generate valid SQL clauses more than a token-level model?

**Deliverable:** `week_04/reflection.md`.

---

## Stretch Goals

- Add gradient clipping (`torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)`) to the training loop. Does it help stability? Log the gradient norm before and after clipping to W&B.
- Implement scheduled sampling: with probability `p` (linearly increasing from 0 to 0.5 over training), feed the model's own prediction instead of the ground truth at each step. Does it improve sample quality?
- Train a GRU (`nn.GRU`) with the same hyperparameters. Compare loss curves and sample quality. Which trains faster?
