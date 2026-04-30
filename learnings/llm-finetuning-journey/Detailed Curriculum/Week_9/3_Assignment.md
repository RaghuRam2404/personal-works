# Week 9 Assignment — Bahdanau Attention from Scratch

## Setup Checklist

- [ ] PyTorch installed (any version ≥ 2.0)
- [ ] `matplotlib` installed for attention heatmap plotting
- [ ] GitHub repo initialized with a `week-09-bahdanau-attn` branch
- [ ] W&B account set up (free tier) — project name `week-09-bahdanau-attn`

No GPU required. This runs comfortably on Mac CPU in under 5 minutes.

---

## Task 1 — Implement Bahdanau Attention Module

**Goal:** Build a standalone `BahdanauAttention` module that takes decoder hidden state and all encoder hidden states and returns a context vector plus attention weights.

**Requirements:**
- Class `BahdanauAttention(nn.Module)` with parameters `hidden_dim`
- Learnable: `W_a` (maps decoder hidden → hidden_dim), `U_a` (maps encoder hidden → hidden_dim), `v_a` (maps hidden_dim → scalar)
- Forward signature: `forward(decoder_hidden, encoder_outputs, src_mask=None) -> (context, attn_weights)`
  - `decoder_hidden`: shape `[batch, hidden_dim]`
  - `encoder_outputs`: shape `[batch, src_len, hidden_dim * 2]` (bidirectional encoder)
  - `src_mask`: shape `[batch, src_len]`, bool tensor — True where token is padding
  - Returns `context` of shape `[batch, hidden_dim * 2]` and `attn_weights` of shape `[batch, src_len]`
- Apply `src_mask` by setting masked scores to `-1e9` before softmax
- Unit test: for a batch of 2 with `src_len=5`, assert `attn_weights.sum(dim=-1)` is all-ones (within 1e-5)

**Deliverable:** `attention.py` containing the module and a `if __name__ == "__main__":` block running the unit test.

---

## Task 2 — Build the Seq2Seq + Attention Model

**Goal:** A full seq2seq model (encoder + attention + decoder) that you train to reverse short character sequences.

**Requirements:**
- `Encoder`: bidirectional GRU (not LSTM, simpler to implement), returns all hidden states per time step plus a final hidden state
- `Decoder`: unidirectional GRU that at each step: receives prev token embedding + context vector from attention
- Vocabulary: 26 lowercase letters + `<SOS>`, `<EOS>`, `<PAD>` (total 29 tokens)
- Training data: generate 5000 random strings of length 4–12 characters. Target is the reversed string.
- Batch size: 64, hidden dim: 128, embedding dim: 64
- Train for at least 30 epochs with teacher forcing ratio 0.5
- Log train loss to W&B at every epoch
- Acceptance check: val accuracy (exact-match, character-level) must exceed 95% on 500 held-out examples

**Hints:**
- When initializing decoder hidden state from encoder: take the forward final hidden state concatenated with backward final hidden state (or just the forward — explain your choice in a comment)
- Teacher forcing: with probability `p`, feed the ground truth previous token to the decoder instead of the predicted token. Set `p = 0.5` during training, `p = 0.0` during inference.
- Pack padded sequences with `nn.utils.rnn.pack_padded_sequence` for efficiency, but this is optional for sequences this short.

**Deliverable:** `seq2seq.py` and a W&B run showing loss curve.

---

## Task 3 — Attention Heatmap Visualization

**Goal:** Generate and save attention weight heatmaps for 4 example reversals.

**Requirements:**
- After training, run inference (no teacher forcing) on 4 manually chosen strings (pick strings of length 6, 8, 10, 12)
- For each string: collect the `attn_weights` at every decoder step — this gives a matrix of shape `[target_len, src_len]`
- Plot this matrix as a heatmap using `matplotlib.pyplot.imshow` with `cmap='Blues'`
- X-axis: source characters, Y-axis: target characters
- Title: "Input: {src_str} | Output: {pred_str}"
- Save as `attention_heatmap.png`
- The heatmap should show a clear anti-diagonal pattern (attending to position `T-i` when generating position `i`)

**Deliverable:** `attention_heatmap.png` committed to the repo.

---

## Stretch Goals

- **Luong comparison:** Implement Luong dot-product attention alongside Bahdanau. Train both and compare accuracy and attention sharpness.
- **SQL reversal:** Create a mini "SQL keyword shuffler" dataset (e.g., "SELECT name FROM users" → "users FROM name SELECT") and train your model on it. Does attention learn SQL-meaningful alignment?
- **Beam search:** Implement beam search with beam width 3 for the decoder. Does it improve exact-match accuracy over greedy decoding?
