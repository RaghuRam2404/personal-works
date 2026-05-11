# Week 9 Assignment Solutions

## Task 1 — Key Snippet: BahdanauAttention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # decoder hidden is [batch, hidden_dim], encoder is [batch, src_len, hidden_dim*2]
        self.W_a = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.U_a = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        self.v_a = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, src_mask=None):
        # decoder_hidden: [batch, hidden_dim] -> [batch, 1, hidden_dim]
        dec = self.W_a(decoder_hidden).unsqueeze(1)
        # encoder_outputs: [batch, src_len, hidden_dim*2]
        enc = self.U_a(encoder_outputs)              # [batch, src_len, hidden_dim]
        scores = self.v_a(torch.tanh(dec + enc))     # [batch, src_len, 1]
        scores = scores.squeeze(-1)                  # [batch, src_len]
        if src_mask is not None:
            scores = scores.masked_fill(src_mask, -1e9)
        attn_weights = F.softmax(scores, dim=-1)     # [batch, src_len]
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)                 # [batch, hidden_dim*2]
        return context, attn_weights
```

The critical shape to track: `dec + enc` broadcasts `[batch,1,hidden_dim]` against `[batch,src_len,hidden_dim]` correctly.

**Expected output (unit test):** `attn_weights.sum(dim=-1)` → tensor of all ones (exactly).

**Common gotchas:**
- Passing `hidden_dim*2` to `W_a` instead of `hidden_dim` — the decoder hidden is unidirectional, it's `hidden_dim`, not `hidden_dim*2`.
- Forgetting `.squeeze(-1)` on scores — leaves a trailing dim-1 that silently broadcasts wrong.
- Applying softmax on `dim=0` or `dim=1` — must be `dim=-1` (across src positions).
- Not setting `masked_fill` before softmax, only after — this gives NaN gradients.
- Using `encoder_outputs[:, -1, :]` as context instead of the weighted sum — defeats the entire purpose.

---

## Task 2 — Key Training Loop Snippet

```python
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for src, tgt, src_lens in dataloader:
        optimizer.zero_grad()
        # encoder
        enc_outputs, enc_hidden = encoder(src, src_lens)
        # decoder init
        dec_hidden = enc_hidden  # [1, batch, hidden_dim]
        dec_input = tgt[:, 0]   # <SOS> token
        loss = 0
        for t in range(1, tgt.shape[1]):
            context, _ = attention(dec_hidden.squeeze(0), enc_outputs)
            dec_out, dec_hidden = decoder(dec_input, dec_hidden, context)
            loss += criterion(dec_out, tgt[:, t])
            # teacher forcing
            use_teacher = torch.rand(1).item() < teacher_forcing_ratio
            dec_input = tgt[:, t] if use_teacher else dec_out.argmax(-1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
```

**Expected output:** Loss should drop from ~3.3 (random) to < 0.05 by epoch 30. Val exact-match accuracy > 95%.

**Common gotchas:**
- Not clipping gradients — loss will spike/diverge after a few epochs.
- Feeding `<EOS>` as the initial decoder input instead of `<SOS>`.
- Forgetting to convert encoder bidirectional hidden to decoder unidirectional: sum or concatenate the two directions.

---

## Task 3 — Collecting Attention Weights

```python
all_attn = []
dec_input = torch.tensor([[SOS_IDX]])
for t in range(max_len):
    context, attn = attention(dec_hidden.squeeze(0), enc_outputs)
    all_attn.append(attn.squeeze(0).detach().cpu())
    out, dec_hidden = decoder(dec_input, dec_hidden, context)
    dec_input = out.argmax(-1, keepdim=True)
    if dec_input.item() == EOS_IDX:
        break
attn_matrix = torch.stack(all_attn)  # [tgt_len, src_len]
```

**Expected heatmap:** Near-perfect anti-diagonal for the string reversal task. If it's noisy, your model may not have converged fully — train longer or reduce LR.

---

## How to Verify You Did It Right

1. `attn_weights.sum(dim=-1).allclose(torch.ones(...))` — must be True.
2. Inference on "abcdef" → predicted "fedcba" with >95% reliability.
3. Heatmap for "abcdef" shows the anti-diagonal: position 0 output attends to position 5 input, position 1 output attends to position 4 input, etc.
4. W&B loss curve is smooth and decreasing (with some noise from teacher forcing stochasticity).
