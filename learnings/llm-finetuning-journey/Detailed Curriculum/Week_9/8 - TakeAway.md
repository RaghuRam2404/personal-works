# Week 9 TakeAway — Bahdanau Attention

**This week in 15 words:** Attention lets the decoder dynamically query all encoder states instead of one fixed vector.

---

## Key Formula

```
# Score for each source position j, at decoder step i
e_{i,j} = v_a^T * tanh(W_a * s_{i-1} + U_a * h_j)

# Normalize
alpha_{i,j} = softmax(e_{i,j})   # sum over j = 1

# Context vector
c_i = sum_j( alpha_{i,j} * h_j )
```

---

## Key Code Pattern

```python
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W_a = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.U_a = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        self.v_a = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, dec_hidden, enc_outputs, mask=None):
        # dec_hidden: [B, H], enc_outputs: [B, T, 2H]
        scores = self.v_a(torch.tanh(
            self.W_a(dec_hidden).unsqueeze(1) + self.U_a(enc_outputs)
        )).squeeze(-1)                           # [B, T]
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        weights = F.softmax(scores, dim=-1)      # [B, T]
        context = torch.bmm(weights.unsqueeze(1), enc_outputs).squeeze(1)
        return context, weights
```

---

## Decision Rules

- If attention heatmap is uniform: gradients are not flowing through attention — check softmax dim and context vector is in computation graph.
- If training loss is low but inference quality is poor: exposure bias from 100% teacher forcing — use scheduled sampling.
- If loss diverges: clip gradients to norm 1.0 (`clip_grad_norm_`).
- Padding in source: always apply a mask before softmax.

---

## Numbers to Remember

- Teacher forcing ratio start: 0.5 (good default)
- Gradient clipping norm: 1.0
- Hidden dim for this toy task: 128 (encoder), decoder initialized from encoder final state

---

## Red Flags During Training

- Loss stuck at log(vocab_size) ≈ 3.37 after 5 epochs → model is predicting uniform distribution; attention likely not connected.
- Attention heatmap is all one row or all one column → softmax on wrong dimension.
- Val accuracy never exceeds 50% on string reversal → check that target input starts with `<SOS>` and target loss is computed on all tokens except `<SOS>`.
