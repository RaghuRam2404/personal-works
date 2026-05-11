# Week 20 Assignment Solutions

## Task 3 — Key Snippet: GPT Model

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).split(C, dim=2)
        q, k, v = [t.view(B, T, self.n_heads, self.d_head).transpose(1,2)
                   for t in qkv]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)

class MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4 * d_model, bias=False)
        self.fc2 = nn.Linear(4 * d_model, d_model, bias=False)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size=32000, context_len=1024,
                 d_model=768, n_heads=12, n_layers=8):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(context_len, d_model)
        self.blocks  = nn.ModuleList(
            [TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1))
        return logits, loss
```

**Expected parameter count:** `8 × (768² × 12 + ...) ≈ 56.7M` parameters.

**Common gotchas:**
- Forgetting to tie weights (`self.lm_head.weight = self.tok_emb.weight`) — run will appear correct but param count will be off by 24M
- Incorrect split in `qkv.split(C, dim=2)` — use `.split(C, dim=-1)` or `.chunk(3, dim=-1)` equivalently
- Positional embedding `torch.arange(T)` must be on the same device as `idx` — always add `device=idx.device`

---

## Task 4 — Key Snippet: LR Scheduler + Training Step

```python
import math

def get_lr(step, warmup_steps=100, max_steps=10000, max_lr=3e-4, min_lr=3e-5):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

# Inside training loop:
for step, (x, y) in enumerate(train_loader):
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    with accelerator.accumulate(model):
        logits, loss = model(x, y)
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
```

---

## How to Verify You Did It Right

**Initial loss check:**
```python
# At step 0, before any training:
assert 9.5 < initial_loss < 11.0, f"Initial loss {initial_loss} seems wrong"
# log(32000) = 10.37 — initial loss should be near this
```

**After 200 steps:**
- Loss should drop from ~10.4 to somewhere in 6–8 range
- If loss drops below 5.0 in 200 steps: your dataset is too small and the model is overfitting (or something is wrong)
- If loss does not drop below 9.0 in 200 steps: check that labels (targets) are the shifted inputs, not the same as inputs

**Red flags:**
- Loss stuck at exactly 10.37 for all 200 steps → model output is not connected to loss (check `loss.backward()` is being called)
- Loss oscillates wildly (9 → 3 → 11 → 4) → learning rate too high; try lr=3e-5 for sanity check
- CUDA out of memory → batch_size is too large; try batch_size=4 with gradient_accumulation=8
