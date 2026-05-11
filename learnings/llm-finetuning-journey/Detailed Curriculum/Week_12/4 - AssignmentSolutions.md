# Week 12 Assignment Solutions

## Task 1 — RMSNorm

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: [..., dim]
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight
```

**Expected:** Output shape matches input shape. `weight` has gradients during backward.

**Common gotchas:**
- Using `x.norm(2, dim=-1)` instead of `sqrt(mean(x^2))` — norm divides by `sqrt(dim)` implicitly, not what RMSNorm computes.
- Forgetting `keepdim=True` in the mean — causes broadcasting error.
- Adding eps inside the sqrt vs. outside: `sqrt(mean+eps)` is correct. `sqrt(mean)+eps` is slightly different numerically (both work but be consistent).

---

## Task 2 — SwiGLU

```python
class SwiGLUMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = int(8 / 3 * config.n_embd)
        hidden = ((hidden + 255) // 256) * 256  # round up to multiple of 256
        self.gate_proj = nn.Linear(config.n_embd, hidden, bias=False)
        self.up_proj   = nn.Linear(config.n_embd, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, config.n_embd, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

`F.silu` is SiLU / Swish: `x * sigmoid(x)`.

**Common gotchas:**
- Applying SiLU to `up_proj(x)` instead of `gate_proj(x)` — the gate is the one that gets the nonlinearity.
- Using `4 * n_embd` for hidden dim — gives 50% more FLOPs than intended.

---

## Task 3 — RoPE (LLaMA convention)

```python
def precompute_rope_freqs(dim, max_seq_len, base=10000, device='cpu'):
    # dim = d_k (per head)
    theta = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    positions = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(positions, theta)   # [T, dim//2]
    return freqs.cos(), freqs.sin()         # each [T, dim//2]

def apply_rotary_emb(q, k, cos, sin):
    # LLaMA convention: split in half
    # q: [B, H, T, D], cos/sin: [T, D//2]
    cos = cos.unsqueeze(0).unsqueeze(0)     # [1, 1, T, D//2]
    sin = sin.unsqueeze(0).unsqueeze(0)
    q1, q2 = q[..., :q.shape[-1]//2], q[..., q.shape[-1]//2:]
    k1, k2 = k[..., :k.shape[-1]//2], k[..., k.shape[-1]//2:]
    q_rot = torch.cat([q1*cos - q2*sin, q1*sin + q2*cos], dim=-1)
    k_rot = torch.cat([k1*cos - k2*sin, k1*sin + k2*cos], dim=-1)
    return q_rot, k_rot
```

**Common gotchas:**
- Applying RoPE to V as well — RoPE only goes on Q and K.
- Not slicing `cos[:T], sin[:T]` when T < max_seq_len — causes shape mismatch.
- Computing freqs over `range(dim)` instead of `range(0, dim, 2)` — duplicates frequencies.

---

## Task 5 — Expected Benchmark Numbers

On a 256-dim, 4-layer, 4-head model trained on SQL for 3000 steps (batch=32, block=128):

| | Val Loss | Params |
|---|---|---|
| Baseline (LN, ReLU/GELU, abs PE, MHA) | ~1.35 | ~4.2M |
| Modernized (RMSNorm, SwiGLU, RoPE, GQA n_kv=2) | ~1.28 | ~4.1M |

The modernized model typically converges slightly faster and achieves marginally better val loss. The improvement is small at toy scale — the benefits compound at 7B+.

---

## How to Verify You Did It Right

1. `RMSNorm`: `x = torch.randn(2,10,256); norm = RMSNorm(256); assert norm(x).shape == x.shape`
2. `SwiGLU`: check that `gate_proj` and `up_proj` have identical shapes; `down_proj` input dim matches hidden.
3. `RoPE`: generate 5 tokens; check that `apply_rotary_emb` returns Q,K with the same shape as inputs.
4. `GQA`: `keys.shape[1] == n_kv_heads` before repeat; `keys.shape[1] == n_heads` after.
5. `model.lm_head.weight.data_ptr() == model.transformer.wte.weight.data_ptr()` still True after modernization.
