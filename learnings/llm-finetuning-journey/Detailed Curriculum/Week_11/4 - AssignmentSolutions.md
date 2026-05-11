# Week 11 Assignment Solutions

## Task 1 — Key Snippets

### CausalSelfAttention

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # causal mask: register as buffer (not a parameter)
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size))
                                         .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))
```

### Weight Tying

```python
# In GPT.__init__:
self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
self.transformer.wte.weight = self.lm_head.weight  # tie
```

**Expected outputs:**
- Shakespeare val loss after 5000 iters: ~1.55–1.65
- Generated Shakespeare: coherent word structure, plausible style, not grammatically perfect
- SQL val loss: ~1.2–1.5 depending on corpus size
- Generated SQL: `SELECT id, name FROM orders WHERE user_id = 12 GROUP BY date`-style sequences

**Common gotchas:**
- Using `torch.zeros` for the causal mask buffer instead of `torch.tril(torch.ones(...))` — inverts the mask.
- Registering the causal mask as a parameter (`nn.Parameter`) instead of a buffer — causes it to be included in `model.parameters()` and updated by the optimizer.
- Not slicing `self.bias[:, :, :T, :T]` — fails on sequences shorter than `block_size`.
- Forgetting `model.eval()` before `model.generate(...)` — dropout stays active.
- Using integer `0` as `-inf` replacement — softmax of 0 is not zero; use `float('-inf')`.

---

## Task 2 — SQL Training Tips

The key is data preparation:
```python
# Combine queries with newline separator
text = "\n".join(sql_queries)
# Character-level tokenization
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
data = torch.tensor(encode(text), dtype=torch.long)
```

SQL character set is typically 60–75 characters. Val loss should reach 1.2–1.4. If you see val loss stuck at 4.0+, your training data didn't encode properly (check that `chars` has more than 10 unique characters).

---

## How to Verify You Did It Right

1. `model.generate(torch.zeros(1,1,dtype=torch.long), 200)` should produce 200 tokens of legible Shakespeare-ish text (not random characters).
2. Count unique tokens generated from the SQL model — should include SELECT, FROM, WHERE, JOIN, GROUP, ORDER within any 100-character window.
3. `model.lm_head.weight.data_ptr() == model.transformer.wte.weight.data_ptr()` → True (same memory, weight tying confirmed).
4. W&B shows val loss decreasing below train loss temporarily during warmup, then tracking together — expected behavior.
