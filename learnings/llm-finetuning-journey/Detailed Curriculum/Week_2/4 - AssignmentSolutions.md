# Week 2 — Assignment Solutions

## Task 1 — Key Snippets (MLP Language Model)

The core MLP block and training step:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters
block_size = 3   # context length
n_embd     = 10  # embedding dim
n_hidden   = 200
vocab_size  = 27  # 26 letters + '.'

class MakemoreMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.C  = nn.Embedding(vocab_size, n_embd)
        self.fc1 = nn.Linear(block_size * n_embd, n_hidden)
        self.bn1 = nn.BatchNorm1d(n_hidden)
        self.fc2 = nn.Linear(n_hidden, vocab_size)
        # Kaiming init for fc1
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')

    def forward(self, x):
        emb = self.C(x)                        # (B, block_size, n_embd)
        emb = emb.view(emb.size(0), -1)        # (B, block_size * n_embd)
        h   = F.relu(self.bn1(self.fc1(emb)))  # batch norm BEFORE activation
        return self.fc2(h)

# Training step
optimizer.zero_grad()
logits = model(Xb)
loss = F.cross_entropy(logits, Yb)
loss.backward()
optimizer.step()
```

**Expected output:** Val loss should reach ~2.17 after ~30k steps with LR schedule (start 0.1, decay to 0.01 around step 100k — see Karpathy's video for exact schedule).

**Common gotchas:**
- `emb.view(emb.size(0), -1)` — the batch size is `emb.size(0)`. Do not hardcode the batch size.
- `BatchNorm1d` expects input shape `(N, D)` or `(N, D, L)` — not 3D tensors from the embedding if you pass them directly. Flatten first.
- The final layer `fc2` should NOT have batch norm. Batch norm on logits distorts the probability distribution.
- If you use `model.eval()` for val loss but forget `model.train()` afterward, batch norm in training mode will use stale running stats and performance degrades.

---

## Task 2 — Key Snippets (SQL Token Extraction)

```python
import json, re

with open('spider/train_spider.json') as f:
    data = json.load(f)

queries = [ex['query'] for ex in data]

# Tokenize: split on non-alphanumeric, keep only alphabetic tokens
tokens_all = []
for q in queries:
    toks = re.findall(r'[A-Za-z]+', q)
    tokens_all.extend([t.lower() for t in toks])

vocab = sorted(set(tokens_all))
print(f"Vocab size: {len(vocab)}")   # expect ~50-120 unique SQL keywords

with open('week_02/sql_vocab.txt', 'w') as f:
    f.write('\n'.join(vocab))

with open('week_02/sql_corpus.txt', 'w') as f:
    f.write(' '.join(tokens_all))
```

**Expected output for 5 generated SQL samples:**
```
select count from where t table
select id name where join on
select max from group by having
from where id and t count order
select distinct count from where t
```
(Tokens will form grammatically reasonable patterns but not valid SQL — that is fine.)

**Common gotchas:**
- Spider's JSON has nested structure. The `query` field is at the top level of each example, but `train_spider.json` is a list of dicts — use `ex['query']` not `ex['sql']['query']`.
- Very short queries (1–2 tokens) produce degenerate n-grams. Filter out queries with fewer than 5 tokens.

---

## Task 3 — Key Snippets (Initialization Diagnostics)

```python
import matplotlib.pyplot as plt

# Collect pre-activation statistics
def get_preact_stats(model, data, n_batches=10):
    stats = []
    for i in range(n_batches):
        x = data[i*32:(i+1)*32]
        emb = model.C(x).view(32, -1)
        preact = model.fc1(emb)  # BEFORE nonlinearity
        stats.append((preact.mean().item(), preact.std().item()))
    return stats

# Before Kaiming: initialize with default (which for nn.Linear is Kaiming by default in PyTorch 2.x)
# To test "bad" init, override with small values:
model_bad = MakemoreMLP()
nn.init.normal_(model_bad.fc1.weight, std=0.01)  # too small
stats_bad = get_preact_stats(model_bad, X_train)

model_good = MakemoreMLP()  # default init is already Kaiming
stats_good = get_preact_stats(model_good, X_train)
```

**What you should observe:** With `std=0.01` init, pre-activation std is ~0.01 (collapsed). With Kaiming init, std is ~1.0 (healthy). Gradient norms on `fc1.weight` will be near-zero with bad init because the upstream gradients through ReLU on near-zero values are tiny.

**Common gotchas:**
- `nn.Linear`'s default init in PyTorch is actually Kaiming Uniform (not Normal). To see the "bad" case, you must override it.
- Gradient norms: access with `model.fc1.weight.grad.norm().item()` after one backward pass.

---

## How to Verify You Did It Right

1. **Task 1:** Val loss ≤ 2.2 on names dataset. W&B shows smooth downward curve with no spikes.
2. **Task 2:** `sql_vocab.txt` has 50–120 entries. Generated samples show repeated SQL keywords (`select`, `from`, `where`) in reasonable positions.
3. **Task 3:** Activation std with bad init is << 1.0; with Kaiming init it is ~1.0. Your plot shows this difference clearly.
