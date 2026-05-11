# Week 8 — Assignment Solutions

## Option A — Key Snippets (nanoGPT Data Loader)

```python
# data.py — character-level data loader for SQL corpus
import torch

class CharDataset:
    def __init__(self, text, block_size):
        chars   = sorted(set(text))
        self.stoi    = {c: i for i, c in enumerate(chars)}
        self.itos    = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(chars)
        self.block_size = block_size
        self.data   = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + 1 + self.block_size]
        return x, y
```

**Core training loop for Option A:**
```python
# train.py — simplified
from torch.utils.data import DataLoader

dataset   = CharDataset(text, block_size=128)
loader    = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)
model     = GPT(GPTConfig(vocab_size=dataset.vocab_size, ...)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
scaler    = torch.cuda.amp.GradScaler(enabled=(device == 'cuda'))

for step, (x, y) in enumerate(loader):
    x, y = x.to(device), y.to(device)
    lr = get_lr(step, warmup=200, total=5000, max_lr=3e-4)
    for pg in optimizer.param_groups: pg['lr'] = lr
    optimizer.zero_grad()
    with torch.cuda.amp.autocast(enabled=(device == 'cuda')):
        logits, loss = model(x, y)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
```

**Expected performance:**
- Model: 4 layers, 4 heads, 128 emb → ~800K params.
- Final train loss: ~1.0–1.2 nats after 5000 steps on Colab T4 (~25 minutes).
- Generated samples: recognizable SQL structure (`SELECT`, `FROM`, `WHERE`) with garbled column/table names.

**Common gotchas:**
- nanoGPT's `GPTConfig` uses `n_layer`, `n_head`, `n_embd` — not `num_layers`, `num_heads`. Match the config field names exactly.
- The `forward()` in nanoGPT returns `(logits, loss)` when `targets` is provided. Without targets, it returns `(logits, None)`. Handle both cases in your training vs. generation code.
- Character-level vocab_size for SQL corpus is typically 70–90 characters. Verify this matches your embedding table size.
- `block_size` must match both `GPTConfig.block_size` and your `CharDataset.block_size`. Mismatches cause shape errors in the positional embedding lookup.

---

## Option B — Key Snippets (distilgpt2 Fine-Tuning)

```python
# finetune.py
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from datasets import load_from_disk

tok   = AutoTokenizer.from_pretrained("distilgpt2")
tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)

ds = load_from_disk("week_07/spider_tokenized")

def collate(batch):
    input_ids  = torch.stack([torch.tensor(b['input_ids'][:128]) for b in batch])
    attn_mask  = torch.stack([torch.tensor(b['attention_mask'][:128]) for b in batch])
    # Pad to same length
    # (assume already at max_length; for variable length, use tok.pad)
    labels = input_ids.clone()
    labels[attn_mask == 0] = -100
    return {'input_ids': input_ids, 'attention_mask': attn_mask, 'labels': labels}

loader = DataLoader(ds['train'], batch_size=8, shuffle=True, collate_fn=collate)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

for step, batch in enumerate(loader):
    if step >= 200: break
    batch = {k: v.to(device) for k, v in batch.items()}
    optimizer.zero_grad()
    out  = model(**batch)
    loss = out.loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if step % 10 == 0:
        print(f"step {step}: loss={loss.item():.4f}")
```

**Expected performance:** Loss should decrease from ~3.5 to ~2.5–3.0 in 200 steps. distilgpt2 is not initialized for SQL, so progress is limited — the point is the workflow, not final quality.

**Common gotchas:**
- `out.loss` is only returned by `AutoModelForCausalLM` when `labels` is passed. Without `labels`, `out.loss = None`.
- `tok.pad_token = tok.eos_token` is required; without it, padding will use `None` and crash.
- The `labels` must be passed to `model(**batch)` — not just `input_ids` and `attention_mask`. Forgetting `labels` means the model runs but doesn't return a loss.

---

## Task 3 — What a Passing Timed Loop Looks Like

A passing `timed_loop.py` in under 10 minutes:

```python
# Completed in 8 minutes. PASS.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 2)
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

model  = Net().to(device)
optim  = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
scaler = GradScaler()
total_steps = 100

def get_lr(step):
    if step < 10: return 1e-3 * step / 10
    return 1e-4 + 0.5*(1e-3-1e-4)*(1+math.cos(math.pi*(step-10)/90))

for step in range(total_steps):
    for pg in optim.param_groups: pg['lr'] = get_lr(step)
    X = torch.randn(32, 10, device=device)
    y = torch.randint(0, 2, (32,), device=device)
    optim.zero_grad()
    with autocast():
        loss = F.cross_entropy(model(X), y)
    scaler.scale(loss).backward()
    scaler.unscale_(optim)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optim)
    scaler.update()
    if step % 10 == 0:
        print(f"step {step}: loss={loss.item():.4f}")

model.eval()
with torch.no_grad():
    val_X = torch.randn(64, 10, device=device)
    val_y = torch.randint(0, 2, (64,), device=device)
    val_loss = F.cross_entropy(model(val_X), val_y)
    print(f"val_loss={val_loss.item():.4f}")
```

**How to verify you passed:**
- Code runs without errors.
- You did not look at any references.
- Time was under 10 minutes.
- The loop includes all 7 required elements (zero_grad, autocast, clip, scaler, scheduler, eval block, logging).
