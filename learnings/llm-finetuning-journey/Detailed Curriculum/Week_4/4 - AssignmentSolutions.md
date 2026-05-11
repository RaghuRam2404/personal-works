# Week 4 — Assignment Solutions

## Task 2 — Key Snippets (LSTM Cell)

```python
class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # Pack all gate weights into one matrix for efficiency
        self.W = nn.Linear(input_size + hidden_size, 4 * hidden_size)

    def forward(self, x, state):
        h_prev, c_prev = state
        combined = torch.cat([x, h_prev], dim=1)   # (B, input + hidden)
        gates    = self.W(combined)                 # (B, 4*hidden)
        # Split into 4 gates
        f, i, g, o = gates.chunk(4, dim=1)
        f = torch.sigmoid(f)
        i = torch.sigmoid(i)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_new = f * c_prev + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new
```

**Expected output:** Verification assertion passes — your `(h_new, c_new)` matches `nn.LSTMCell`'s output to 4 decimal places (after copying weights manually).

**Common gotchas:**
- `nn.LSTMCell` packs its weights as `weight_ih` `(4*hidden, input)` and `weight_hh` `(4*hidden, hidden)`, split in order `i, f, g, o` — **not** `f, i, g, o`. The order in PyTorch is input gate first, then forget gate. Read the docs carefully when copying weights.
- `chunk(4, dim=1)` splits evenly — your hidden size must divide cleanly.
- Do not forget both `h` and `c` must be detached for TBPTT, not just `h`.

---

## Task 3 — Key Snippets (Character LSTM Training)

```python
class CharLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size, n_layers):
        super().__init__()
        self.emb    = nn.Embedding(vocab_size, emb_dim)
        self.lstm   = nn.LSTM(emb_dim, hidden_size, n_layers, batch_first=True)
        self.head   = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, state):
        emb    = self.emb(x)              # (B, T, emb_dim)
        out, state = self.lstm(emb, state) # out: (B, T, hidden)
        logits = self.head(out)           # (B, T, vocab_size)
        return logits, state

# TBPTT training loop
state = None
for step, (X, Y) in enumerate(loader):
    if state is not None:
        state = (state[0].detach(), state[1].detach())
    logits, state = model(X, state)
    loss = F.cross_entropy(logits.view(-1, vocab_size), Y.view(-1))
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # always clip for RNNs
    optimizer.step()

# Generation with temperature
def generate(model, seed_text, length=200, temperature=0.8):
    model.eval()
    chars = [stoi[c] for c in seed_text]
    state = None
    with torch.no_grad():
        for _ in range(length):
            x = torch.tensor([[chars[-1]]], device=device)
            logits, state = model(x, state)
            probs = F.softmax(logits[0, -1] / temperature, dim=-1)
            next_char = torch.multinomial(probs, 1).item()
            chars.append(next_char)
    return ''.join(itos[c] for c in chars)
```

**Expected sample output (training loss ~1.3 nats):**
```
SELECT t1.id, t2.name FROM table1 AS t1 JOIN table2 AS t2 ON t1.id = t2.id WHERE t1.value > 5 GROUP
SELECT COUNT(*) FROM orders WHERE statua = 'complet' AND price > 100 ORDER BY dae DESC LIMIT 10
```
(Mostly grammatical structure correct; column names and table names are hallucinated.)

**Common gotchas:**
- `nn.LSTM` with `batch_first=True`: input shape `(B, T, E)`, output `(B, T, H)`. Without `batch_first=True`, shapes are `(T, B, H)` — easy to get backwards.
- `F.cross_entropy` expects logits of shape `(N, C)`. Use `.view(-1, vocab_size)` on both logits and targets.
- Always clip gradients for RNNs. Without clipping, explosive gradients are common and will cause NaN loss.
- State initialization: pass `state=None` on the first call — PyTorch initializes hidden/cell states to zeros.

---

## How to Verify You Did It Right

1. **Task 1:** Your `VanillaRNNCell` output matches `nn.RNNCell` for the same weights. Loss decreases from ~4.0 to < 1.0 on "hello world" after 200 steps.
2. **Task 2:** Assertion passes. Comment block contains correct gate descriptions.
3. **Task 3:** W&B shows loss decreasing below 1.5 nats. Generated samples contain recognizable SQL keywords (`SELECT`, `FROM`, `WHERE`) but garbled column/table names — this is the expected behavior.
4. **Task 4:** Reflection addresses all three questions with concrete technical answers.
