# Week 1 — Assignment Solutions

## Task 1 — Key Snippets

The most commonly tripped-up part is broadcasting. Rule of thumb: align shapes from the right.

```python
# (4, 3) + (3,) -> PyTorch expands (3,) to (1, 3) -> (4, 3). Fine.
# (4, 3) + (4,) -> Error! (4,) aligns as rightmost; (3,) != (4,).
a = torch.randn(4, 3)
b = torch.randn(3)
print((a + b).shape)  # torch.Size([4, 3])

# Moving device on MPS (Apple Silicon)
x = torch.randn(3, 3)
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
x = x.to(device)
print(x.device)   # mps:0
x = x.to('cpu')
print(x.device)   # cpu
```

**Expected output:** All shape prints match the values in the task. No errors.

**Common gotchas:**
- Trying to use `view` on a non-contiguous tensor (after `.permute()` or `.transpose()`) raises an error. Use `.contiguous().view()` or just `.reshape()`.
- `torch.randn` creates float32 by default. Mixing with `torch.ones` (also float32) is fine. Mixing with `torch.randint` (int64) will error on arithmetic.
- MPS availability: `torch.backends.mps.is_available()` is only `True` on macOS 12.3+ with Apple Silicon and PyTorch 2.0+.

---

## Task 2 — Key Snippets (Micrograd)

The trickiest part is the `_backward` for multiplication, since both operands need gradients:

```python
class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad  += other.data * out.grad   # chain rule
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
```

**Expected output for verification example:**
```
x1.grad ≈ -1.5000
x2.grad ≈  0.5000
w1.grad ≈  1.0000
w2.grad ≈  0.0000
```

**Common gotchas:**
- Using `=` instead of `+=` in `_backward` — this breaks graphs where a node is used more than once.
- Forgetting to handle the case where `other` is a plain float/int in `__mul__` and `__add__` — wrap with `Value(other)`.
- The topological sort must add a node to `topo` **after** visiting all its children, not before.
- Calling `.backward()` on a non-scalar `Value` — add a guard or always ensure your final output is scalar.

---

## Task 3 — Key Snippets (PyTorch NN)

```python
import torch
import torch.nn as nn

class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(2, 16) * 0.1)
        self.b1 = nn.Parameter(torch.zeros(16))
        self.W2 = nn.Parameter(torch.randn(16, 1) * 0.1)
        self.b2 = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        h = torch.tanh(x @ self.W1 + self.b1)
        return (h @ self.W2 + self.b2).squeeze(-1)

model = TinyMLP()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

for step in range(500):
    optimizer.zero_grad()          # MUST be first
    logits = model(X_train)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, y_train)
    loss.backward()
    optimizer.step()
    if step % 100 == 0:
        print(f"step {step}: loss={loss.item():.4f}")
```

**Expected output:** Loss should drop from ~0.69 (random) to below 0.1 within 500 steps on `make_moons`.

**Common gotchas:**
- Not calling `model.train()` at the start of the loop (matters more when there's dropout, but build the habit now).
- Using `loss` directly in a print statement — always call `.item()` to detach from the graph.
- Shapes: `binary_cross_entropy_with_logits` expects logits and targets both of shape `(N,)` or `(N, 1)`. Ensure they match.

---

## How to Verify You Did It Right

1. **Micrograd:** Your gradients must match PyTorch's exactly. Run this check:
   ```python
   import torch
   x1 = torch.tensor(2.0, requires_grad=True)
   x2 = torch.tensor(0.0, requires_grad=True)
   w1 = torch.tensor(-3.0, requires_grad=True)
   w2 = torch.tensor(1.0, requires_grad=True)
   b  = torch.tensor(6.8813736, requires_grad=True)
   o  = ((x1*w1 + x2*w2) + b).tanh()
   o.backward()
   print(x1.grad, x2.grad, w1.grad, w2.grad)
   ```
   Values must match your micrograd output to 4+ decimal places.

2. **PyTorch NN:** Final loss < 0.1 and decision boundary is correct (visualize with matplotlib if you have time).
