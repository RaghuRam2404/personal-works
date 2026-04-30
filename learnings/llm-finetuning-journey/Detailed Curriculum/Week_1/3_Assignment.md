# Week 1 — Assignment

## Setup Checklist

Before starting, verify the following:

- [ ] Python 3.11+ installed via `pyenv` or `uv`. Run `python --version`.
- [ ] PyTorch installed with correct backend: `python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"` — should show `True` on Apple Silicon.
- [ ] Git repo `llm-finetuning-journey` exists on GitHub with an initial commit.
- [ ] W&B account created and `wandb login` works in your terminal.
- [ ] `journal.md` file exists in the repo root.

---

## Task 1 — Tensor Drills

**Goal:** Build reflexive fluency with tensor operations so they never slow you down in later weeks.

**Requirements:**
- Create a script `week_01/tensor_drills.py`.
- Complete each of the following operations and print the result shape after every operation:
  1. Create a random tensor of shape `(4, 3)` and a tensor of shape `(3,)`. Add them (broadcasting). Print resulting shape.
  2. Create a tensor of shape `(2, 3, 4)`. Compute the mean over `dim=1`. Print resulting shape.
  3. Create a `(5, 5)` identity matrix without using `torch.eye` (use `torch.zeros` + index assignment).
  4. Matrix multiply a `(10, 8)` tensor by an `(8, 6)` tensor. Print the result shape.
  5. Reshape a `(6, 4)` tensor to `(2, 3, 4)` using `view`, then to `(24,)` using `reshape`.
  6. Create a float tensor on CPU, move it to MPS (or CUDA if on Colab), and back to CPU. Verify the device at each step.
- All operations must complete without errors.

**Deliverable:** `week_01/tensor_drills.py` committed to GitHub. Output pasted into `journal.md`.

---

## Task 2 — Implement Micrograd from Scratch

**Goal:** Prove you understand backpropagation at the scalar level.

**Requirements:**
- Watch Karpathy's [micrograd video](https://www.youtube.com/watch?v=VMj-3S1tku0) in full. Code along — type every line yourself, no copy-paste.
- Your implementation must live in `week_01/micrograd.py`.
- Implement these operations on the `Value` class: `+`, `*`, `-`, `/`, `**`, `exp`, `tanh`, `relu`.
- Implement `backward()` with topological sort.
- Verify correctness by running a 2-layer network:
  ```python
  x1 = Value(2.0)
  x2 = Value(0.0)
  w1 = Value(-3.0)
  w2 = Value(1.0)
  b  = Value(6.8813736)
  x1w1 = x1 * w1
  x2w2 = x2 * w2
  x1w1x2w2 = x1w1 + x2w2
  n = x1w1x2w2 + b
  o = n.tanh()
  o.backward()
  ```
  Expected: `x1.grad ≈ -1.5`, `x2.grad ≈ 0.5`, `w1.grad ≈ 1.0`, `w2.grad ≈ 0.0`. Match Karpathy's video results exactly.
- Include a `draw_dot()` function (or just print the graph manually) to visualize the computation graph for this example.

**Deliverable:** `week_01/micrograd.py` committed. Gradient values printed and included in `journal.md`.

**Hints:**
- The topological sort must visit children before parents when traversing backwards.
- Don't forget to accumulate (not overwrite) `self.grad` in each `_backward` function — multiple paths can flow into one node.

---

## Task 3 — Port Your Old Python NN to PyTorch

**Goal:** Translate your existing 1–2 layer plain-Python neural network into PyTorch, then train it.

**Requirements:**
- Create `week_01/nn_pytorch.py`.
- Use `torch.nn.Module` to define the network. No `nn.Linear` shortcuts yet — define weights as `nn.Parameter` explicitly so you see what's happening.
- Train on any toy dataset (XOR is fine; `sklearn.datasets.make_moons` is better).
- Log: step number, train loss every 100 steps.
- Run for at least 500 steps and achieve loss < 0.1 on the toy dataset.
- The training loop must be written from scratch — no `nn.Trainer` or HuggingFace wrappers.

**Deliverable:** `week_01/nn_pytorch.py` with a printed loss curve (step vs. loss). Commit message: `week-01-micrograd`.

**Hints:**
- `model.parameters()` returns an iterator over all `nn.Parameter`s — pass it to `torch.optim.SGD` or `torch.optim.Adam`.
- Use `torch.nn.functional.binary_cross_entropy_with_logits` for binary classification to avoid the log-of-zero problem.

---

## Stretch Goals

- Implement a `Neuron`, `Layer`, and `MLP` class on top of your `micrograd` engine (Karpathy shows this in the second half of the video). Train a tiny 2-class classification problem using only your micrograd engine — no PyTorch.
- Add a live W&B logging run to Task 3. Project name: `week-01-pytorch-basics`. Log `train/loss` every 10 steps.
- Read the PyTorch source for `torch.autograd.Function` and understand how custom backward passes are written. You will need this in Phase 4.
