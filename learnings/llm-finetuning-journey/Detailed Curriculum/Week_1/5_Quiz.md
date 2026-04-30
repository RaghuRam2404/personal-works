# Week 1 — Quiz

Calibration: junior ML engineer interview level. These are real scenarios, not trivia.

---

**Q1.** You run the following code and the loss does not decrease at all across 1000 steps. The model architecture is correct and the data is loaded properly. What is the most likely bug?

```python
for step, (x, y) in enumerate(train_loader):
    y_hat = model(x)
    loss = criterion(y_hat, y)
    loss.backward()
    optimizer.step()
```

A) The learning rate is too high.
B) `optimizer.zero_grad()` is missing, so gradients accumulate across batches.
C) `loss.backward()` should be called before `criterion`.
D) `model.train()` was not called before the loop.

---

**Q2.** A tensor `a` has shape `(5, 1, 4)` and a tensor `b` has shape `(3, 4)`. What is the shape of `a + b` under PyTorch broadcasting rules?

A) Error — shapes are incompatible.
B) `(5, 3, 4)`
C) `(5, 1, 4)`
D) `(15, 4)`

---

**Q3.** You are logging training loss every step using `loss_history.append(loss)`. After 1000 steps you notice GPU memory is nearly full. What is the cause and fix?

A) The model is too large. Use a smaller batch size.
B) You are appending the tensor `loss` instead of `loss.item()`, holding the entire computation graph in memory. Fix: `loss_history.append(loss.item())`.
C) You forgot `torch.no_grad()` in the training loop.
D) The optimizer is accumulating gradients. Add `optimizer.zero_grad()`.

---

**Q4.** In your micrograd implementation, a `Value` node `z = x * y` is used twice in downstream operations. If you use `=` instead of `+=` in the `_backward` function for multiplication, what happens?

A) A Python `AttributeError` is raised.
B) The gradient from the second path overwrites the first, giving an incorrect total gradient for `z`.
C) Both gradients are summed correctly because Python handles it automatically.
D) The computation graph becomes circular and `.backward()` hangs.

---

**Q5.** What does `requires_grad=False` on a tensor mean, and why would you set it during inference?

A) The tensor cannot be moved to GPU.
B) PyTorch will not track operations on this tensor in the computational graph, so no gradient memory is allocated. This saves memory and compute during inference when you do not need gradients.
C) The tensor values are frozen and cannot be changed by in-place operations.
D) The optimizer will skip this parameter during `optimizer.step()`.

---

**Q6.** You have `x = torch.randn(3, 4)`. You call `x.view(2, 6)`. This succeeds. You then call `x.transpose(0, 1)` and try to call `.view(4, 3)` on the result. This raises a `RuntimeError`. Why?

A) `view` does not support transposed dimensions.
B) `transpose` returns a non-contiguous tensor (it changes the stride without copying data), and `view` requires contiguous memory. Fix: call `.contiguous().view(4, 3)` or use `.reshape(4, 3)`.
C) `(4, 3)` is not a valid reshape of a `(3, 4)` tensor.
D) You must call `.detach()` before `view`.

---

**Q7 (short answer).** Explain what happens, step by step, when you call `loss.backward()` on a scalar loss tensor in a two-layer network. Your answer should mention: the computational graph, the chain rule, and where gradients end up.

---

**Q8 (short answer).** You are training a model and you notice the loss spikes sharply at step 450, then partially recovers. List three hypotheses for what caused the spike, ranked from most to least likely.

---

**Q9 (short answer).** A colleague says: "I always use `.data` to access the tensor values and bypass autograd when I want to do in-place modifications." What is the risk of this approach, and what is the safer alternative?
