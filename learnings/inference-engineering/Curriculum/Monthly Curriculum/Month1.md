## Overview and Placement in the Curriculum

Month 1 is Week 1–4 of Phase A ("PyTorch, autograd, GPU basics") in the 24-month Inference Engineering curriculum, whose stated theme is bridging plain-Python/NumPy neural network intuition into fluent, debuggable PyTorch practice. The month has three ordered material tracks — manual NumPy-to-PyTorch derivation, the official PyTorch tutorial pair, and `einops` tensor notation — culminating in Project A1, an autograd parity test harness. The goal is not "learn PyTorch syntax" but "understand precisely what autograd computes," so that later phases (Transformers from scratch, pretraining, distributed training) never require debugging `.backward()` as a black box.[^1]

This report expands the month into a week-by-week execution plan with exact inputs/outputs, quizzes, glossary, and resource links, so each session has a concrete artifact to produce and verify.

## Week-by-Week Breakdown

### Week 1 — Manual NumPy derivation and PyTorch port

**Objective:** Re-derive the forward/backward pass of your existing 1–2 layer plain-Python network using explicit matrix calculus, with no autograd, then port it line-by-line to PyTorch and numerically diff the gradients.[^1]

**Exact input:** A synthetic dataset — e.g., 200 samples of \(X \in \mathbb{R}^{200 \times 4}\) drawn from `numpy.random.seed(0)`, and a target \(y \in \mathbb{R}^{200 \times 1}\) generated from a known linear-plus-noise function, so gradient correctness can be checked against a computable ground truth. Fix the network shape: input dim 4, hidden dim 8, output dim 1, one hidden ReLU layer.

**Exact output:** Two Python scripts producing identical numeric artifacts:
- `numpy_manual.py` — implements forward pass \(z_1 = XW_1 + b_1\), \(a_1 = \text{ReLU}(z_1)\), \(z_2 = a_1 W_2 + b_2\), a mean-squared-error loss, and hand-derived backward pass computing \(\partial L/\partial W_1, \partial L/\partial b_1, \partial L/\partial W_2, \partial L/\partial b_2\) via chain rule, printed as NumPy arrays.
- `pytorch_port.py` — the same architecture built with `torch.tensor(..., requires_grad=True)` (no `nn.Module` yet), calling `.backward()` once, and printing `.grad` for each parameter.
- A comparison script asserting `torch.allclose(torch_grad, torch.from_numpy(numpy_grad), atol=1e-6)` for every parameter tensor.[^1]

**Quiz (answer without notes):**
1. Why does the chain rule for \(\partial L/\partial W_1\) require the gradient of ReLU, and what is \(\text{ReLU}'(z)\) at \(z=0\)?
2. If you forget to transpose \(W_2\) when backpropagating through the second linear layer, what shape-mismatch error will NumPy throw, and where does the transpose come from mathematically?
3. What does `atol=1e-6` guard against that a naive `==` comparison would not, given floating-point non-associativity between NumPy's and PyTorch's BLAS backends?

**Glossary:** forward pass, backward pass, chain rule, Jacobian, computational graph, `requires_grad`, `.grad`, `.backward()`, numerical vs. analytical gradient, `atol`/`rtol` tolerance.

**Resources:** Your existing plain-Python 1–2 layer network code (prerequisite artifact); NumPy documentation for `np.dot`/broadcasting rules; PyTorch `torch.allclose` docs.[^1]

**Takeaway:** You should be able to write out, on paper, the five-line chain-rule derivation for a 2-layer MLP's gradients before touching any framework — this derivation is the mental model every later PyTorch abstraction (autograd, `nn.Module`, distributed gradient sync) sits on top of.

### Week 2 — PyTorch "Learn the Basics" + break-it-deliberately drill

**Objective:** Work through PyTorch's official **Learn the Basics** series (tensors, `Dataset`/`DataLoader`, transforms, `nn.Module`, autograd, optimization loop, save/load) and the **60 Minute Blitz** (tensors, autograd, neural networks, training a CIFAR classifier), doing every code cell twice — once correct, once deliberately broken.[^2][^3][^4]

**Exact input:** The nine official notebook modules under Learn the Basics — `intro`, `quickstart_tutorial`, `tensorqs_tutorial`, `data_tutorial`, `transforms_tutorial`, `buildmodel_tutorial`, `autogradqs_tutorial`, `optimization_tutorial`, `saveloadrun_tutorial` — plus the five Blitz modules — `tensor_tutorial`, `autograd_tutorial`, `neural_networks_tutorial`, `cifar10_tutorial`, `data_parallel_tutorial`. Both series run on the FashionMNIST dataset (Basics) and CIFAR-10 (Blitz).[^5][^4][^6][^2][^1]

**Exact output:** A single annotated notebook (`week2_basics_and_blitz.ipynb`) with, for each of the 14 modules, two sub-cells: (a) the tutorial code run as-is with its correct output, and (b) an intentionally broken variant — for example, feeding a `(64, 28, 28)` batch into a `Linear(784, 512)` layer without flattening (shape error), calling `.backward()` twice without `retain_graph=True` (autograd error), or setting `requires_grad=False` on a leaf tensor you then expect a gradient from (silent `None` gradient) — with the raw PyTorch/CUDA stack trace pasted below each broken cell and a one-sentence diagnosis written by hand.[^1]

**Quiz:**
1. What is the actual difference between a `Dataset` and a `DataLoader`, and why does `DataLoader` need a `batch_size` argument rather than the `Dataset` itself?
2. Reading a `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` — what three things could cause this, and how do you fix each?
3. In `optimizer.step()`, why must `optimizer.zero_grad()` be called before `loss.backward()` on the next iteration, and what artifact appears in your gradients if you forget it?
4. What does `.detach()` do to a tensor's place in the computational graph, and name one legitimate use case besides "fixing an error."

**Glossary:** `nn.Module`, `nn.Parameter`, `Dataset`, `DataLoader`, `transforms`, forward hook, `loss.backward()`, `grad_fn`, leaf tensor, `optimizer.zero_grad()`, `torch.no_grad()`, `retain_graph`, `state_dict`, checkpoint.

**Resources:** PyTorch Learn the Basics — pytorch.org/tutorials/beginner/basics; Deep Learning with PyTorch: A 60 Minute Blitz — pytorch.org/tutorials/beginner/blitz; Quickstart tutorial for the save/load workflow.[^4][^7][^5][^1]

**Takeaway:** Reading a raw PyTorch/CUDA stack trace fluently — distinguishing a shape mismatch from an autograd-graph error from a device-mismatch error — is a debugging skill that saves hours in every later phase; deliberately breaking each concept is what builds that pattern-matching muscle rather than only ever seeing the "happy path."

### Week 3 — `nn.Module`, optimization loop, and einops rewrite

**Objective:** Consolidate the `nn.Module` + optimizer + training-loop pattern from Week 2, then rewrite three of your Week 1 NumPy forward passes using `einops.rearrange`/`einops.einsum` instead of `.view()`/`.permute()` chains.[^1]

**Exact input:** Your Week 1 NumPy forward-pass functions (linear layer, 2-layer MLP, and one additional shape-manipulation-heavy example such as a batched attention-style dot product), plus the `einops` "Einops basics" and "Writing better code with einops" notebooks.[^1]

**Exact output:** `einops_rewrite.py` containing three functions, each with two implementations side by side: the original `.view()`/`.transpose()`/`.permute()` version and an `einops.rearrange`/`einops.einsum` version, both asserting identical output via `torch.allclose`. Example transformation: replacing `x.view(B, H, T, D).permute(0, 2, 1, 3).reshape(B, T, H*D)` with `einops.rearrange(x, 'b h t d -> b t (h d)')`.

**Quiz:**
1. What bug class does `einops.rearrange` eliminate that `.view()` cannot catch at call time (hint: silent reshapes on non-contiguous tensors)?
2. Write the `einops.rearrange` pattern string that converts a tensor of shape `(batch, seq, heads, head_dim)` into `(batch, heads, seq, head_dim)`.
3. What is the difference between `einops.einsum` and `torch.einsum` in terms of readability, and why does explicit axis-naming reduce off-by-one transpose errors?

**Glossary:** `rearrange`, `reduce`, `repeat`, `einsum`, axis pattern string, contiguous memory layout, stride, view vs. copy.

**Resources:** einops.rocks — "Einops basics" and "Writing better code with einops" notebooks.[^1]

**Takeaway:** Einops patterns make tensor-shape intent self-documenting; once you can read `'b h t d -> b t (h d)'` at a glance, later Transformer attention-head reshaping code (Phase B onward) becomes trivial to audit for correctness instead of a guessing game with `.permute()` argument order.

### Week 4 — Project A1: Autograd parity harness

**Objective:** Build a `pytest` harness with 5 test cases, each pairing a hand-derived NumPy gradient implementation against a PyTorch implementation, verifying `max_abs_diff < 1e-6` across 20 random seeds.[^1]

**Exact input:** Five architectures, each with fixed small dimensions for fast testing:
1. Linear layer: \(y = Wx + b\), MSE loss.
2. 2-layer MLP: linear → ReLU → linear, MSE loss.
3. MLP + ReLU + softmax: linear → ReLU → linear → softmax → cross-entropy loss.
4. MLP with manual L2 regularizer: same as case 2, loss \(= \text{MSE} + \lambda \sum W_i^2\).
5. Toy RNN cell: a single-step Elman cell \(h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b)\), unrolled 3–5 timesteps, MSE loss on final hidden state.

For each case: random inputs generated from `numpy.random.default_rng(seed)` for `seed in range(20)`.

**Exact output:** A repository structure:
```
project_a1/
  numpy_impl/{linear,mlp,mlp_softmax,mlp_l2,rnn_cell}.py
  torch_impl/{linear,mlp,mlp_softmax,mlp_l2,rnn_cell}.py
  test_parity.py   # 5 test functions x 20 seeds = 100 assertions
  report.md         # max_abs_diff table per case per seed, pass/fail summary
```
`test_parity.py` uses `@pytest.mark.parametrize("seed", range(20))` for each of the 5 test functions, asserting `np.max(np.abs(torch_grad.numpy() - numpy_grad)) < 1e-6` for every learnable parameter in the case.

**Acceptance bar (from the curriculum):** All 5 cases pass on all 20 seeds; you can explain out loud, without notes, what `.backward()` does for each case.[^1]

**Quiz (oral, no notes — this is the actual acceptance test):**
1. For the RNN cell case, why does `.backward()` need to traverse the unrolled graph across all timesteps, and what would happen numerically if you called `.backward()` after every timestep instead of once at the end?
2. For the softmax + cross-entropy case, why is it numerically preferable to combine them into a single fused operation (as `nn.CrossEntropyLoss` does) rather than computing softmax then log then NLL separately?
3. For the L2-regularized case, derive by hand the extra gradient term contributed by \(\lambda \sum W_i^2\), and confirm it matches what PyTorch's autograd produces when you add the penalty into the loss graph directly.
4. Why does testing across 20 seeds catch bugs that a single fixed seed would miss (e.g., a gradient formula that happens to be correct only when an input is non-negative)?

**Glossary:** parity test, hand-derived gradient, `pytest.mark.parametrize`, `max_abs_diff`, Elman RNN cell, backpropagation through time (BPTT), fused kernel, `nn.CrossEntropyLoss`, gradient checking.

**Resources:** Your Week 1–3 NumPy and PyTorch implementations as the direct code base; PyTorch autograd mechanics documentation for the "Gentle Introduction to torch.autograd" referenced in the Blitz; `pytest` parametrize documentation for the 20-seed test matrix.[^4]

**Takeaway:** Passing 100/100 assertions is necessary but not sufficient — the real acceptance bar is being able to narrate, case by case, exactly which computational-graph nodes `.backward()` visits and what local derivative each node contributes, since this is the debugging lens you will reuse for every custom loss, regularizer, and recurrent structure in later phases.

## Month 1 Resource Table

| Resource | Exact scope | Link |
|---|---|---|
| PyTorch "Learn the Basics" | Tensors, `Dataset`/`DataLoader`, transforms, build model, autograd, optimization, save/load — 9 modules | pytorch.org/tutorials/beginner/basics[^1][^5] |
| PyTorch "60 Minute Blitz" | Tensors, autograd, neural networks, CIFAR-10 classifier, optional data parallelism — 5 modules | pytorch.org/tutorials/beginner/blitz[^4] |
| einops documentation | "Einops basics" + "Writing better code with einops" | einops.rocks[^1] |
| `torch.allclose` reference | Numerical gradient-diffing with `atol`/`rtol` | PyTorch tensor comparison docs[^1] |
| Your existing plain-Python NN | Prerequisite: the 1–2 layer network to re-derive | Your own prior artifact[^1] |

## Consolidated Month 1 Glossary

| Term | Definition |
|---|---|
| Autograd | PyTorch's automatic differentiation engine that records operations on tensors with `requires_grad=True` into a dynamic computational graph and computes gradients via reverse-mode chain rule on `.backward()`[^4]. |
| Computational graph | The DAG of tensor operations built during the forward pass, traversed in reverse during `.backward()` to compute gradients. |
| `requires_grad` | Tensor flag marking it as a leaf node whose gradient should be tracked and accumulated in `.grad`. |
| `.detach()` | Returns a tensor sharing the same data but detached from the computational graph, stopping gradient flow through it. |
| `nn.Module` | PyTorch's base class for composable, parameterized network layers with automatic parameter registration. |
| `einops.rearrange` | Declarative tensor reshape/transpose operation using named-axis pattern strings instead of positional `.view()`/`.permute()` calls[^1]. |
| Numerical gradient parity | Verifying an analytical (hand-derived or autograd) gradient matches a reference implementation within a small tolerance, here `atol=1e-6`[^1]. |
| BPTT (backpropagation through time) | Backpropagation applied across the unrolled timesteps of a recurrent computational graph. |

## Milestone Checklist for Month 1

- Week 1 artifact: NumPy-vs-PyTorch single-layer/MLP gradient diff script passing `torch.allclose(atol=1e-6)`.[^1]
- Week 2 artifact: Annotated notebook covering all 14 Basics + Blitz modules, each with a deliberately-broken variant and stack-trace diagnosis.[^5][^4][^1]
- Week 3 artifact: Three NumPy forward passes rewritten with `einops.rearrange`/`einsum`, verified equivalent to the `.view()`/`.permute()` originals.[^1]
- Week 4 artifact: Project A1 pytest harness — 5 cases × 20 seeds, 100% passing at `max_abs_diff < 1e-6`, plus the ability to verbally explain `.backward()` for each case without notes.[^1]

---

## References

1. [Learn the Basics — PyTorch Tutorials 2.13.0+cu130 documentation](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)

2. [Intro — PyTorch Tutorials 2.12.0+cu130 documentation](https://docs.pytorch.org/tutorials/intro.html)

3. [Deep Learning with PyTorch: A 60 Minute Blitz](https://docs.pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

4. [Deep Learning with PyTorch: A 60 Minute Blitz](https://docs.pytorch.org/tutorials/beginner/blitz/index.html) - PyTorch: A 60 Minute Blitz Learning. Visualizing Models, Data, and Training with TensorBoard. Get in...

5. [Learn the Basics — PyTorch Tutorials 2.11.0+cu130 ...](https://docs.pytorch.org/tutorials/beginner/basics/index.html)

6. [Deep Learning with PyTorch: A 60 Minute Blitz](https://brsoff.github.io/tutorials/beginner/deep_learning_60min_blitz.html) - Goal of this tutorial: Understand PyTorch's Tensor library and neural networks at a high level. Trai...

7. [Quickstart — PyTorch Tutorials 2.13.0+cu130 documentation](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)

