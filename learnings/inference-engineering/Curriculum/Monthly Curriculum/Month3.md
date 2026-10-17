# Month 3 — Autograd and Character Language Models from Scratch

## Month purpose

Month 3 begins Phase B: **Transformers from scratch**. This month does not build attention or GPT yet. It establishes the three ideas that Month 4 will assume without stopping to reteach them:

1. A neural network is a differentiable computation graph whose parameters are improved through reverse-mode automatic differentiation.
2. A language model estimates a conditional distribution for the next token and can sample from that distribution autoregressively.
3. Discrete token IDs become learnable vectors, and a finite context can be mapped through an MLP to next-token logits.

The required spine is the first three lectures of Andrej Karpathy's *Neural Networks: Zero to Hero*: micrograd, the bigram makemore model, and the MLP makemore model. Their official runtimes are 2h25m, 1h57m, and 1h15m respectively.[cite:125][cite:131]

The course describes the progression as scalar reverse-mode autograd, then character-level bigram language modeling, then an MLP language model with train/dev/test splits and hyperparameter reasoning.[cite:131][cite:132] This sequence is important: Month 4 will extend the same language-modeling problem through activation diagnostics, manual tensor backpropagation, WaveNet-like hierarchy, and finally causal self-attention.

## Starting position

Months 1 and 2 have already established PyTorch model construction, training and evaluation loops, gradient checking, checkpointing, CUDA execution, AMP, and PyTorch Profiler usage. The learning log also shows that you learn best by implementing, taking notes, and answering checkpoint questions, but a nominal one-hour session can expand when coding and debugging are included.[cite:18]

Therefore, Month 3 is organized as **11 substantial sessions**, not as dozens of tiny sessions. Expect most sessions to require 2.5–4 focused hours. If a session takes longer, split it across days without changing the order.

## Scope guardrails

### Required this month

- Scalar-valued reverse-mode autodiff from scratch.
- A tiny neural-network library built on that scalar engine.
- A count-based character bigram language model.
- A one-layer neural bigram language model.
- A finite-context MLP character language model.
- Negative log-likelihood, cross-entropy, sampling, embeddings, train/dev/test splits, mini-batches, and learning-rate experiments.
- Gradient and shape tests throughout.

### Deferred to Month 4

- Activation and gradient distribution diagnostics in deep MLPs.
- Batch normalization internals.
- Full manual backpropagation through tensor-level cross-entropy, linear layers, tanh, and BatchNorm.
- WaveNet-style hierarchical networks.
- Causal self-attention and multi-head attention.
- A full GPT implementation.

### Deferred to Month 5

- Byte Pair Encoding.
- Production-style tokenizer behavior.
- KV caching.
- Naive-versus-cached generation benchmarks.

Do not pull Month 4 or Month 5 work forward. Month 3 is complete when the computation-graph and autoregressive-language-model foundations are explainable from first principles.

## Learning method

Apply this loop to each lecture:

1. **Predict:** Before the lecture section, write what you think the code must do.
2. **Watch:** Watch for the idea and causal explanation, not merely the final code.
3. **Close the source:** Rebuild the section in a fresh file without looking.
4. **Test:** Compare values, gradients, shapes, loss, or generated samples against an independent reference.
5. **Explain:** Answer the session's first-principles questions aloud.
6. **Consult the reference last:** Use the official notebook only after your own implementation fails or remains blocked for about 20 minutes.
7. **Record the gap:** Write what assumption or shape caused the failure.

The official repository contains notebooks created during the lectures and exercises linked from the videos; use it to check work, not as code to copy.[cite:131]

## Month deliverable

Create one repository containing three progressively stronger systems:

```text
month3/
  README.md
  requirements.txt
  data/
    names.txt
  micrograd/
    engine.py
    nn.py
    train.py
    graph.py                  # optional visualization helper
  makemore/
    data.py
    bigram_counts.py
    bigram_nn.py
    mlp.py
    sample.py
    metrics.py
  tests/
    test_engine.py
    test_nn.py
    test_bigram.py
    test_mlp_shapes.py
  outputs/
    micrograd_loss.csv
    bigram_results.md
    mlp_results.md
    samples.txt
    loss_curves.png
  notes/
    session-00.md
    ...
    session-10.md
```

Notebooks are acceptable for exploration, but the final implementations should be runnable from clean Python modules or scripts.

## Session map

| Session | Main work | Required result |
|---:|---|---|
| 0 | Setup and baseline diagnostic | Repository, dataset, written predictions |
| 1 | Micrograd lecture | Computation-graph mental model |
| 2 | Scalar forward engine | Correct `Value` operations and graph metadata |
| 3 | Reverse-mode autodiff | Correct topological backward pass |
| 4 | Neural-network library | MLP trained using your scalar engine |
| 5 | Bigram lecture and dataset | Vocabulary, training pairs, count matrix |
| 6 | Count-based bigram model | NLL evaluation and seeded generation |
| 7 | Neural bigram model | PyTorch implementation and equivalence explanation |
| 8 | MLP lecture and data pipeline | Context windows, embeddings, shape ledger |
| 9 | MLP training and evaluation | Train/dev/test experiment and generated names |
| 10 | Consolidation and oral defense | Refactored repository, comparison, memory rebuild |

---

## Session 0 — Setup and diagnostic

### Objective

Prepare a clean environment and expose what you already understand before the videos influence your answers.

### Work

- Create the repository structure shown above.
- Clone or bookmark the official `nn-zero-to-hero` and `micrograd` repositories, but do not open their implementation files yet.
- Download `names.txt` from the official makemore lecture materials.
- Record Python and PyTorch versions.
- Set a fixed seed helper for Python and PyTorch.
- Add a short README describing the Month 3 goal.

### Baseline questions

Answer in `notes/session-00.md` without searching:

1. If `z = x * y + x`, what information must each intermediate value retain so that `z.backward()` can compute both input gradients?
2. Why must the gradient entering the final scalar output start at 1?
3. Why must gradients from two downstream paths be added rather than overwritten?
4. What probability is a bigram language model estimating?
5. What information is lost when a model uses only the previous character?
6. What is the difference between a token ID and an embedding vector?
7. What must the final dimension of language-model logits represent?
8. Why should a test set not be used repeatedly to choose hyperparameters?

Mark every answer as **confident**, **uncertain**, or **guess**. Re-answer the same questions in Session 10.

### Assignment M3.0

Write tensor-shape predictions for a hypothetical MLP language model:

```text
vocabulary size V = 27
batch size B = 32
context length T = 3
embedding dimension C = 10
hidden width H = 200
```

Predict the shapes of:

- Input token IDs.
- Embedding table.
- Embedded input.
- Flattened context.
- First-layer weights and bias.
- Hidden activation.
- Output-layer weights and bias.
- Logits.
- Targets.
- Scalar loss.

Do not correct the predictions yet.

### Exit criteria

- Repository and environment are ready.
- Dataset is available.
- Baseline answers and shape predictions are committed.

---

## Session 1 — Watch micrograd

### Objective

Develop the first-principles model of derivatives, computation graphs, local derivatives, the chain rule, and reverse-mode automatic differentiation.

### Material

Watch **The spelled-out intro to neural networks and backpropagation: building micrograd**. The official course presents it as a step-by-step introduction requiring only Python and a basic recollection of calculus.[cite:131]

Video: <https://youtu.be/VMj-3S1tku0>

### Watching rules

- Do not attempt to maintain a perfect copy of the implementation.
- Pause after each major concept and predict the next data structure or algorithm.
- Record no more than two sentences per concept.
- Write every derivative calculation by hand at least once.

### Concepts to capture

- Numerical derivative and the role of a small perturbation.
- Scalar computation graph.
- Leaf and intermediate values.
- Forward value versus gradient.
- Local derivative.
- Upstream gradient.
- Chain rule.
- Reverse-mode autodiff.
- Topological ordering.
- Gradient accumulation at branches.
- Neuron, layer, and MLP.
- Forward pass, loss, backward pass, parameter update.

The reference micrograd implementation performs reverse-mode autodiff over a dynamically constructed scalar DAG and places a small PyTorch-like neural-network library on top.[cite:133]

### Stop-and-predict checkpoints

Before the video reveals each answer, predict:

1. What fields must a scalar `Value` object contain?
2. How can an operation remember how to propagate its gradient?
3. Why can recursive backward execution fail without a dependency-respecting order?
4. What bug appears if a node is used more than once and its gradient is assigned with `=`?
5. How can subtraction, division, and reverse multiplication be expressed using simpler primitives?
6. Why are parameters themselves `Value` objects?

### Assignment M3.1

After finishing the video, close it and draw this graph by hand:

```python
a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
e = a * b
d = e + c
f = d.tanh()
```

For every node, write:

- Forward value.
- Immediate parents.
- Operation.
- Local derivative with respect to each parent.
- Final gradient after backpropagation from `f`.

Then verify the final gradients using PyTorch.

### First-principles questions

1. Is backpropagation a calculus algorithm, a graph algorithm, or both?
2. Why does reverse mode suit a scalar loss with many parameters?
3. What exactly does `x.grad` mean?
4. Why does calling backward twice accumulate gradients unless they are cleared?
5. Why does an update such as `p.data += -lr * p.grad` not itself belong in the differentiable graph in this tiny engine?

### Deliverable

- `notes/session-01.md`.
- Hand-derived graph and gradients.
- PyTorch verification.

### Exit criteria

- You can explain local derivative multiplied by upstream gradient.
- You can explain why branches require gradient accumulation.
- You can describe the need for reverse topological order.

---

## Session 2 — Build scalar forward mode

### Objective

Build the `Value` object and all required forward operations from a blank file before implementing general backward propagation.

### Required API

Implement:

```python
class Value:
    data
    grad
    _prev
    _op
    label
    _backward

    __add__
    __mul__
    __pow__
    __neg__
    __sub__
    __radd__
    __rmul__
    __truediv__
    tanh
    exp
    relu
    backward
```

`backward()` may remain unimplemented until Session 3.

### Work order

1. Implement construction and `repr`.
2. Implement addition.
3. Implement multiplication.
4. Support Python scalars by converting them to `Value` objects.
5. Implement reverse operators such as `2 + x` and `3 * x`.
6. Build subtraction and negation from addition and multiplication.
7. Implement powers needed for division.
8. Implement `exp`, `tanh`, and `relu`.
9. Store parent references and operation labels for every produced node.
10. Manually inspect a small graph.

### Tests

Test forward values for:

```python
x + y
x * y
x**2
-x
x - y
x / y
2 + x
3 * x
x.tanh()
x.exp()
x.relu()
```

Compare each value against Python `math` or PyTorch.

### Assignment M3.2

Build one expression using at least:

- A repeated input.
- A branch.
- Multiplication.
- Addition.
- One nonlinearity.

Print every node's label, operation, parents, and value. Explain why the graph is a DAG rather than necessarily a tree.

### First-principles questions

1. Why does the output node store its parents instead of each parent storing only its children?
2. Why is a closure convenient for `_backward`?
3. Why does `Value + float` need coercion?
4. Which state belongs to the forward pass and which state exists for backward?
5. What would break if graph edges were not retained?

### Deliverable

- `micrograd/engine.py` with all forward operations.
- `tests/test_engine.py` forward tests.
- `notes/session-02.md`.

### Exit criteria

- Every required forward operation matches its reference.
- Mixed `Value`/Python-scalar expressions work.
- Graph metadata is correct.

---

## Session 3 — Implement reverse-mode autodiff

### Objective

Implement correct backward propagation through arbitrary scalar DAGs.

### Work order

1. Write the local derivative of addition.
2. Write the local derivative of multiplication.
3. Write derivatives for power, `exp`, `tanh`, and ReLU.
4. Store each operation's local backward rule in `_backward`.
5. Build a topological ordering with depth-first traversal.
6. Seed the final output gradient with 1.
7. Traverse nodes in reversed topological order.
8. Accumulate parent gradients using `+=`.
9. Add a helper to clear gradients.

### Critical tests

Include separate tests for:

- A single operation.
- A long chain.
- A branch.
- A shared node: `y = x + x`.
- Reuse through different operations: `y = x * x + x`.
- A nonlinearity.
- A mixed expression using all required operations.
- Calling backward twice.

For each test:

- Compute with your engine.
- Compute the same expression with a scalar PyTorch tensor.
- Compare forward value and every leaf gradient.
- Use at least 20 random seeds for the mixed-expression test.

The official micrograd tests use PyTorch as the gradient reference, which is the correct pattern for this validation.[cite:133]

### Finite-difference check

For at least one expression, independently check:

\[
\frac{\partial f}{\partial x}
\approx
\frac{f(x+h)-f(x-h)}{2h}
\]

Compare:

- Analytical derivative if manageable.
- Your engine's derivative.
- PyTorch autograd derivative.
- Central finite-difference estimate.

### Assignment M3.3

Deliberately replace one `+=` in a backward rule with `=`. Identify which tests fail and explain why. Restore the correct code afterward.

Then deliberately execute nodes in forward topological order during backward. Identify the resulting failure and explain why some nodes receive incomplete upstream gradients.

### First-principles questions

1. Why is the final gradient initialized to 1 rather than 0?
2. Why does topological order matter?
3. Why must gradients accumulate?
4. What is the difference between a local derivative and a final derivative?
5. What part of this implementation is specific to scalars?
6. What conceptual work does PyTorch perform that your engine now reproduces?

### Deliverable

- Complete `Value.backward()`.
- Passing gradient-parity tests.
- Finite-difference result.
- Failure-analysis notes.

### Exit criteria

- All gradient tests pass.
- You can reconstruct the backward algorithm without notes.
- You can explain both intentional failures.

---

## Session 4 — Build and train an MLP

### Objective

Use your scalar autograd engine to build a neural-network library and complete a full optimization loop.

### Required classes

Implement:

```python
class Module:
    parameters()
    zero_grad()

class Neuron(Module):
    __call__(x)
    parameters()

class Layer(Module):
    __call__(x)
    parameters()

class MLP(Module):
    __call__(x)
    parameters()
```

### Work order

1. Implement a neuron as a weighted sum plus bias.
2. Add a nonlinear activation.
3. Group neurons into a layer.
4. Stack layers into an MLP.
5. Recursively collect parameters.
6. Implement `zero_grad()`.
7. Create a tiny binary-classification dataset such as XOR or two moons.
8. Write a scalar loss.
9. Run forward, zero-grad, backward, and update steps.
10. Record loss over time.

The reference micrograd repository demonstrates that a scalar engine is sufficient to train a two-hidden-layer MLP for binary classification.[cite:133]

### Required invariants

- Every learnable weight and bias appears exactly once in `parameters()`.
- Gradients are cleared before every backward pass.
- Parameter updates happen outside graph construction.
- Loss decreases substantially from its initial value.
- At least one prediction changes in the correct direction.

### Assignment M3.4

Add tests that verify:

- Parameter count for a known architecture.
- Output count for every layer.
- `zero_grad()` clears every parameter gradient.
- One MLP output and all parameter gradients match an equivalent small PyTorch MLP for one fixed example.

### First-principles questions

1. What makes a number a learnable parameter rather than an activation?
2. Why does a neuron need a bias?
3. Why do stacked linear layers without nonlinearities collapse into one linear transformation?
4. Why must gradients be cleared before the next optimization step?
5. What changes during training: graph structure, parameter values, or both?

### Deliverable

- `micrograd/nn.py`.
- `micrograd/train.py`.
- Parameter and parity tests.
- `outputs/micrograd_loss.csv`.
- A short explanation of one complete training step.

### Exit criteria

- The tiny classifier learns.
- The MLP gradient test matches PyTorch.
- You can write the five-step training loop from memory: forward, loss, zero, backward, update.

---

## Session 5 — Watch makemore bigram

### Objective

Move from generic supervised learning to autoregressive next-character language modeling.

### Material

Watch **The spelled-out intro to language modeling: building makemore**. The lecture builds a character bigram model while introducing tensor operations, model training, sampling, and negative log-likelihood evaluation.[cite:131]

Video: <https://youtu.be/PaCmpygFfXo>

Notebook: <https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part1_bigrams.ipynb>

### Concepts to capture

- Corpus and examples.
- Character vocabulary.
- String-to-index and index-to-string mappings.
- Beginning/end boundary token.
- Bigram training examples.
- Count matrix.
- Row-wise probability distribution.
- Smoothing.
- Likelihood, log-likelihood, and average negative log-likelihood.
- Sampling with a seeded generator.
- One-hot encoding.
- Logits and softmax.
- Neural bigram model.
- Regularization as count smoothing's neural analogue.

### Pre-code reasoning

Given the name `emma`, write the complete sequence of input-target pairs when `.` is the boundary token.

Then answer:

1. Why does `.` appear as both an input and a target in different pairs?
2. What does row `i` of a bigram count matrix represent?
3. Why must each probability row sum to 1?
4. Why is the probability of a whole name a product of conditional probabilities?
5. Why does taking logs turn that product into a sum?
6. Why minimize negative log-likelihood rather than maximize likelihood directly?

### Dataset implementation

Implement:

```python
load_words(path)
build_vocabulary(words)
build_bigram_examples(words, stoi)
```

Required checks:

- Mappings are mutual inverses.
- Boundary token exists once.
- Every target ID lies in `[0, V)`.
- Number of bigram examples equals the sum of `len(word) + 1` over words.
- Decoding the examples reconstructs expected pairs.

### Assignment M3.5

Create `makemore/data.py` without copying the notebook. Print:

- Number of words.
- Vocabulary size.
- Five encoded words.
- Five decoded words.
- First 20 bigram pairs in character and integer form.

### Deliverable

- `makemore/data.py`.
- Dataset tests.
- `notes/session-05.md`.

### Exit criteria

- You can derive all pairs for any name.
- You can define the distribution estimated by a bigram model.
- The dataset checks pass.

---

## Session 6 — Count-based bigram model

### Objective

Build, evaluate, and sample from a statistical bigram language model before introducing neural-network parameters.

### Work order

1. Allocate a count matrix `N` with shape `[V, V]`.
2. Increment `N[x, y]` for every training pair.
3. Add a chosen smoothing value.
4. Normalize each row to produce `P`.
5. Assert every row sums approximately to 1.
6. Calculate average NLL for the corpus.
7. Build a uniform-model baseline.
8. Generate names autoregressively.
9. Use a dedicated seeded `torch.Generator` for reproducible samples.

`torch.multinomial` samples indices from non-negative row weights or probabilities, and accepts a generator for reproducibility.[cite:145] A dedicated generator lets sampling be repeatable without coupling every random operation to global state.[cite:168]

### Required derivations

Write and explain:

\[
P(y\mid x)=\frac{N_{x,y}+\alpha}{\sum_j(N_{x,j}+\alpha)}
\]

\[
\text{NLL}=-\frac{1}{M}\sum_{m=1}^{M}\log P(y_m\mid x_m)
\]

For a uniform model over `V` characters, derive why the expected per-token NLL is `log(V)`.

### Controlled smoothing experiment

Evaluate at least four smoothing values, including:

- No smoothing.
- A small positive value.
- A moderate value.
- A deliberately large value.

Record training NLL and explain:

- Why no smoothing can assign zero probability to unseen pairs.
- Why excessive smoothing approaches a uniform distribution.
- Why smoothing changes generated samples.

Do not use the final test split to tune smoothing in this session. Formal train/dev/test usage begins with the MLP sessions.

### Assignment M3.6

Generate at least 20 names with a fixed seed and save them. For five generated transitions, print the previous character, sampled character, and sampled probability.

Then deliberately normalize the count matrix along the wrong dimension. Write which probability statement that incorrect tensor represents and why generation becomes conceptually wrong.

### First-principles questions

1. What exactly is learned by counting?
2. Where is the model's memory stored?
3. Why is a count matrix already a language model?
4. Why does a lower NLL mean the model assigned more probability to observed targets?
5. Why can a model with good average NLL still generate strange names?
6. Why does autoregressive generation need to feed the sampled character back as the next input?

### Deliverable

- `makemore/bigram_counts.py`.
- Smoothing-results table.
- Seeded samples.
- Wrong-dimension failure analysis.

### Exit criteria

- NLL is computed correctly.
- Samples are reproducible.
- You can explain each matrix axis and normalization dimension.

---

## Session 7 — Neural bigram model

### Objective

Show that the count model's conditional-probability table can be represented and learned as a one-layer neural network.

### Model

For input character `x`:

1. Convert `x` to a one-hot vector of length `V`.
2. Multiply by `W` with shape `[V, V]`.
3. Interpret the result as next-character logits.
4. Apply cross-entropy against target `y`.
5. Update `W` with gradient descent.

PyTorch one-hot encoding appends a class dimension and places a 1 at the indexed class.[cite:167] `torch.nn.functional.cross_entropy` expects unnormalized logits and targets, so do not apply softmax before calling it.[cite:142]

### Shape ledger

Record the shape and meaning of:

| Tensor | Shape | Meaning |
|---|---|---|
| `xs` | `[M]` | Input character IDs |
| `ys` | `[M]` | Target character IDs |
| `xenc` | `[M, V]` | One-hot inputs |
| `W` | `[V, V]` | Learnable transition scores |
| `logits` | `[M, V]` | Scores for every next character |
| `loss` | `[]` | Mean cross-entropy |

### Work order

1. Implement the explicit educational path: logits, exponentiation, normalization, log probabilities, mean NLL.
2. Implement the stable path with `F.cross_entropy`.
3. Verify the losses and gradients are close on ordinary logits.
4. Train `W` with gradient descent.
5. Record the loss curve.
6. Generate names from the learned model.
7. Compare count-based and neural samples.

### Equivalence experiment

Take the smoothed count probabilities `P` and set:

```python
W = P.log()
```

Feed one-hot inputs through `W` and apply softmax. Verify that the resulting probabilities match the count model.

Explain why:

```text
one_hot(x) @ W
```

selects row `x` of `W`.

This experiment is central: it demonstrates that the count model and one-layer neural bigram model represent the same family of conditional distributions, although they estimate parameters differently.

### Regularization experiment

Train with several L2 regularization strengths. Explain why pulling `W` toward zero pushes logits toward equality, which pushes softmax probabilities toward uniformity, making the effect analogous to smoothing.

### Assignment M3.7

Write tests for:

- One-hot dtype and shape.
- Logits shape.
- Cross-entropy equivalence to manual mean NLL.
- Count-table-to-softmax probability equivalence.
- Finite gradients.
- Loss decrease over a short deterministic run.

### First-principles questions

1. Why is a linear layer enough for a bigram model?
2. Why is there no hidden layer?
3. What does each row of `W` mean?
4. Why are logits not probabilities?
5. Why is explicit `exp` and division less numerically stable than cross-entropy?
6. How is L2 regularization related to smoothing?
7. What additional capability would a second previous character require?

### Deliverable

- `makemore/bigram_nn.py`.
- Tests.
- Count-versus-neural comparison.
- Loss curve and generated samples.

### Exit criteria

- Neural bigram training works.
- You can prove the one-hot row-selection behavior.
- You can explain statistical smoothing and neural regularization as related biases toward less extreme distributions.

### Optional stretch

Implement a trigram model using two previous characters. Compare its development NLL with the bigram model. Keep this optional; it must not delay the MLP work.

---

## Session 8 — Watch and build the MLP dataset

### Objective

Extend the context from one character to a fixed window and map discrete character IDs into learned vectors.

### Material

Watch **Building makemore Part 2: MLP**. The official course describes this lecture as an MLP character language model that introduces training, learning-rate tuning, hyperparameters, evaluation, train/dev/test splits, and under/overfitting.[cite:131]

Video: <https://youtu.be/TCH_1BHY58I>

Notebook: <https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part2_mlp.ipynb>

The architecture follows the neural probabilistic language-model idea of learning continuous token representations jointly with a neural conditional-probability model.[cite:143]

### Pre-watch questions

1. Why can a bigram model never distinguish `...ann` from another context ending in `n`?
2. If context length is 3, what training examples are produced by `emma`?
3. Why is a one-hot vector an inefficient input representation for a larger vocabulary?
4. Why can similar tokens benefit from nearby learned representations?
5. Why does concatenating embeddings preserve position within the context?

### Build the dataset

Implement:

```python
build_dataset(words, block_size, stoi)
```

For every word:

1. Start with `block_size` boundary-token IDs.
2. Predict the first character.
3. Shift the context left.
4. Append the observed character.
5. Continue through the word.
6. Predict the final boundary token.

Split words—not individual character pairs—into:

- Training set.
- Development set.
- Test set.

Use a fixed seed. The exact percentages may follow the lecture, but the invariants matter more: training updates parameters, development selects configurations, and test is used only once for the final unbiased estimate.

### Embedding lookup

Create an embedding table `C` of shape `[V, C]`. Given input IDs of shape `[B, T]`, indexing produces embeddings of shape `[B, T, C]`. PyTorch describes `nn.Embedding` as a lookup table mapping index tensors to vectors with one extra embedding dimension.[cite:170]

Implement the lookup first with direct tensor indexing. Then verify the equivalent `nn.Embedding` behavior.

### Shape ledger

For each forward operation, record:

- Symbolic shape.
- Concrete shape.
- Meaning of each axis.
- Whether the tensor is a parameter, activation, target, or scalar metric.

Required path:

```text
X:          [B, T]
C:          [V, C]
emb = C[X]: [B, T, C]
flat:       [B, T*C]
W1:         [T*C, H]
b1:         [H]
h:          [B, H]
W2:         [H, V]
b2:         [V]
logits:     [B, V]
y:          [B]
loss:       []
```

### Assignment M3.8

Return to the Session 0 predictions. Correct each wrong shape and explain the mistaken assumption.

Write tests that verify:

- Dataset context/target alignment.
- Split disjointness at the word level.
- Embedding output shape.
- Flattened shape.
- Logits shape.
- Target range.

### Deliverable

- MLP dataset builder.
- Fixed splits.
- Completed shape ledger.
- Dataset and shape tests.

### Exit criteria

- You can derive every forward shape without executing code.
- You can explain why embedding lookup is row selection.
- You can explain the role of each split.

---

## Session 9 — Train and evaluate the MLP

### Objective

Train the fixed-context MLP correctly, choose hyperparameters using development loss, and generate samples autoregressively.

### Part A — Forward pass

Implement the educational tensor version first:

```python
emb = C[X_batch]
flat = emb.reshape(batch_size, -1)
h = torch.tanh(flat @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Y_batch)
```

Do not begin with `nn.Sequential`; explicit tensors make parameter roles and shapes visible.

### Part B — Mini-batch training

Implement:

1. Sample a mini-batch of training indices.
2. Run the forward pass.
3. Clear gradients.
4. Run backward.
5. Update every parameter.
6. Record step, batch loss, and periodic full train/dev loss.

Explain why mini-batch loss is noisy and why a smoothed or periodic full-split loss is more interpretable.

### Part C — Learning-rate experiment

Run a coarse logarithmic learning-rate sweep. For each candidate, start from the same initialization or clearly state when you do not.

Identify:

- Learning rates that barely change the loss.
- Learning rates that reduce it efficiently.
- Learning rates that become unstable or diverge.

Select a learning rate using development behavior, then train a fresh model with a simple decay schedule.

### Part D — Capacity experiment

Change one capacity variable at a time:

- Embedding dimension.
- Hidden width.
- Context length.

Record training and development NLL. Interpret:

- Both losses high: possible underfitting, poor optimization, or insufficient context.
- Training lower while development worsens: possible overfitting.
- Both improve: useful capacity or context increase.

Do not perform a broad search. Three controlled changes are sufficient.

### Part E — Final evaluation

After selecting the final configuration using development loss:

- Restore or retrain the selected configuration.
- Evaluate the test set once.
- Compare count bigram, neural bigram, and MLP losses on consistent splits where possible.
- The MLP should normally improve over the bigram model; if it does not, treat that as a debugging task rather than hiding it.

### Part F — Generation

Generate at least 30 names:

1. Initialize context with boundary tokens.
2. Embed and flatten the context.
3. Compute next-character logits.
4. Convert to a distribution.
5. Sample one character.
6. Append it to output.
7. Shift the context and append the sampled ID.
8. Stop at the boundary token or a safety length.

Use a fixed seed for the official sample file and a second seed to verify the model is not returning a memorized deterministic list.

### Assignment M3.9

Produce this table with measured values:

| Model | Context | Parameters | Train NLL | Dev NLL | Test NLL | Notes |
|---|---:|---:|---:|---:|---:|---|
| Uniform baseline | 0 | 0 | Measured/derived | Measured/derived | Measured/derived | No learned context |
| Count bigram | 1 | Count table | Measured | Measured | Measured | Smoothed counts |
| Neural bigram | 1 | Measured | Measured | Measured | Measured | Learned logits |
| MLP | `T` | Measured | Measured | Measured | Measured | Learned embeddings and hidden layer |

Do not invent expected losses. Record actual runs.

### First-principles questions

1. Why do embeddings receive gradients only through selected rows?
2. Why is the flattening operation not a learned layer?
3. Why does increasing context length change the first weight matrix's shape?
4. Why should cross-entropy receive logits rather than probabilities?
5. Why does development loss guide hyperparameters while test loss should not?
6. Why can training loss continue falling while development loss rises?
7. Why does generation remain autoregressive even though training examples can be processed in parallel?
8. What information does this MLP still lack that self-attention will later provide?

### Deliverable

- `makemore/mlp.py`.
- Training logs and loss curves.
- Learning-rate experiment.
- Capacity experiment.
- Model comparison table.
- `outputs/samples.txt`.

### Exit criteria

- Training, development, and test roles are correct.
- The MLP trains without non-finite loss.
- The final test result is measured once after selection.
- Samples are generated autoregressively.
- You can derive every forward-pass shape.

---

## Session 10 — Consolidate and defend

### Objective

Turn the month's notebooks and experiments into a coherent implementation you can reproduce and explain without the videos.

### Part A — Refactor

Ensure the repository provides commands equivalent to:

```bash
pytest -q
python -m micrograd.train
python -m makemore.bigram_counts
python -m makemore.bigram_nn
python -m makemore.mlp
python -m makemore.sample --model mlp --seed 2147483647
```

Add a README containing:

- Setup instructions.
- What each model predicts.
- How to run tests and training.
- Actual model/loss comparison.
- Ten generated samples.
- Known limitations.

### Part B — Rebuild from memory

Without notes or reference code, implement on paper or in a blank file:

1. The core `Value` fields.
2. Topological sorting.
3. `backward()`.
4. One addition backward rule.
5. One multiplication backward rule.
6. Bigram dataset construction for one word.
7. Count-matrix normalization.
8. Neural bigram forward pass.
9. MLP shape path from IDs to logits.
10. Autoregressive sampling loop.

This is not a speed contest. The purpose is to reveal missing causal links.

### Part C — Re-answer Session 0

Answer the original eight questions again. For every changed answer, write what caused the correction.

### Part D — Oral examination

Answer aloud without notes:

1. What is reverse-mode automatic differentiation?
2. Why does a scalar loss make reverse mode efficient?
3. Why must backpropagation follow reversed topological order?
4. Why are gradients accumulated at graph branches?
5. What does a bigram language model estimate?
6. Why is average NLL a useful language-model metric?
7. Why is a neural bigram model equivalent in capacity to a conditional table?
8. Why are logits preferable to explicitly normalized probabilities as model outputs during training?
9. What is an embedding table?
10. Why does indexing an embedding table remain differentiable with respect to the selected rows?
11. How does context length affect the MLP input?
12. Why are train, development, and test sets separate?
13. What is underfitting?
14. What is overfitting?
15. Why is generation sequential while training is batch-parallel?
16. What limitation of a fixed-window MLP motivates architectures that can combine information across positions more flexibly?

### Part E — Month report

Write `outputs/month3-report.md` with:

```text
1. What was built
2. Micrograd design and gradient tests
3. Bigram count model
4. Neural bigram model
5. MLP model and tensor shapes
6. Training setup
7. Model comparison table
8. Generated samples
9. Bugs encountered and their causes
10. Remaining uncertainties before Month 4
```

### Exit criteria

- All tests pass from a clean environment.
- The repository runs without hidden notebook state.
- All three model families generate samples.
- The model comparison uses actual measurements.
- You can rebuild the essential algorithms without reference code.
- You can answer at least 13 of the 16 oral questions clearly; any missed question becomes the first review item before Month 4.

---

## Acceptance checklist

### Micrograd

- [ ] `Value` tracks data, gradient, parents, operation, and local backward rule.
- [ ] Addition, multiplication, power, negation, subtraction, division, `exp`, `tanh`, and ReLU work.
- [ ] Backward uses a reverse topological traversal.
- [ ] Gradients accumulate with `+=` at branches.
- [ ] Mixed-expression gradients match PyTorch across 20 seeds.
- [ ] At least one gradient is independently checked with central finite differences.
- [ ] `Neuron`, `Layer`, and `MLP` are implemented.
- [ ] The scalar MLP learns a tiny classification problem.
- [ ] MLP parameter gradients match an equivalent PyTorch model for a fixed example.

### Bigram models

- [ ] Vocabulary and inverse mappings are correct.
- [ ] Boundary-token pairs are correct.
- [ ] Count matrix axes can be explained.
- [ ] Count probabilities normalize over the next-character axis.
- [ ] NLL is computed correctly.
- [ ] Sampling is seeded and reproducible.
- [ ] Neural bigram uses one-hot inputs and raw logits.
- [ ] Manual NLL and `F.cross_entropy` agree closely.
- [ ] Count probabilities can be reproduced by softmax over logged probabilities.
- [ ] The relationship between smoothing and regularization is explained.

### MLP language model

- [ ] Words are split into train/dev/test sets before examples are built.
- [ ] Context windows and targets are correct.
- [ ] Embedding lookup and every forward shape are documented.
- [ ] Mini-batch training works.
- [ ] Learning-rate behavior is measured.
- [ ] At least three controlled capacity/context experiments are recorded.
- [ ] Hyperparameters are selected using development loss.
- [ ] Test loss is measured only after final selection.
- [ ] At least 30 samples are generated.
- [ ] The MLP is compared with both bigram implementations.

### Understanding

- [ ] Reverse-mode autodiff can be explained as calculus plus graph traversal.
- [ ] A complete training step can be written from memory.
- [ ] Bigram conditional probability can be written from memory.
- [ ] Average NLL can be derived from sequence likelihood.
- [ ] The neural bigram/count-table equivalence can be explained.
- [ ] MLP tensor shapes can be derived without running code.
- [ ] Autoregressive generation can be implemented from memory.
- [ ] Fixed-context MLP limitations can be stated clearly before Month 4.

---

## Glossary

### Autodiff and graphs

| Term | Meaning |
|---|---|
| Derivative | Local rate at which an output changes when an input changes |
| Partial derivative | Derivative with respect to one input while treating other inputs as fixed |
| Gradient | Collection of partial derivatives of a scalar output with respect to inputs or parameters |
| Chain rule | Rule for composing derivatives through nested operations |
| Computation graph | Graph whose nodes are values/operations and whose edges describe dependencies |
| DAG | Directed acyclic graph; dependencies have direction and contain no cycle |
| Leaf node | Input or parameter not produced by another tracked operation |
| Intermediate node | Value produced by an operation and consumed by later operations |
| Forward pass | Evaluation of values from inputs and parameters to output/loss |
| Local derivative | Derivative of one operation's output with respect to one immediate input |
| Upstream gradient | Gradient arriving from later graph nodes during backward propagation |
| Backpropagation | Application of the chain rule from output toward inputs/parameters |
| Reverse-mode autodiff | Autodiff mode propagating one scalar output's sensitivity backward to many inputs |
| Topological order | Ordering in which every dependency appears before its consumer |
| Gradient accumulation | Addition of contributions reaching a value through multiple graph paths |
| Finite difference | Numerical derivative estimate produced by perturbing an input |
| Parameter | Learnable value updated by optimization |
| Activation | Intermediate output produced while evaluating a network |
| Zeroing gradients | Clearing accumulated parameter gradients before a new backward pass |
| Gradient descent | Updating parameters in the direction opposite the gradient |

### Language modeling

| Term | Meaning |
|---|---|
| Corpus | Collection of text used to train or evaluate a language model |
| Token | Discrete unit represented by an integer ID; this month, one character |
| Vocabulary | Set of possible tokens |
| Token ID | Integer index assigned to a vocabulary item |
| Boundary token | Special token marking the beginning/end context of an example |
| Language model | Model assigning probabilities to token sequences or next tokens |
| Autoregressive model | Model factorizing sequence probability into next-token conditionals |
| Context | Tokens supplied to predict the next token |
| Bigram | Model using one previous token as context |
| Trigram | Model using two previous tokens as context |
| Conditional probability | Probability of the next token given a context |
| Count matrix | Table containing observed transition frequencies |
| Smoothing | Adding pseudo-counts to avoid extreme or zero probabilities |
| Likelihood | Probability the model assigns to observed data |
| Log-likelihood | Sum of logarithms of probabilities assigned to observations |
| NLL | Negative log-likelihood; minimized during training |
| Cross-entropy | Loss comparing predicted logits/distribution with target classes |
| Logit | Unnormalized real-valued score for a possible class/token |
| Softmax | Function converting logits into a categorical probability distribution |
| Sampling | Randomly choosing a token according to model probabilities |
| Greedy decoding | Choosing the highest-probability token rather than sampling |
| Seed | Initial random-generator state used for reproducibility |

### MLP modeling

| Term | Meaning |
|---|---|
| One-hot vector | Vector with one 1 at a token's index and zeros elsewhere |
| Embedding | Learned dense vector associated with a discrete token |
| Embedding table | Matrix with one learned embedding row per vocabulary item |
| Embedding dimension | Number of learned features in each token vector |
| Context length/block size | Number of previous tokens used to predict the next token |
| Flatten/concatenate | Reshape that joins positional embeddings into one feature vector |
| Hidden layer | Intermediate learned transformation between inputs and outputs |
| Hidden width | Number of units in a hidden layer |
| Nonlinearity | Function such as tanh that prevents stacked layers collapsing into one linear map |
| Tanh | Bounded nonlinear activation with outputs between -1 and 1 |
| Hyperparameter | Configuration chosen outside gradient descent, such as width or learning rate |
| Mini-batch | Subset of training examples used for one parameter update |
| Learning rate | Step-size multiplier applied to gradients during updates |
| Learning-rate decay | Reduction of learning rate later in training |
| Regularization | Bias or penalty intended to discourage overly extreme/flexible solutions |
| Capacity | Amount of function complexity a model can represent |
| Underfitting | Model or optimization fails to fit even training patterns sufficiently |
| Overfitting | Training performance improves while generalization deteriorates |
| Training set | Data used for parameter updates |
| Development set | Held-out data used for model and hyperparameter selection |
| Test set | Held-out data used once for final unbiased evaluation |
| Generalization | Performance on examples not used for parameter updates |

### Tensor reasoning

| Term | Meaning |
|---|---|
| Shape | Length of every tensor axis |
| Axis/dimension | One indexing direction in a tensor |
| Broadcasting | Automatic alignment/expansion of compatible tensor shapes |
| Indexing | Selecting elements or rows using integer IDs |
| Matrix multiplication | Contracting an inner dimension to combine learned features |
| Batch dimension | Axis indexing independent training examples |
| Vocabulary dimension | Axis containing one score or probability per token |
| View/reshape | Reinterpretation of tensor dimensions when layout permits |
| Contiguous tensor | Tensor whose elements follow a standard contiguous memory layout |
| Scalar loss | Zero-dimensional value summarized over examples and used for backward |

---

## Learning materials

Use these materials in order.

### Required spine

1. **Zero to Hero course page** — lecture order, runtimes, and descriptions  
   <https://karpathy.ai/zero-to-hero.html>
2. **Lecture 1: micrograd**  
   <https://youtu.be/VMj-3S1tku0>
3. **Official micrograd repository** — check implementation and tests only after your attempt  
   <https://github.com/karpathy/micrograd>
4. **Official micrograd lecture notebooks**  
   <https://github.com/karpathy/nn-zero-to-hero/tree/master/lectures/micrograd>
5. **Lecture 2: makemore bigram**  
   <https://youtu.be/PaCmpygFfXo>
6. **Official bigram notebook**  
   <https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part1_bigrams.ipynb>
7. **Lecture 3: makemore MLP**  
   <https://youtu.be/TCH_1BHY58I>
8. **Official MLP notebook**  
   <https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part2_mlp.ipynb>

### Targeted PyTorch references

- `torch.nn.functional.one_hot` — use when implementing the neural bigram input representation.[cite:167]  
  <https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.one_hot.html>
- `torch.nn.functional.cross_entropy` — use raw logits and class-index targets.[cite:142]  
  <https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html>
- `torch.multinomial` — use for categorical sampling with an explicit generator.[cite:145]  
  <https://docs.pytorch.org/docs/stable/generated/torch.multinomial.html>
- `torch.nn.Embedding` — use to verify direct embedding-table indexing behavior.[cite:170]  
  <https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html>
- `torch.Generator` — use for reproducible isolated random streams.[cite:168]  
  <https://docs.pytorch.org/docs/stable/generated/torch.Generator.html>

### Optional conceptual reference

- Bengio et al., **A Neural Probabilistic Language Model**. Read only the abstract, model idea, and embedding/context diagram after implementing the MLP. The paper expresses sequence probabilities through next-token conditionals and jointly learns token feature vectors with the probability model.[cite:143]  
  <https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf>

Do not add unrelated transformer courses this month. The three assigned lectures, official notebooks, focused PyTorch API pages, and your own implementation are sufficient.

---

## If time slips

Preserve the conceptual spine in this order:

1. Micrograd backward correctness.
2. Count-based bigram and NLL.
3. Neural bigram equivalence.
4. MLP context/embedding shapes.
5. Correct train/dev/test experiment.
6. Generation.

Reduce optional work before reducing understanding:

- Skip graph visualization styling.
- Skip the optional trigram.
- Run fewer capacity experiments, but keep at least one.
- Reduce sample count.
- Do not skip gradient tests, shape tests, data-split correctness, or the final oral explanation.

## Month 4 readiness gate

Proceed to Month 4 only when you can explain this chain without notes:

> A text corpus is converted into context-target examples. Token IDs select rows from an embedding table. The context embeddings are combined and transformed into logits over the vocabulary. Cross-entropy measures how much probability the model assigns to the observed next token. Reverse-mode autodiff propagates the scalar loss gradient backward through the computation graph, accumulating contributions into every parameter. Gradient descent updates those parameters. During generation, the model repeatedly samples a next token and feeds it back into the context.

If any sentence is merely memorized, use the corresponding oral question and implementation test to reconstruct it from first principles before beginning BatchNorm, WaveNet, or attention.
