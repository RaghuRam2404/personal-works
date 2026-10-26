# Month 4 — Deep-Network Internals, Causal Attention, and GPT from Scratch

## Month purpose

Month 4 is the architectural bridge from the fixed-context character MLP built in Month 3 to a decoder-only Transformer. The month has four linked stages:

1. Diagnose why deep networks can train badly even when the code is mathematically valid.
2. Backpropagate through a tensor-level MLP and BatchNorm manually, without `loss.backward()`.
3. Replace one-shot context mixing with a hierarchical sequence model and strengthen multidimensional tensor-shape reasoning.
4. Derive causal self-attention, implement it three ways, and assemble a character-level GPT without the `transformers` library.

The required video spine is Karpathy's lectures 4–7: activations and BatchNorm, manual tensor backpropagation, WaveNet-style hierarchy, and GPT. Their approximate runtimes are 1h55m, 1h55m, 56m, and 1h56m.[cite:160] The official course describes Lecture 4 as an examination of forward activations, backward gradients, bad scaling, diagnostics, and BatchNorm; Lecture 5 manually differentiates cross-entropy, two linear layers, tanh, BatchNorm, and the embedding table; Lecture 6 builds a deeper tree-like model before the GPT lecture.[cite:185][cite:132][cite:131]

Month 3 already established scalar reverse-mode autodiff, bigram language models, embeddings, fixed context windows, MLPs, cross-entropy, train/dev/test splits, and autoregressive sampling.[cite:183] Month 4 assumes those foundations and uses them rather than repeating them.

## Workload and pacing

This month is substantially denser than Month 3. It contains four lectures, manual tensor calculus, three attention implementations, a Transformer, tests, and a real training run. It is organized into **17 sequential sessions**.

Most sessions are designed for roughly 2–3 focused hours. Manual BatchNorm backward and multi-head attention may need 3–4 hours. If a session exceeds the available time, stop at a marked checkpoint and resume it the next day; do not skip the explanation or test merely to preserve the word “month.”

The required path is approximately 38–48 focused hours plus unattended model-training time. Optional work is explicitly marked and must not delay the required path.

## Scope guardrails

### Required this month

- Activation, gradient, and parameter-update diagnostics.
- Principled initialization intuition.
- BatchNorm forward behavior and train/evaluation differences.
- Manual tensor-level backward pass for the Month 3 MLP with BatchNorm.
- Hierarchical context aggregation and shape tracking.
- Character-level TinyShakespeare data pipeline.
- Causal self-attention implemented three ways:
  1. Explicit Python loops over batch/query/key positions.
  2. Vectorized batched matrix multiplication with a causal mask.
  3. A reusable multi-head `nn.Module` with explicit head splitting.
- Decoder-only Transformer blocks.
- A from-scratch character-level GPT training and generation pipeline.
- Correctness tests for causality, shapes, outputs, and gradients.

### Deferred to Month 5

- Byte Pair Encoding.
- Production-quality tokenization edge cases.
- KV-cache design and implementation.
- Incremental single-token decoding.
- Cached-versus-naive tokens/second benchmarks.
- The final 3x-at-800-tokens cache acceptance test.

### Deferred to later phases

- FlashAttention implementation.
- Triton or CUDA attention kernels.
- Fused QKV and MLP kernels.
- RoPE, RMSNorm, SwiGLU, and grouped-query attention; these enter the modernized architecture in Phase C.
- Distributed pretraining.
- Full GPT-2 reproduction.
- Quantization, serving, and production inference.

### Library restriction

For the core GPT implementation, do not use:

- Hugging Face `transformers`.
- `nn.MultiheadAttention`.
- `F.scaled_dot_product_attention`.
- A copied nanoGPT model.

Those APIs may be used **afterward as references** for parity experiments. The core implementation must expose the attention mathematics and tensor transformations directly.

## Important metric note

Month 4 uses a character tokenizer so its loss is measured in **nats per character token**. Month 5 changes tokenization by adding BPE. Raw nats/token from two different tokenizers are not directly comparable because one “token” represents different amounts of text. Keep the character-level checkpoint and loss log; when BPE is added, report the tokenizer and consider bits/byte if comparing across tokenizations.

The Phase B validation-loss target of 1.6 should therefore be interpreted on the same character-level TinyShakespeare setup used for the reference range. It is a Phase B project target, not a reason to hide or relabel a BPE-token loss.

## Learning method

Use the following loop throughout the month:

1. **Predict:** Write expected shapes, statistics, or gradients before running code.
2. **Observe:** Run one controlled experiment and record what changed.
3. **Derive:** Explain the observation using equations or graph dependencies.
4. **Implement:** Rebuild the idea in a clean module without copying lecture code.
5. **Test:** Compare against autograd, a slower implementation, or a known invariant.
6. **Break:** Introduce one targeted mistake and diagnose its effect.
7. **Explain:** Answer the session's first-principles questions aloud.
8. **Consult references last:** Check official code only after making and testing your own attempt.

The goal is not to remember Karpathy's code. The goal is to be able to reconstruct the code from the probability model, dependency graph, and tensor shapes.

## Month repository

Extend the Month 3 repository or create a dedicated Month 4 repository with this structure:

```text
month4/
  README.md
  requirements.txt
  data/
    names.txt
    tinyshakespeare.txt
  makemore_deep/
    data.py
    layers.py
    diagnostics.py
    batchnorm.py
    manual_backward.py
    wavenet.py
    train.py
  attention/
    loop_attention.py
    vector_attention.py
    multihead_attention.py
    reference.py
  gpt/
    data.py
    tokenizer.py
    model.py
    train.py
    generate.py
    config.py
  tests/
    test_batchnorm.py
    test_manual_backward.py
    test_wavenet_shapes.py
    test_attention_forward.py
    test_attention_gradients.py
    test_causality.py
    test_gpt_shapes.py
    test_gpt_overfit.py
  outputs/
    activation_stats.csv
    gradient_stats.csv
    update_ratios.csv
    initialization_comparison.md
    manual_backward_results.md
    wavenet_results.md
    attention_parity.md
    gpt_training.csv
    gpt_samples.txt
    month4-report.md
  notes/
    session-00.md
    ...
    session-16.md
```

Exploration may happen in notebooks, but the final model and tests must run without hidden notebook state.

## Session map

| Session | Main work | Required result |
|---:|---|---|
| 0 | Bridge and baseline diagnostic | Reproduce Month 3 MLP and record initial health |
| 1 | Lecture 4 | Mental model of saturation, initialization, and BatchNorm |
| 2 | Initialization experiments | Measured logits/activation/gradient effects |
| 3 | BatchNorm from scratch | Train/eval-correct custom BatchNorm with parity tests |
| 4 | Deep-network diagnostics | Activation, gradient, and update-ratio report |
| 5 | Lecture 5 | Complete tensor-level backward dependency map |
| 6 | Manual backward, expanded graph | Every intermediate gradient matches autograd |
| 7 | Manual backward, compact forms | Compact cross-entropy and BatchNorm gradients; training works |
| 8 | Lecture 6 | Hierarchical receptive-field and shape model |
| 9 | WaveNet-style hierarchy | Working hierarchical character model |
| 10 | GPT data and baseline | TinyShakespeare batches and bigram baseline |
| 11 | Attention version 1 | Explicit loop implementation |
| 12 | Attention version 2 | Vectorized causal attention with forward/gradient parity |
| 13 | Attention version 3 | Multi-head module with causality tests |
| 14 | Lecture 7 | Complete decoder-only Transformer mental model |
| 15 | Assemble GPT | Tested from-scratch model and one-batch overfit |
| 16 | Train, evaluate, and defend | Checkpoint, samples, report, memory rebuild |

---

## Session 0 — Bridge from Month 3

### Objective

Recreate the Month 3 MLP in a controlled form and establish a baseline before modifying its initialization or adding normalization.

### Required setup

Use the same names dataset, vocabulary mapping, train/dev/test word split, context length, batch sampler, and MLP structure from Month 3. Freeze these choices for Sessions 0–7 so changes in behavior can be attributed to the experiment rather than to changing data.

### Work

1. Set deterministic Python and PyTorch seeds.
2. Rebuild or import the Month 3 dataset.
3. Instantiate the embedding table, first linear layer, tanh, and output layer.
4. Run one mini-batch forward and backward pass.
5. Save the initial parameters so every initialization experiment can start from a controlled state.
6. Record train and development loss before training.
7. Train briefly and confirm the baseline still learns.

### Baseline measurements

Record:

- Vocabulary size.
- Context length.
- Embedding dimension.
- Hidden width.
- Parameter count.
- Initial logits mean and standard deviation.
- Initial softmax entropy.
- Initial loss and the uniform-distribution loss `log(V)`.
- Hidden pre-activation mean and standard deviation.
- Tanh activation mean, standard deviation, and saturated fraction.
- Gradient standard deviation for each parameter.

Define “saturated tanh” consistently, for example `abs(h) > 0.97`.

### Assignment M4.0

Before looking at Lecture 4, answer:

1. If all initial logits are equal, what probability distribution results?
2. What initial cross-entropy should that imply for vocabulary size `V`?
3. Why can very large logits create an unnecessarily high initial loss?
4. What happens to the derivative of tanh when its input magnitude is large?
5. Why can a network have finite loss and still train poorly?
6. Why might changing only weight scale alter gradient flow through many layers?
7. What statistics would reveal an unhealthy network before waiting for the final loss curve?

Mark each answer as **confident**, **uncertain**, or **guess**.

### Deliverable

- Reproducible baseline script.
- `notes/session-00.md`.
- One row of baseline statistics in each diagnostics CSV.

### Exit criteria

- The Month 3 MLP trains from a clean run.
- Initial and backward statistics are recorded.
- The baseline checkpoint can be restored exactly.

---

## Session 1 — Activations, gradients, BatchNorm

### Objective

Understand why initialization influences logits, activation saturation, backward gradients, and early optimization.

### Material

Watch **Building makemore Part 3: Activations & Gradients, BatchNorm**.

- Video: <https://youtu.be/P6sfmUTpUmc>
- Official notebook: <https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part3_bn.ipynb>

The official description emphasizes forward activation statistics, backward gradient statistics, scaling failures, diagnostic visualizations, and Batch Normalization.[cite:185]

### Watching rules

- Pause before every “fix” and predict which statistic will change.
- Do not copy the final layer classes during the first watch.
- For every plot, write what is on the x-axis, y-axis, and what failure it is intended to reveal.
- Distinguish observed evidence from an explanation.

### Required concepts

- Overconfident initial logits.
- Expected initial loss under a uniform prediction.
- Saturating nonlinearities.
- Fan-in and fan-out.
- Variance propagation.
- Gain for an activation.
- Kaiming initialization intuition.
- Activation distributions.
- Activation-gradient distributions.
- Parameter-gradient distributions.
- Gradient-to-data ratio.
- Update-to-data ratio.
- Batch normalization.
- Learnable gain and bias.
- Running mean and variance.
- Training versus evaluation behavior.

### Assignment M4.1

Draw the Month 3 MLP as a dependency chain:

```text
IDs → embeddings → concatenate → linear → tanh → linear → logits → loss
```

For each arrow, write:

- Input shape.
- Output shape.
- Whether the operation can expand or shrink variance.
- Whether it can saturate.
- Which local derivative controls backward flow.

Then answer aloud:

1. Why is a high initial loss not automatically an optimizer problem?
2. Why does scaling the final layer toward zero make initial predictions less confident?
3. Why can shrinking hidden pre-activations improve tanh gradients?
4. Why is “mean near zero” alone insufficient to diagnose a layer?
5. Why is a healthy forward distribution not proof of healthy backward gradients?
6. Why does BatchNorm introduce dependence between examples in a training batch?

### Deliverable

- Lecture notes limited to causal explanations.
- Annotated dependency chain.
- Six oral answers.

### Exit criteria

- You can explain overconfident logits and tanh saturation without referring to the video.
- You can name the statistics needed to inspect forward, backward, and parameter-update health.

---

## Session 2 — Initialization experiments

### Objective

Turn initialization claims into controlled measurements on the Month 3 MLP.

### Experiment A — Output confidence

Create at least three output-layer initializations:

- Unscaled random weights.
- Small random weights.
- Zero output weights with an explicitly justified bias choice.

Keep all other parameters identical. For each initialization, record:

- Logits mean and standard deviation.
- Mean softmax entropy.
- Maximum predicted probability averaged over examples.
- Initial train and development loss.
- First 50 optimization-step losses.

Explain why the desired initial loss is close to `log(V)` when no character class should initially dominate.

### Experiment B — Tanh saturation

Vary only the first-layer weight scale. For each scale, record:

- Hidden pre-activation mean and standard deviation.
- Tanh activation mean and standard deviation.
- Saturated fraction.
- Gradient standard deviation entering and leaving tanh.
- Loss after the same number of updates.

Relate the observations to:

$$
\frac{d}{dx}\tanh(x)=1-\tanh^2(x)
$$

### Experiment C — Fan-in scaling

For a linear input with fan-in `fan_in`, compare:

- Fixed standard deviation independent of fan-in.
- Standard deviation proportional to `1/sqrt(fan_in)`.
- PyTorch Kaiming initialization configured for tanh or the lecture's equivalent gain-based scale.

Do not memorize one magic constant. Explain the objective: prevent signal magnitude from systematically exploding or shrinking as it passes through layers.

### Assignment M4.2

Write `initialization_comparison.md` with a table of measured results and answer:

1. Which intervention corrected initial confidence?
2. Which intervention corrected hidden saturation?
3. Did either intervention solve both problems automatically?
4. Which measurement would have detected each problem fastest?
5. Why must controlled runs share data, seeds, architecture, and update count?

### Intentional break

Initialize every weight to zero. Train briefly and inspect whether hidden units remain identical. Explain the symmetry problem: identical neurons receive identical inputs and gradients and therefore fail to specialize.

### Deliverable

- Initialization experiment script.
- Measured comparison table.
- Activation and gradient plots.
- Zero-initialization failure note.

### Exit criteria

- You can predict how scaling a layer affects logits or tanh saturation.
- You can distinguish output-layer confidence from hidden-layer signal propagation.
- Results are based on controlled measurements, not only visual impressions.

---

## Session 3 — BatchNorm from scratch

### Objective

Implement BatchNorm as a differentiable layer and understand exactly how training and evaluation differ.

### Forward equations

For input `x` with shape `[B, C]`, normalize each feature across the batch:

$$
\mu_c=\frac{1}{B}\sum_{b=1}^{B}x_{b,c}
$$

$$
\sigma_c^2=\frac{1}{B}\sum_{b=1}^{B}(x_{b,c}-\mu_c)^2
$$

$$
\hat{x}_{b,c}=\frac{x_{b,c}-\mu_c}{\sqrt{\sigma_c^2+\epsilon}}
$$

$$
y_{b,c}=\gamma_c\hat{x}_{b,c}+\beta_c
$$

PyTorch BatchNorm normalizes per feature across mini-batches, learns one scale and bias per feature, updates running statistics during training, and normally uses those stored statistics during evaluation.[cite:209]

### Required implementation

Implement a custom `BatchNorm1d` supporting:

- `dim` or number of features.
- `eps`.
- `momentum`.
- Learnable `gamma` and `beta`.
- Non-parameter buffers `running_mean` and `running_var`.
- `training` state.
- Two-dimensional input `[B, C]`.
- Three-dimensional input `[B, T, C]` as required by the WaveNet session.

For `[B, T, C]`, define explicitly which axes are normalized. Match the lecture implementation first, then compare with PyTorch's documented `BatchNorm1d` conventions before claiming parity.

### Train/eval reasoning

During training:

- Normalize with current mini-batch statistics.
- Update running statistics without tracking those updates in autograd.

During evaluation:

- Normalize with stored running statistics.
- Do not update buffers.

Be careful when comparing exact values: PyTorch uses a biased variance for current training normalization but stores an unbiased variance estimate in its running buffer.[cite:209] Align variance conventions and momentum semantics before writing an exact parity assertion.

### Required tests

1. Output shape matches input shape.
2. Training output is approximately zero mean and unit variance before the affine transform.
3. `gamma` and `beta` change scale and location as expected.
4. Running buffers change in training mode.
5. Running buffers do not change in evaluation mode.
6. Evaluation of one example does not require a batch.
7. Outputs and gradients match an aligned PyTorch reference.
8. `gamma` and `beta` appear in parameters; running statistics do not.
9. Serialization restores parameters and buffers.

### Assignment M4.3

Perform three evaluations of the same example:

- As part of one training batch.
- As part of a different training batch.
- Alone in evaluation mode after running statistics have been accumulated.

Explain why the two training outputs can differ and why the evaluation output should not depend on the other examples supplied at inference.

### Intentional break

Evaluate while the layer remains in training mode with batch size 1. Record the result or failure and explain why `model.eval()` is semantically necessary, not merely an optimization switch.

### First-principles questions

1. Across which axis is each feature normalized?
2. Why are `gamma` and `beta` needed after normalization?
3. Why are running statistics buffers rather than optimizer-updated parameters?
4. Why does BatchNorm make a sample's training-time output depend on other samples?
5. Why can small batches make statistics noisy?
6. Why is the bias before a normalized linear output often redundant?

### Deliverable

- `makemore_deep/batchnorm.py`.
- `tests/test_batchnorm.py`.
- Train/eval demonstration.

### Exit criteria

- Custom BatchNorm passes aligned forward and gradient tests.
- Train/eval behavior is correct.
- You can explain every tensor axis and state variable.

---

## Session 4 — Deep-network diagnostics

### Objective

Build reusable diagnostics rather than relying on final loss as the only signal.

### Build a small layer API

Implement simple educational layers similar to PyTorch:

```python
class Linear
class BatchNorm1d
class Tanh
class Sequential
```

Each layer should expose or retain its latest output only when diagnostics are enabled. Avoid permanently retaining every graph during normal training.

### Required diagnostics

For every hidden layer, periodically record:

- Activation mean.
- Activation standard deviation.
- Saturated fraction for tanh.
- Activation-gradient mean.
- Activation-gradient standard deviation.
- Parameter mean and standard deviation.
- Parameter-gradient mean and standard deviation.
- Gradient-to-data ratio.
- Update-to-data ratio.

A useful relative update diagnostic is:

$$
\log_{10}\left(\frac{\eta\,\operatorname{std}(g)}{\operatorname{std}(p)}\right)
$$

where `eta` is the learning rate, `g` the gradient, and `p` the parameter. Treat it as a comparative diagnostic, not a universal pass/fail constant.

### Controlled comparison

Train three deep MLP variants for an equal number of steps:

1. Deliberately poor initialization, no normalization.
2. Principled initialization, no normalization.
3. Principled initialization plus BatchNorm.

Use the same data split, mini-batch sequence where possible, optimizer, and learning-rate schedule.

### Assignment M4.4

Write a one-page “network health report” answering:

- Which layers saturate?
- Where do activation gradients shrink or grow?
- Which parameters receive unusually large relative updates?
- Which intervention improves the observed problem?
- What remains imperfect after BatchNorm?

### Intentional break

Remove `torch.no_grad()` from running-statistic updates or accidentally retain diagnostic outputs across all steps. Observe graph/memory behavior and explain why diagnostic instrumentation can itself change resource usage.

### Deliverable

- `makemore_deep/diagnostics.py`.
- Three measured runs.
- Activation, gradient, and update-ratio plots.
- Health report.

### Exit criteria

- You can diagnose at least one failure before looking at final validation loss.
- You can explain why one scalar loss cannot localize internal pathologies.

---

## Session 5 — Becoming a Backprop Ninja

### Objective

Move from scalar micrograd intuition to vectorized tensor calculus over a full MLP with BatchNorm.

### Material

Watch **Building makemore Part 4: Becoming a Backprop Ninja**.

- Video: <https://youtu.be/q8SA3rM6ckI>
- Official notebook: <https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part4_backprop.ipynb>

The lecture manually backpropagates through cross-entropy, the second linear layer, tanh, BatchNorm, the first linear layer, and the embedding table without calling `loss.backward()`.[cite:132]

### Watching rules

- Do not type the gradient expressions while watching.
- Pause before each derivative and identify the output gradient, local Jacobian action, and desired input shape.
- Write reduction and broadcasting axes explicitly.
- Treat shape restoration as part of differentiation.

### Build the dependency map

Expand the forward pass into elementary operations:

```text
Xb
→ emb
→ embcat
→ hprebn
→ bnmeani
→ bndiff
→ bndiff2
→ bnvar
→ bnvar_inv
→ bnraw
→ hpreact
→ h
→ logits
→ logit_maxes
→ norm_logits
→ counts
→ counts_sum
→ counts_sum_inv
→ probs
→ logprobs
→ loss
```

Your exact names may differ, but every elementary operation needed for the expanded backward pass must be visible.

### Gradient ledger

Create a table with columns:

| Node | Forward shape | Parents | Incoming gradient shape | Reduction/broadcast reversal | Manual-gradient status |
|---|---|---|---|---|---|

Populate it before implementing gradients.

### Assignment M4.5

For these operations, derive the vector-Jacobian product and shape behavior:

- Addition with broadcasting.
- Matrix multiplication.
- Elementwise multiplication.
- Mean.
- Sum with `keepdim=True`.
- Exponentiation.
- Reciprocal.
- Logarithm.
- Tanh.
- Index selection.
- Embedding lookup with repeated indices.

Then answer:

1. Why is writing a full Jacobian usually unnecessary?
2. Why must gradients be summed over broadcast dimensions?
3. Why does an embedding gradient require accumulation for repeated IDs?
4. Why can two algebraically equivalent forward expressions lead to differently convenient backward derivations?

### Deliverable

- Complete forward dependency graph.
- Gradient ledger.
- Operation derivative notes.

### Exit criteria

- Every forward node has a known shape and parent set.
- You can state the reverse traversal order.
- Broadcasting and repeated-index accumulation are understood before coding.

---

## Session 6 — Expanded manual backward

### Objective

Implement the complete backward pass one elementary operation at a time and compare every intermediate gradient with PyTorch autograd.

### Test harness

Retain gradients for intermediate tensors and define a comparison helper that reports:

- Exact-shape agreement.
- Maximum absolute difference.
- Relative difference or `allclose` result.
- Whether both tensors are finite.

Use float64 for a small gradient-check case if numerical noise obscures correctness, then repeat the working implementation in the normal training dtype.

### Work order

Differentiate backward through:

1. Mean negative selected log probabilities.
2. Logarithm.
3. Probability normalization.
4. Reciprocal and sum.
5. Exponentiation.
6. Max-shift used for numerical stability.
7. Second linear layer.
8. Tanh.
9. BatchNorm affine transform.
10. Normalization by inverse standard deviation.
11. Variance.
12. Centering by mean.
13. First linear layer.
14. Reshape/concatenation.
15. Embedding lookup.

Compare each manual gradient immediately instead of implementing everything and testing only at the end.

### Required correctness cases

- A fixed lecture-style mini-batch.
- A batch containing repeated character IDs.
- Batch size greater than 1.
- At least five random seeds.
- Non-symmetric parameter initialization so an incorrect formula is not accidentally masked.

### Assignment M4.6

Deliberately introduce each bug separately:

- Fail to sum a broadcast bias gradient.
- Use assignment rather than accumulation for embedding gradients.
- Drop the `1 / B` factor from a mean.
- Differentiate tanh using the pre-activation rather than the tanh output incorrectly.

For each bug, identify the first intermediate gradient that diverges from autograd and explain why.

### First-principles questions

1. Why compare every intermediate rather than only parameter gradients?
2. Why can a shape-correct gradient still be mathematically wrong?
3. Why do repeated token IDs create a scatter-add pattern?
4. Why does numerical stabilization with a max not alter softmax probabilities?
5. What role does the upstream gradient play at every node?

### Deliverable

- `makemore_deep/manual_backward.py` expanded backward.
- `tests/test_manual_backward.py`.
- Intermediate comparison table.
- Four failure analyses.

### Exit criteria

- Every intermediate and parameter gradient agrees with autograd within justified tolerance.
- You can locate a deliberately introduced bug from the first failed node.

---

## Session 7 — Compact backward and training

### Objective

Collapse the expanded derivation into efficient layer-level gradients and use the manual backward pass in a real optimizer loop.

### Compact cross-entropy gradient

Derive and implement the combined gradient with respect to logits:

$$
\frac{\partial L}{\partial z}
=
\frac{p-y_{\text{one-hot}}}{B}
$$

Do not merely copy the expression. Explain how softmax and negative log-likelihood derivatives simplify into it.

### Compact BatchNorm gradient

Derive or reconstruct the layer-level gradient from the expanded graph. A valid compact form should account for:

- Direct dependence on normalized values.
- Dependence through batch mean.
- Dependence through batch variance.
- Per-feature summations across the batch.
- `gamma` scaling.

Check the compact result against both the expanded manual gradient and autograd.

### Manual training loop

Train the MLP without calling `loss.backward()`:

1. Forward pass.
2. Compute compact manual gradients.
3. Assign or store parameter gradients.
4. Update parameters under `torch.no_grad()`.
5. Clear temporary state.
6. Periodically compute train and development loss.

### Assignment M4.7

Compare three one-step updates from identical starting parameters:

- PyTorch autograd.
- Expanded manual backward.
- Compact manual backward.

Assert that resulting parameter tensors agree within tolerance.

Then run enough manual updates to show the loss falls and generated names remain valid.

### First-principles questions

1. Why is the compact formula faster than traversing every elementary node manually?
2. What information from the forward pass must be cached for backward?
3. Why is a fused backward implementation still just the chain rule?
4. What could go wrong if cached forward values are modified in place?
5. Why does successful loss reduction not replace gradient parity tests?

### Deliverable

- Compact cross-entropy backward.
- Compact BatchNorm backward.
- One-step parameter-update parity test.
- Manual-backward training curve.
- `outputs/manual_backward_results.md`.

### Exit criteria

- Compact and expanded gradients match.
- Manual training reduces loss.
- You can explain cross-entropy's compact gradient from first principles.

---

## Session 8 — WaveNet-style hierarchy

### Objective

Understand why a flat context MLP scales poorly and how hierarchical pairwise aggregation expands the receptive field gradually.

### Material

Watch **Building makemore Part 5: Building a WaveNet**.

- Video: <https://youtu.be/t3YJ5hKiMQ0>
- Official notebook: <https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part5_cnn1.ipynb>

The official course describes this model as a deeper tree-like architecture similar to WaveNet, while emphasizing `torch.nn`-style layers, multidimensional shapes, and normal development workflow.[cite:131]

### Pre-watch reasoning

For context length 8, compare:

- Flat MLP: concatenate all 8 embeddings and mix them in one linear layer.
- Hierarchical model: combine adjacent pairs, then pairs of pair-representations, then the final pair.

Answer:

1. How many aggregation levels are needed to combine 8 positions pairwise?
2. What is the receptive field after each level?
3. Why can the same transformation be reused across positions at one level?
4. Why is hierarchical aggregation not yet attention?
5. Which tensor axis represents time/space at every level?

### Required concepts

- Receptive field.
- Hierarchical aggregation.
- Weight sharing over positions.
- `FlattenConsecutive`.
- Rank-3 activations `[B, T, C]`.
- BatchNorm over multi-dimensional activations.
- Shape-preserving linear transformation over the last dimension.
- Relationship between tree depth and context length.

### Assignment M4.8

Draw the path for context positions 0–7 through three pairwise aggregation levels. For the final representation, identify exactly which input positions can influence it.

Write the complete symbolic shape sequence for:

```text
[B, 8]
→ embedding
→ first pair flatten
→ linear
→ BatchNorm
→ tanh
→ second pair flatten
→ linear
→ BatchNorm
→ tanh
→ third pair flatten
→ linear
→ vocabulary logits
```

### Deliverable

- Lecture notes.
- Receptive-field diagram.
- Symbolic shape ledger.

### Exit criteria

- You can explain the difference between flat, hierarchical, recurrent, and attention-based context mixing at a high level.
- Every hierarchy level's receptive field and tensor shape can be derived.

---

## Session 9 — Build the hierarchy

### Objective

Implement and train the WaveNet-style character model while preserving explicit shape reasoning.

### Required layers

Implement or finalize:

```python
class Linear
class BatchNorm1d
class Tanh
class Embedding
class FlattenConsecutive
class Sequential
```

`Linear` should operate on the final dimension of rank-2 or rank-3 input. `FlattenConsecutive(n)` should group adjacent positions and merge `n` neighboring feature vectors.

### Work order

1. Change the context length from 3 to a power-of-two value such as 8.
2. Verify the flat wider-context MLP first.
3. Implement `FlattenConsecutive(2)`.
4. Test it on a small numbered tensor whose grouping can be inspected manually.
5. Stack hierarchical blocks.
6. Verify each intermediate shape.
7. Ensure BatchNorm handles rank-3 input correctly.
8. Train and record train/development loss.
9. Generate names using the final model.
10. Compare parameter count and loss with the flat MLP.

### Required tests

- Correct grouping for `FlattenConsecutive`.
- Error when the time dimension is not divisible by the group size.
- Rank-2 behavior when the time axis collapses to 1.
- Linear transforms only the final dimension.
- BatchNorm preserves shape.
- No training/evaluation running-stat updates in evaluation mode.
- Final logits shape `[B, V]`.

### Assignment M4.9

Produce a comparison table:

| Model | Context length | Parameters | Aggregation depth | Train NLL | Dev NLL | Notes |
|---|---:|---:|---:|---:|---:|---|
| Month 3 flat MLP | Measured | Measured | 1 | Measured | Measured | Baseline |
| Wider flat MLP | Measured | Measured | 1 | Measured | Measured | All context mixed at once |
| Hierarchical model | Measured | Measured | Measured | Measured | Measured | Pairwise growth of receptive field |

Explain performance differences without claiming causation from one run alone.

### Intentional break

Swap the batch and time dimensions before BatchNorm or normalize across the wrong axes. Record which statistics appear plausible and which semantic meaning is wrong. This exercise is about preventing “shape-valid but conceptually invalid” code.

### Deliverable

- `makemore_deep/wavenet.py`.
- Layer tests.
- Comparison table and generated names.

### Exit criteria

- The hierarchical model trains and generates.
- Every shape and receptive field is explainable.
- You can identify the limitation: the pattern of communication is fixed by the hierarchy rather than selected dynamically from content.

---

## Session 10 — TinyShakespeare and baseline

### Objective

Build a clean character-level sequence dataset and reproduce the simple baseline that the GPT lecture evolves into attention.

### Data

Use TinyShakespeare from the source linked by the GPT lecture repository:

<https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt>

The lecture's reference GPT uses the TinyShakespeare text, a character vocabulary, random fixed-length chunks, and shifted next-character targets.[cite:189]

### Character tokenizer

Implement:

```python
class CharTokenizer:
    encode(text) -> list[int]
    decode(ids) -> str
    vocab_size
```

Tests:

- `decode(encode(text)) == text` for representative text.
- All encoded IDs are in range.
- Vocabulary order is deterministic.
- Unknown-character behavior is explicit.

### Contiguous split

Split the encoded stream into contiguous training and validation regions. Do not randomly distribute individual characters between splits, because overlapping sequence windows could otherwise leak nearly identical local context across them.

### Batch sampler

Given `block_size = T` and `batch_size = B`, return:

```text
x: [B, T]
y: [B, T]
```

For every row:

```text
x = data[i : i + T]
y = data[i + 1 : i + T + 1]
```

Write a manual decoded example proving the one-token shift.

### Baseline model

Implement a token-transition model that maps each input token directly to logits over the next token at every time position. Train it briefly and generate text. Its purpose is to validate the data pipeline, loss flattening, optimizer, evaluation, and generation loop before attention is introduced.

### Assignment M4.10

Answer:

1. Why does each sequence provide `T` next-token training examples?
2. Why can all `T` losses be computed in parallel during training?
3. Why must generation still proceed one token at a time without a cache?
4. Why is random-window sampling different from randomly splitting overlapping windows?
5. Why must logits `[B, T, V]` and targets `[B, T]` be reshaped for cross-entropy?

### Deliverable

- `gpt/tokenizer.py`.
- `gpt/data.py`.
- Tokenizer and batch tests.
- Baseline training and sample.

### Exit criteria

- Encode/decode round-trip passes.
- Every target is the next token after the corresponding input.
- The baseline overfits a tiny batch and learns on the training stream.

---

## Session 11 — Attention version 1: explicit loops

### Objective

Implement causal single-head self-attention in the most explicit form, prioritizing meaning over speed.

### Input and parameters

Let:

```text
x:  [B, T, C]
Wq: [C, D]
Wk: [C, D]
Wv: [C, Dv]
q:  [B, T, D]
k:  [B, T, D]
v:  [B, T, Dv]
```

For each batch `b`, query position `i`, and permitted key position `j <= i`:

$$
s_{b,i,j}=\frac{q_{b,i}\cdot k_{b,j}}{\sqrt{D}}
$$

Normalize scores only over permitted keys and calculate:

$$
y_{b,i}=\sum_{j=0}^{i}\operatorname{softmax}(s_{b,i,:})_jv_{b,j}
$$

The Transformer paper defines scaled dot-product attention using query-key similarities followed by a weighted sum of values; causal masking prevents an autoregressive position from using future tokens.[cite:233]

### Loop restriction

The educational implementation must use explicit Python loops over:

- Batch index.
- Query position.
- Key position.

Linear projections may be vectorized, but score construction and value aggregation must remain explicit. Implement a numerically stable softmax by subtracting the maximum permitted score before exponentiation.

### Required outputs

Return:

- Attention output `[B, T, Dv]`.
- Attention weights `[B, T, T]` for testing and inspection.

Future positions should have exactly zero probability after masking.

### Hand-worked example

Use `B=1`, `T=3`, and very small vectors. Calculate all allowed scores, probabilities, and outputs by hand. Confirm:

- Position 0 can use only position 0.
- Position 1 can use positions 0 and 1.
- Position 2 can use positions 0, 1, and 2.

### Assignment M4.11

Implement and test these invariants:

1. Every attention row sums to 1 over allowed keys.
2. Every probability above the causal diagonal is 0.
3. Changing `x[:, i+1:]` cannot change output positions `<= i`.
4. With all permitted scores equal, attention is uniform over the prefix.
5. With `T=1`, the output equals the single projected value.
6. Output remains finite for deliberately large scores because softmax is stabilized.

### First-principles questions

1. What question does a query represent computationally?
2. What does a key do differently from a value?
3. Why are query-key dot products scores rather than outputs?
4. Why does softmax operate over the key-position axis?
5. Why is the causal mask applied before softmax?
6. Why does dividing by `sqrt(D)` help keep score scale controlled?
7. What is the time and memory complexity in sequence length?

### Deliverable

- `attention/loop_attention.py`.
- Hand-worked example.
- Invariant tests.
- Full shape ledger.

### Exit criteria

- Loop attention is correct and causal.
- You can explain every loop and tensor index without saying “because the code does it.”

---

## Session 12 — Attention version 2: vectorized

### Objective

Replace explicit score and aggregation loops with batched matrix multiplication while preserving identical mathematics.

### Vectorized derivation

Given `q`, `k`, and `v`:

```python
scores = q @ k.transpose(-2, -1) / math.sqrt(D)
scores = scores.masked_fill(~causal_mask, float("-inf"))
weights = F.softmax(scores, dim=-1)
out = weights @ v
```

Required shapes:

```text
q:                [B, T, D]
k.transpose:      [B, D, T]
scores:           [B, T, T]
causal_mask:       [T, T] or broadcastable equivalent
weights:           [B, T, T]
v:                 [B, T, Dv]
out:               [B, T, Dv]
```

### Parity harness

Use the same input and projection parameters for loop and vector implementations. Compare:

- Projected `q`, `k`, and `v`.
- Pre-mask allowed scores.
- Attention weights.
- Outputs.
- Input gradients.
- `Wq`, `Wk`, and `Wv` gradients.

Use a shared scalar test loss such as a weighted sum of outputs so that all output elements contribute non-symmetrically.

Run across at least 20 random seeds and multiple shapes, including:

- `B=1, T=1`.
- `B>1`.
- `D != Dv`.
- A repeated-token input.
- A non-power-of-two sequence length.

### Causal leakage test

For a random cut position `i`:

1. Clone the input.
2. Replace all future positions after `i` with unrelated values.
3. Run both inputs.
4. Assert outputs through position `i` are unchanged.
5. Confirm at least one later output usually changes.

### Assignment M4.12

Benchmark loop and vector implementations for small increasing sequence lengths. Measure only after warm-up and synchronize CUDA if using it. Record the trend rather than expecting a fixed speedup.

Then answer:

1. Did vectorization change asymptotic `T^2` work?
2. Why can it still be dramatically faster?
3. Which intermediate tensor causes quadratic memory growth?
4. Why does a lower-triangular mask preserve parallel training?
5. Why are boolean-mask semantics easy to reverse across APIs?

PyTorch's current scaled-dot-product-attention API treats `True` in its boolean attention mask as allowed to participate, while other attention APIs may use `True` to mean masked; always verify the API rather than assuming mask polarity.[cite:221]

### Optional reference check

After your implementations pass, compare the vectorized output with `F.scaled_dot_product_attention` using aligned mask and dropout settings. This is validation only; do not replace your core implementation.

### Deliverable

- `attention/vector_attention.py`.
- 20-seed forward and gradient parity results.
- Causal leakage test.
- Small timing table.
- `outputs/attention_parity.md`.

### Exit criteria

- Loop and vector outputs and gradients match.
- Causality is tested behaviorally, not only by inspecting the mask.
- All axes of `[B, T, T]` attention weights are explainable.

---

## Session 13 — Attention version 3: multi-head module

### Objective

Build reusable causal multi-head self-attention with explicit head splitting and recombination.

### Module contract

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout): ...
    def forward(self, x): ...
```

Require:

```text
x:   [B, T, C]
C % H == 0
D = C / H
q,k,v after projection/split: [B, H, T, D]
scores:                       [B, H, T, T]
weights:                      [B, H, T, T]
head outputs:                 [B, H, T, D]
merged output:                [B, T, C]
projected output:             [B, T, C]
```

PyTorch's `MultiheadAttention` similarly divides the total embedding dimension across heads, but the required implementation here must perform the split, transpose, mask, merge, and output projection directly.[cite:232]

### Required implementation choices

- One packed `qkv` projection or three explicit projections; document the choice.
- Causal mask registered as a non-parameter buffer.
- Scaling by head dimension, not total embedding dimension.
- Softmax over the key axis.
- Attention dropout.
- Output projection and residual dropout.
- Sequence-length assertion against `block_size`.
- `reshape`/`transpose`/`contiguous` use explained by shape and memory layout.

### Equivalence test

Create a multi-head module with one head and align its parameters with the Session 12 single-head implementation. Assert output and gradient parity with dropout disabled.

### Independent-head test

Construct two heads with intentionally different projection parameters. Confirm their attention matrices differ and that concatenation preserves both head feature subspaces before the output projection.

### Required causality tests

- Future-input perturbation does not change prefix output.
- Every head has zero probability above the diagonal.
- Attention rows sum to 1 before dropout.
- Evaluation mode disables dropout and is deterministic.
- Training mode can vary under dropout while remaining causal.

### Assignment M4.13

Write `attention_from_memory.py` in a blank file without reference material. Time only the coding portion. The Month 4 target is a correct single-head or multi-head implementation in **15 minutes or less** after tests. The final Phase B target remains unaided multi-head attention in under 10 minutes.

Do not optimize the timer by omitting assertions or pretending shapes are understood. Record mistakes and repeat after at least two days.

### First-principles questions

1. Why must `C` be divisible by `H` in this implementation?
2. Why scale by `sqrt(D)` rather than `sqrt(C)`?
3. What does using multiple heads add beyond one larger head?
4. Why concatenate head outputs before an output projection?
5. Why is the causal mask a buffer rather than a parameter?
6. Why does dropout require different train and evaluation behavior?

### Deliverable

- `attention/multihead_attention.py`.
- One-head equivalence test.
- Multi-head causality and shape tests.
- First timed memory implementation and reflection.

### Exit criteria

- Multi-head attention passes all tests.
- Every reshape and transpose can be derived from `[B, T, C]`.
- You can distinguish head count, model dimension, and head dimension.

---

## Session 14 — Build GPT lecture

### Objective

Understand how embeddings, causal self-attention, feed-forward networks, residual streams, LayerNorm, and repeated blocks form a decoder-only Transformer.

### Material

Watch **Let's build GPT: from scratch, in code, spelled out**.

- Video: <https://www.youtube.com/watch?v=kCc8FmEb1nY>
- Reference repository: <https://github.com/karpathy/ng-video-lecture>
- `bigram.py`: <https://github.com/karpathy/ng-video-lecture/blob/master/bigram.py>
- `gpt.py`: <https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py>

The reference repository contains the code produced during the lecture and its Git history, which can be inspected after your implementation attempt.[cite:188] The lecture's published model uses token and positional embeddings, causal multi-head attention, feed-forward layers, residual connections, LayerNorm, and repeated Transformer blocks.[cite:189]

### Watching checkpoints

Pause and answer before each addition:

1. What limitation of the bigram baseline is being fixed?
2. Why does averaging previous token representations not provide content-dependent retrieval?
3. Why are positional embeddings needed when attention scores depend only on token representations?
4. Why does each position need both communication and private computation?
5. Why do attention and feed-forward sublayers preserve `[B, T, C]`?
6. Why are residual connections possible only when input/output shapes match?
7. Why is LayerNorm preferable to BatchNorm for this per-token Transformer path?
8. Why is a final vocabulary projection required?

LayerNorm calculates statistics over specified final feature dimensions and uses input statistics in both training and evaluation, unlike the running-statistic behavior of BatchNorm.[cite:206]

### Transformer block diagram

Draw and annotate the pre-normalization block used in the lecture:

```text
x ────────────────────────────────┐
│                                 │
└→ LayerNorm → causal MHA → + ────┘ = x'

x' ───────────────────────────────┐
│                                 │
└→ LayerNorm → FeedForward → + ───┘ = output
```

For every branch, identify:

- Shape.
- Learned parameters.
- Whether tokens communicate across positions.
- Whether the operation acts independently per position.

### Architecture ledger

For input IDs `[B, T]`, derive:

```text
token embeddings:        [B, T, C]
position embeddings:     [T, C] → broadcast to [B, T, C]
residual stream:         [B, T, C]
per-head q/k/v:          [B, H, T, D]
attention scores:        [B, H, T, T]
attention output:        [B, T, C]
feed-forward hidden:     [B, T, 4C]
feed-forward output:     [B, T, C]
final normalized state:  [B, T, C]
logits:                  [B, T, V]
loss input:              [B*T, V]
targets:                 [B*T]
```

### Assignment M4.14

Write one sentence for each component answering “what unique job does this do?”

- Token embedding.
- Position embedding.
- Query projection.
- Key projection.
- Value projection.
- Causal mask.
- Multiple heads.
- Output projection.
- LayerNorm.
- Residual connection.
- Feed-forward network.
- Dropout.
- Vocabulary projection.

### Deliverable

- Lecture notes.
- Annotated block diagram.
- Full architecture shape ledger.
- Component-purpose table.

### Exit criteria

- You can explain a decoder block as communication plus per-token computation.
- You can derive all major tensor shapes without executing code.
- You can explain why BatchNorm from earlier sessions is not simply inserted into the GPT block.

---

## Session 15 — Assemble GPT

### Objective

Implement a tested decoder-only Transformer from your own attention module.

### Required modules

Implement:

```python
class CausalSelfAttention(nn.Module)
class FeedForward(nn.Module)
class Block(nn.Module)
class GPTLanguageModel(nn.Module)
```

### GPT forward contract

```python
def forward(self, idx, targets=None):
    # idx: [B, T]
    # logits: [B, T, V]
    # loss: scalar or None
```

Required model path:

1. Token embedding lookup.
2. Learned position embedding lookup.
3. Addition into the residual stream.
4. Repeated pre-LayerNorm Transformer blocks.
5. Final LayerNorm.
6. Vocabulary projection.
7. Optional flattened cross-entropy.

### Feed-forward network

Use the lecture-style two-layer position-wise MLP:

```text
C → 4C → C
```

with nonlinearity and dropout. Explain why this network processes each position independently even though all positions are stored in the same tensor.

### Required tests

#### Shape tests

- Input `[B, T]` returns logits `[B, T, V]`.
- With targets, loss is scalar and finite.
- Every block preserves `[B, T, C]`.
- Parameter count is reported.

#### Causality test

Change future token IDs after position `i`. In evaluation mode, logits through position `i` must remain unchanged.

#### Gradient test

After one backward pass:

- Every expected trainable parameter has a finite gradient.
- No gradient exists for registered masks or integer token IDs.
- At least one gradient in each block is nonzero.

#### Tiny-batch overfit test

Train repeatedly on one tiny batch. The model should drive its loss substantially downward. This tests that the architecture, targets, gradients, and optimizer are connected before expensive full training.

#### Generation test

Implement naive generation:

```python
for _ in range(max_new_tokens):
    idx_cond = idx[:, -block_size:]
    logits, _ = model(idx_cond)
    next_logits = logits[:, -1, :]
    probs = softmax(next_logits)
    idx_next = multinomial(probs)
    idx = cat([idx, idx_next], dim=1)
```

Do not implement a KV cache this month. This intentionally recomputes the context and becomes Month 5's baseline.

### Parameter-count exercise

Estimate the parameter count symbolically before calling `sum(p.numel())`. Include:

- Token embedding.
- Position embedding.
- Per-block attention projections.
- Per-block feed-forward layers.
- LayerNorm affine parameters.
- Final vocabulary projection.

Explain why the `4C` feed-forward expansion usually contributes a large share of each block's parameters.

### Required configurations

#### Smoke configuration

Use a small model that can rapidly pass tests and overfit one batch, for example:

```text
block_size = 64 or 128
n_embd = 96 or 128
n_head = 4
n_layer = 2–4
dropout = 0.0 for deterministic tests
```

#### Project-shaped configuration

Instantiate a configuration near the Phase B project's approximately 10M parameters. The lecture reference uses `n_embd=384`, `n_head=6`, `n_layer=6`, and `block_size=256` on TinyShakespeare.[cite:189] Compute the actual count for your vocabulary and implementation rather than assuming the number.

### Assignment M4.15

Run this debugging ladder in order:

1. Shape-only forward pass.
2. Finite scalar loss.
3. Backward and gradient inspection.
4. One optimizer step changes parameters.
5. One-batch overfit.
6. Short training run.
7. Generation.

Record the first failure at each stage and its cause. Do not launch the long training run until all earlier stages pass.

### Deliverable

- `gpt/model.py`.
- Full test suite.
- Parameter estimate versus actual count.
- One-batch overfit curve.
- Naive generation implementation.

### Exit criteria

- GPT passes shape, causality, gradient, overfit, and generation tests.
- The approximately 10M-parameter configuration can perform forward and backward passes on the selected device.
- Every model component can be mapped to its function and shape.

---

## Session 16 — Train, evaluate, and defend

### Objective

Train the Month 4 character GPT, preserve a useful Month 5 baseline, and prove that the implementation is understood rather than copied.

### Training protocol

Use the character tokenizer and TinyShakespeare pipeline from Session 10.

Record:

- Device and dtype.
- Vocabulary size.
- Block size.
- Batch size.
- Model dimension.
- Head count and head dimension.
- Layer count.
- Dropout.
- Parameter count.
- Optimizer and learning rate.
- Evaluation interval and number of evaluation batches.
- Random seed.
- Total updates and wall-clock time.

Use the mixed-precision and profiling skills from Month 2 only after the FP32 training path is correct. If AMP is used, state the dtype and hardware support explicitly.

### Evaluation

At fixed intervals:

- Switch to evaluation mode.
- Estimate training loss across multiple mini-batches.
- Estimate validation loss across multiple mini-batches.
- Restore training mode.
- Save the best validation checkpoint and a resumable latest checkpoint.

Stop when one of these occurs:

- Validation loss has clearly plateaued under your defined criterion.
- The allocated compute budget is exhausted.
- Repeated validation degradation indicates overfitting.

Do not report a single favorable mini-batch loss as validation performance.

### Month 4 minimum versus Phase B target

#### Required Month 4 minimum

- Small GPT trains with decreasing train and validation loss.
- Approximately 10M model is instantiated, counted, and passes forward/backward.
- At least one meaningful full-stream training run is completed.
- A 200-character sample is generated from the best checkpoint.
- All attention correctness tests pass.

#### Preferred Month 4 completion

- Train the project-shaped character GPT until validation loss plateaus.
- Reach or approach the Phase B character-level target of 1.6 nats/token.
- Generate locally English-like text.

#### Not a Month 4 failure

If compute or time prevents the final loss target but the architecture, tests, training curve, checkpointing, and causal attention are correct, carry the checkpoint into Month 5. The final Project B1 acceptance remains at the end of Phase B.

### Sampling experiment

Generate from the same prompt with several temperatures, including a low, medium, and high value. Optionally add top-k sampling only after basic multinomial sampling works.

For each setting, describe:

- Repetition.
- Diversity.
- Local spelling/punctuation structure.
- Failure patterns.

Do not select a sample and call the model better based only on aesthetics; retain validation loss as the main training metric.

### Timed memory implementation

In a blank file and without references, implement causal multi-head self-attention. Include:

- Q/K/V projections.
- Head split.
- Scaled score matrix.
- Causal mask.
- Softmax over keys.
- Value aggregation.
- Head merge.
- Output projection.

Month 4 exit target: correct code in 15 minutes, then tests. Continue practicing in Month 5 until the Phase B under-10-minute target is met.

### Oral examination

Answer without notes:

1. Why should initial logits be low-confidence?
2. Why can tanh saturation suppress gradients?
3. What does fan-in scaling try to preserve?
4. Which axes does BatchNorm normalize in the MLP?
5. Why does BatchNorm need running statistics?
6. Why does LayerNorm not need running batch statistics?
7. Why must broadcast gradients be reduced over expanded axes?
8. Why do repeated embedding IDs require gradient accumulation?
9. Derive the cross-entropy gradient with respect to logits conceptually.
10. How does a hierarchical model grow its receptive field?
11. What makes attention content-dependent?
12. Distinguish queries, keys, and values.
13. Why scale query-key scores?
14. Why mask before softmax?
15. What does each axis in `[B, H, T, T]` mean?
16. Why can Transformer training process every position in parallel despite causality?
17. Why is autoregressive generation sequential?
18. Why are residual connections useful for deep optimization?
19. What separate roles do attention and the feed-forward network play?
20. Why are raw nats/token not comparable across different tokenizers?
21. What computation will KV caching avoid next month?

### Final report

Write `outputs/month4-report.md`:

```text
1. Month objective
2. Initialization experiments
3. BatchNorm implementation and train/eval behavior
4. Activation/gradient/update diagnostics
5. Manual backward parity
6. Hierarchical model and shape analysis
7. Attention v1/v2/v3 design
8. Forward and gradient parity results
9. Causality tests
10. GPT architecture and parameter count
11. Training configuration and curves
12. Validation result and samples
13. Bugs and root causes
14. Remaining risks before BPE and KV caching
```

### Exit criteria

- All tests pass from a clean checkout.
- Best and latest checkpoints are loadable.
- A 200-character sample is reproducible from a saved checkpoint and seed.
- Attention can be reconstructed without reference material.
- At least 17 of the 21 oral questions can be answered clearly; misses become Month 5 review items.

---

## Core acceptance checklist

### Initialization and diagnostics

- [ ] Uniform initial prediction loss is related to `log(V)`.
- [ ] Output confidence and hidden saturation are diagnosed separately.
- [ ] At least three controlled initialization runs are recorded.
- [ ] Zero-initialization symmetry failure is explained.
- [ ] Activation distributions are recorded per hidden layer.
- [ ] Activation-gradient distributions are recorded.
- [ ] Parameter-gradient and relative-update statistics are recorded.
- [ ] A health report identifies at least one failure before final validation loss.

### BatchNorm

- [ ] Custom BatchNorm normalizes the intended axes.
- [ ] Learnable gain and bias are implemented.
- [ ] Running mean and variance are buffers.
- [ ] Training updates buffers; evaluation does not.
- [ ] Evaluation uses stored statistics.
- [ ] Forward and gradient parity with an aligned PyTorch reference passes.
- [ ] Rank-3 behavior required by the hierarchy is tested.

### Manual backpropagation

- [ ] Expanded cross-entropy backward matches autograd.
- [ ] Second linear-layer gradients match.
- [ ] Tanh backward matches.
- [ ] Expanded BatchNorm backward matches.
- [ ] First linear-layer gradients match.
- [ ] Embedding gradients accumulate repeated indices correctly.
- [ ] Compact cross-entropy gradient matches the expanded graph.
- [ ] Compact BatchNorm gradient matches the expanded graph.
- [ ] One-step parameter updates match autograd.
- [ ] Manual-backward training reduces loss.

### Hierarchical model

- [ ] Context length and receptive field are explained at every level.
- [ ] `FlattenConsecutive` is tested on inspectable data.
- [ ] Every rank-3 shape is documented.
- [ ] BatchNorm axes remain semantically correct.
- [ ] Flat and hierarchical models are compared with measured values.
- [ ] Hierarchical generation works.

### Attention version 1

- [ ] Explicit batch/query/key loops are used.
- [ ] Stable softmax is implemented.
- [ ] Future attention probability is zero.
- [ ] Prefix outputs are invariant to future changes.
- [ ] Hand-worked example matches code.

### Attention version 2

- [ ] Batched score multiplication is correct.
- [ ] Causal mask broadcasts correctly.
- [ ] Softmax uses the key axis.
- [ ] Value aggregation has the correct shape.
- [ ] Forward parity with loop attention passes across 20 seeds.
- [ ] Input and parameter gradient parity passes.
- [ ] Loop/vector timing trend is measured.

### Attention version 3

- [ ] Head dimension is derived from model dimension and head count.
- [ ] Q/K/V split produces `[B, H, T, D]`.
- [ ] Scores have shape `[B, H, T, T]`.
- [ ] Heads merge back to `[B, T, C]` correctly.
- [ ] Causal mask is a registered buffer.
- [ ] One-head parity test passes.
- [ ] Per-head causality tests pass.
- [ ] Dropout behaves correctly in train/evaluation modes.

### GPT

- [ ] Character tokenizer round-trips text.
- [ ] TinyShakespeare input/target batches are shifted correctly.
- [ ] Bigram baseline trains and generates.
- [ ] Token and position embeddings are implemented.
- [ ] Pre-LayerNorm Transformer block is implemented.
- [ ] Feed-forward expansion and projection are implemented.
- [ ] Residual connections preserve shape.
- [ ] Final logits have shape `[B, T, V]`.
- [ ] GPT causality test passes.
- [ ] All expected parameters receive finite gradients.
- [ ] Tiny-batch overfit succeeds.
- [ ] Approximately 10M configuration is instantiated and counted.
- [ ] A full-stream training run is logged.
- [ ] Best and latest checkpoints are saved.
- [ ] At least 200 characters are generated.
- [ ] Naive generation is preserved as Month 5's cache baseline.

### First-principles understanding

- [ ] BatchNorm and LayerNorm axes/state are contrasted.
- [ ] Tensor-level manual backprop can be explained as vector-Jacobian products.
- [ ] Attention equations can be reconstructed from the query/key/value roles.
- [ ] Every attention axis can be named.
- [ ] Causality is explained as an information-flow constraint.
- [ ] Training parallelism and generation sequentiality are reconciled.
- [ ] A decoder block is explained as communication plus per-token computation.
- [ ] Causal multi-head attention can be written unaided within 15 minutes.

---

## Glossary

### Deep-network health

| Term | Meaning |
|---|---|
| Logit scale | Magnitude/spread of unnormalized output scores |
| Prediction entropy | Uncertainty of the predicted categorical distribution |
| Saturation | Region where an activation changes little and its derivative is small |
| Vanishing gradient | Gradient magnitude becomes too small to drive useful earlier-layer updates |
| Exploding gradient | Gradient magnitude becomes excessively large and destabilizes updates |
| Fan-in | Number of input features contributing to one output unit |
| Fan-out | Number of output units receiving contributions from an input |
| Variance propagation | How activation or gradient variance changes across layers |
| Gain | Scale chosen to account for an activation's effect on signal magnitude |
| Kaiming initialization | Fan-based initialization intended to preserve useful signal variance through deep networks |
| Activation distribution | Statistical distribution of a layer's forward outputs |
| Activation-gradient distribution | Statistical distribution of gradients with respect to activations |
| Gradient-to-data ratio | Parameter-gradient scale relative to parameter scale |
| Update-to-data ratio | Intended parameter-update scale relative to parameter scale |
| Symmetry breaking | Initial differences that allow neurons to learn distinct features |
| Dead neuron | Unit that remains in a region producing no useful activation/gradient response |

### Normalization

| Term | Meaning |
|---|---|
| Batch normalization | Normalization using feature statistics aggregated over training examples/positions according to defined axes |
| Layer normalization | Normalization across selected feature dimensions within each input element/token |
| Batch statistic | Mean or variance computed from the current mini-batch |
| Running statistic | Moving estimate stored for BatchNorm evaluation |
| Affine transform | Learnable rescaling and shifting after normalization |
| Gamma | Learnable normalization scale parameter |
| Beta | Learnable normalization shift parameter |
| Epsilon | Small positive denominator term for numerical stability |
| Momentum | Weight controlling running-statistic updates |
| Buffer | Persistent model state saved in checkpoints but not optimized by gradient descent |
| Train mode | Module state enabling behaviors such as dropout and BatchNorm batch-stat updates |
| Evaluation mode | Module state using deterministic inference behavior and frozen BatchNorm buffers |
| Biased variance | Variance estimate dividing by `N` |
| Unbiased variance | Corrected sample estimate typically dividing by `N-1` |

### Tensor backpropagation

| Term | Meaning |
|---|---|
| Jacobian | Matrix of all output partial derivatives with respect to all inputs |
| Vector-Jacobian product | Backward operation combining an upstream vector with a local Jacobian without materializing the full Jacobian |
| Reduction axis | Axis removed or contracted by sum/mean; backward must broadcast appropriately |
| Broadcast axis | Implicitly expanded axis whose backward gradient must be summed back to original shape |
| Scatter-add | Accumulating updates into indexed locations, as in repeated embedding IDs |
| Cached activation | Forward value retained because backward needs it |
| Fused gradient | Algebraically simplified backward for several forward operations together |
| Gradient parity | Agreement between independent gradient implementations |
| Relative error | Difference normalized by reference magnitude |
| Finite-difference check | Numerical derivative estimate used as an independent sanity check |

### Hierarchical sequence models

| Term | Meaning |
|---|---|
| Receptive field | Input positions capable of influencing one output representation |
| Hierarchical aggregation | Combining local groups repeatedly so later layers represent wider context |
| Weight sharing | Reusing the same learned transformation across positions |
| Dilated convolution | Convolution sampling spaced positions to enlarge receptive field efficiently |
| Causal convolution | Convolution whose output does not use future positions |
| FlattenConsecutive | Operation grouping adjacent time positions and joining their feature dimensions |
| Rank-3 activation | Tensor commonly interpreted as batch, time, and channel/features |
| Fixed communication pattern | Architecture-defined token interaction independent of current content |

### Attention

| Term | Meaning |
|---|---|
| Self-attention | Attention where queries, keys, and values derive from the same sequence |
| Query | Projected representation used to request relevant information |
| Key | Projected representation used to score whether a position matches a query |
| Value | Projected content mixed according to attention probabilities |
| Attention score | Query-key similarity before softmax |
| Attention weight | Normalized non-negative coefficient applied to a value |
| Scaled dot-product attention | Dot-product attention divided by the square root of key/head dimension |
| Causal mask | Constraint preventing a position from attending to future positions |
| Attention matrix | For each query position, distribution over key positions |
| Key axis | Axis over which each query's attention scores are normalized |
| Head | Independent attention subspace with its own projections |
| Head dimension | Feature size handled by one attention head |
| Multi-head attention | Parallel attention heads whose outputs are combined |
| Output projection | Learned mixing of concatenated head outputs back into model space |
| Prefix invariance | Earlier outputs remain unchanged when only future inputs change |
| Causal leakage | Incorrect influence from a future token on an earlier representation |
| Quadratic attention | `T^2` score/storage growth with sequence length |

### Transformer and GPT

| Term | Meaning |
|---|---|
| Transformer | Architecture built from attention and position-wise feed-forward blocks |
| Decoder-only Transformer | Autoregressive Transformer using causal self-attention without an encoder |
| GPT | Generative Pre-trained Transformer; here, a decoder-only next-token language model |
| Token embedding | Learned vector representing token identity |
| Position embedding | Learned or computed vector representing sequence position |
| Residual stream | Main `[B, T, C]` representation updated through residual additions |
| Residual connection | Addition of a sublayer's input to its output |
| Pre-LayerNorm | Applying LayerNorm before attention or feed-forward transformation |
| Feed-forward network | Position-wise MLP transforming each token independently |
| Expansion ratio | Hidden FFN width relative to model dimension, commonly 4 in this lecture model |
| Transformer block | Attention, feed-forward, normalization, and residual sublayers |
| Vocabulary projection | Final linear mapping from model features to token logits |
| Block size | Maximum context length processed by the model |
| Context cropping | Keeping only the most recent `block_size` tokens during naive generation |
| Temperature | Logit scaling controlling sampling sharpness |
| Top-k sampling | Sampling only among the `k` highest-scoring tokens |
| Nats per token | Mean negative log-likelihood using natural logarithms per tokenizer-defined token |
| Validation plateau | Period during which validation loss no longer improves meaningfully |
| Checkpoint | Saved model/optimizer/training state |
| Naive generation | Recomputing all retained context representations for each generated token |
| KV cache | Future Month 5 mechanism retaining past attention keys/values to avoid recomputation |

### Shape symbols

| Symbol | Meaning |
|---|---|
| `B` | Batch size |
| `T` | Sequence or context length |
| `C` | Model/embedding dimension |
| `H` | Number of attention heads |
| `D` | Per-head dimension, usually `C / H` |
| `V` | Vocabulary size |
| `L` | Number of Transformer blocks |
| `F` | Feed-forward hidden dimension, often `4C` in this model |

---

## Learning materials

Use these in order. Do not read all references before implementation; consult the relevant item when its session begins.

### Required lecture spine

1. **Zero to Hero course page**  
   <https://karpathy.ai/zero-to-hero.html>
2. **Lecture 4 — Activations, Gradients, BatchNorm**  
   <https://youtu.be/P6sfmUTpUmc>
3. **Lecture 4 notebook**  
   <https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part3_bn.ipynb>
4. **Lecture 5 — Becoming a Backprop Ninja**  
   <https://youtu.be/q8SA3rM6ckI>
5. **Lecture 5 notebook**  
   <https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part4_backprop.ipynb>
6. **Lecture 6 — Building a WaveNet**  
   <https://youtu.be/t3YJ5hKiMQ0>
7. **Lecture 6 notebook**  
   <https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part5_cnn1.ipynb>
8. **Lecture 7 — Let's build GPT**  
   <https://www.youtube.com/watch?v=kCc8FmEb1nY>
9. **Lecture 7 reference repository**  
   <https://github.com/karpathy/ng-video-lecture>

### Required API references

- PyTorch `BatchNorm1d` — axes, running statistics, momentum, and train/evaluation behavior.[cite:209]  
  <https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html>
- PyTorch `LayerNorm` — normalized dimensions and evaluation behavior.[cite:206]  
  <https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html>
- PyTorch `MultiheadAttention` — read only for documented shape conventions after implementing your own module.[cite:232]  
  <https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html>
- PyTorch scaled dot-product attention — use only for optional post-implementation parity and pay attention to mask semantics.[cite:221]  
  <https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html>

### Focused paper reading

#### Attention Is All You Need

<https://arxiv.org/abs/1706.03762>

Read only these parts this month:

- Architecture overview.
- Scaled dot-product attention.
- Multi-head attention.
- Masked decoder self-attention.
- Position-wise feed-forward network.
- Positional encoding overview.

The original paper introduced a Transformer architecture based on attention rather than recurrence or convolution.[cite:233] Do not attempt to reproduce the translation experiments.

#### Optional: WaveNet

Read the architecture figures and causal/dilated-convolution explanation only after completing the hierarchy implementation. Do not implement a production WaveNet convolution stack this month.

### Data and reference code

- TinyShakespeare text  
  <https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt>
- Official `ng-video-lecture` `bigram.py`  
  <https://github.com/karpathy/ng-video-lecture/blob/master/bigram.py>
- Official `ng-video-lecture` `gpt.py`  
  <https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py>

Reference code should be opened only after your corresponding implementation and tests exist. When you consult it, write down what architectural assumption you missed rather than replacing your file wholesale.

---

## If time slips

Preserve this order:

1. Initialization and BatchNorm concepts.
2. Manual backward parity.
3. Attention version 1.
4. Attention version 2 forward and gradient parity.
5. Multi-head attention with causality tests.
6. GPT assembly and tiny-batch overfit.
7. One real TinyShakespeare training run.
8. WaveNet comparison and extended plots.

Reduce optional work first:

- Skip the optional WaveNet paper reading.
- Use fewer initialization scales while keeping at least three controlled cases.
- Reduce the loop/vector benchmark sizes.
- Train the hierarchical model for fewer steps after correctness is established.
- Use the smoke GPT configuration for debugging.
- Carry the final 10M convergence target into the remaining Phase B time if compute is the blocker.

Never skip:

- BatchNorm train/evaluation tests.
- Manual-gradient parity.
- Behavioral causality tests.
- Loop/vector attention parity.
- Tiny-batch GPT overfit.
- Saving the naive generation baseline for Month 5.

## Month 5 readiness gate

Proceed to BPE and KV caching only when you can explain this chain without notes:

> Token IDs are converted into token and position embeddings. Each causal-attention head projects the residual stream into queries, keys, and values. A query scores only present and previous keys, softmax converts those scores into a distribution, and that distribution mixes the corresponding values. Multiple heads perform this communication in different learned subspaces. The output is projected back to the residual stream. A position-wise feed-forward network then transforms each token independently. LayerNorm and residual connections stabilize repeated blocks. The final projection produces next-token logits. Cross-entropy creates one scalar loss, and reverse-mode autodiff propagates its gradient through all blocks. During training all positions are processed in parallel under the causal mask; during generation tokens are produced sequentially, and Month 4's naive loop recomputes earlier keys and values—exactly the redundancy Month 5's KV cache will remove.

If any sentence is memorized but cannot be connected to an equation, tensor shape, or test, revisit the corresponding session before beginning Month 5.
