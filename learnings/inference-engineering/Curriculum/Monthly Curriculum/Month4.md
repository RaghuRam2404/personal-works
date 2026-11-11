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

This month is substantially denser than Month 3. It contains four lectures, manual tensor calculus, three attention implementations, a Transformer, tests, and a real training run. It is organized into **17 sequential sessions**, plus one optional experimentation session (Session 17, RoPE).

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

Each session lives in its own file under [Month 4/](<Month 4/>). This file holds the month-level plan, the acceptance checklist, the glossary, learning materials, and the readiness gate.

| Session | Main work | Required result |
|---:|---|---|
| [0](<Month 4/Month4CurrSession0.md>) | Bridge and baseline diagnostic | Reproduce Month 3 MLP and record initial health |
| [1](<Month 4/Month4CurrSession1.md>) | Lecture 4 | Mental model of saturation, initialization, and BatchNorm |
| [2](<Month 4/Month4CurrSession2.md>) | Initialization experiments | Measured logits/activation/gradient effects |
| [3](<Month 4/Month4CurrSession3.md>) | BatchNorm from scratch | Train/eval-correct custom BatchNorm with parity tests |
| [4](<Month 4/Month4CurrSession4.md>) | Deep-network diagnostics | Activation, gradient, and update-ratio report |
| [5](<Month 4/Month4CurrSession5.md>) | Lecture 5 | Complete tensor-level backward dependency map |
| [6](<Month 4/Month4CurrSession6.md>) | Manual backward, expanded graph | Every intermediate gradient matches autograd |
| [7](<Month 4/Month4CurrSession7.md>) | Manual backward, compact forms | Compact cross-entropy and BatchNorm gradients; training works |
| [8](<Month 4/Month4CurrSession8.md>) | Lecture 6 | Hierarchical receptive-field and shape model |
| [9](<Month 4/Month4CurrSession9.md>) | WaveNet-style hierarchy | Working hierarchical character model |
| [10](<Month 4/Month4CurrSession10.md>) | GPT data and baseline | TinyShakespeare batches and bigram baseline |
| [11](<Month 4/Month4CurrSession11.md>) | Attention version 1 | Explicit loop implementation |
| [12](<Month 4/Month4CurrSession12.md>) | Attention version 2 | Vectorized causal attention with forward/gradient parity |
| [13](<Month 4/Month4CurrSession13.md>) | Attention version 3 | Multi-head module with causality tests |
| [14](<Month 4/Month4CurrSession14.md>) | Lecture 7 | Complete decoder-only Transformer mental model |
| [15](<Month 4/Month4CurrSession15.md>) | Assemble GPT | Tested from-scratch model and one-batch overfit |
| [16](<Month 4/Month4CurrSession16.md>) | Train, evaluate, and defend | Checkpoint, samples, report, memory rebuild |
| [17](<Month 4/Month4CurrSession17.md>) *(optional)* | Rotary position embeddings | Tested RoPE, learned vs RoPE vs none comparison, extrapolation result |

## What Month 4 achieves

### Stage 1 — Deep-network health (Sessions 0–4)

- **[Session 0](<Month 4/Month4CurrSession0.md>):** Recreate the Month 3 MLP in a controlled form and record a baseline (initial and backward statistics, restorable checkpoint) before changing initialization or adding normalization.
- **[Session 1](<Month 4/Month4CurrSession1.md>):** Understand why initialization influences logits, activation saturation, backward gradients, and early optimization.
- **[Session 2](<Month 4/Month4CurrSession2.md>):** Turn initialization claims into controlled measurements: output confidence, tanh saturation, and fan-in scaling.
- **[Session 3](<Month 4/Month4CurrSession3.md>):** Implement BatchNorm as a differentiable layer and understand exactly how training and evaluation differ.
- **[Session 4](<Month 4/Month4CurrSession4.md>):** Build reusable activation, gradient, and update-ratio diagnostics rather than relying on final loss as the only signal.

### Stage 2 — Manual tensor backpropagation (Sessions 5–7)

- **[Session 5](<Month 4/Month4CurrSession5.md>):** Move from scalar micrograd intuition to vectorized tensor calculus over a full MLP with BatchNorm; build the dependency map and gradient ledger.
- **[Session 6](<Month 4/Month4CurrSession6.md>):** Implement the complete backward pass one elementary operation at a time and match every intermediate gradient against PyTorch autograd.
- **[Session 7](<Month 4/Month4CurrSession7.md>):** Collapse the expanded derivation into compact cross-entropy and BatchNorm gradients and train with the manual backward pass.

### Stage 3 — Hierarchical sequence modelling (Sessions 8–9)

- **[Session 8](<Month 4/Month4CurrSession8.md>):** Understand why a flat context MLP scales poorly and how hierarchical pairwise aggregation grows the receptive field gradually.
- **[Session 9](<Month 4/Month4CurrSession9.md>):** Implement and train the WaveNet-style character model while keeping shape reasoning explicit.

### Stage 4 — Causal attention and GPT (Sessions 10–16)

- **[Session 10](<Month 4/Month4CurrSession10.md>):** A runnable TinyShakespeare pipeline: character tokenizer, train/val split, shifted random-window batches, a bigram baseline, training with train/val loss, and generation (no attention yet).
- **[Session 11](<Month 4/Month4CurrSession11.md>):** Single-head causal self-attention written with explicit Python loops, a hand-worked example it reproduces, and tests for six invariants.
- **[Session 12](<Month 4/Month4CurrSession12.md>):** Vectorized causal attention proven identical to the loops by forward and gradient parity across 20 seeds, a behavioral leakage test, and a timing analysis.
- **[Session 13](<Month 4/Month4CurrSession13.md>):** Grow Session 12 into multi-head attention step by step (loop over heads → head axis → `nn.Module`), then prove one-head equivalence, head independence, causality and dropout behavior, and do a first timed from-memory rebuild.
- **[Session 14](<Month 4/Month4CurrSession14.md>):** Understand how embeddings, causal self-attention, feed-forward networks, residual streams, LayerNorm, and repeated blocks form a decoder-only Transformer.
- **[Session 15](<Month 4/Month4CurrSession15.md>):** Assemble a tested decoder-only Transformer from your own attention module, including tiny-batch overfit and parameter-count checks.
- **[Session 16](<Month 4/Month4CurrSession16.md>):** Train the character GPT, preserve a Month 5 baseline (checkpoint, samples, naive generation), write the report, and pass the oral examination and memory rebuild.
- **[Session 17](<Month 4/Month4CurrSession17.md>) (optional):** Replace learned position embeddings with RoPE inside attention, prove its relative-position property with tests, compare learned vs RoPE vs no-position training, and measure length extrapolation as preparation for Month 5 KV caching.

### End-of-month outcome

A from-scratch character-level GPT trained on TinyShakespeare, backed by tests for shapes, causality, outputs, and gradients; a manual-backprop and diagnostics toolkit for deep networks; and the ability to explain the full chain in the [Month 5 readiness gate](#month-5-readiness-gate) without notes.

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
