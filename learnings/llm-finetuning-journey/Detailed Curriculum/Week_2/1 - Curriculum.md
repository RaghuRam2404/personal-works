# Week 2 — MLPs, Activations, Initialization, and the Bag of Tricks

## Learning Objectives

By the end of this week, you will be able to:

- Build a multi-layer perceptron in PyTorch using both `nn.Parameter` (manual) and `nn.Linear` (idiomatic), and explain the difference.
- Choose between ReLU, GELU, and Tanh activations for a given situation and justify the choice.
- Derive Kaiming (He) initialization on paper, and explain why it is the default for ReLU networks.
- Implement batch normalization from scratch and explain what it computes during training vs. inference.
- Diagnose overfitting from a W&B loss curve and apply dropout as a mitigation.
- Train a bigram and MLP language model on the makemore names dataset, then swap it to a SQL keyword dataset.

---

## Concepts

### Multi-Layer Perceptrons (MLPs)

An MLP is a stack of linear transformations alternated with nonlinear activations. For a single layer:

```
h = activation(x @ W + b)
```

where `W` is a weight matrix of shape `(d_in, d_out)` and `b` is a bias of shape `(d_out,)`. The power of depth comes from composing these — a network with enough layers and nonlinearities is a universal function approximator in theory.

In PyTorch, the idiomatic way is `nn.Linear(d_in, d_out)` which internally holds `weight` of shape `(d_out, d_in)` and handles the transpose. The computation is `F.linear(x, weight, bias)` = `x @ weight.T + bias`.

For language modeling, Karpathy's makemore builds an MLP that takes N previous characters (as embeddings) and predicts the next character — a direct precursor to transformer language models.

### Activations

**Sigmoid:** `σ(x) = 1 / (1 + exp(-x))`. Saturates for large |x|, causing vanishing gradients. Avoid as hidden activation; still valid for output of binary classification.

**Tanh:** `tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`. Also saturates, but zero-centered (unlike sigmoid). Still used in LSTMs and RNNs.

**ReLU:** `max(0, x)`. Does not saturate for positive x. Dying ReLU problem: neurons can get stuck outputting 0 permanently if they receive consistently negative pre-activations. Cheap to compute — the dominant choice for CNNs.

**GELU:** `x * Φ(x)` where Φ is the CDF of the standard normal. Smooth approximation of ReLU. The activation of choice in modern transformers (GPT-2/3, BERT). Slightly more expensive but empirically better for language tasks.

**Rule:** For CNNs and simple MLPs → ReLU. For transformers and LLMs → GELU. For output layers → depends on the task (sigmoid for binary, softmax for multiclass, linear for regression).

### Weight Initialization

The problem: if you initialize weights too small, signals shrink layer by layer (vanishing). Too large, they explode. The goal is to preserve the variance of activations through layers.

**Xavier (Glorot) init:** Designed for Tanh/Sigmoid. Sets `std = sqrt(2 / (fan_in + fan_out))`. The `2` comes from the two-sided nature of the activation.

**Kaiming (He) init:** Designed for ReLU. Because ReLU kills exactly half of activations (the negative half), the effective fan is half what Xavier assumes. The correction: `std = sqrt(2 / fan_in)`. The factor of 2 compensates for the ReLU zero region.

In PyTorch: `nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')`. This is the PyTorch default for `nn.Linear` in many contexts.

Karpathy's makemore Part 3 video is the most important video of this week — it shows you what happens to activation distributions and gradients with different initializations, which is the kind of diagnostic skill you will use in every project.

### Batch Normalization

Batch norm addresses the problem of internal covariate shift: as layers update, the distribution of their inputs changes, forcing downstream layers to constantly readjust.

For a mini-batch of activations `x` with shape `(N, D)`:

```
μ = mean(x, dim=0)           # (D,)
σ² = var(x, dim=0)           # (D,)
x_hat = (x - μ) / sqrt(σ² + ε)   # normalize
y = γ * x_hat + β            # learned scale and shift
```

`γ` and `β` are learned parameters — this allows the network to "undo" normalization if that is what the task requires.

**Training vs. inference:** During training, `μ` and `σ²` are computed from the current mini-batch. During inference (after `model.eval()`), PyTorch uses exponential moving averages `running_mean` and `running_var` accumulated during training. This is why `model.eval()` matters: batch norm behaves differently in the two modes.

**Practical placement:** Batch norm typically goes after the linear layer and before the activation. Some modern architectures (Transformers) use layer norm instead, which normalizes across the feature dimension rather than the batch dimension.

### Dropout

Dropout is a regularization technique. During training, each neuron's output is set to zero with probability `p` (the dropout rate). The surviving outputs are scaled by `1/(1-p)` to maintain expected values (inverted dropout).

During inference (`model.eval()`), dropout is disabled — all neurons are active.

Typical values: `p = 0.1`–`0.5`. Higher dropout for larger models or smaller datasets. Dropout after fully-connected layers is standard; before the final output layer is uncommon.

The intuition: by randomly removing neurons, dropout prevents any single neuron from becoming a "load-bearing wall" — it forces redundant representations, which generalizes better.

### Bigram and MLP Language Models (makemore)

Karpathy's makemore series builds language models on a names dataset using progressively more powerful architectures:
- **Bigram model:** Predict next character from only the previous character. Essentially a lookup table — the logits are log-probabilities.
- **MLP model:** Use the previous N characters (embedded) as input to a multi-layer network to predict the next character.

The loss is cross-entropy between the predicted distribution and the true next character. The key insight is that this is exactly the same objective as modern LLMs — just at a tiny scale.

**SQL domain tie-in:** Swapping the names dataset for SQL keyword sequences (extracted from the Spider dataset) gives you a toy text-to-SQL generative model. The SQL "vocabulary" is smaller and more structured than English, so the bigram and MLP models train faster and produce more intelligible samples.

---

## Connections

**Builds on:** Week 1's training loop, autograd, and `nn.Module`.

**Unlocks:** Week 3's CNN uses the same initialization and batch norm. Week 4's LSTM builds on the same language modeling objective. Week 8's capstone reuses the MLP architecture. Every future week that trains a model uses these regularization techniques.

---

## Common Misconceptions and Pitfalls

- **"Batch norm automatically fixes bad initialization."** Partially true — it normalizes activations, but with very bad init (e.g., all zeros), all neurons learn the same thing even with batch norm.
- **"Dropout should be used everywhere."** Wrong. Using dropout inside recurrent connections (LSTMs) requires special treatment. In transformers, dropout rate is usually small (0.1). Never put dropout before the final softmax.
- **"Higher learning rate + batch norm = always better."** Batch norm does allow higher LRs, but there is still a ceiling. Too high will cause loss spikes.
- **Forgetting that batch norm has two modes.** Train/eval switching is the most common batch-norm bug. Validate that `model.eval()` is called before every eval loop.

---

## Time Allocation (6–8 hours this week)

| Activity | Time |
|---|---|
| Read Karpathy "Yes you should understand backprop" | 30 min |
| Read Deep Learning Book Ch. 6 (activations + init sections) | 1 h |
| Watch makemore Part 1 (bigram LM) — code along | 1.25 h |
| Watch makemore Part 2 (MLP LM) — code along | 1.25 h |
| Watch makemore Part 3 (Activations, Gradients, BatchNorm) — code along | 2 h |
| Swap dataset to SQL keywords, train, log to W&B | 1 h |
| Write journal entry and commit | 30 min |
