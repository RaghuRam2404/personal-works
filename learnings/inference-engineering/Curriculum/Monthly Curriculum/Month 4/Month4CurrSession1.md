[← Month 4 overview](<../Month4.md>) · [← Session 0](<Month4CurrSession0.md>) · [Session 2 →](<Month4CurrSession2.md>)

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
