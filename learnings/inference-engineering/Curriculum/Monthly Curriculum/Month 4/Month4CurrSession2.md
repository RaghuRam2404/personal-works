[← Month 4 overview](<../Month4.md>) · [← Session 1](<Month4CurrSession1.md>) · [Session 3 →](<Month4CurrSession3.md>)

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
