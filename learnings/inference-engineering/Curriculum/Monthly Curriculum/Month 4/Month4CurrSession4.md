[← Month 4 overview](<../Month4.md>) · [← Session 3](<Month4CurrSession3.md>) · [Session 5 →](<Month4CurrSession5.md>)

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
