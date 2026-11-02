[← Month 4 overview](<../Month4.md>) · [Session 1 →](<Month4CurrSession1.md>)

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
