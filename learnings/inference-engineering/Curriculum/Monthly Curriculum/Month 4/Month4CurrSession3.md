[← Month 4 overview](<../Month4.md>) · [← Session 2](<Month4CurrSession2.md>) · [Session 4 →](<Month4CurrSession4.md>)

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
