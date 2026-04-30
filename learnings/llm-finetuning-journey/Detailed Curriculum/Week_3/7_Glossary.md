# Week 3 — Glossary

**Convolution (2D)**: Operation sliding a filter over a 2D input, computing dot products at each position; produces a feature map.

**Kernel (filter)**: The weight tensor (e.g., 3×3) that defines what feature a conv layer detects; shared across all spatial positions.

**Parameter sharing**: Using the same kernel weights at every spatial location in a convolutional layer, reducing parameter count drastically vs. fully-connected.

**Receptive field**: The region of the original input that affects a given output neuron; grows with depth and with pooling.

**Padding**: Zero-valued border added around the input before convolution; controls output spatial size.

**Stride**: Step size of the kernel as it slides across the input; stride > 1 reduces output spatial dimensions.

**Max pooling**: Downsampling operation that takes the maximum value in each local window; reduces spatial size and provides translation invariance.

**Global average pooling**: Collapses each feature map to a single value by averaging all spatial positions; outputs `(N, C)` from `(N, C, H, W)`.

**BatchNorm2d**: Batch normalization applied to 2D spatial feature maps; normalizes over N, H, W dimensions per channel C.

**Translation equivariance**: Property of CNNs where shifting the input shifts the output by the same amount; a consequence of parameter sharing.

**Receptive field (dilated)**: With dilated convolutions, the effective receptive field grows exponentially with dilation factor, not linearly.

**Dilated convolution**: Convolution where the kernel elements are spaced `d` steps apart (dilation factor d), expanding the receptive field without increasing parameters.

**Causal convolution**: 1D convolution that only looks at past positions (left side), not future; required for autoregressive models.

**WaveNet**: DeepMind's autoregressive audio model using dilated causal convolutions; precursor to hierarchical sequence modeling.

**Scatter-add**: Operation that accumulates values into a target tensor at specified indices; used in embedding backward pass to handle repeated token indices.
