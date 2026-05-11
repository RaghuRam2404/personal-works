# Week 73 Glossary — Anthropic Interpretability

**Superposition Hypothesis**: The theory that neural networks store more features than they have neurons by representing features as directions in high-dimensional space, exploiting the sparsity of natural data.

**Monosemanticity**: The property of a neuron or SAE feature that activates on exactly one interpretable concept; the ideal target of interpretability research.

**Polysemanticity**: The property of a neuron that responds to multiple unrelated concepts simultaneously; makes individual neurons difficult to interpret.

**Sparse Autoencoder (SAE)**: A neural network trained to reconstruct model activations using a much larger dictionary, with a sparsity penalty that drives it to learn monosemantic features.

**Dictionary learning**: The technique of training an SAE to find a sparse, overcomplete basis for model activations; the dictionary elements correspond to features.

**Residual stream**: The running sum of all layer outputs in a transformer; each layer (attention + MLP) reads from and writes to the residual stream.

**Logit lens**: A technique that projects the residual stream at each intermediate layer through the unembedding matrix to see what the model would predict at that point in the forward pass.

**Activation patching**: Replacing the activations at a specific position or layer with values from another example to test whether those activations causally determine the output.

**Activation steering**: Adding a learned direction to the residual stream at inference time to shift the model's behavior; a form of inference-time behavioral editing.

**Induction head**: A two-attention-head circuit that implements in-context pattern copying; one of the best-characterized computational primitives in transformer models.

**Transformer Circuits**: The framework (Elhage et al. 2021) for analyzing information flow in transformers as "circuits" composed of attention heads and MLPs; the theoretical basis for Anthropic's interpretability work.
