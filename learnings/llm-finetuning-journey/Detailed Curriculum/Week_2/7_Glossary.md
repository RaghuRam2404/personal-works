# Week 2 — Glossary

**MLP (Multi-Layer Perceptron)**: A neural network with one or more hidden layers of fully connected (linear) transformations followed by nonlinear activations.

**Activation function**: A nonlinear function applied after each linear transformation, enabling the network to learn non-linear mappings.

**ReLU (Rectified Linear Unit)**: `max(0, x)` — the most common hidden-layer activation; fast, not saturating for x > 0, but can "die".

**GELU (Gaussian Error Linear Unit)**: Smooth approximation of ReLU; default activation in GPT-2, BERT, and most modern transformers.

**Dying ReLU**: A pathology where a neuron's pre-activation is always negative, so it always outputs 0 and receives zero gradient — it can never recover.

**Xavier (Glorot) initialization**: Weight init designed for Tanh/Sigmoid: `std = sqrt(2 / (fan_in + fan_out))`.

**Kaiming (He) initialization**: Weight init designed for ReLU: `std = sqrt(2 / fan_in)`; compensates for ReLU zeroing half of inputs.

**fan_in**: Number of input connections to a neuron (= number of columns in a weight matrix, or `d_in`).

**Batch normalization (BN)**: Normalizes activations over the batch dimension per feature; adds learned scale (`gamma`) and shift (`beta`); has distinct train/eval behavior.

**Running mean / running variance**: Exponential moving averages of batch statistics accumulated during training, used by batch norm during inference.

**Layer normalization (LN)**: Normalizes over the feature dimension per example; batch-size independent; preferred in transformers.

**Dropout**: Regularization that randomly zeros neurons with probability `p` during training; uses inverted scaling so inference requires no change.

**Inverted dropout**: Scales surviving neuron outputs by `1/(1-p)` during training so the expected value is preserved at inference time without scaling.

**Embedding table**: A lookup matrix of shape `(vocab_size, d_embd)` that maps discrete token indices to continuous vectors; trained via backprop.

**Language model loss**: Cross-entropy between the model's predicted next-token distribution and the actual next token; lower is better; the baseline is `log(vocab_size)`.

**Bigram model**: A language model that predicts the next token conditioning only on the immediately preceding token.
