# Week 3 — Quiz

---

**Q1.** An input of shape `(1, 3, 28, 28)` passes through `nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2)`. What is the output shape?

A) `(1, 64, 14, 14)`
B) `(1, 64, 28, 28)`
C) `(1, 64, 12, 12)`
D) `(1, 64, 15, 15)`

---

**Q2.** You are training your CIFAR-10 CNN and notice train accuracy is 85% but test accuracy plateaus at 65%. You have already added batch norm. What is the most effective next step?

A) Increase the number of convolutional filters.
B) Add data augmentation (RandomCrop, RandomHorizontalFlip) and dropout before the classifier head.
C) Reduce the learning rate to 1e-5.
D) Remove batch normalization — it is preventing generalization.

---

**Q3.** In the combined cross-entropy + softmax backward pass, the gradient with respect to logits is:

A) `probs - 1` for all positions.
B) `probs`, with no modification needed.
C) `probs`, but with 1 subtracted only at the index of the correct class, then divided by batch size N.
D) `softmax(probs)` applied a second time.

---

**Q4.** WaveNet uses dilated causal convolutions with dilation schedule `d = 1, 2, 4, 8`. Why causal (one-sided) and not standard (two-sided)?

A) Causal convolutions are computationally cheaper.
B) The model must not see future tokens when predicting the current one — it is an autoregressive model. Using both sides of the context would be data leakage.
C) Two-sided dilated convolutions cause gradient vanishing.
D) Causal convolutions produce larger receptive fields.

---

**Q5.** Your CNN uses `MaxPool2d(2, 2)` twice. The input to the first conv is `(1, 3, 32, 32)`. After two max pool layers with no padding changes, the spatial size going into the classifier's `nn.Flatten` is:

A) `(1, C, 16, 16)` — only one pool reduces the size.
B) `(1, C, 8, 8)` — each pool halves spatial dimensions.
C) `(1, C, 4, 4)` — each pool reduces by 4.
D) `(1, C, 32, 32)` — pooling doesn't change size in PyTorch.

---

**Q6.** The "backprop ninja" skill refers to implementing backward passes manually. In the embedding lookup backward, why must you use scatter-add rather than simple index assignment?

A) Scatter-add is faster on GPU.
B) The same token index can appear multiple times in a batch. Simple assignment would overwrite gradients from earlier occurrences; scatter-add correctly accumulates them.
C) Index assignment does not work with integer indices.
D) Scatter-add is required by PyTorch for all backward implementations.

---

**Q7 (short answer).** Explain why parameter sharing is the defining property of convolutions, and give a concrete example of the parameter saving it provides compared to a fully-connected layer operating on CIFAR-10 images.

---

**Q8 (short answer).** Your CIFAR-10 CNN achieves 76% test accuracy after 20 epochs. Your colleague says "add more layers to get above 80%." You know from CS231n that naively adding layers to a plain CNN without residual connections often hurts performance. Explain why, and describe one architectural change (available this week) and one training-procedure change that could improve accuracy without residuals.

---

**Q9 (short answer).** Describe the receptive field advantage that WaveNet's dilated convolutions provide over a standard (non-dilated) stack of 1D convolutions with the same number of parameters.
