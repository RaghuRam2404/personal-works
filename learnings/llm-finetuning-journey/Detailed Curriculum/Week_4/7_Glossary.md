# Week 4 — Glossary

**RNN (Recurrent Neural Network)**: A neural network with a recurrent hidden state; processes sequences by passing hidden state from one step to the next.

**Hidden state (h_t)**: The vector summary of all inputs seen up to timestep t; the "memory" of a vanilla RNN.

**BPTT (Backpropagation Through Time)**: The algorithm for training RNNs; unrolls the recurrent computation and applies standard backprop through all time steps.

**Vanishing gradient**: Gradients that shrink exponentially as they propagate backward through many time steps, preventing learning of long-range dependencies.

**Exploding gradient**: Gradients that grow exponentially during BPTT; mitigated by gradient clipping.

**LSTM (Long Short-Term Memory)**: Recurrent architecture with a cell state and three gates (forget, input, output) that enable learning of long-range dependencies.

**Cell state (c_t)**: The "conveyor belt" in an LSTM; updated additively by the forget and input gates, enabling gradients to flow more freely backward.

**Forget gate**: Sigmoid gate that decides what fraction of the previous cell state to retain; output near 0 erases, near 1 preserves.

**Input gate**: Sigmoid gate that controls how much of the new candidate values to write into the cell state.

**Output gate**: Sigmoid gate that determines what portion of the cell state is exposed as the hidden state h_t.

**GRU (Gated Recurrent Unit)**: Simpler gated RNN with update and reset gates; fewer parameters than LSTM, similar performance on many tasks.

**Teacher forcing**: Training technique where ground-truth previous tokens are fed as inputs at each step, regardless of model predictions.

**Truncated BPTT (TBPTT)**: Limits BPTT to a fixed chunk length; requires detaching the hidden state between chunks to stop gradients crossing chunk boundaries.

**detach()**: PyTorch operation that creates a tensor sharing data but disconnected from the computation graph; stops gradient flow at that point.

**Temperature (sampling)**: Scalar that scales logits before softmax; lower = sharper (less diverse), higher = flatter (more random) sampling distribution.

**Gradient clipping**: Rescaling gradient norms that exceed a threshold before the optimizer step; prevents exploding gradients in RNN training.

**Autoregressive generation**: Generating a sequence one token at a time, where each new token is conditioned on all previously generated tokens.
