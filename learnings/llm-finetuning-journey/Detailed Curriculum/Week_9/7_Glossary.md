# Week 9 Glossary

**Seq2seq**: Encoder-decoder architecture that maps a variable-length input sequence to a variable-length output sequence.

**Context vector**: Weighted sum of encoder hidden states; what the decoder uses at each step in attention-based models.

**Alignment**: The learned soft correspondence between source tokens and target tokens in attention.

**Attention weights (alpha)**: Softmax-normalized scores indicating how much to attend to each source position.

**Additive attention**: Bahdanau's score function using an MLP (W_a * s + U_a * h) rather than a dot product.

**Multiplicative attention**: Luong-style attention using a dot product between decoder and encoder states.

**Annotation vector (h_j)**: Encoder hidden state at position j; in Bahdanau, a concatenation of forward and backward GRU outputs.

**Bidirectional RNN**: RNN that processes input both left-to-right and right-to-left, concatenating both hidden states at each position.

**Teacher forcing**: Training trick where ground-truth previous tokens are fed to the decoder instead of its own predictions.

**Exposure bias**: Train/inference mismatch caused by always using teacher forcing; model is fragile to its own errors.

**Scheduled sampling**: Gradually reducing teacher forcing ratio over training to reduce exposure bias.

**Soft attention**: Attention that computes a weighted sum over all positions (differentiable). Contrast with hard attention (argmax, not differentiable).

**Padding mask**: Boolean tensor marking pad tokens; applied before softmax to zero out irrelevant positions.
