# Week 9 Quiz — Bahdanau Attention

Calibration: mid-junior ML interview level. You should be able to answer these without looking at code.

---

**Q1.** In the vanilla seq2seq model (no attention), what is the bottleneck that causes performance to degrade on long sequences?

A) The embedding dimension is too small  
B) The encoder compresses the entire source sequence into a single fixed-size vector  
C) Backpropagation through time is too slow on long sequences  
D) The decoder has no dropout, causing overfitting  

---

**Q2.** The Bahdanau attention score function is:

```
e_{i,j} = v_a^T * tanh(W_a * s_{i-1} + U_a * h_j)
```

Why is a `tanh` nonlinearity included here?

A) To normalize the scores to the range [-1, 1] before softmax  
B) To introduce nonlinearity so the alignment model can learn non-linear relationships between decoder state and encoder annotations  
C) To prevent the dot product from growing too large  
D) Because tanh is faster to compute than ReLU on CPUs  

---

**Q3.** You train your seq2seq+attention model on string reversal. After 30 epochs, the attention heatmap for input "hello" looks like a uniform matrix (all weights ≈ 0.2). What is the most likely problem?

A) Your model is overfitting — the attention is too sharp  
B) Your softmax is applied along the wrong dimension, or attention scores are not backpropagating through the context vector  
C) Your hidden dimension is too large  
D) Teacher forcing ratio is too high  

---

**Q4.** In Bahdanau attention, the encoder is bidirectional. What is the purpose of using a bidirectional encoder (vs. unidirectional)?

A) To double the hidden state size so the decoder has more information  
B) To allow `h_j` to capture context from both left and right of position `j`, giving a richer annotation  
C) To speed up training by running two GPUs in parallel  
D) The bidirectionality is only needed during inference, not training  

---

**Q5.** After computing attention weights `alpha_{i,j}` (shape `[batch, src_len]`) and encoder outputs (shape `[batch, src_len, hidden_dim]`), what is the correct PyTorch operation to compute the context vector?

A) `torch.matmul(alpha, encoder_outputs)`  
B) `torch.bmm(alpha.unsqueeze(1), encoder_outputs).squeeze(1)`  
C) `(alpha.unsqueeze(-1) * encoder_outputs).sum(dim=0)`  
D) `encoder_outputs[alpha.argmax(dim=-1)]`  

---

**Q6 (short answer).** Explain the difference between additive (Bahdanau) attention and multiplicative (Luong/dot-product) attention. What are the trade-offs in terms of expressiveness and computational cost?

---

**Q7 (short answer).** You have a batch of source sentences padded to length 30. The actual sentence lengths are [10, 15, 30, 8]. Describe exactly how you would construct and apply the source padding mask in your attention module to prevent the model from attending to padding tokens. What happens to training if you skip this step?

---

**Q8 (scenario).** You train your seq2seq model for 50 epochs with teacher forcing ratio = 1.0 (always feed ground truth). Training loss reaches 0.01. But at inference time (teacher forcing = 0.0), the model generates garbage — it gets confused after the first token. What is this phenomenon called, and how do you fix it?
