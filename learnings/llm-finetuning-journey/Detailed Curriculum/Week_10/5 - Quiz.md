# Week 10 Quiz — "Attention Is All You Need"

Calibration: mid-junior ML interview level. Focus on "why" not "what".

---

**Q1.** You implement scaled dot-product attention and accidentally divide by `d_k` instead of `sqrt(d_k)`. During training on a 512-dimensional model, what is the most likely observable symptom?

A) Training loss immediately goes to NaN  
B) Gradients through the softmax vanish too quickly; the model trains much slower or stalls  
C) The model learns faster because scores are more concentrated  
D) The attention weights become non-positive  

---

**Q2.** In the original Transformer (Vaswani et al. 2017), what is the purpose of the Feed-Forward sublayer that follows multi-head attention in each encoder layer?

A) To apply a second round of attention across a different projection  
B) To apply a position-wise nonlinear transformation, allowing the model to store and retrieve information beyond what attention alone can compute  
C) To normalize the residual stream before the next layer  
D) To project down from d_model to d_k for the next attention layer  

---

**Q3.** The decoder has three sublayers per layer. In what order do they appear, and why does masked self-attention come before cross-attention?

A) FFN → masked self-attention → cross-attention; FFN stabilizes the input first  
B) Masked self-attention → cross-attention → FFN; the decoder must first contextualize its own generated tokens before querying the encoder  
C) Cross-attention → masked self-attention → FFN; encoder context must be injected first  
D) Masked self-attention → FFN → cross-attention; FFN processes the self-attended representation before using the encoder  

---

**Q4.** Positional encoding uses sinusoidal functions. Which property of sinusoidal encodings is most important for transformers trained on short sequences but evaluated on longer ones?

A) Sinusoidal values are bounded in [-1, 1], preventing gradient explosion from positional inputs  
B) The encoding for position `pos+k` is a linear function of the encoding for position `pos`, allowing the model to generalize to unseen lengths  
C) Sinusoidal encodings are orthogonal to each other, preventing information overlap between positions  
D) They require no learned parameters, saving memory  

---

**Q5.** Multi-head attention with `h=8` heads and `d_model=512` uses 8 parallel attention computations each with `d_k=64`. How does the total parameter count compare to a single-head attention with `d_k=512`?

A) 8x more parameters, because there are 8 sets of projection matrices  
B) The same total parameters, because each head's projection matrices are 8x smaller  
C) 2x fewer, because each head only needs half the parameters  
D) It depends on whether biases are included  

---

**Q6.** You notice that in your Transformer implementation, you accidentally omitted the residual connections (i.e., each sublayer output replaces the input rather than being added to it). You have 6 encoder layers. What will happen during training?

A) Nothing — layer normalization compensates for the missing residual  
B) Training will work but converge much slower  
C) Gradients will vanish through 6 layers of attention/FFN with no skip path; training will likely stall or diverge  
D) Only the last layer will train; earlier layers will have near-zero gradients  

---

**Q7 (short answer).** Explain what causal masking is in the decoder self-attention layer. How is it implemented in practice (be specific about the mask values and where they are applied)? What would go wrong during training if you forgot the causal mask?

---

**Q8 (short answer).** You are training your Annotated Transformer implementation and notice the loss plateaus at 2.3 (approximately log of the vocabulary size for a 10-token vocabulary). List 3 specific hypotheses ranked by likelihood and how you would diagnose each.
