# Week 11 Quiz — Decoder-Only Transformers and GPT

Calibration: mid-junior ML interview level.

---

**Q1.** In a decoder-only transformer trained with causal language modeling, what is the training target at each position?

A) The embedding of the current token  
B) The next token in the sequence (shifted by 1)  
C) A reconstruction of the entire sequence  
D) A binary label indicating whether the token is grammatically correct  

---

**Q2.** You are implementing nanoGPT and register the causal mask as `nn.Parameter` instead of `self.register_buffer`. What is the consequence?

A) The mask will not be saved in the model checkpoint  
B) The mask will be updated by the optimizer during training, corrupting the causal attention structure  
C) PyTorch will raise an error because parameters must be floating point  
D) The mask will work correctly but consume extra GPU memory  

---

**Q3.** In nanoGPT, the position embedding has shape `[block_size, n_embd]`. During a forward pass with a sequence of length T < block_size, what is the correct way to use it?

A) Pad the sequence with zeros to length block_size, then use all embeddings  
B) Use only the first T rows: `self.wpe(torch.arange(T))`  
C) Resize the embedding matrix to `[T, n_embd]` using interpolation  
D) Skip positional embeddings for short sequences  

---

**Q4.** GPT-2 uses Pre-LN (layer norm before each sublayer) while the original Transformer (Week 10) uses Post-LN (layer norm after each sublayer). What practical advantage does Pre-LN provide?

A) Pre-LN uses fewer parameters because the normalization is applied before projection  
B) Pre-LN provides a cleaner gradient path through the residual stream, making training more stable especially at initialization  
C) Pre-LN is faster to compute because it's applied to smaller tensors  
D) Pre-LN prevents the model from memorizing training data  

---

**Q5.** Your nanoGPT generates coherent text during training (steps 1000–4000) but then starts generating nonsense at step 5000. Training loss is still decreasing. What is most likely happening?

A) The LR is too low — the model stopped learning  
B) The model is overfitting on the training set; val loss is increasing while train loss decreases  
C) Weight tying is causing gradient cancellation  
D) The position embeddings exceeded block_size  

---

**Q6 (short answer).** Explain weight tying between the embedding table and the LM head. Write the one-line PyTorch code that implements it. Why does weight tying improve generation coherence, beyond just saving parameters?

---

**Q7 (short answer).** A decoder-only Transformer and an encoder-decoder Transformer both process a 100-token input. For a text generation task (no separate source text), which architecture is more appropriate and why? What computation does the decoder-only model avoid?

---

**Q8 (scenario).** You train your nanoGPT on a SQL corpus for 3000 steps. Val loss reaches 1.3. You sample from the model at temperature 1.0 and get: `SELECT SELECT WHERE FROM id GROUP`. At temperature 0.3, you get: `SELECT name FROM users`. Why does lowering temperature produce better SQL, and what is the risk of setting temperature too low?
