# Week 10 Resources

## Papers

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al. 2017. Read it 3 times this week. Take handwritten notes on every equation.
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) — Xiong et al. 2020. Explains why Pre-LN is more stable than Post-LN. Read the main theorem and Figure 1.
- [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) — Dosovitskiy et al. 2020. Optional — shows Transformer applied to vision, confirms architecture generality.

## Videos

- [Yannic Kilcher — Attention Is All You Need (full 28 min)](https://www.youtube.com/watch?v=iDulhoQ2pro) — Watch the full video this week (you watched only the intro in Week 9).
- [3Blue1Brown — But what is a GPT? Visual intro to transformers](https://www.youtube.com/watch?v=wjZofJX0v4M) — 27 min. Exceptional visual intuition for the attention mechanism.
- [3Blue1Brown — Attention in transformers, visually explained](https://www.youtube.com/watch?v=eMlx5fFNoYc) — 26 min. Goes deeper into multi-head attention geometry.

## Blog Posts / Articles

- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) — Sasha Rush (Harvard NLP). Your primary coding reference. Every line of the paper's architecture with working PyTorch code interleaved.
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Jay Alammar. Best visual explanation of multi-head attention and encoder-decoder flow. Read before coding.
- [Transformer Architecture: The Positional Encoding](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/) — Amirhossein Kazemnejad. Deep dive on why sinusoidal encoding works the way it does.

## GitHub Repos

- [harvardnlp/annotated-transformer](https://github.com/harvardnlp/annotated-transformer) — Official repo for The Annotated Transformer. Reference code — type it yourself, don't clone.
- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) — Preview for Week 11. The decoder-only simplification of the full encoder-decoder Transformer you build this week.

## Documentation

- [PyTorch nn.MultiheadAttention](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html) — The built-in MHA module. Understand its `key_padding_mask` and `attn_mask` parameters after you implement your own.
- [PyTorch nn.TransformerEncoderLayer](https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html) — After building from scratch, compare your implementation to PyTorch's official module.

## Optional / Bonus

- [Formal Algorithms for Transformers](https://arxiv.org/abs/2207.09238) — DeepMind 2022. Pseudocode-level specification of every Transformer variant. Good reference for precision.
- [Transformers from Scratch](https://peterbloem.nl/blog/transformers) — Peter Bloem. Another excellent from-scratch tutorial with different notation than The Annotated Transformer. Good second reference.
