# Week 11 Resources

## Papers

- [Improving Language Understanding by Generative Pre-Training (GPT-1)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) — Radford et al. 2018. The first decoder-only pretraining paper. Focus on Section 3 (model architecture) and Section 4 (fine-tuning).
- [Language Models are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — Radford et al. 2019. Key change: Pre-LN, scale, and zero-shot task framing.
- [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165) — Brown et al. 2020. Skim Section 2 (architecture) and Section 3 (few-shot prompting emergence). Full paper is 75 pages — focus on the architecture table and Figure 1.

## Videos

- [Andrej Karpathy — Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) — 1h56m. The primary resource this week. Code along. Do not skip anything.

## Blog Posts / Articles

- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) — Jay Alammar. Excellent visual walkthrough of the GPT-2 architecture and autoregressive generation, including the KV cache (preview of Week 13).
- [Transformer Feed-Forward Layers Are Key-Value Memories](https://arxiv.org/abs/2012.14913) — Geva et al. 2020. The theoretical basis for the "FFN as memory" framing mentioned in the curriculum.

## GitHub Repos

- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) — Karpathy's clean implementation. Reference only — type everything from the video yourself first.
- [karpathy/minGPT](https://github.com/karpathy/minGPT) — Older, more documented version. Good for cross-referencing if nanoGPT video moves too fast.
- [Spider dataset (Yale)](https://yale-seas.github.io/spider/) — SQL corpus for your second training run. Download `train.json` and extract the SQL queries.

## Documentation

- [PyTorch nn.Embedding](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html) — Understand `num_embeddings`, `embedding_dim`, and the `weight` attribute you use for weight tying.
- [PyTorch AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html) — Default optimizer for transformer training. Note the `weight_decay` parameter.

## Optional / Bonus

- [GPT-2 Weight Tying Explanation — EleutherAI blog](https://blog.eleuther.ai/tied-embeddings/) — Detailed analysis of tied embeddings in language models, including its effect on the loss landscape.
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — Kaplan et al. 2020. Preview for Week 17. Explains why bigger models trained on more data always win.
