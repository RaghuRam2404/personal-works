# Week 73 Resources — Anthropic Interpretability

## Papers

- [Toy Models of Superposition](https://arxiv.org/abs/2209.10652) — Elhage et al. 2022 (Anthropic). Foundational paper on polysemanticity and superposition in neural networks; explains why neurons represent multiple features simultaneously. Sections 1–3 required.
- [Towards Monosemanticity: Decomposing Language Models With Dictionary Learning](https://arxiv.org/abs/2310.04625) — Bricken et al. 2023 (Anthropic). Sparse autoencoder applied to a one-layer transformer to extract monosemantic features. The core methodology paper for this week.
- [Scaling and Evaluating Sparse Autoencoders](https://arxiv.org/abs/2406.04093) — Gao et al. 2024 (OpenAI). Companion paper on how SAE quality scales with width; covers the k-sparse variant and evaluation via downstream loss.
- [Interpretability in the Wild: A Circuit for Indirect Object Identification](https://arxiv.org/abs/2211.00593) — Wang et al. 2022. Best worked example of a complete circuits analysis; identifies specific attention heads responsible for a concrete reasoning task.

## Primary Reading (transformer-circuits.pub)

[Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html) — Elhage et al. 2022; the foundational paper for understanding why neurons are polysemantic; Sections 1–3 are essential reading.

[Towards Monosemanticity: Decomposing Language Models With Dictionary Learning](https://transformer-circuits.pub/2023/monosemanticity/index.html) — Bricken et al. 2023; the sparse autoencoder paper applied to a one-layer transformer; shows specific monosemantic features found.

[Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) — Templeton et al. 2024; SAEs applied to a frontier model; includes safety-relevant features and activation steering experiments.

[A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) — Elhage et al. 2021; formal treatment of transformer computation as circuits; the theoretical foundation.

[In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) — Olsson et al. 2022; explains the universal two-head circuit that enables in-context learning.

## Blog Posts / Articles

[Toy Models of Superposition — Summary and Commentary](https://www.lesswrong.com/posts/z6QQJbyKxPsHBJFuF/toy-models-of-superposition-key-claims-and-results) — LessWrong community post summarizing the key findings from the superposition paper; useful companion before reading the full transformer-circuits.pub article.

[Understanding Anthropic’s Interpretability Research (Jack Clark)](https://jack-clark.net/2023/10/06/import-ai-340/) — Import AI newsletter issue covering the monosemanticity paper; useful for framing the research program in context.

[Chris Olah — Zoom In: An Introduction to Circuits](https://distill.pub/2020/circuits/zoom-in/) — Distill.pub. The original circuits thread that motivated the Anthropic interpretability program; explains how features and circuits emerge in vision models and why the same framework applies to transformers.

## Videos

[Chris Olah — Circuits in Neural Networks (Lex Fridman Podcast)](https://www.youtube.com/watch?v=vJRkEzoAc9Q) — ~2h; long but the best explanation of the circuits research program from the main author.

[Sparse Autoencoders for LLM Interpretability (Neel Nanda)](https://www.youtube.com/watch?v=SvigCqH8KZQ) — ~60 min; practical guide to SAE training and feature analysis.

## GitHub Repos

[EleutherAI/sae](https://github.com/EleutherAI/sae) — Open-source SAE training code; supports GPT-2 and Pythia models; good for the stretch goal.

[neelnanda-io/TransformerLens](https://github.com/neelnanda-io/TransformerLens) — The main toolkit for mechanistic interpretability; supports GPT-2, Pythia, and some Llama models.

[jbloomAus/SAELens](https://github.com/jbloomAus/SAELens) — A more recent SAE training library with better scaling properties; includes pre-trained SAEs for GPT-2 and Gemma.

## Documentation

[TransformerLens Documentation](https://transformerlensorg.github.io/TransformerLens/) — API reference; the `run_with_cache` and `HookPoint` documentation is most relevant for attention visualization.

[circuitsvis](https://github.com/alan-cooney/CircuitsVis) — Visualization library for attention patterns; works directly with TransformerLens cache outputs.

## Optional / Bonus

[Interpretability in the Wild: a Circuit for Indirect Object Identification (Wang et al.)](https://arxiv.org/abs/2211.00593) — The best worked example of a complete circuits analysis; identifies specific heads responsible for a reasoning task.

[Neuronpedia](https://www.neuronpedia.org) — Interactive browser for SAE features in GPT-2 and Claude; look up features by searching "SQL" or "JOIN" to see what current SAEs find.
