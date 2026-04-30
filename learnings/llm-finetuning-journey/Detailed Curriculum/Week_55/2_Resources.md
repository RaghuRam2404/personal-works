# Week 55 Resources

## Papers

- [Alpagasus: Training a Better Alpaca with Fewer Data](https://arxiv.org/abs/2307.08701) — Chen et al., 2023. The foundational LLM-as-judge paper for training data filtering.
- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) — Zheng et al., 2023. Systematic study of LLM judge reliability and biases.
- [PRE: A Peer Review Inspired LLM-as-Judge Framework](https://arxiv.org/abs/2409.00511) — 2024. Multi-judge consensus approach; relevant for borderline examples.
- [DEITA: What Makes Good Data for Alignment?](https://arxiv.org/abs/2312.15685) — Liu et al., 2023. Automated complexity + quality scoring as an alternative to LLM judging.
- [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264) — Muennighoff et al., 2023. What happens when data quality varies; supports the filtering argument.

## Videos

- [Yannic Kilcher — LLM-as-judge paper walkthrough](https://www.youtube.com/watch?v=TtxiSbDxJqo) — Yannic Kilcher, ~35m. Covers MT-Bench and judge biases.
- [Sebastian Raschka — Data quality in LLM training](https://www.youtube.com/watch?v=3ZlFMSp4O10) — Sebastian Raschka, ~40m.

## Blog Posts / Articles

- [Hamel Husain — Your AI product needs evals](https://hamel.dev/blog/posts/evals/) — Broader argument for systematic evaluation including data quality.
- [Eugene Yan — Patterns for LLM evaluation](https://eugeneyan.com/writing/llm-patterns/) — Practical catalog of LLM judging patterns.
- [Argilla blog — LLM judge calibration](https://argilla.io/blog/llm-judge-calibration/) — Step-by-step guide to calibrating a judge against human labels.

## GitHub Repos

- [fastchat-lmsys/FastChat](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge) — The MT-Bench judge implementation; adapt their judge prompt template.
- [allenai/open-instruct](https://github.com/allenai/open-instruct) — Tulu 3 codebase; study their quality filtering scripts.
- [argilla-io/distilabel](https://github.com/argilla-io/distilabel) — Full pipeline including judge-based filtering with built-in calibration utilities.

## Documentation

- [sklearn.metrics.cohen_kappa_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html) — For computing inter-rater agreement.
- [OpenAI API — structured outputs](https://platform.openai.com/docs/guides/structured-outputs) — Use structured outputs for reliable JSON judge responses.

## Optional / Bonus

- [Chatbot Arena Elo methodology](https://arxiv.org/abs/2403.04132) — Understanding how LLM quality is measured at scale via human comparison.
- [Constitution AI (Anthropic)](https://arxiv.org/abs/2212.08073) — Using an AI to evaluate outputs against principles; similar to but different from LLM-as-judge.
