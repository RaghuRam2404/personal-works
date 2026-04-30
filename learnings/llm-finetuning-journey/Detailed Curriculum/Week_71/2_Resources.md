# Week 71 Resources — Frontier Reading: Tulu 3, SmolLM2, OLMo 2

## Papers (primary reading this week)

[Tulu 3: Pushing Frontiers in Open Language Model Post-Training](https://arxiv.org/abs/2411.15124) — Ivison et al. 2024; covers RLVR, on-policy data generation, and the full Tulu 3 training pipeline; focus on Sections 3 (data) and 4 (training).

[SmolLM2: When Smol Goes Big (Efficient Language Models)](https://arxiv.org/abs/2502.05654) — Allal et al. 2024; covers FineWeb-Edu curriculum, tokenizer efficiency, and small-model performance; focus on Sections 2 (data) and 4 (evaluation).

[OLMo 2: The Second Generation of Truly Open Language Models](https://arxiv.org/abs/2501.00656) — Groeneveld et al. 2024; covers fully open training stack, mid-training data mixing, and the OLMES evaluation harness; focus on Sections 3 (training) and 5 (ablations).

## Supporting Papers

[Tulu 2: Instruction-Tuned Language Models](https://arxiv.org/abs/2311.10702) — Ivison et al. 2023; read this first if Tulu 3 references techniques without explaining them; contains the preference data pipeline that Tulu 3 builds on.

[FineWeb: Decanting the Web for the Finest Text Data at Scale](https://arxiv.org/abs/2406.17557) — Penedo et al. 2024; explains the FineWeb-Edu quality filtering methodology that SmolLM2 uses.

## Videos

[Tulu 3 Overview (Allen AI talk)](https://www.youtube.com/watch?v=2kSYFx4N5Nw) — ~40 min; one of the Tulu 3 authors explains RLVR and on-policy generation.

[OLMo 2 Release Stream (Allen AI)](https://www.youtube.com/watch?v=XtD-iYGa1Uo) — ~60 min; complete walkthrough of OLMo 2's training decisions and evaluation results.

## Blog Posts / Articles

[SmolLM2 HuggingFace Blog](https://huggingface.co/blog/smollm2) — Launch post with benchmarks, model sizes, and usage examples; good entry point before reading the full paper.

[The Alignment Handbook (HuggingFace)](https://github.com/huggingface/alignment-handbook) — The open-source training toolkit that Tulu 3-style RLVR is built on; contains working implementations of DPO, GRPO, and on-policy generation.

## GitHub Repos

[allenai/open-instruct](https://github.com/allenai/open-instruct) — Tulu 3 training code; includes RLVR implementation and on-policy data generation pipeline.

[allenai/OLMo](https://github.com/allenai/OLMo) — OLMo 2 training code and the OLMES evaluation harness.

[huggingface/smollm](https://github.com/huggingface/smollm) — SmolLM2 training code and data curation scripts.

## Optional / Bonus

[DataComp for Language Models (DCLM)](https://arxiv.org/abs/2406.11794) — The benchmark behind SmolLM2's data curriculum; explains what makes high-quality pretraining data.

[On the Token Count Analysis of LLM Tokenizers for Code](https://arxiv.org/abs/2402.15093) — Quantitative study of tokenizer efficiency across SQL and code tasks; directly relevant to SmolLM2's tokenizer findings.
