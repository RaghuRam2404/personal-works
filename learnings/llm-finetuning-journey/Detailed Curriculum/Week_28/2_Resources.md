# Week 28 Resources

## Papers

- [InstructGPT: Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155) — Ouyang et al., OpenAI 2022. The foundational SFT+RLHF paper. Read sections 1–3 this week.
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Hu et al. 2021. Introduces the intrinsic low-rank hypothesis. Required for Week 30; skim the intro now.

## Videos

- [Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI) — Andrej Karpathy, YouTube, 3h31m. **Required this week.** The single best conceptual walkthrough of pretraining → post-training pipeline.

## Blog Posts / Articles

- [LLM Year in Review 2025](https://karpathy.bearblog.dev/year-in-review-2025/) — Andrej Karpathy. Current state of the field as of 2025; read after the video.
- [HuggingFace LLM Course Chapter 11: Supervised Fine-Tuning](https://huggingface.co/learn/llm-course/chapter11/1) — HuggingFace. Practical intro to SFT concepts.

## GitHub Repos

- [TRL (Transformer Reinforcement Learning)](https://github.com/huggingface/trl) — HuggingFace. The library you will use for SFTTrainer, DPOTrainer, and GRPOTrainer in Phase 4–5. Bookmark now.

## Documentation

- [TRL SFTTrainer documentation](https://huggingface.co/docs/trl/sft_trainer) — Reference for Week 29's hands-on SFT work. Skim the parameter list now.

## Optional / Bonus

- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — Anthropic 2022. Background on RLAIF, referenced in Karpathy's video.
- [Llama 3 Technical Report](https://arxiv.org/abs/2407.21783) — Meta 2024. Section 4 describes their post-training pipeline; useful for seeing how production systems implement the SFT → DPO → RLVR pipeline you are building.
- [Karpathy's "Let's build ChatGPT from scratch" (2023)](https://www.youtube.com/watch?v=kCc8FmEb1nY) — YouTube, 1h56m. Older but complementary; compare with the 2024 deep dive to see how his framing evolved.
