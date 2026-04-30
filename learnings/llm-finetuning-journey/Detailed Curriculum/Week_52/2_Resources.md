# Week 52 Resources — Phase 5 Gate

This is the consolidated reference list for everything covered in Phase 5 (Weeks 41–52). Use it to revisit any topic during gate preparation and to orient yourself before Phase 6 begins.

---

## Papers

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — Schulman et al., 2017. The clipping objective, GAE advantage estimation, and the KL-penalty variant that InstructGPT used. You should be able to explain why clipping stabilizes training without needing a trust region solve.

- [Learning to Summarize from Human Feedback](https://arxiv.org/abs/2009.01325) — Stiennon et al., 2020. OpenAI. The clearest end-to-end demonstration of the reward model + PPO loop before InstructGPT. Read if you want the reference model role explained more carefully.

- [Training Language Models to Follow Instructions with Human Feedback (InstructGPT)](https://arxiv.org/abs/2203.02155) — Ouyang et al., 2022. The paper that operationalized RLHF at scale. Section 3 (Methods) is required reading for Phase 5 gate preparation.

- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) — Rafailov et al., 2023. The closed-form derivation in Section 4 is what the gate asks you to reproduce. Pay attention to the implicit reward interpretation.

- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — Bai et al., 2022. Anthropic. The RLAIF framing used in Week 44 to generate SQL preference labels without human annotators. Skim Sections 2–3.

- [RLAIF: Scaling Reinforcement Learning from Human Feedback using AI Feedback](https://arxiv.org/abs/2309.00267) — Lee et al., 2023. Google DeepMind. Companion to Constitutional AI. Focuses on using an AI judge to replace human preference labelers at scale.

- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) — Shao et al., 2024. The paper that introduced GRPO (Group Relative Policy Optimization). Section 3.3 defines the group-normalized advantage. This derivation appears on the Phase 5 gate.

- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) — DeepSeek-AI, 2025. Shows GRPO applied to reasoning tasks with verifiable rewards. Relevant to Weeks 46–48. Section 2 explains the RLVR framing.

- [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306) — Ethayarajh et al., 2024. Week 49 alignment zoo entry. The key contribution: aligning on unpaired feedback using Kahneman-Tversky utility.

- [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691) — Hong et al., 2024. Week 49. Combines SFT and preference optimization in a single loss. No reference model required.

- [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734) — Meng et al., 2024. Week 49. Uses average log-likelihood as the implicit reward. Simpler than DPO, often matches or exceeds it on benchmarks.

- [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206) — Zhou et al., 2023. Meta AI. Argues that 1,000 carefully curated SFT examples can match RLHF-tuned models. Relevant to data curation strategy in Phase 6.

- [Tülu 3: Pushing Frontiers in Open Language Model Post-Training](https://arxiv.org/abs/2411.15124) — Lambert et al., 2024. Allen AI. A comprehensive post-training recipe covering SFT, DPO, and PPO at production scale. A useful map of Phase 6 territory.

## Videos

- [HuggingFace Deep RL Course — Full Playlist](https://www.youtube.com/playlist?list=PLo-OljUVeFCYJ8cpglH9pwRrIyMftpMP1) — HuggingFace — variable per unit (30–60 min each). The course units covering REINFORCE, PPO, and policy gradients that formed the Week 41–42 foundation. Use for gate remediation.
- [Andrej Karpathy — Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU) — Andrej Karpathy — 4h01m. The full pretraining walkthrough referenced across Phase 3 and Phase 5. Rewatch the RL sections if policy gradient intuition is shaky.

## Blog Posts / Articles

- [HuggingFace — Illustrating Reinforcement Learning from Human Feedback (RLHF)](https://huggingface.co/blog/rlhf) — A well-structured walkthrough of the SFT → reward model → PPO pipeline. Good refresher before the gate.
- [HuggingFace — Fine-tune Mistral-7b with DPO](https://huggingface.co/blog/dpo-trl) — The blog post behind the DPO notebook used in Week 43. Read alongside the DPO paper.
- [HuggingFace — ORPO: An Efficient LLM Alignment Approach](https://huggingface.co/blog/orpo) — Concise explanation of the ORPO loss with code. Pairs with the ORPO paper for Week 49.
- [HuggingFace — Open-R1: A Fully Open Reproduction of DeepSeek-R1](https://huggingface.co/blog/open-r1) — Covers GRPO training details and the verifiable reward setup used in Weeks 46–48. Directly applicable to your SQL reward function design.
- [Lilian Weng — Policy Gradient Algorithms](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/) — The reference for Week 41 RL foundations. Covers REINFORCE, actor-critic, PPO, and the policy gradient theorem derivations.

## GitHub Repos

- [huggingface/trl](https://github.com/huggingface/trl) — The primary training library for Phase 5. The `examples/` directory contains runnable scripts for DPO, PPO, and GRPO.
- [huggingface/open-r1](https://github.com/huggingface/open-r1) — Open reproduction of DeepSeek-R1's GRPO training. The reward function structure used here directly informed the SQL reward design in Week 47.
- [huggingface/alignment-handbook](https://github.com/huggingface/alignment-handbook) — Production-grade recipes for SFT, DPO, and ORPO. Reference for how config-driven training pipelines are structured.
- [unslothai/unsloth](https://github.com/unslothai/unsloth) — Memory-efficient fine-tuning used in Weeks 45 and 48. The `GRPOTrainer` integration and LoRA GRPO notebooks are in `notebooks/`.
- [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) — Quantized CPU/Metal inference. Relevant if you want to run your Phase 5 models locally on Apple Silicon without a GPU server.

## Documentation

- [TRL Trainer Overview](https://huggingface.co/docs/trl/index) — The entry point for all TRL trainers. Links to PPOTrainer, DPOTrainer, GRPOTrainer, KTOTrainer, ORPOTrainer, and CPOTrainer documentation.
- [TRL PPOTrainer](https://huggingface.co/docs/trl/ppo_trainer) — Annotated in Week 42. The `step()` method signature and the `generate()` → reward → `step()` loop.
- [TRL DPOTrainer](https://huggingface.co/docs/trl/dpo_trainer) — Used in Weeks 43 and 45. Pay attention to the `beta` parameter and the `loss_type` argument.
- [TRL GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer) — Central to Weeks 46–48. The `reward_funcs` argument is where your SQL execution reward plugs in.
- [HuggingFace Model Card Guide](https://huggingface.co/docs/hub/model-cards) — Required for the gate artifact. Covers the model card metadata YAML header and expected sections.
- [Unsloth Documentation](https://docs.unsloth.ai) — Used for memory-efficient DPO (Week 45) and GRPO (Week 48) on Apple Silicon and RunPod.
- [W&B Quickstart](https://docs.wandb.ai/quickstart) — The gate requires a confirmed W&B run link. Use this if your GRPO run is not already logged.
- [W&B Tables for Model Evaluation](https://docs.wandb.ai/guides/tables) — Useful for logging per-example SQL execution results alongside predicted and ground-truth queries.
- [vLLM Documentation](https://docs.vllm.ai) — Fast inference engine likely to be used in Phase 6 for serving your fine-tuned models.

## Optional / Bonus

- [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) — Rich Sutton, 2019. Reflect on whether your capstone (domain knowledge + task specificity beating a larger general model) confirms or challenges this thesis.
