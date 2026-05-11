# Week 58 Resources

## Papers

- [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206) — Revisit; their SFT methodology is directly applicable.
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Hu et al., 2021. The foundational LoRA paper; review the rank selection discussion.
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — Dettmers et al., 2023. The quantization + LoRA combination used in 4-bit mode.
- [Instruction Tuning with GPT-4](https://arxiv.org/abs/2304.03277) — Peng et al., 2023. How data quality from strong teachers affects SFT quality.

## Videos

- [Unsloth — Fine-tuning Qwen2.5 with SFTTrainer](https://www.youtube.com/watch?v=2trUMnVVAhg) — Unsloth, 30m. Practical walkthrough.
- [Sebastian Raschka — SFT implementation details](https://www.youtube.com/watch?v=nKM_dFVzL_Q) — Sebastian Raschka, 45m.

## Blog Posts / Articles

- [Hugging Face — SFT Trainer docs](https://huggingface.co/docs/trl/sft_trainer) — Comprehensive TRL SFT documentation.
- [Weights & Biases — Debugging Neural Networks with PyTorch and W&B](https://wandb.ai/wandb_fc/articles/reports/Debugging-Neural-Networks-with-PyTorch-and-W-B-Using-Gradients-and-Visualizations--Vmlldzo1NDQxNTA5) — How to read loss curves, gradient histograms, and diagnose training issues.
- [Unsloth blog — SFT fine-tuning guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide) — Practical guide specific to Unsloth.

## GitHub Repos

- [huggingface/trl](https://github.com/huggingface/trl/blob/main/trl/trainer/sft_trainer.py) — SFTTrainer source code; study the DataCollatorForCompletionOnlyLM implementation.
- [unslothai/unsloth](https://github.com/unslothai/unsloth) — Unsloth SFT notebooks for Qwen models.
- [philschmid/deep-learning-pytorch-huggingface](https://github.com/philschmid/deep-learning-pytorch-huggingface) — Phil Schmid's SFT notebooks (formerly `llm-sft-examples`); comprehensive and well-documented; key notebook at `training/fine-tune-llms-in-2024-with-trl.ipynb`.

## Documentation

- [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer) — Official docs including DataCollatorForCompletionOnlyLM.
- [Qwen2.5 model card](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) — Chat template specification for your base model.
- [RunPod documentation — PyTorch templates](https://docs.runpod.io/pods/templates/overview) — Setting up the RunPod environment for training.

## Optional / Bonus

- [When Scaling Meets LLM Finetuning: The Effect of Data, Model and Finetuning Method](https://arxiv.org/abs/2402.17193) — Empirical study of how dataset size and model size interact in SFT.
- [Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?](https://arxiv.org/abs/2405.05904) — Important caveat: fine-tuning can introduce new forms of hallucination if done carelessly.
