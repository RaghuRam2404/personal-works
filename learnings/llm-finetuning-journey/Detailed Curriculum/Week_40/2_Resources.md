# Week 40 Resources — Phase 4 Gate

This week is consolidation. Resources here are reference material for the full Phase 4 arc and a preview of Phase 5.

## Papers (Phase 4 Master References)

[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Hu et al. 2021. The foundational paper for all adapter-based fine-tuning in Phase 4.

[QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — Dettmers et al. 2023. NF4 quantization + paged optimizers; the technique behind your 7B fine-tune.

[DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353) — Liu et al. 2024. Magnitude + direction decomposition for LoRA.

[RSLoRA: A Rank Stabilization Scaling Factor for Fine-Tuning of Large Language Models](https://arxiv.org/abs/2312.03732) — Kalajdzievski 2024. `use_rslora=True` fix for rank scaling instability.

[LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models](https://arxiv.org/abs/2310.08659) — Li et al. 2023. Better initialization when quantizing before LoRA fine-tuning.

[PEFT: Parameter-Efficient Fine-Tuning of Large Language Models Survey](https://arxiv.org/abs/2403.14608) — Xu et al. 2024. Comprehensive survey covering LoRA, QLoRA, DoRA, and 20+ variants.

## Phase 5 Preview Papers

[GRPO: DeepSeekMath — Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) — DeepSeek-AI 2024. Introduces GRPO; Section 3 covers the algorithm you will implement in Phase 5.

[DPO: Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.20223) — Rafailov et al. 2023. Phase 5's alternative alignment technique; read the abstract now, full paper in Week 43.

## Videos

[Andrej Karpathy: Let's Reproduce GPT-2](https://www.youtube.com/watch?v=l8pRSuU81PU) — Karpathy. ~4h. The Phase 4 anchor video (Week 28); re-watch the fine-tuning sections as a consolidation exercise.

[Hugging Face: How to Share a Model on the Hub](https://www.youtube.com/watch?v=XvSGPZFEjDY) — Hugging Face. ~10 min. Exact walkthrough of `push_to_hub` and model card creation.

[Tim Dettmers: QLoRA Explained](https://www.youtube.com/watch?v=y0NWt5k7B4c) — Tim Dettmers. ~30 min. Author walkthrough of QLoRA — good consolidation before Phase 5.

## Blog Posts / Articles

[HuggingFace: Model Cards Guide](https://huggingface.co/docs/hub/model-cards) — The official guide to writing a compliant model card. Follow this for Task 2.

[Sebastian Raschka: Practical Tips for Fine-Tuning LLMs Using LoRA](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms) — Raschka. Summarizes Phase 4 lessons in one article; excellent consolidation read.

[Lilian Weng: Fine-Tuning Large Language Models](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/) — Weng. Deep technical treatment of fine-tuning paradigms.

## GitHub Repos

[huggingface/peft](https://github.com/huggingface/peft) — The library behind all your LoRA runs. Re-read the LoraConfig docstring before Phase 5.

[unslothai/unsloth](https://github.com/unslothai/unsloth) — Phase 5 GRPO training will likely use Unsloth's GRPO support — bookmark the repo now.

[defog-ai/sql-eval](https://github.com/defog-ai/sql-eval) — Your eval harness is based on this. Star it so you can track updates.

## Documentation

[HuggingFace Hub: Uploading Models](https://huggingface.co/docs/hub/models-uploading) — Official upload guide covering `push_to_hub`, upload_folder, and large file (LFS) handling.

[HuggingFace Datasets: Share a Dataset](https://huggingface.co/docs/datasets/upload_dataset) — Covers `push_to_hub` for datasets, including DatasetDict with multiple splits.

[PEFT: Quicktour](https://huggingface.co/docs/peft/quicktour) — A 10-minute reference for LoraConfig parameters; bookmark for Phase 5 when you adapt the config for GRPO.

## Optional / Bonus

[Weights & Biases: Model Registry](https://docs.wandb.ai/guides/model_registry) — Version your model artifacts in W&B as an alternative to HuggingFace Hub. Useful if you want reproducibility metadata alongside your runs.

[Defog AI: sqlcoder model series](https://huggingface.co/defog) — The professional version of what you built in Phase 4. Study their model cards for how production text-to-SQL adapters are documented and versioned.
