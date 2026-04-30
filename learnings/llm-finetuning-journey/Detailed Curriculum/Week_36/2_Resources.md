# Week 36 Resources

## Papers

- [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353) — Liu et al. 2024. **Required reading.** Read sections 1–4 for the decomposition theory and sections 5–6 for empirical results.
- [RSLoRA: A Rank Stabilization Scaling Factor for Fine-Tuning of Large Language Models](https://arxiv.org/abs/2312.03732) — Kalajdzic et al. 2023. Short paper; read fully (6–8 pages). The math is in sections 2–3.
- [LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models](https://arxiv.org/abs/2310.08659) — Li et al. 2023. Read sections 1–3 for the initialization problem and SVD solution.
- [Parameter-Efficient Fine-Tuning Methods Survey](https://arxiv.org/abs/2403.14608) — Comprehensive survey covering LoRA, DoRA, and other variants in a unified framework.

## Videos

- [DoRA: Weight-Decomposed Low-Rank Adaptation (Paper Review)](https://www.youtube.com/watch?v=VnHoFRXHF5g) — Yannic Kilcher — ~35 min. Paper walkthrough covering DoRA's magnitude/direction decomposition, why it outperforms LoRA on instruction tuning, and how to enable it via `use_dora=True` in PEFT.
- [DoRA and LoRA Variants Explained](https://www.youtube.com/watch?v=t1caDsMzWBk) — AI Coffee Break — ~20 min. Clear visual explanation of DoRA's decomposition versus standard LoRA, with intuition for when DoRA improves performance over vanilla LoRA.
- [LoRA Variants: DoRA, RSLoRA, and LoftQ Compared](https://www.youtube.com/watch?v=NqDqF3aA8Xg) — Trelis Research — ~25 min. Side-by-side comparison of DoRA, RSLoRA, and LoftQ implementations in PEFT; covers which variant to choose based on quantization setup and rank.

## Blog Posts / Articles

- [DoRA: A New Fine-tuning Method that Performs Better than LoRA](https://magazine.sebastianraschka.com/p/lora-and-dora-from-scratch) — Sebastian Raschka. Written walkthrough of DoRA with code. Highly recommended.
- [PEFT LoRA variants guide](https://huggingface.co/docs/peft/conceptual_guides/lora) — HuggingFace. Updated documentation covering DoRA and RSLoRA configuration.

## GitHub Repos

- [peft library](https://github.com/huggingface/peft) — DoRA and RSLoRA are in `peft/tuners/lora/layer.py`. Look at `DoraLayer` for the decomposition implementation.
- [LoftQ reference implementation](https://github.com/yxli2123/LoftQ) — Original LoftQ authors' code.

## Documentation

- [PEFT DoRA documentation](https://huggingface.co/docs/peft/package_reference/lora#peft.LoraConfig.use_dora) — `use_dora` parameter reference.
- [LoftQ config documentation](https://huggingface.co/docs/peft/package_reference/lora#peft.LoftQConfig) — `LoftQConfig` parameters.

## Optional / Bonus

- [VeRA: Vector-based Random Matrix Adaptation](https://arxiv.org/abs/2310.11454) — 2024 variant that uses shared random matrices to further reduce adapter parameters while maintaining quality.
- [Flora: Low-Rank Adapters Are Secretly Gradient Compressors](https://arxiv.org/abs/2402.03293) — Theoretical connection between LoRA and gradient compression; deep reading for understanding why low-rank works.
