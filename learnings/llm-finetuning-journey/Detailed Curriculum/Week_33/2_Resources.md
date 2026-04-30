# Week 33 Resources

## Papers

- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — Dettmers et al. 2023. **Read fully this week.** The core QLoRA paper; pay special attention to Section 2 (background on NF4 and double quantization) and Section 4 (experiments showing QLoRA matches full SFT quality).

## Videos

- [QLoRA: Efficient Finetuning of Quantized LLMs (Paper Walkthrough)](https://www.youtube.com/watch?v=TPcXVJ1VSRI) — Yannic Kilcher — ~40 min. Deep dive into the QLoRA paper covering NF4 quantization, double quantization, and paged optimizers; essential before running your first 7B fine-tune.
- [Fine-tune Any LLM with Unsloth and QLoRA](https://www.youtube.com/watch?v=aQmoog_s8_k) — Daniel Han / Unsloth AI — ~25 min. Practical demonstration of QLoRA setup with Unsloth; covers BitsAndBytesConfig, adapter merging, and VRAM usage on an A100.
- [QLoRA Fine-Tuning Walkthrough: 7B Model on a Single GPU](https://www.youtube.com/watch?v=tgr0VNBbZ-k) — Trelis Research — ~30 min. End-to-end QLoRA tutorial on a 7B model; covers rank selection, target_modules, and interpreting training loss curves during the first 7B run.

## Blog Posts / Articles

- [Making LLMs Even More Accessible with bitsandbytes, 4-bit Quantization and QLoRA](https://huggingface.co/blog/4bit-transformers-bitsandbytes) — HuggingFace blog. Practical walkthrough of the full QLoRA setup.
- [Fine-Tune Your Own Llama 2 Model in a Colab Notebook](https://mlabonne.github.io/blog/posts/Fine_Tune_Your_Own_Llama_2_Model_in_a_Colab_Notebook.html) — Maxime Labonne. Same pattern as your Qwen2.5 fine-tune; useful reference for debugging.

## GitHub Repos

- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) — Source for NF4 implementation, paged optimizers, and 8-bit Adam.
- [QLoRA reference implementation](https://github.com/artidoro/qlora) — The original QLoRA paper authors' training code.
- [peft](https://github.com/huggingface/peft) — LoRA adapter API used in QLoRA setup.

## Documentation

- [bitsandbytes quantization configuration](https://huggingface.co/docs/transformers/main/en/quantization/bitsandbytes) — Complete `BitsAndBytesConfig` parameter reference.
- [Gradient checkpointing docs](https://huggingface.co/docs/transformers/main/en/perf_train_gpu_one#gradient-checkpointing) — When and how to enable it.

## Optional / Bonus

- [Efficient Fine-Tuning: Comparing LoRA and QLoRA](https://huggingface.co/blog/lora-adapters-dynamic-loading) — HuggingFace. Discussion of when to prefer QLoRA vs. standard LoRA.
- [NVIDIA's Flash Attention 2](https://arxiv.org/abs/2307.08691) — Not directly used this week but dramatically reduces activation memory and speeds up training; available via `attn_implementation="flash_attention_2"` in `from_pretrained`. Relevant for Week 34's Unsloth discussion.
