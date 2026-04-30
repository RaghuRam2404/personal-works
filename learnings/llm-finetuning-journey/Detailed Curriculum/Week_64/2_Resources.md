# Week 64 Resources — Quantize and Publish Your Final Model

## Papers

[GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) — Frantar et al. 2022; the original Hessian-based INT4 method you implement this week.

[AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978) — Lin et al. 2023; explains the salient-channel scaling that makes AWQ more domain-aware than GPTQ.

## Videos

[How to Quantize LLMs: GGUF, GPTQ, AWQ Explained (Trelis Research)](https://www.youtube.com/watch?v=pF7UxBRxMOI) — ~25 min; side-by-side walkthrough of all three formats with code.

[Publishing a Model to Hugging Face Hub (HuggingFace official)](https://www.youtube.com/watch?v=XvSGPZFEjDY) — ~15 min; covers model cards, `push_to_hub`, upload_file, and licensing.

## Blog Posts / Articles

[Hugging Face GPTQ Integration Guide](https://huggingface.co/blog/gptq-integration) — Full walkthrough of GPTQConfig API, calibration data preparation, and saving quantized models.

[Hugging Face AWQ Blog Post](https://huggingface.co/blog/awq-inference) — Covers AutoAWQ installation, quantization config options, and throughput benchmarks on various GPU types.

[The Illustrated Guide to LLM Quantization (Maarten Grootendorst)](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization) — Visual reference for all methods covered in Weeks 63–64.

## GitHub Repos

[casper-hansen/AutoAWQ](https://github.com/casper-hansen/AutoAWQ) — The AWQ library you use this week; check the `examples/` directory for Qwen2.5 examples.

[PanQiWei/AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ) — GPTQ library; `examples/quantization/basic_usage_wikitext2.py` is the clearest starting point to adapt for SQL calibration.

[ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) — Build instructions in `docs/build.md`; quantization types table in `docs/quantization.md`.

[unslothai/unsloth](https://github.com/unslothai/unsloth) — If you trained with Unsloth, `model.save_pretrained_gguf()` handles conversion + quantization in one call.

## Documentation

[HuggingFace Hub — Upload Files](https://huggingface.co/docs/huggingface_hub/guides/upload) — API reference for `upload_file`, `upload_folder`, and `create_repo`; covers large file handling.

[HuggingFace Model Cards Guide](https://huggingface.co/docs/hub/model-cards) — What a good model card includes; links to the model card metadata spec for tags and license fields.

[llama.cpp quantization types reference](https://github.com/ggerganov/llama.cpp/blob/master/docs/quantization.md) — Official table of Q-types with perplexity benchmarks on Llama models; use to interpret your own measurements.

## Optional / Bonus

[ExLlamaV2](https://github.com/turboderp/exllamav2) — GPU-optimized GPTQ inference engine with significantly higher throughput than AutoGPTQ; worth testing after you have your GPTQ model.

[imatrix quantization (llama.cpp)](https://github.com/ggerganov/llama.cpp/blob/master/examples/imatrix/README.md) — Importance-matrix guided quantization for GGUF; improves Q4 accuracy by 0.3–0.8 pp with a calibration pass; good stretch goal.

[GGUF model card template (community)](https://huggingface.co/TheBloke) — Browse any TheBloke repo for a battle-tested model card format that the community expects for GGUF files.
