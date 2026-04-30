# Week 32 Glossary

**FP32 (float32)**: 32-bit IEEE floating-point format with 1 sign, 8 exponent, 23 mantissa bits; the default precision for PyTorch tensors.

**FP16 (float16)**: 16-bit floating-point with 5 exponent bits; can overflow during training, producing NaN; used for inference and mixed-precision training.

**BF16 (bfloat16)**: 16-bit format with 8 exponent bits (same range as FP32) and 7 mantissa bits; numerically stable for training, standard on A100/H100 hardware.

**INT8**: 8-bit integer format; requires a scale factor to approximate floating-point weights; used in LLM.int8() for inference at ~2x memory reduction.

**INT4**: 4-bit integer format with 16 uniformly spaced levels; aggressive compression but suboptimal for normally distributed LLM weights.

**NF4 (NormalFloat4)**: 4-bit data type with 16 quantile-based levels placed at the quantiles of a standard normal distribution; information-theoretically optimal for normally distributed weights.

**Weight-only quantization**: Compressing model weight matrices to lower precision while keeping activations in full precision during inference.

**Activation quantization**: Quantizing both weights and intermediate activations; harder due to activation outliers in large LLMs.

**LLM.int8()**: Dettmers 2022 quantization method using mixed INT8/FP16 matrix multiplication to handle activation outliers; implements 8-bit inference for large LLMs.

**GPTQ**: Post-training quantization method using per-layer Hessian-based weight adjustment to compensate for quantization error; typically applied at 4-bit.

**AWQ (Activation-aware Weight Quantization)**: Post-training 4-bit quantization that protects salient weight channels by scaling them before quantization; faster to apply than GPTQ.

**Double quantization**: QLoRA technique that quantizes the 4-bit quantization scale factors themselves to 8-bit, saving an additional ~0.37 bits per parameter.

**BitsAndBytesConfig**: HuggingFace configuration class for loading models in 4-bit or 8-bit using the bitsandbytes library.

**Calibration dataset**: A small set of representative examples used by post-training quantization methods (GPTQ, AWQ) to estimate weight importance and minimize quantization error.

**GGUF**: File format for quantized models used by llama.cpp; enables CPU-based inference on Mac and other non-CUDA hardware.
