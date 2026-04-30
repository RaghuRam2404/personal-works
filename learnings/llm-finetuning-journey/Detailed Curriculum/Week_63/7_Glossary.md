# Week 63 Glossary — Quantization Formats: GGUF, GPTQ, AWQ

**GGUF**: The file format used by llama.cpp; encodes quantized weights plus all model metadata in a single portable binary.

**GPTQ**: Post-training quantization algorithm that uses second-order Hessian information to minimize per-layer quantization error with calibration data.

**AWQ**: Activation-aware Weight Quantization; protects salient channels (high-activation weights) from aggressive rounding to preserve accuracy.

**K-quants**: llama.cpp's family of mixed-precision quantization schemes (e.g., Q4_K_M) that quantize different tensor types at different bit depths.

**Q4_K_M**: A specific K-quant level using 4-bit weights with medium mixed-precision for attention and feed-forward layers; the most popular GGUF deployment target.

**Salient channels**: Output channels of a weight matrix whose corresponding input activations have consistently large magnitude; erroneously quantizing these causes disproportionate output error.

**Group quantization**: Dividing a weight row into groups (e.g., 128 elements) and computing separate scale/zero-point per group, improving accuracy at modest overhead.

**Calibration data**: A small representative dataset (128–512 samples) fed through the model during GPTQ or AWQ to compute activation statistics used to guide quantization decisions.

**Dequantization**: Runtime operation that converts stored integer weights back to float16 (or bfloat16) before the matrix multiply; happens on-the-fly in GGUF/GPTQ inference.

**Memory bandwidth bottleneck**: At inference time with small batch sizes, GPU throughput is limited by how fast weights can be read from VRAM, not by FLOPs; quantization directly reduces this bottleneck.

**Perplexity degradation**: The increase in per-token perplexity caused by quantization; used as a proxy for quality loss when full benchmark evaluation is too slow.

**Absmax quantization**: The simplest quantization scheme that maps the range [-max(|W|), max(|W|)] to the integer range; sensitive to outliers.

**Zero-point quantization**: Quantization scheme that allows asymmetric mapping using both a scale and an integer offset (zero-point), improving accuracy for non-symmetric weight distributions.
