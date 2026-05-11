# Week 66 Resources — Cloud Deployment with vLLM

## Papers

[Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — Kwon et al. 2023; the original vLLM paper introducing paged attention and continuous batching.

## Videos

[vLLM: Easy, Fast, and Cheap LLM Serving (vLLM team, UC Berkeley)](https://www.youtube.com/watch?v=5ZlavKF_98U) — ~30 min; architecture walkthrough, benchmarks vs Hugging Face TGI, and deployment demo.

[Deploy Any LLM to the Cloud in 10 Minutes with RunPod (Trelis Research)](https://www.youtube.com/watch?v=1P-VmUidtAw) — ~20 min; RunPod setup, SSH, port exposure, and vLLM launch walkthrough.

## Blog Posts / Articles

[vLLM Blog: High-throughput and Memory-efficient LLM Serving](https://blog.vllm.ai/2023/06/20/vllm.html) — Official launch post with benchmark comparisons (10–20x throughput vs naive HF inference).

[FastAPI Official Tutorial](https://fastapi.tiangolo.com/tutorial/) — Comprehensive docs; focus on the "Request Body" and "Response Model" sections for your SQL API.

[RunPod Documentation — Expose Ports](https://docs.runpod.io/pods/configuration/expose-ports) — Official guide for setting up TCP port exposure and getting the public endpoint URL.

## GitHub Repos

[vllm-project/vllm](https://github.com/vllm-project/vllm) — vLLM source; `examples/` contains OpenAI-compatible server examples; check supported quantization formats in `vllm/model_executor/layers/quantization/`.

[tiangolo/fastapi](https://github.com/tiangolo/fastapi) — FastAPI source; issues and discussions are useful for production deployment patterns.

[gradio-app/gradio](https://github.com/gradio-app/gradio) — For the HF Spaces stretch goal; `gr.Interface` and `gr.ChatInterface` docs.

## Documentation

[vLLM OpenAI-Compatible Server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) — All supported endpoints, request parameters, and response schemas.

[vLLM Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html) — Verify Qwen2.5 AWQ support before launching (it is listed under AWQ-quantized models).

[HuggingFace Spaces — Gradio SDK](https://huggingface.co/docs/hub/spaces-sdks-gradio) — How to deploy a Gradio app to HF Spaces with a `requirements.txt`.

## Optional / Bonus

[Text Generation Inference (TGI) — HuggingFace](https://github.com/huggingface/text-generation-inference) — Alternative to vLLM; Docker-based, tighter HF integration; worth knowing as a comparison point.

[LiteLLM](https://github.com/BerriAI/litellm) — Proxy layer that provides a unified OpenAI-compatible API over vLLM, Ollama, Anthropic, and others; useful for switching providers without changing client code.

[CloudFlare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/) — Free way to expose a RunPod pod's HTTP port over HTTPS with a stable subdomain, without a static IP.
