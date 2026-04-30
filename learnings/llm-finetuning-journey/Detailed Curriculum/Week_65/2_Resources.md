# Week 65 Resources — Local Deployment

## Papers / Technical Docs

[llama.cpp: Efficient LLM Inference in C/C++ (project README)](https://github.com/ggerganov/llama.cpp/blob/master/README.md) — Comprehensive build instructions, backend options, and performance notes including Metal.

## Videos

[Run Any LLM Locally with Ollama (Matt Williams / Ollama official)](https://www.youtube.com/watch?v=ZoxJcPkjirs) — ~20 min; covers install, Modelfile creation, and REST API usage.

[llama.cpp Deep Dive — Build, Quantize, and Run (Andrej Karpathy adjacent community video)](https://www.youtube.com/watch?v=ms-2F5PpqhU) — ~35 min; Metal backend walkthrough and performance profiling.

## Blog Posts / Articles

[Ollama Modelfile Reference](https://github.com/ollama/ollama/blob/main/docs/modelfile.md) — Official spec for all Modelfile directives: FROM, PARAMETER, SYSTEM, TEMPLATE, MESSAGE.

[Ollama OpenAI Compatibility](https://github.com/ollama/ollama/blob/main/docs/openai.md) — Documents which OpenAI API endpoints Ollama supports and what parameters are accepted.

[How to Run LLMs on Apple Silicon with llama.cpp (Simon Willison)](https://simonwillison.net/2023/Nov/29/llamafile/) — Practical walkthrough of Metal performance and common pitfalls on M1/M2/M3.

## GitHub Repos

[ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) — Source; build instructions in `docs/build.md`; grammar files in `grammars/` including `sql.gbnf`.

[ollama/ollama](https://github.com/ollama/ollama) — Ollama source and issue tracker; search issues for Qwen2.5 model-specific quirks.

[janhq/jan](https://github.com/janhq/jan) — GUI frontend for local models built on llama.cpp; useful for quick testing without writing code.

## Documentation

[llama.cpp GBNF Grammar Guide](https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md) — How to write grammar files to constrain output to SQL or any other formal language.

[Ollama REST API Reference](https://github.com/ollama/ollama/blob/main/docs/api.md) — All API endpoints, request/response schemas, and streaming format documentation.

[PyInstaller Quickstart](https://pyinstaller.org/en/stable/usage.html) — For packaging sql_ask.py as a single-file binary for distribution.

## Optional / Bonus

[llamafile — Single-file distributable LLMs (Mozilla)](https://github.com/Mozilla-Ocho/llamafile) — Package your GGUF + llama.cpp runtime as a single executable that runs on any platform with no install.

[LM Studio](https://lmstudio.ai) — GUI app for running GGUF models locally; useful for demoing your model to stakeholders without a CLI.

[Ollama JavaScript and Python Libraries](https://github.com/ollama/ollama-python) — Official Python client that wraps the REST API with type hints; cleaner than raw httpx for production tooling.
