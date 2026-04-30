# Week 22 Resources — LM Evaluation

## Papers

- [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) — Holtzman et al. 2019. Explains why top-p sampling outperforms top-k for long-form generation; introduces nucleus sampling.
- [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) — Elhage et al. 2021, Anthropic. Deep dive into what small transformers actually learn.
- [BabyLM Challenge](https://babylm.github.io/) — Challenge for training LMs from scratch on limited data; useful for calibrating small-model performance expectations.

## Videos

- [Andrej Karpathy — Building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo) — (~1.5h). Language model evaluation and sampling, clearly explained.
- [How to sample from language models](https://www.youtube.com/watch?v=dR8tnK1F_bQ) — HuggingFace, shorter (~20 min). Practical guide to temperature, top-k, top-p.

## Blog Posts / Articles

- [How to generate text using different decoding methods](https://huggingface.co/blog/how-to-generate) — HuggingFace. The definitive guide to generation strategies. Required reading.
- [Perplexity of Fixed-Length Models](https://huggingface.co/docs/transformers/perplexity) — HuggingFace. Shows the sliding window approach and gotchas; explains the stride parameter.
- [GPT-2 OpenAI Blog](https://openai.com/research/better-language-models) — The original GPT-2 announcement with sample analysis. Compare their samples to yours.

## GitHub Repos

- [huggingface/transformers generation utils](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py) — The production implementation of temperature, top-k, top-p, beam search. Reading this is educational.
- [karpathy/nanoGPT generate.py](https://github.com/karpathy/nanoGPT/blob/master/sample.py) — A clean reference implementation for sampling from a GPT model.

## Documentation

- [HuggingFace text generation API](https://huggingface.co/docs/transformers/main_classes/text_generation) — Full reference for generation parameters; useful for when you move to HuggingFace model generation in later weeks.
- [HuggingFace Hub model cards](https://huggingface.co/docs/hub/model-cards) — How to write a good model card for your uploaded checkpoint.

## Optional / Bonus

- [Holistic Evaluation of Language Models (HELM)](https://crfm.stanford.edu/helm/latest/) — Stanford CRFM. The comprehensive evaluation framework you will use in Week 23.
- [Lm-evaluation-harness paper](https://arxiv.org/abs/2404.12253) — EleutherAI 2024. The paper behind the tool you will use in Week 23.
- [On the Opportunities and Risks of Foundation Models](https://arxiv.org/abs/2108.07258) — Bommasani et al. 2021. Section 3 on evaluation is relevant background.
