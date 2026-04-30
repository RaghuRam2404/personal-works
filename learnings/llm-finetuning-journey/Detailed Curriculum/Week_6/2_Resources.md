# Week 6 — Resources

## Videos

- [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) — Andrej Karpathy, YouTube, 2h13m. **Required. Code along. This is the entire Week 6 in one video.** Covers BPE from scratch, GPT-2 and GPT-4 regex patterns, and tiktoken.

## Blog Posts / Articles

- [HuggingFace LLM Course — Chapter 6: The Tokenizers Library](https://huggingface.co/learn/llm-course/chapter6/1) — Interactive course covering BPE, WordPiece, Unigram, and the HuggingFace tokenizers API. Do all exercises.
- [Summary of the Tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary) — HuggingFace docs. Concise reference comparing BPE, WordPiece, SentencePiece, and Unigram with diagrams.
- [Tokenization is NLP's Dirty Little Secret](https://moultano.wordpress.com/2023/06/28/tokenization-nlp/) — Ryan Moulton. Excellent rant about tokenization artifacts and why they matter for code generation.

## Papers

- [Neural Machine Translation of Rare Words with Subword Units (BPE)](https://arxiv.org/abs/1508.07909) — Sennrich et al., ACL 2016. The original BPE tokenization paper. Read sections 1, 2, and 3.1.
- [SentencePiece: A simple and language independent subword tokenizer and detokenizer](https://arxiv.org/abs/1808.06226) — Kudo & Richardson, 2018. Read the abstract + Section 3 (the algorithm). Qwen and LLaMA use SentencePiece variants.
- [Language-Agnostic BERT Sentence Embedding](https://arxiv.org/abs/2007.01852) — Feng et al. Uses multilingual tokenization; Section 3.1 discusses vocabulary trade-offs across languages.

## Documentation

- [tiktoken GitHub](https://github.com/openai/tiktoken) — OpenAI's tokenizer. Use `tiktoken.get_encoding("cl100k_base")` for GPT-4. Read the README for the supported encodings and how to add custom tokens.
- [HuggingFace tokenizers library docs](https://huggingface.co/docs/tokenizers/index) — The fast Rust-based library. Covers `ByteLevelBPETokenizer`, training, and saving/loading.
- [transformers AutoTokenizer docs](https://huggingface.co/docs/transformers/main_classes/tokenizer) — The interface you will use for all fine-tuning. Key methods: `encode`, `decode`, `batch_encode_plus`, `add_special_tokens`, `save_pretrained`.

## GitHub Repos

- [karpathy/minbpe](https://github.com/karpathy/minbpe) — The reference implementation from Karpathy's video. Do NOT look at the code until after you have completed your own implementation.
- [openai/tiktoken](https://github.com/openai/tiktoken) — Source for the `cl100k_base` and `o200k_base` GPT-4 tokenizers.
- [taoyds/spider](https://github.com/taoyds/spider) — Spider SQL dataset. Used this week for training your SQL tokenizer.

## Optional / Bonus

- [Tokenization Tutorial — Stanford CS224N](https://web.stanford.edu/class/cs224n/) — If available, the tokenization lecture from the current year's CS224N covers WordPiece and the BERT connection.
- [What Is ChatGPT Doing and Why Does It Work?](https://writings.stephenwolfram.com/2023/02/what-is-chatgpt-doing-and-why-does-it-work/) — Stephen Wolfram. Section on tokenization is a useful lay perspective.
- [Efficient Tokenization (ByT5 paper)](https://arxiv.org/abs/2105.13626) — Xue et al., 2022. Argues for byte-level models (no tokenization at all) for multilingual robustness. A good thought experiment for understanding trade-offs.
