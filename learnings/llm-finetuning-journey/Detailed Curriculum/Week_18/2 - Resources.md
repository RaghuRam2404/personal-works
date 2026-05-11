# Week 18 Resources — Pretraining Data

## Papers

- [The Pile: An 800GB Dataset of Diverse Text for Language Modeling](https://arxiv.org/abs/2101.00027) — Gao et al. 2021, EleutherAI. The multi-source mixing approach to pretraining data.
- [The RefinedWeb Dataset for Falcon LLM](https://arxiv.org/abs/2306.01116) — Penedo et al. 2023, TII. Shows aggressive CC filtering matches multi-source data.
- [Dolma: An Open Corpus of Three Trillion Tokens for Language Model Pretraining Research](https://arxiv.org/abs/2402.00159) — Soldaini et al. 2024, AI2. Another large-scale open pretraining dataset with detailed methodology.
- [Data Curation via Joint Example Selection Further Accelerates Multimodal Learning](https://arxiv.org/abs/2406.17711) — Relevant for understanding how dataset curation interacts with model learning.
- [RedPajama-Data-v2](https://arxiv.org/abs/2411.12372) — Together AI's 30T-token dataset with quality signals attached to every document; good reference for advanced filtering.

## Videos

- [How LLMs are trained — data section](https://www.youtube.com/watch?v=zjkBMFhNj_g) — Andrej Karpathy, State of GPT (Microsoft Build 2023, ~1h). The data section (first 20 min) is required.
- [FineWeb: Making a large-scale web dataset](https://www.youtube.com/watch?v=zVAtgOXw-ao) — HuggingFace, official walkthrough (~45 min).

## Blog Posts / Articles

- [FineWeb Technical Blog Post](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) — HuggingFace. Extremely detailed write-up of the full FineWeb pipeline. Required reading.
- [FineWeb Dataset Card](https://huggingface.co/datasets/HuggingFaceFW/fineweb) — HuggingFace. Read the full dataset card, including the quality filter ablations table.
- [The Pile Dataset Card](https://huggingface.co/datasets/EleutherAI/pile) — EleutherAI. Source proportions and mixing rationale.
- [Common Crawl documentation](https://commoncrawl.org/get-started) — Common Crawl. Understanding the raw data before filtering.

## GitHub Repos

- [ekzhu/datasketch](https://github.com/ekzhu/datasketch) — MinHash and LSH implementation. Required for this week's assignment.
- [huggingface/datatrove](https://github.com/huggingface/datatrove) — HuggingFace's industrial data processing library, used to build FineWeb. Great reference for production pipelines.
- [zytedata/trafilatura](https://github.com/adbar/trafilatura) — HTML text extraction. Superior to BeautifulSoup for web content.
- [facebookresearch/fastText](https://github.com/facebookresearch/fastText) — Language detection model used throughout this week.

## Documentation

- [HuggingFace datasets streaming docs](https://huggingface.co/docs/datasets/stream) — Required for Task 1: how to use streaming mode to avoid downloading full datasets.
- [datasketch MinHashLSH docs](http://ekzhu.com/datasketch/lsh.html) — Reference for MinHashLSH API used in Task 3.

## Optional / Bonus

- [MassiveText: The training data behind Gopher](https://arxiv.org/abs/2112.11446) — Rae et al. 2021, DeepMind. How Gopher's data was filtered; includes the Gopher quality heuristics used in FineWeb.
- [The Stack v2](https://arxiv.org/abs/2402.19173) — BigCode 2024. The code pretraining dataset, relevant for your PostgreSQL dataset in Weeks 25–26.
- [ROOTS corpus](https://arxiv.org/abs/2303.03915) — BigScience 2023. Multilingual pretraining dataset construction methodology.
