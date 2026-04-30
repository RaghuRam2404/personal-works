# Week 67 Resources — Technical Report Writing

## Papers (reference technical reports to read)

[Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288) — Touvron et al. 2023; the gold standard for LLM technical report structure; study its dataset and training sections.

[Tulu 2: Instruction-Tuned Language Models](https://arxiv.org/abs/2311.10702) — Ivison et al. 2023; excellent dataset construction section with filtering methodology and statistics tables.

[LIMA: Less is More for Alignment](https://arxiv.org/abs/2305.11206) — Zhou et al. 2023; a shorter, focused technical report; good for abstract and contribution writing style.

[Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task](https://arxiv.org/abs/1809.08887) — Yu et al. 2018; the benchmark you compare against; needed for correct citation details.

[BIRD: A Big Bench for Large-Scale Database Grounded Text-to-SQL Evaluation](https://arxiv.org/abs/2305.03111) — Li et al. 2023; second benchmark you compare against.

## Videos

[How to Write a Great Research Paper (Simon Peyton Jones, Microsoft Research)](https://www.youtube.com/watch?v=VK51E3gHENc) — ~45 min; best practical advice on structuring a research paper; focuses on clarity and contribution framing.

[How to Read a Paper (Andrew Ng, DeepLearning.AI)](https://www.youtube.com/watch?v=733m6qBH-jI) — ~10 min; covers how reviewers read papers, which tells you what to prioritize in your writing.

## Blog Posts / Articles

[Lessons from Thousands of ML Papers (Andrej Karpathy blog)](https://karpathy.github.io/2016/09/07/phd/) — Advice on research communication style from the perspective of an experienced ML researcher.

[How to Write the Methods Section of a Machine Learning Paper](https://www.cs.jhu.edu/~jason/advice/write-the-paper-first.html) — Jason Eisner's guide; focused on ML methods section clarity.

## GitHub Repos

[huggingface/hub-docs](https://github.com/huggingface/hub-docs) — Source for HuggingFace Hub documentation; includes the model card specification and metadata YAML schema you must follow.

[paperswithcode/paperswithcode-data](https://github.com/paperswithcode/paperswithcode) — Papers With Code infrastructure; understanding how results tables are structured helps you format your benchmark results section correctly.

[EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — Evaluation framework referenced in the report; link to your eval harness fork so readers can reproduce your Custom-200 numbers.

[defog-ai/sql-eval](https://github.com/defog-ai/sql-eval) — The standardized text-to-SQL evaluation framework you used; link directly so your methodology is reproducible.

## Tools

[Overleaf](https://www.overleaf.com) — Online LaTeX editor; if you plan to submit to arXiv or a venue, LaTeX is expected. Use the NeurIPS or EMNLP template.

[Markdown to PDF via Pandoc](https://pandoc.org/MANUAL.html) — For a clean PDF without LaTeX: `pandoc report.md -o report.pdf --pdf-engine=xelatex`.

[Connected Papers](https://www.connectedpapers.com) — Visualize the citation graph around key papers; useful for finding related work you might have missed.

## Optional / Bonus

[Papers Without Code — reproducibility checklist](https://paperswithcode.com/rc2020) — ML reproducibility checklist used by NeurIPS; running your paper through this checklist reveals gaps before submission.

[arXiv submission guide](https://arxiv.org/help/submit) — When you are ready to publish: formatting requirements, LaTeX compilation, and metadata fields for ML papers.
