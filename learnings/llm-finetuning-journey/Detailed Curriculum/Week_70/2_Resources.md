# Week 70 Resources — Limitations, Polish, and Publication

## Papers (model technical reports to study for limitations sections)

[Llama 2 Technical Report](https://arxiv.org/abs/2307.09288) — Touvron et al. 2023; Section 6 (Limitations) is one of the best examples of honest, specific limitations writing in recent LLM papers.

[GPT-4 Technical Report](https://arxiv.org/abs/2303.08774) — OpenAI 2023; Section 8 covers limitations and risks; a different style (broader scope) but instructive.

[Mistral 7B](https://arxiv.org/abs/2310.06825) — Jiang et al. 2023; very concise technical report; good example of how to be dense without being vague.

## Videos

[How to Submit to arXiv (LaTeX to PDF walkthrough)](https://www.youtube.com/watch?v=ks-e6Z-UM7g) — Murad Abusufiyan — ~20 min; covers arXiv account setup, LaTeX compilation, and the submission interface.

[Writing Research Limitations — How to Be Honest Without Undermining Your Paper](https://www.youtube.com/watch?v=2LnPjGKZjLg) — Academic English Now — ~15 min; academic writing advice; directly applicable to ML technical reports.

## Blog Posts / Articles

[How to Write a Good Scientific Paper: Conclusion and Future Work (SPIE)](https://spie.org/news/photonics-focus/janfeb-2020/how-to-write-a-good-scientific-paper) — Practical guidance on limitations vs future work distinction.

[arXiv Submission Guide for Authors](https://arxiv.org/help/submit) — Official guide; covers LaTeX requirements, figure embedding, and how to update a preprint after submission.

[HuggingFace Model Cards Documentation](https://huggingface.co/docs/hub/model-cards) — How to write a model card that links to a technical report PDF; covers metadata tags for discoverability.

## GitHub Repos

[huggingface/hub-docs](https://github.com/huggingface/hub-docs) — Source for HuggingFace model card specification; the YAML schema and required fields are defined here.

[defog-ai/sqlcoder](https://github.com/defog-ai/sqlcoder) — The closest public analog to your capstone model; compare its model card structure, limitations section, and benchmark reporting against your own.

[EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — Link to this in your limitations section when documenting which benchmarks you did and did not run.

## Tools

[Pandoc](https://pandoc.org) — Convert Markdown → LaTeX → PDF: `pandoc report.md -o report.pdf --pdf-engine=xelatex`. Install via `brew install pandoc` on macOS.

[Overleaf NeurIPS 2024 Template](https://www.overleaf.com/latex/templates/neurips-2024/tpsbbrdqcmsh) — If you want the standard ML venue paper format; use this template for arXiv submission.

[Grammarly](https://grammarly.com) — Useful for a final pass on tense consistency and clarity; paste section by section.

## Optional / Bonus

[Papers With Code — Submit Your Paper](https://paperswithcode.com/submit) — Once published on arXiv, submit to Papers With Code to get automatic benchmark tracking; your Custom-200 numbers will appear on a leaderboard.

[HuggingFace Daily Papers Submission](https://huggingface.co/papers) — Community-curated daily papers; submit after arXiv publication; strong results on public benchmarks are prioritized.

[Research Debt (Chris Olah and Shan Carter)](https://distill.pub/2017/research-debt/) — Reflection on the cost of unclear research communication; motivates writing the clearest technical report you can.
