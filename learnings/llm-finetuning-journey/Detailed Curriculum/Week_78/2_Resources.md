# Week 78 Resources — Final Phase 6 Gate and Course Wrap

## Papers

- [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993) — Mitchell et al., 2019. The original model card paper; foundational reading on why and how to document models for transparency.
- [What Will It Take to Fix Benchmarking in NLP?](https://arxiv.org/abs/2106.07282) — Bowman & Dahl, 2021. Critical analysis of NLP benchmark limitations; directly applicable to thinking rigorously about Custom-200's weaknesses.
- [Are NLP Leaderboards Broken?](https://arxiv.org/abs/2004.04997) — Ethayarajh & Jurafsky, 2020. Argues that single-score leaderboards obscure important performance variation; motivates your per-stratum reporting on Custom-200.

## Videos

- [Andrej Karpathy — The State of GPT (Microsoft Build 2023)](https://www.youtube.com/watch?v=bZQun8Y4L2A) — Andrej Karpathy — 42 min. Year-in-review style talk covering the full LLM pipeline from pretraining through RLHF; a fitting capstone watch for the end of the course.
- [Yannic Kilcher — How to Read a Machine Learning Paper](https://www.youtube.com/watch?v=733m6qBH-jI) — Yannic Kilcher — ~10 min. Covers how reviewers read papers, which tells you what to prioritize in your final report writing.

## Blog Posts / Articles

- [HuggingFace Model Card Guideline](https://huggingface.co/docs/hub/model-cards) — Official specification for model cards on the Hub; covers all required and recommended fields.
- [arXiv Submission Guidelines for cs.CL](https://arxiv.org/help/submit) — Step-by-step guide to submitting a paper; includes endorsement requirements and LaTeX formatting instructions.
- [Eugene Yan — Practical MLOps for the Individual Practitioner](https://eugeneyan.com/writing/practical-mlops/) — Blog post on what production ML looks like at a small scale; actionable for a single engineer deploying their own model.
- [Andrej Karpathy — The Unreasonable Effectiveness of Data](https://cs231n.github.io/transfer-learning/) — On what actually drives model quality improvements in production settings.
- [Sebastian Ruder — NLP Research Highlights](https://ruder.io/research/) — Regular summaries of significant NLP papers; useful for staying current after the course ends.
- [Swyx — The AI Engineer (latent.space)](https://www.latent.space/p/ai-engineer) — On the "AI Engineer" role and what distinguishes it from traditional ML engineering; relevant for your job search positioning.
- [Weights & Biases — ML Engineering Roles Explained](https://wandb.ai/site/articles/ml-engineering-roles) — Breakdown of ML Engineer vs Applied Scientist vs Research Engineer; helps you target the right job title.

## GitHub Repos

- [stanford-crfm/helm](https://github.com/stanford-crfm/helm) — Holistic Evaluation of Language Models; study how a rigorous evaluation framework is built and documented.
- [defog-ai/sqlcoder](https://github.com/defog-ai/sqlcoder) — The closest public analog to your capstone model; compare your approach, model card, and results against this reference implementation.
- [eosphoros-ai/Awesome-Text2SQL](https://github.com/eosphoros-ai/Awesome-Text2SQL) — Curated list of Text-to-SQL resources, papers, and models; your reference for staying current in this specific research area.

## Documentation

- [HuggingFace Papers](https://huggingface.co/papers) — Alternative to arXiv for publishing ML work; no endorsement required, community-moderated.

## Optional / Bonus

- [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) — Rich Sutton, 2019. Reflect on whether your capstone (domain knowledge + task specificity beating a larger general model) confirms or challenges this thesis.
- [Chip Huyen — Designing Machine Learning Systems (O'Reilly, 2022)](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — The industry-standard reference for taking ML prototypes to production; chapters on data pipelines, monitoring, and deployment are directly applicable.
- [ML Collective — Open Research Community](https://mlcollective.org/) — A community for independent ML researchers without institutional affiliation; relevant if you want to collaborate on the Tamil NL→SQL or Custom-200 extension research directions.
- [ACL Anthology](https://aclanthology.org/) — Definitive archive of NLP research papers; search "text-to-SQL" filtered to 2023–2025 to see where your work fits in the current literature.
