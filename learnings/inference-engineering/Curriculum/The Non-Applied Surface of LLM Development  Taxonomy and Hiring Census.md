# The Non-Applied Surface of LLM Development: A Taxonomy and Hiring Census Across US and China Frontier Labs

## Overview

Building a frontier large language model involves far more distinct technical disciplines than the commonly cited stages of data collection, curation, architecture design, pretraining, RL training, post-training, and inference/serving. This report does two things. First, it maps the fuller set of "non-applied" sub-disciplines — research and engineering work that happens before a model is hosted and used, as distinct from anything that consumes a finished, hosted model (chatbots, agents, RAG pipelines, coding assistants, enterprise deployments, and other downstream "applied" work). Second, it turns that taxonomy into a hiring census: every individual open job posting from five US frontier labs (OpenAI, Anthropic, xAI, Google DeepMind, Thinking Machines Lab) and three Chinese frontier labs (DeepSeek, Alibaba's Qwen/Tongyi team, Moonshot AI/Kimi) was read directly from each lab's own careers site, classified into the taxonomy, and tallied. Meta's Superintelligence Labs/FAIR postings and several smaller Chinese labs (MiniMax, Zhipu AI, ByteDance Seed, 01.AI, SenseTime) were intentionally excluded from the quantitative count at the user's direction, to keep the comparison focused and the classification rigorous rather than broad and shallow.

The headline finding: across 366 classified postings, distributed training/ML infrastructure (78) and data collection & curation (73) are the two largest non-applied job categories overall, followed by safety/alignment/interpretability research (32, almost entirely concentrated in US labs) and multimodal research (26, more concentrated in Chinese labs). US labs post roughly 2.1x as many non-applied roles as the three Chinese labs sampled (248 vs. 118), but the *shape* of hiring differs sharply by category, not just by volume.

## Part 1 — Refined Taxonomy of Non-Applied Work

The taxonomy below extends the well-known headline stages (data collection, curation, architecture, pretraining, RL, post-training, inference/serving) with the sub-disciplines that sit inside, around, and between them — each one a distinct area that a person could be hired specifically to do, and none of which involves consuming a hosted model for a downstream task.

### Data Layer
- **Data collection & curation** — sourcing, web-scale crawling, near-duplicate detection (MinHash/vector dedup), quality/toxicity/PII filtering, data provenance and licensing governance, and the human-labeling/annotation pipelines (including domain-expert "AI tutors") that feed both pretraining and post-training.
- **Synthetic data generation** — using existing models to generate additional training examples, verification data, or RL rollouts to fill capability gaps that natural data cannot cover.
- **Tokenization & data representation** — vocabulary size, segmentation algorithm choice, and multilingual/code/number handling; a distinct governance decision that shapes efficiency and fairness (e.g., Anthropic's dedicated "Research Engineer, Tokens" role).

### Model & Systems Layer
- **Model architecture / foundation model research** — attention variants, mixture-of-experts routing, positional encoding, scaling-law research, and exploration of post-scaling paradigms (continual learning, self-evolution, recursive self-improvement).
- **Pretraining research & engineering** — the algorithmic and methodological work of running and improving the base next-token pretraining objective itself, distinct from the infrastructure that executes it.
- **Distributed training systems & ML infrastructure** — cluster orchestration and scheduling (Kubernetes/Slurm/torchrun), GPU/TPU kernel engineering (CUDA/Triton/Pallas), networking (RDMA/InfiniBand/NVLink), fault tolerance and checkpointing, hardware/silicon co-design, and the numerics (low-precision training formats) that make multi-week, multi-thousand-GPU runs possible.

### Training-the-Behavior Layer
- **Post-training / RL / alignment training** — supervised fine-tuning, reward-model training, PPO/DPO/RLHF/RLAIF policy optimization, and domain-specific RL (coding, cybersecurity, chip design) that shapes what a trained base model actually does.
- **Reasoning research** — RL and training-recipe research specifically targeting multi-step chain-of-thought and long-horizon reasoning behavior, distinct from general instruction-following post-training.
- **Agent systems research** — core, model-level research into agentic capability (computer use, tool use, multi-agent collaboration, RL training "environments" for agents) — distinct from applied agent product engineering that wires a finished model into a customer workflow.
- **Multimodal research** — architecture, data, and training work specific to vision, audio, and video understanding/generation models.

### Evaluation, Efficiency, and Safety Layer
- **Evaluation & benchmarking** — building benchmark datasets, evaluation harnesses/environments, and metrics that isolate and measure specific capabilities, including LLM-as-judge methodology.
- **Model compression / quantization / efficiency** — post-training quantization, quantization-aware training, pruning, knowledge distillation, and low-rank factorization that make a trained model small and fast enough to serve.
- **Inference & serving infrastructure** — the internal runtime engines, KV-cache management, batching, and autoscaling systems that host a lab's own models — not the applied engineering that builds a customer-facing product on top of an API.
- **Safety, alignment research & interpretability** — mechanistic interpretability, red-teaming, frontier risk research (cyber/bio/chemical), and safety-oversight research into model internals and behavior, distinct from production trust-and-safety operations that enforce policy on live traffic.
- **Research engineering / platform tooling** — internal tooling, experiment tracking, and research-productivity platforms that make all of the above disciplines move faster, without being any one of them directly.

## Part 2 — Methodology and Scope

Job-board aggregators and search results for career pages are frequently stale or contain expired listings, so every posting counted in this report was read directly from each lab's own live careers site or ATS (Greenhouse, Ashby, Google Careers, or the lab's own portal) rather than from secondary aggregators. For each lab, every open posting was reviewed and classified using two decision rules applied consistently across all eight labs:

- **Included**: any role whose description involves collecting/curating training data, designing model architecture, building or operating training/inference systems used to build or run the lab's own model, conducting post-training or RL research, evaluating models, compressing/optimizing models, or researching model safety/interpretability/agentic capability at the core-model level.
- **Excluded ("applied")**: sales, partnerships, forward-deployed/solutions/"Applied AI" engineering for customers, product engineering for consumer-facing app surfaces (e.g., the ChatGPT or Claude.ai UI), production trust-and-safety enforcement operations, marketing, legal, recruiting, finance, physical data-center construction/facilities, and corporate/IT security.

Roles that touch two categories (for example, an "RL Infrastructure Engineer" who builds systems for RL training) were assigned to the systems/infrastructure category when the role is fundamentally about building or operating training/serving systems, and to the algorithmic category (post-training/RL, pretraining, etc.) when the role is about designing the training recipe, reward signal, or model behavior itself. This rule was the single largest source of classification judgment calls and is the main place two careful readers might disagree by a few roles per lab.

**Labs and sources used:**

| Lab | Country | Source | Total open roles (site) | Relevant non-applied postings |
|---|---|---|---|---|
| OpenAI | US | [openai.com/careers/search](https://openai.com/careers/search/) | 750 | 74 |
| Anthropic | US | [anthropic.com/careers/jobs](https://www.anthropic.com/careers/jobs) | 518 | 72 |
| xAI | US | [job-boards.greenhouse.io/xai](https://job-boards.greenhouse.io/xai) | 259 | 60 |
| Google DeepMind | US | [Google Careers, DeepMind filter](https://www.google.com/about/careers/applications/jobs/results?company=DeepMind) | 73 | 22 |
| Thinking Machines Lab | US | [jobs.ashbyhq.com/ThinkingMachines](https://jobs.ashbyhq.com/ThinkingMachines) | 34 | 20 |
| DeepSeek (深度求索) | China | [talent.deepseek.com](https://talent.deepseek.com/) | 33 | 15 |
| Alibaba Qwen (通义千问 / Token Foundry) | China | [careers-tongyi.alibaba.com](https://careers-tongyi.alibaba.com/off-campus/position-list?lang=zh&search=) | 58 | 50 |
| Moonshot AI / Kimi (月之暗面) | China | [app.mokahr.com/apply/moonshot](https://app.mokahr.com/apply/moonshot/148506) + [campus board](https://app.mokahr.com/campus-recruitment/moonshot/148507) | 193 | 53 |

**Scope notes and limitations:**
- Meta's Superintelligence Labs/FAIR postings were dropped from this comparison at the user's direction after the data-collection run was interrupted; no Meta figures appear anywhere in the tables below.
- Per the user's direction, the Chinese sample was deliberately narrowed to three labs (DeepSeek, Alibaba Qwen, Moonshot AI) rather than the fuller set (MiniMax, Zhipu AI, ByteDance Seed, 01.AI, SenseTime) that a broader census would include; the China totals below should be read as a three-lab sample, not a market-wide figure.
- Postings are a live, daily-changing snapshot taken August 23, 2026. Absolute counts will drift; the categorical *pattern* is the more durable finding.
- Several sub-agent research passes initially mis-tallied their own summary headline count against their own itemized list (a known LLM self-counting error). Every count in this report was independently rebuilt from the itemized job titles each pass actually listed, not from its stated summary number, and cross-checked against the itemized total to make sure the two reconcile.
- Job titles at Chinese labs were read and translated from Chinese; classification confidence for these is slightly lower than for English-language postings, though most titles were sufficiently descriptive (e.g., "预训练数据工程师" = Pretraining Data Engineer) to classify with confidence.

## Part 3 — The Aggregated Table: Role Area vs. Company

![Open non-applied postings by lab](https://d2z0o16i8xm8ak.cloudfront.net/abf1048d-47ee-4bed-aafa-603c6d9bb51c/b452b6da-8801-4eca-b5d5-168460b518a7/postings_by_lab.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9kMnowbzE2aTh4bThhay5jbG91ZGZyb250Lm5ldC9hYmYxMDQ4ZC00N2VlLTRiZWQtYWFmYS02MDNjNmQ5YmI1MWMvYjQ1MmI2ZGEtODgwMS00ZWNhLWI1ZDUtMTY4NDYwYjUxOGE3L3Bvc3RpbmdzX2J5X2xhYi5wbmc~KiIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc4ODA4ODcyMX19fV19&Signature=YcAmBgAUH~etuH0Yp5E62oUZG7E4LgzeKslvX6lzLmjV5N8ZdIWx2SzlyiyTGakbppRpMhUVv3u5ML~-kBLg4U8H~rrQZinCugeqMelJ5oQhT04OYTHkNz0NdVrc3gEFXPAuLj2dPiG5MtNQFDOodDhi87clXJgo~88-gbTBbWDBcC-VtgwWlnPRXvfLb4sdNmI87nEouPLXf6mxy6gxLAjOkGslBvtlYz72zzWFk8X0acPopqkdJnY7bLhbonai55lbai6Rw9cElwQg7-MJqpul6NhRE386P6UqQRyrZbP9RJTDMF-4pUPr28m37VJ31aa1IT7Ruz3NJKG5JEEXUg__&Key-Pair-Id=K1BF7XGXAIMYNX)

The chart above shows the volume disparity: even after applying an identical, strict "non-applied" filter, the three largest US labs (OpenAI, Anthropic, xAI) each post 60–74 non-applied research/engineering roles at any given time, roughly 3x Google DeepMind or Thinking Machines Lab and comparable to or larger than Alibaba Qwen (50) and Moonshot AI (53). DeepSeek, despite its outsized research reputation, has the smallest open non-applied headcount of any lab sampled (15), consistent with its famously lean team structure.

### Full count: role area vs. company

| Role area (non-applied) | OpenAI | Anthropic | xAI | Google DeepMind | Thinking Machines Lab | DeepSeek | Alibaba Qwen | Moonshot AI (Kimi) | **US total** | **China total** | **Grand total** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Data collection & curation | 10 | 5 | 45 | 0 | 1 | 1 | 1 | 10 | **61** | **12** | 73 |
| Synthetic data generation | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | **2** | **1** | 3 |
| Model architecture / foundation model research | 6 | 5 | 0 | 0 | 1 | 2 | 8 | 2 | **12** | **12** | 24 |
| Pretraining research & engineering | 2 | 5 | 2 | 0 | 0 | 1 | 1 | 2 | **9** | **4** | 13 |
| Distributed training systems & ML infrastructure | 15 | 23 | 7 | 5 | 8 | 5 | 7 | 8 | **58** | **20** | 78 |
| Post-training / RL / alignment training | 9 | 10 | 1 | 5 | 3 | 1 | 5 | 4 | **28** | **10** | 38 |
| Reasoning research | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 3 | **1** | **4** | 5 |
| Evaluation & benchmarking | 2 | 1 | 1 | 0 | 0 | 0 | 4 | 5 | **4** | **9** | 13 |
| Model compression / quantization / efficiency | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | **1** | **1** | 2 |
| Inference & serving infrastructure | 4 | 8 | 1 | 2 | 1 | 1 | 1 | 7 | **16** | **9** | 25 |
| Safety, alignment research & interpretability | 16 | 10 | 0 | 5 | 0 | 0 | 1 | 0 | **31** | **1** | 32 |
| Agent systems research | 7 | 2 | 0 | 2 | 0 | 2 | 6 | 7 | **11** | **15** | 26 |
| Multimodal research | 0 | 1 | 3 | 3 | 2 | 1 | 14 | 2 | **9** | **17** | 26 |
| Research engineering / platform tooling | 0 | 2 | 0 | 0 | 3 | 1 | 1 | 1 | **5** | **3** | 8 |
| **Total** | **74** | **72** | **60** | **22** | **20** | **15** | **50** | **53** | **248** | **118** | **366** |

![Non-applied role areas: US vs China](https://d2z0o16i8xm8ak.cloudfront.net/abf1048d-47ee-4bed-aafa-603c6d9bb51c/f52adbd3-6931-4841-95f9-622b6d3fb2c0/category_us_vs_china.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9kMnowbzE2aTh4bThhay5jbG91ZGZyb250Lm5ldC9hYmYxMDQ4ZC00N2VlLTRiZWQtYWFmYS02MDNjNmQ5YmI1MWMvZjUyYWRiZDMtNjkzMS00ODQxLTk1ZjktNjIyYjZkM2ZiMmMwL2NhdGVnb3J5X3VzX3ZzX2NoaW5hLnBuZz8qIiwiQ29uZGl0aW9uIjp7IkRhdGVMZXNzVGhhbiI6eyJBV1M6RXBvY2hUaW1lIjoxNzg4MDg4NzIxfX19XX0_&Signature=aLBuX-LrwKIm6nZi9dcDz3CGadXXhk-HuVFslcFHYtGm9m-Eh7k4SXRN2QN1GS1dIxrYl3XscO3L6naAfjwyyAb68ilCXo44kqKVdjZq6zHhJumVWTXClQ8zwlE81VjzPA~IDZ9sSZUbW~GngvbJZTVJclywsowS8VjuXVSK8GrEYF9CV7AyZaKax~7JqbkfWVESWrB5o8J9r7KdUVWaSBxv18LVBcLq2Fvncb1rWYAh8rTvMRzKYiZ9EvozpVzH-tOYLv33K-U7b4XHx0ejqILsqT3bb0AOpXS8VGKbHcJTCKvgp-b0qe4SZdshI~aD9j4fc6YRytahlbAzA1X9hQ__&Key-Pair-Id=K1BF7XGXAIMYNX)

## Part 4 — What the Pattern Says

**Infrastructure and data dominate everywhere, but for different reasons.** Distributed training/ML infrastructure is the single largest category in both regions (58 US, 20 China) — every lab, regardless of size or region, needs people who can keep tens of thousands of GPUs training reliably. Data collection & curation is nearly as large (73 total) but its US total is dramatically skewed by xAI, whose 45 relevant roles are almost entirely "AI Tutor" and domain-expert data-labeling positions across nearly 40 languages and STEM subjects — a distinctive, human-annotation-heavy approach to building training data that is not mirrored at this scale by any other lab in the sample, including Moonshot AI's own sizable annotation-operations footprint (10 roles).

**Safety and interpretability research is almost entirely a US phenomenon in this sample.** US labs post 31 dedicated safety/alignment-research/interpretability roles versus just 1 across the three Chinese labs sampled (an "AI Cybersecurity LLM Algorithm Engineer" at Alibaba Qwen). This gap is the sharpest asymmetry in the entire dataset. It is concentrated in two labs specifically: Anthropic (10 roles, including a "Frontier Red Team" lead and multiple dedicated interpretability research-engineer and research-scientist tracks) and OpenAI (16 roles spanning "Model Policy," frontier cyber/bio risk research, and recursive-self-improvement safety). Google DeepMind adds 5 more (AGI Safety and Alignment team, Frontier Safety Framework governance, Safety Oversight). Whether this reflects a genuine difference in safety-research investment, a difference in how each lab's public job board is organized (some safety work may exist internally at Chinese labs without a standalone public posting), or both, cannot be fully resolved from job-board data alone — but the near-total absence of any comparable public posting category at DeepSeek, Qwen, or Moonshot is a striking and repeatable finding across the full itemized list, not just the summary count.

**Multimodal and agent-systems research skew toward China.** Chinese labs post nearly twice as many multimodal-research roles (17 vs. 9) and more agent-systems-research roles (15 vs. 11) than the sampled US labs, despite having under half the total headcount. Alibaba Qwen alone accounts for 14 multimodal roles — spanning its Qwen-Audio, Qwen-Image, and Wan (video generation) sub-teams — reflecting a deliberate multi-product family strategy (separate text, audio, image, and video foundation models under one umbrella) rather than a single flagship multimodal LLM. Moonshot AI and DeepSeek both show meaningful agent-systems-research investment (Agent Harness / "Agentic RL" research tracks) that is proportionally larger relative to their total non-applied headcount than most US labs except OpenAI, whose "Agent Post-Training" sub-team (7 distinct roles: computer use, connectors, context, personality, artifacts, and more) is the most granularly subdivided agent-research organization found in the entire sample.

**Evaluation & benchmarking and reasoning research are both larger, proportionally, in China.** Alibaba Qwen (4 eval, 1 reasoning) and Moonshot AI (5 eval, 3 reasoning — including a dedicated "long chain-of-thought RL" research track) together outweigh the entire US sample's reasoning-research postings (1, at OpenAI) and come close to matching the US eval total (4). This is notable given that reasoning-focused training (extended chain-of-thought, verifier-guided RL) has been a publicly stated technical priority for both DeepSeek (R1) and Moonshot (Kimi's reasoning-focused releases) — the hiring pattern here is consistent with, and corroborates, each lab's public technical narrative.

**Post-training/RL is the single largest algorithmic (non-infrastructure) category everywhere.** At 38 total roles it outsizes pretraining research (13) by roughly 3x across the whole sample — a strong signal that, in 2026, marginal researcher headcount at the frontier is being allocated far more to shaping model behavior after the base model exists than to the base pretraining run itself. Anthropic (10) and OpenAI (9, split across a dedicated "Personal AGI" post-training sub-team) lead the US side; Alibaba Qwen (5) and Moonshot AI (4) lead China.

**Company-level specialization is visible even within a small sample.** xAI's headcount is overwhelmingly weighted toward data annotation (45 of 60 relevant roles), a stark contrast to Anthropic, where infrastructure (23) and safety (10) dominate and there is no annotation-labeling job category at all on the public board. Thinking Machines Lab, the newest and smallest lab sampled, shows a hiring profile weighted almost entirely toward infrastructure (8 of 20) and platform tooling (3) — consistent with a lab still building its core training stack (including its public "Tinker" fine-tuning platform) rather than running many parallel algorithmic research programs yet.

## Part 5 — Notable Distinctly-Titled Roles by Category

A sample of concretely-titled postings illustrating how granularly some labs subdivide these categories, gathered directly from each lab's job board:

- **Agent systems research**: OpenAI's "Agent Post-Training, Computer Use Research," "…Connectors Research," and "…Context Research" ([openai.com/careers](https://openai.com/careers/search/)); Anthropic's "Research Engineer, Universes" (builds RL training environments for agentic capability) and "Research Engineer, Computer Use" ([anthropic.com/careers/jobs](https://www.anthropic.com/careers/jobs)); DeepSeek's "Agent Harness Team — Deep Learning Researcher" and Alibaba's "Computer-Use Agent Algorithm Expert" ([careers-tongyi.alibaba.com](https://careers-tongyi.alibaba.com/off-campus/position-list?lang=zh&search=)).
- **Interpretability**: Anthropic's "Research Manager, Interpretability," "Research Scientist, Interpretability," and "Software Engineer, Infrastructure, Interpretability" — three distinct interpretability-specific job tracks at one lab ([anthropic.com/careers/jobs](https://www.anthropic.com/careers/jobs)); OpenAI's "Researcher, Interpretability" under its Safety Systems/Model Policy/Preparedness org ([openai.com/careers/search](https://openai.com/careers/search/)).
- **Reasoning**: Moonshot AI's "RL / LLM Researcher-Engineer — Reasoning," explicitly framed around "long chain-of-thought RL research" ([Moonshot job board](https://app.mokahr.com/apply/moonshot/148506)); Alibaba's "Model Self-Evolution Algorithm Expert" under the Qwen Model Training team.
- **Hardware/silicon co-design**: Google DeepMind's "Staff Silicon Architect" (architects TPU-class accelerator hardware) and "Software Engineer, GenAI Silicon Automation" ([Google Careers, DeepMind](https://www.google.com/about/careers/applications/jobs/results?company=DeepMind)); Thinking Machines Lab's "Research Engineer, Infrastructure, Numerics" (low-precision BF16/MXFP8/NVFP4 formats) ([Ashby board](https://jobs.ashbyhq.com/ThinkingMachines)).
- **Data annotation specialization**: xAI's nearly 40 language-specific "AI Tutor" roles plus subject-specific STEM tutors (Physics, Pure Math, Statistics, Data Science) ([job-boards.greenhouse.io/xai](https://job-boards.greenhouse.io/xai)); DeepSeek's separate "Pretraining Data Engineer" (algorithmic/pipeline) versus data-annotation-product-manager tracks for general, domain-expert, creative, and "emotional intelligence" data ([talent.deepseek.com](https://talent.deepseek.com/)).

## Conclusion

The taxonomy in Part 1 confirms that "non-applied" LLM work is not one job but a dense stack of at least fourteen distinct, independently-hireable disciplines. The hiring census in Parts 2–4 shows those disciplines are not weighted equally anywhere, and the weighting differs meaningfully by lab and by region: US labs sampled invest disproportionately in safety/interpretability research and post-training/RL breadth, while the three Chinese labs sampled invest disproportionately in multimodal model families and reasoning/agent research relative to their overall size. Infrastructure and data curation remain the largest single functions everywhere, underscoring that even at the frontier, the majority of non-applied headcount is systems and data engineering rather than the algorithmic research work that tends to dominate public discussion of "AI research."
