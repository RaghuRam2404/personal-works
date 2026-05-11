# LLM Course — Mismatch Detail Report

**Total mismatches: 496** across 78 weeks  
**Source:** `.verify/llm_2026-05-01T09-52-32.json`  

> A "mismatch" means the URL is live and reachable, but the stored `link_text` in the markdown file does not exactly match the actual H1 heading on the target page. The link itself is not broken.

---

## Table of Contents
1. [Mismatch categories & counts](#1-mismatch-categories--counts)
2. [Category A — URL used as link text](#category-a--url-used-as-link-text-5-links)
3. [Category B — Repository transferred or renamed](#category-b--repository-transferred-or-renamed-8-links)
4. [Category C — Page redirect / wrong page](#category-c--page-redirect--wrong-page-15-links)
5. [Category D — Abbreviated paper title](#category-d--abbreviated-paper-title-~160-links)
6. [Category E — Informal/descriptive label for docs](#category-e--informaldescriptive-label-for-docs-~200-links)
7. [Category F — Capitalisation / punctuation only](#category-f--capitalisation--punctuation-only-~25-links)
8. [Category G — Model card name mismatch](#category-g--model-card-name-mismatch-~15-links)
9. [Full per-week mismatch table](#full-per-week-mismatch-table)

---

## 1. Mismatch categories & counts

| Category | Description | Approx. count |
|----------|-------------|:-------------:|
| A | URL used verbatim as link text | 5 |
| B | GitHub repository transferred or renamed | 8 |
| C | Link redirects to different page than described | 15 |
| D | Abbreviated paper title (subtitle/parenthetical dropped) | ~160 |
| E | Informal or descriptive label for a docs/blog page | ~200 |
| F | Capitalisation, punctuation, or whitespace difference only | ~25 |
| G | HuggingFace model card display name mismatch | ~15 |

**The vast majority (≈75%) fall into categories D and E** — the link works fine, the text just describes rather than quotes the page heading.

---

## Category A — URL used as link text (5 links)

These are cases where the raw URL was pasted in as the display text.

| ID | Week | File | Stored text | Actual title |
|----|------|------|-------------|--------------|
| 654 | Week_42 | 0 - Overview.md | `arxiv.org/abs/1707.06347` | Proximal Policy Optimization Algorithms |
| 655 | Week_42 | 0 - Overview.md | `arxiv.org/abs/2203.02155` | Training language models to follow instructions with human feedback |
| 671 | Week_42 | 3 - Assignment.md | `arxiv.org/abs/1707.06347` | Proximal Policy Optimization Algorithms |
| 672 | Week_42 | 3 - Assignment.md | `arxiv.org/abs/2203.02155` | Training language models to follow instructions with human feedback |
| 960 | Week_57 | 3 - Assignment.md | `archive.org/details/stackexchange` | Stack Exchange Data Dump |

**Fix:** Replace the raw URL display text with the actual paper/resource title.

---

## Category B — Repository transferred or renamed (8 links)

GitHub repos where the organisation or repository was moved — the URL now redirects to a new slug.

| ID | Week | File | Stored text | Actual slug (redirected to) | Notes |
|----|------|------|-------------|---------------------------|-------|
| 165 | Week_18 | 2 - Resources.md | `zytedata/trafilatura` | `adbar/trafilatura` | Ownership transferred |
| 185 | Week_19 | 2 - Resources.md | `microsoft/DeepSpeed` | `deepspeedai/DeepSpeed` | Org renamed |
| 186 | Week_19 | 2 - Resources.md | `pytorch/torchtune` | `meta-pytorch/torchtune` | Org renamed |
| 196 | Week_2 | 0 - Overview.md | `yale-lily/spider` | `taoyds/spider` | Repo moved |
| 215 | Week_2 | 3 - Assignment.md | `yale-lily/spider` | `taoyds/spider` | Repo moved |
| 283 | Week_23 | 2 - Resources.md | `yale-lily/spider` | `taoyds/spider` | Repo moved |
| 264 | Week_22 | 2 - Resources.md | `karpathy/nanoGPT generate.py` | `karpathy/nanoGPT sample.py` | File renamed in repo |
| 588 | Week_39 | 2 - Resources.md | `taoyds/spider` | `yale-lily/spider (redirected from taoyds/spider)` | Repo moved back |

**Fix for spider:** Update all `yale-lily/spider` references to `taoyds/spider` (the current canonical URL). There are 6+ occurrences across multiple weeks.

---

## Category C — Link redirects to different page than described (15 links)

These are cases where the URL loads a different page than what the stored text says — either the resource moved, the site was rebranded, or the description is about a subsection that doesn't match the landing page H1.

| ID | Week | Stored text | Actual H1 | Issue |
|----|------|-------------|-----------|-------|
| 6 | Week_1 | Yes You Should Understand Backprop | Medium | Medium domain catches all karpathy.medium.com links and shows generic "Medium" |
| 7 | Week_1 | PyTorch Autograd Explained | Towards Data Science | Same — generic subdomain H1 |
| 45 | Week_11 | GPT-2 Weight Tying — EleutherAI blog | The pragmatic tradeoff of tied embeddings \| Silen | Blog moved from EleutherAI to silennai.com |
| 56 | Week_12 | Understanding GQA and MQA | Unlocking Longer Generation with Key-Value Cache Quantization | Wrong URL — goes to KV cache article, not GQA/MQA |
| 98 | Week_15 | Training Neural Networks — Andrej Karpathy | A Recipe for Training Neural Networks | Minor — subtitle vs title |
| 136 | Week_17 | A Hitchhiker's Guide to Scaling Laws | New Scaling Laws for Large Language Models | LessWrong post title vs article H1 |
| 138 | Week_17 | EleutherAI Scaling Laws post | Chinchilla scaling: A replication attempt | Post moved to epoch.ai |
| 180 | Week_19 | HuggingFace Accelerate: Concept Guides | Loading big models into memory | URL links to specific subpage, not concept guides index |
| 245 | Week_21 | Loss spikes and how to handle them | Efficient Training on a Single GPU | URL leads to single-GPU efficiency page, not a spikes-specific one |
| 252 | Week_21 | Training Neural Networks: A Road to Hell | Deep Learning for NLP Best Practices | ruder.io article title differs from stored description |
| 554 | Week_37 | TimescaleDB documentation | Tiger Data Documentation | TimescaleDB rebranded as TigerData — docs now at tigerdata.com |
| 289 | Week_23 | Spider benchmark | Spider 1.0 | yale-lily.github.io/spider page title is "Spider 1.0" |
| 1016 | Week_60 | Zephyr: Direct Distillation of LM Alignment | Illustrating Reinforcement Learning from Human Feedback (RLHF) | Stored text is about Zephyr but URL goes to HF RLHF blog post |
| 1017 | Week_60 | willthompson.io — GRPO from scratch | The Illustrated GRPO: Group Relative Policy Optimization Explained | URL goes to abderrahmanskiredj.github.io, not willthompson.io |
| 1141 | Week_67 | Lessons from Thousands of ML Papers (Andrej Karpathy blog) | A Survival Guide to a PhD | URL goes to karpathy's PhD survival guide, not ML papers post |

**Most actionable fixes:**
- **ID 56** (Week_12): Wrong URL — should link to a GQA/MQA article, not the KV cache article.
- **ID 554** (all occurrences): Update TimescaleDB doc links — the site is now TigerData.
- **ID 1016** (Week_60): Stored description is about Zephyr but URL is for the HF RLHF post. Either fix the URL or fix the text.
- **ID 1017** (Week_60): URL domain doesn't match stored description — verify intended link.
- **ID 1141** (Week_67): URL goes to Karpathy's PhD guide, but text says "Lessons from Thousands of ML Papers" — wrong URL.

---

## Category D — Abbreviated paper title (~160 links)

The most common pattern: arxiv paper titles stored without their subtitle, parenthetical clarification, or slight capitalisation variation. The papers are accessible; only the display text is abbreviated.

### Sub-pattern D1: Subtitle in parentheses dropped

| ID | Week | Stored text | Actual title |
|----|------|-------------|--------------|
| 670 | Week_42 | Secrets of RLHF in Large Language Models Part I | Secrets of RLHF in Large Language Models Part I: PPO |
| 684 | Week_43 | A General Theoretical Paradigm to Understand Learning from Human Feedback | A General Theoretical Paradigm to Understand Learning from Human Preferences |
| 688 | Week_44 | RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback | RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback |
| 701 | Week_44 | Self-Play Fine-Tuning (SPIN) | Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models |
| 702 | Week_44 | Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs | Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing |
| 715 | Week_45 | From r to Q*: Your Language Model is Secretly a Q-Function (Rafailov et al. 2024) | From r to Q*: Your Language Model is Secretly a Q-Function |
| 716 | Week_45 | SPIN: Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models | Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models |
| 733 | Week_46 | STILL-2 / Imitate, Explore, and Self-Improve | Imitate, Explore, and Self-Improve: A Reproduction Report on Slow-thinking Reasoning Systems |
| 734 | Week_46 | DAPO: An Open-Source LLM Reinforcement Learning System | DAPO: An Open-Source LLM Reinforcement Learning System at Scale |
| 736 | Week_47 | Let's Verify Step by Step (Lightman et al. 2023) | Let's Verify Step by Step |
| 737 | Week_47 | Scaling LLM Test-Time Compute Optimally Improves Reasoning | Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters |
| 756 | Week_48 | DeepSeekMath: Pushing the Limits of Mathematical Reasoning | DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models |
| 757 | Week_48 | DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL | DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning |
| 764 | Week_48 | DAPO: An Open-Source LLM RL System at Scale | DAPO: An Open-Source LLM Reinforcement Learning System at Scale |
| 830 | Week_52 | Learning to Summarize from Human Feedback | Learning to summarize from human feedback |
| 831 | Week_52 | Training Language Models to Follow Instructions with Human Feedback (InstructGPT) | Training language models to follow instructions with human feedback |
| 834 | Week_52 | RLAIF: Scaling Reinforcement Learning from Human Feedback using AI Feedback | RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback |
| 841 | Week_52 | Tülu 3: Pushing Frontiers in Open Language Model Post-Training | Tulu 3: Pushing Frontiers in Open Language Model Post-Training |
| 867 | Week_53 | DEITA: What Makes Good Data for Alignment? | What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning |
| 868 | Week_53 | Alpagasus: Training a Better Alpaca with Fewer Data | AlpaGasus: Training A Better Alpaca with Fewer Data |
| 883 | Week_53 | Dolma dataset paper | Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research |
| 889 | Week_54 | WizardLM: Empowering Large Language Models to Follow Complex Instructions | WizardLM: Empowering large pre-trained language models to follow complex instructions |
| 903 | Week_54 | Evol-Instruct paper | WizardLM: Empowering large pre-trained language models to follow complex instructions |
| 906 | Week_55 | Alpagasus: Training a Better Alpaca with Fewer Data | AlpaGasus: Training A Better Alpaca with Fewer Data |
| 908 | Week_55 | Auto-PRE: An Automatic and Cost-Efficient Peer-Review Framework for LLM Evaluation | Auto-PRE: An Automatic and Cost-Efficient Peer-Review Framework for Language Generation Evaluation |
| 921 | Week_55 | Chatbot Arena Elo methodology | Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference |
| 922 | Week_55 | Constitution AI (Anthropic) | Constitutional AI: Harmlessness from AI Feedback |
| 923 | Week_56 | CoSQL: A Conversational Text-to-SQL Challenge | CoSQL: A Conversational Text-to-SQL Challenge Towards Cross-Domain Natural Language Interfaces to Databases |
| 942 | Week_57 | Efficient Online Data Mixing for Language Model Pre-Training | Efficient Online Data Mixing For Language Model Pre-Training |
| 944 | Week_57 | StarCoder: May the Source Be with You! | StarCoder: may the source be with you! |
| 957 | Week_57 | LIMA: Less Is More for Alignment (Zhou et al. 2023) | LIMA: Less Is More for Alignment |
| 958 | Week_57 | Dolma: An Open Corpus of Three Trillion Tokens | Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research |
| 979 | Week_59 | A General Theoretical Paradigm to Understand Learning from Human Feedback | A General Theoretical Paradigm to Understand Learning from Human Preferences |
| 991 | Week_59 | IPO: A General Framework for Preference Optimization | A General Theoretical Paradigm to Understand Learning from Human Preferences |
| 1012 | Week_60 | Open Problems and Fundamental Limitations of RLHF (Casper et al. 2023) | Open Problems and Fundamental Limitations of Reinforcement Learning from Human Feedback |
| 1026 | Week_60 | Evaluating Large Language Models Trained on Code (HumanEval, pass@K metric) | Evaluating Large Language Models Trained on Code |
| 1029 | Week_61 | BIRD: A Big Bench for Large-Scale Database Grounded Text-to-SQL Evaluation | Can LLM Already Serve as A Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs |
| 1046 | Week_61 | WikiSQL: A Large Crowd-Sourced Dataset for Developing Natural Language Interfaces | Seq2SQL: Generating Structured Queries from Natural Language using Reinforcement Learning |
| 1049 | Week_62 | Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena (Zheng et al. 2023) | Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena |
| 1063 | Week_62 | Chatbot Arena methodology paper | Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference |
| 1071 | Week_63 | Tim Dettmers — A Gentle Introduction to 8-bit Matrix Multiplication for Transformers | A Gentle Introduction to 8-bit Matrix Multiplication for transformers at scale using Hugging Face Transformers, Accelerate and bitsandbytes |
| 1082 | Week_63 | QuaRot: Outlier-Free 4-bit Inference in Rotated LLMs | QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs |
| 1135 | Week_67 | Tulu 2: Instruction-Tuned Language Models | Camels in a Changing Climate: Enhancing LM Adaptation with Tulu 2 |
| 1136 | Week_67 | LIMA: Less is More for Alignment | LIMA: Less Is More for Alignment |
| 1138 | Week_67 | BIRD: A Big Bench for Large-Scale Database Grounded Text-to-SQL Evaluation | Can LLM Already Serve as A Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs |
| 1166 | Week_68 | Chinchilla Scaling Laws for Neural Language Models | Training Compute-Optimal Large Language Models |
| 1168 | Week_69 | BIRD: A Big Bench for Large-Scale Database Grounded Text-to-SQL Evaluation | Can LLM Already Serve as A Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs |
| 1169 | Week_69 | Spider: A Large-Scale Human-Labeled Dataset | Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task |
| 1171 | Week_69 | Beyond the Imitation Game: Quantifying and Extrapolating the Capabilities of Language Models (BIG-Bench) | Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models |
| 1174 | Week_69 | The Hitchhiker's Guide to Testing Statistical Significance in NLP (Dror et al. 2018) | Appendix - Recommended Statistical Significance Tests for NLP Tasks |
| 1179 | Week_69 | HELM: Holistic Evaluation of Language Models | Holistic Evaluation of Language Models |
| 1180 | Week_69 | Chatbot Arena: Judging LLMs as a Judge | Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference |
| 1181 | Week_7 | HuggingFace Transformers: State-of-the-Art Natural Language Processing | HuggingFace's Transformers: State-of-the-art Natural Language Processing |
| 1205 | Week_70 | Llama 2 Technical Report | Llama 2: Open Foundation and Fine-Tuned Chat Models |
| 1223 | Week_71 | SmolLM2: When Smol Goes Big (Efficient Language Models) | SmolLM2: When Smol Goes Big -- Data-Centric Training of a Small Language Model |
| 1224 | Week_71 | OLMo 2: The Second Generation of Truly Open Language Models | 2 OLMo 2 Furious |
| 1225 | Week_71 | Tulu 2: Instruction-Tuned Language Models | Camels in a Changing Climate: Enhancing LM Adaptation with Tulu 2 |
| 1226 | Week_71 | FineWeb: Decanting the Web for the Finest Text Data at Scale | The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale |
| 1234 | Week_71 | DataComp for Language Models (DCLM) | DataComp-LM: In search of the next generation of training sets for language models |
| 1235 | Week_71 | CodeGeeX: A Pre-Trained Model for Code Generation with Multilingual Benchmarking (Zheng et al. 2023) | CodeGeeX: A Pre-Trained Model for Code Generation with Multilingual Benchmarking on HumanEval-X |
| 1249 | Week_72 | Unified Scaling Laws for Routed Language Models (Clark et al. 2022) | Unified Scaling Laws for Routed Language Models |
| 1252 | Week_73 | Scaling and Evaluating Sparse Autoencoders | Scaling and evaluating sparse autoencoders |
| 1253 | Week_73 | Interpretability in the Wild: A Circuit for Indirect Object Identification | Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small |
| 1269 | Week_73 | Interpretability in the Wild: a Circuit for Indirect Object Identification (Wang et al.) | Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small |
| 1279 | Week_74 | Extending Context Length with YaRN (Nous Research blog) | YaRN: Efficient Context Window Extension of Large Language Models |
| 1280 | Week_74 | Lost in the Middle: How Language Models Use Long Contexts (Liu et al. 2023) | Lost in the Middle: How Language Models Use Long Contexts |
| 1285 | Week_74 | Towards Complex Text-to-SQL in Cross-Domain Database with Intermediate Representation (IRNet, Guo et al. 2019) | Towards Complex Text-to-SQL in Cross-Domain Database with Intermediate Representation |
| 1289 | Week_75 | Llama 3 Herd of Models | The Llama 3 Herd of Models |
| 1302 | Week_75 | Scaling Laws for Neural Language Models (Kaplan et al., 2020) | Scaling Laws for Neural Language Models |
| 1303 | Week_75 | Are Emergent Abilities of Large Language Models a Mirage? (Schaeffer et al., 2023) | Are Emergent Abilities of Large Language Models a Mirage? |
| 1304 | Week_76 | ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2022) | ReAct: Synergizing Reasoning and Acting in Language Models |
| 1305 | Week_76 | DIN-SQL: Decomposed In-Context Learning of Text-to-SQL with Self-Correction (Pourreza & Rafiei, 2023) | DIN-SQL: Decomposed In-Context Learning of Text-to-SQL with Self-Correction |
| 1306 | Week_76 | DAIL-SQL: Efficient Prompt Engineering for Text-to-SQL (Gao et al., 2023) | Text-to-SQL Empowered by Large Language Models: A Benchmark Evaluation |
| 1307 | Week_76 | CHESS: Contextual Harnessing for Efficient SQL Synthesis (Talaei et al., 2024) | CHESS: Contextual Harnessing for Efficient SQL Synthesis |
| 1308 | Week_76 | Toolformer: Language Models Can Teach Themselves to Use Tools (Schick et al., 2023) | Toolformer: Language Models Can Teach Themselves to Use Tools |
| 1309 | Week_76 | FireActs: Toward Language Agent Fine-tuning (Chen et al., 2023) | FireAct: Toward Language Agent Fine-tuning |
| 1320 | Week_76 | Gorilla: Large Language Model Connected with Massive APIs (Patil et al., 2023) | Gorilla: Large Language Model Connected with Massive APIs |
| 1321 | Week_76 | AgentBench: Evaluating LLMs as Agents (Liu et al., 2023) | AgentBench: Evaluating LLMs as Agents |
| 1323 | Week_77 | IndicTrans2: Towards High-Quality and Accessible Machine Translation for All 22 Scheduled Indian Languages (Gala et al., 2023) | IndicTrans2: Towards High-Quality and Accessible Machine Translation Models for all 22 Scheduled Indian Languages |
| 1324 | Week_77 | Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model (Üstün et al., 2024) | Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model |
| 1325 | Week_77 | RAT-SQL: Relation-Aware Schema Encoding and Linking for Text-to-SQL Parsers (Wang et al.) | RAT-SQL: Relation-Aware Schema Encoding and Linking for Text-to-SQL Parsers |
| 1326 | Week_77 | CodeGeeX: A Pre-Trained Model for Code Generation with Multilingual Benchmarking (Zheng et al. 2023) | CodeGeeX: A Pre-Trained Model for Code Generation with Multilingual Benchmarking on HumanEval-X |
| 1340 | Week_78 | What Will It Take to Fix Benchmarking in NLP? | What Will it Take to Fix Benchmarking in Natural Language Understanding? |
| 1341 | Week_78 | Utility is in the Eye of the User: A Critique of NLP Leaderboards (Ethayarajh & Jurafsky) | Utility is in the Eye of the User: A Critique of NLP Leaderboards |
| 388 | Week_27 | Visualizing a Neural Machine Translation Model | Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention) |
| 1382 | Week_9 | Visualizing A Neural Machine Translation Model | Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention) |
| 1388 | Week_9 | The Illustrated BERT, ELMo, and co. | The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer Learning) |
| 610 | Week_4 | An Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling | Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling |
| 436 | Week_30 | Parameter-Efficient Fine-Tuning Methods Survey | Parameter-Efficient Fine-Tuning for Large Models: A Comprehensive Survey |
| 448 | Week_31 | Parameter-Efficient Fine-Tuning Methods Survey | Parameter-Efficient Fine-Tuning for Large Models: A Comprehensive Survey |
| 520 | Week_36 | Parameter-Efficient Fine-Tuning Methods Survey | Parameter-Efficient Fine-Tuning for Large Models: A Comprehensive Survey |
| 616 | Week_40 | PEFT: Parameter-Efficient Fine-Tuning of Large Language Models Survey | Parameter-Efficient Fine-Tuning for Large Models: A Comprehensive Survey |
| 458 | Week_31 | LoRA Land: Fine-Tuned Open-Source LLMs that Outperform GPT-4 | LoRA Land: 310 Fine-tuned LLMs that Rival GPT-4, A Technical Report |
| 564 | Week_38 | LoRA Land: Fine-Tuned Open-Source LLMs that Outperform GPT-4 | LoRA Land: 310 Fine-tuned LLMs that Rival GPT-4, A Technical Report |
| 391 | Week_28 | Llama 3 Technical Report | The Llama 3 Herd of Models |
| 395 | Week_28 | Llama 3 technical report | The Llama 3 Herd of Models |
| 474 | Week_32 | SqueezeLLM: Sparse-Quantization for Efficient LLM Inference | SqueezeLLM: Dense-and-Sparse Quantization |
| 488 | Week_33 | NVIDIA's Flash Attention 2 | FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning |
| 644 | Week_41 | Asynchronous Methods for Deep Reinforcement Learning (A3C) | Asynchronous Methods for Deep Reinforcement Learning |
| 645 | Week_41 | High-Dimensional Continuous Control Using Generalized Advantage Estimation (GAE) | High-Dimensional Continuous Control Using Generalized Advantage Estimation |

### Sub-pattern D2: Author/year citation used instead of paper title

| ID | Week | Stored text | Actual title |
|----|------|-------------|--------------|
| 1166 | Week_68 | Chinchilla Scaling Laws for Neural Language Models | Training Compute-Optimal Large Language Models |
| 1046 | Week_61 | WikiSQL: A Large Crowd-Sourced Dataset... | Seq2SQL: Generating Structured Queries from Natural Language using Reinforcement Learning |

### Sub-pattern D3: "DAIL-SQL" title mismatch (appears 3×)

| ID | Week | Stored text | Actual title |
|----|------|-------------|--------------|
| 593 | Week_39 | DAIL-SQL: Efficient Prompt Engineering for Text-to-SQL | Text-to-SQL Empowered by Large Language Models: A Benchmark Evaluation |
| 1032 | Week_61 | DAIL-SQL: Efficient Prompt Engineering for Text-to-SQL | Text-to-SQL Empowered by Large Language Models: A Benchmark Evaluation |
| 1306 | Week_76 | DAIL-SQL: Efficient Prompt Engineering for Text-to-SQL (Gao et al., 2023) | Text-to-SQL Empowered by Large Language Models: A Benchmark Evaluation |

> The stored text uses the model/system name "DAIL-SQL" as the title, but the paper's actual arXiv title is "Text-to-SQL Empowered by Large Language Models: A Benchmark Evaluation".

---

## Category E — Informal/descriptive label for docs (~200 links)

Docs pages, GitHub READMEs, blog posts, and HuggingFace resources where the stored text is a description of the content rather than the page's actual H1 heading.

### E1: TRL documentation pages

All TRL docs pages have short H1s ("PPO Trainer", "DPO Trainer", etc.) but are referenced with longer descriptive labels.

| ID | Week | Stored text | Actual H1 |
|----|------|-------------|-----------|
| 656 | Week_42 | TRL PPOTrainer source | PPO Trainer |
| 665 | Week_42 | TRL PPOTrainer source | PPO Trainer |
| 667 | Week_42 | TRL PPOTrainer docs | PPO Trainer |
| 668 | Week_42 | TRL PPOConfig reference | PPO Trainer |
| 682 | Week_43 | TRL DPOTrainer docs | DPO Trainer |
| 705 | Week_45 | TRL DPOTrainer docs | DPO Trainer |
| 708 | Week_45 | DPO implementation tips from TRL developers | DPO Trainer |
| 718 | Week_45 | length normalization | DPO Trainer |
| 732 | Week_46 | TRL GRPOConfig documentation | GRPO Trainer |
| 751 | Week_48 | TRL GRPOConfig reference | GRPO Trainer |
| 774 | Week_49 | TRL KTOTrainer docs | KTO Trainer |
| 775 | Week_49 | TRL ORPOTrainer docs | ORPO Trainer |
| 776 | Week_49 | TRL SimPOTrainer docs | SimPO Trainer |
| 777 | Week_49 | TRL DPO variants overview | DPO Trainer |
| 856 | Week_52 | TRL PPOTrainer | PPO Trainer |
| 857 | Week_52 | TRL DPOTrainer | DPO Trainer |
| 858 | Week_52 | GRPO Trainer | GRPO Trainer (exact — verified) |
| 855 | Week_52 | TRL Trainer Overview | TRL - Transformers Reinforcement Learning |
| 985 | Week_59 | Hugging Face — DPO Trainer documentation | DPO Trainer |
| 989 | Week_59 | TRL DPOConfig | DPO Trainer |
| 1015 | Week_60 | Hugging Face GRPO Trainer documentation | GRPO Trainer |
| 1021 | Week_60 | GRPO docs | GRPO Trainer |
| 1022 | Week_60 | TRL GRPOConfig | GRPO Trainer |
| 1164 | Week_68 | TRL SFTTrainer | SFT Trainer |
| 1165 | Week_68 | TRL DPOTrainer | DPO Trainer |

### E2: HuggingFace Transformers / Datasets / Hub docs

| ID | Week | Stored text | Actual H1 |
|----|------|-------------|-----------|
| 1369 | Week_8 | HuggingFace Trainer documentation | Trainer |
| 1163 | Week_68 | HuggingFace Trainer — All Hyperparameters | Trainer |
| 1187 | Week_7 | HuggingFace LLM Course — Chapters 1–3 | Introduction |
| 1188 | Week_7 | HuggingFace `datasets` Tutorial | Overview |
| 1189 | Week_7 | transformers AutoModelForCausalLM docs | Auto Classes |
| 1190 | Week_7 | transformers generation docs | Generation strategies |
| 1191 | Week_7 | datasets library docs | Datasets |
| 1192 | Week_7 | huggingface_hub library docs | 🤗 Hub client library |
| 1193 | Week_7 | The HuggingFace Ecosystem Explained | Hugging Face Hub documentation |
| 1194 | Week_7 | Fine-tuning a Pretrained Model (HuggingFace Blog) | Fine-tuning |
| 994 | Week_6 | HuggingFace LLM Course — Chapter 6: The Tokenizers Library | Introduction |
| 995 | Week_6 | Summary of the Tokenizers | Tokenization algorithms |
| 1001 | Week_6 | HuggingFace tokenizers library docs | Tokenizers |
| 1002 | Week_6 | transformers AutoTokenizer docs | Tokenizer |
| 1074 | Week_63 | Hugging Face Quantization Concepts | Overview |
| 1079 | Week_63 | Hugging Face GPTQ Integration | GPTQ |
| 1080 | Week_63 | Hugging Face AWQ Integration | AWQ |
| 1090 | Week_64 | Hugging Face AWQ Blog Post | AWQ |
| 1096 | Week_64 | HuggingFace Hub — Upload Files | Upload files to the Hub |
| 1097 | Week_64 | HuggingFace Model Cards Guide | Model Cards |
| 1130 | Week_66 | HuggingFace Spaces — Gradio SDK | Gradio Spaces |
| 1283 | Week_74 | HuggingFace — rope_scaling configuration | Llama |
| 1293 | Week_75 | Hugging Face — Chat Templating Guide | Chat templates |
| 1312 | Week_76 | Hugging Face — Tool Use with Transformers | Chat templates |
| 1330 | Week_77 | Hugging Face — Multilingual Tokenization Guide | Components |
| 1335 | Week_77 | HuggingFace — Tokenizer Vocabulary Extension | Utilities for Tokenizers |
| 820 | Week_51 | Model Evaluation Best Practices | Evaluate on the Hub |
| 821 | Week_51 | How to Write a Good Model Card | Model Cards |
| 825 | Week_51 | HuggingFace Hub model versioning | The Model Hub |
| 859 | Week_52 | HuggingFace Model Card Guide | Model Cards |
| 879 | Week_53 | HuggingFace Datasets documentation | Datasets |
| 947 | Week_57 | Hugging Face — continued pretraining guide | Auto Classes |
| 1044 | Week_61 | HuggingFace evaluate library | Evaluate on the Hub |
| 1212 | Week_70 | HuggingFace Model Cards Documentation | Model Cards |
| 1344 | Week_78 | HuggingFace Model Card Guideline | Model Cards |

### E3: Weights & Biases docs

| ID | Week | Stored text | Actual H1 |
|----|------|-------------|-----------|
| 1372 | Week_8 | Weights & Biases Quickstart | W&B Quickstart |
| 1373 | Week_8 | W&B Reports | Reports overview |
| 810 | Week_50 | W&B reports for comparing runs | Reports overview |
| 826 | Week_51 | W&B Compare Runs | Compare run metrics |
| 1295 | Week_75 | Weights & Biases — Comparing Runs Across Experiments | Compare run metrics |
| 631 | Week_40 | Weights & Biases: Model Registry | Registry overview |
| 753 | Week_48 | W&B GRPO metrics | Hugging Face |
| 862 | Week_52 | W&B Tables for Model Evaluation | Tables overview |
| 990 | Week_59 | Weights & Biases — DPO metrics guide | Direct Preference Optimization Fine tuning Mistral 7B with DPO |
| 1158 | Week_68 | Weights & Biases — Logging Hyperparameters | Configure experiments |
| 968 | Week_58 | Weights & Biases — Debugging Neural Networks with PyTorch and W&B | Debugging Neural Networks with PyTorch and W&B Using Gradients and Visualizations |

### E4: GitHub file/README links

GitHub blob/file URLs and README fragment URLs whose H1 shows the repo slug (`owner/repo`), not the stored description.

| ID | Week | Stored text | Actual H1 (repo slug) |
|----|------|-------------|----------------------|
| 679 | Week_43 | philschmid's DPO notebook | rl-with-llms-in-2025-dpo.ipynb |
| 680 | Week_43 | TRL DPOTrainer source | dpo_trainer.py |
| 681 | Week_43 | Unsloth DPO examples | README.md |
| 704 | Week_45 | Unsloth DPO training guide | README.md |
| 706 | Week_45 | Unsloth Colab notebooks | unslothai/unsloth |
| 710 | Week_45 | TRL source — DPOTrainer.compute_loss() | dpo_trainer.py |
| 719 | Week_46 | TRL's GRPOTrainer | grpo_trainer.py |
| 728 | Week_46 | TRL GRPOTrainer source | grpo_trainer.py |
| 729 | Week_46 | TRL GRPO examples | huggingface/trl |
| 730 | Week_46 | Unsloth GRPO | unslothai/unsloth |
| 731 | Week_46 | Open-Reasoner-Zero | Open-Reasoner-Zero/Open-Reasoner-Zero |
| 740 | Week_47 | TRL GRPOTrainer reward_fn interface | grpo_trainer.py |
| 741 | Week_47 | Unsloth GRPO examples | unslothai/unsloth |
| 742 | Week_47 | Open-R1 SQL examples | huggingface/open-r1 |
| 750 | Week_48 | Unsloth GRPO training guide | unslothai/unsloth |
| 758 | Week_48 | Open-R1 | huggingface/open-r1 |
| 759 | Week_48 | Unsloth notebooks | unslothai/unsloth |
| 760 | Week_48 | Open-Reasoner | Open-Reasoner-Zero/Open-Reasoner-Zero |
| 783 | Week_49 | Alignment handbook | huggingface/alignment-handbook |
| 808 | Week_50 | Open-R1 training scripts | huggingface/open-r1 |
| 809 | Week_50 | TRL GRPO examples | huggingface/trl |
| 892 | Week_54 | OpenAI API cookbook — async parallelism | openai/openai-cookbook |
| 895 | Week_54 | Magpie official code | magpie-align/magpie |
| 899 | Week_54 | OpenAI Python async client | openai/openai-python |
| 1018 | Week_60 | huggingface/trl — GRPOTrainer | huggingface/trl |
| 1020 | Week_60 | TRL GRPOTrainer source | huggingface/trl |
| 1023 | Week_60 | Unsloth — saving and merging guide | unslothai/unsloth |
| 1064 | Week_62 | EleutherAI LM Evaluation Harness | EleutherAI/lm-evaluation-harness |
| 1081 | Week_63 | llama.cpp GGUF quantization types reference | ggml-org/llama.cpp |
| 1083 | Week_63 | GGUF format specification | ggerganov/ggml |
| 1084 | Week_63 | Exllamav2 | turboderp/exllamav2 |
| 1098 | Week_64 | llama.cpp quantization types reference | ggml-org/llama.cpp |
| 1099 | Week_64 | ExLlamaV2 | turboderp/exllamav2 |
| 1100 | Week_64 | imatrix quantization (llama.cpp) | ggml-org/llama.cpp |
| 1104 | Week_65 | llama.cpp: Efficient LLM Inference in C/C++ (project README) | ggml-org/llama.cpp |
| 1113 | Week_65 | llama.cpp GBNF Grammar Guide | ggml-org/llama.cpp |
| 1114 | Week_65 | Ollama REST API Reference | ollama/ollama |
| 1116 | Week_65 | llamafile — Single-file distributable LLMs (Mozilla) | Mozilla-Ocho/llamafile |
| 1118 | Week_65 | Ollama JavaScript and Python Libraries | ollama/ollama-python |
| 1131 | Week_66 | Text Generation Inference (TGI) — HuggingFace | huggingface/text-generation-inference |
| 1132 | Week_66 | LiteLLM | BerriAI/litellm |
| 1144 | Week_67 | paperswithcode/paperswithcode-data | paperswithcode/paperswithcode |
| 1176 | Week_69 | spider/evaluation.py | taoyds/spider |
| 1177 | Week_69 | DAIL-SQL evaluation | BeachWang/DAIL-SQL |
| 1230 | Week_71 | The Alignment Handbook (HuggingFace) | huggingface/alignment-handbook |
| 1247 | Week_72 | unslothai/unsloth — supported models | unslothai/unsloth |
| 1268 | Week_73 | circuitsvis | TransformerLensOrg/CircuitsVis |
| 1294 | Week_75 | EleutherAI — Eval Harness Model Comparison Guide | EleutherAI/lm-evaluation-harness |
| 1297 | Week_75 | ggml-org/llama.cpp — Model Support Matrix | ggml-org/llama.cpp |
| 1315 | Week_76 | langchain-ai/langchain — SQL Agent | langchain-ai/langchain |
| 1316 | Week_76 | huggingface/trl — SFTTrainer | huggingface/trl |
| 1371 | Week_8 | nanoGPT README | karpathy/nanoGPT |
| 1374 | Week_8 | nanoGPT README — baby GPT config | karpathy/nanoGPT |
| 1375 | Week_8 | karpathy/nanoGPT/model.py | karpathy/nanoGPT |

### E5: Miscellaneous external docs and blogs

| ID | Week | Stored text | Actual H1 |
|----|------|-------------|-----------|
| 253 | Week_21 | Gradient Descent Variants | An overview of gradient descent optimization algorithms |
| 306 | Week_24 | Nathan Lambert's post on open LLM training recipes | We aren't running out of training data, we are running out of open training data |
| 441 | Week_30 | LoRA from scratch — Explained | Parameter-Efficient LLM Finetuning With Low-Rank Adaptation (LoRA) |
| 453 | Week_31 | LoRA insights from 1000+ experiments | Finetuning LLMs with LoRA and QLoRA: Insights from Hundreds of Experiments |
| 501 | Week_34 | Unsloth blog post on 80% memory reduction | Finetune Mistral 14x faster (Unsloth benchmark blog) |
| 509 | Week_35 | Hyperparameter Search for LoRA | Hyperparameter Search with Transformers and Ray Tune |
| 518 | Week_36 | RSLoRA: A Rank Stabilization Scaling Factor... | A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA |
| 524 | Week_36 | DoRA: A New Fine-tuning Method that Performs Better than LoRA | Improving LoRA: Implementing Weight-Decomposed Low-Rank Adaptation (DoRA) from Scratch |
| 548 | Week_37 | Synthetic Data Generation for SQL: Best Practices | SQLCoder 2: technical insights for fine-tuning and dataset generation |
| 565 | Week_38 | How to Fine-Tune LLMs with Unsloth: Step-by-Step | Fine-tuning LLMs Guide |
| 578 | Week_38 | Text-to-SQL Benchmarking | A Survey of Text-to-SQL in the Era of LLMs: Where are we, and where are we going? |
| 585 | Week_39 | How We Evaluate Text-to-SQL Models at Defog | Defog Data (blog index) |
| 617 | Week_40 | GRPO: DeepSeekMath — Pushing the Limits... | DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models |
| 618 | Week_40 | DPO: Direct Preference Optimization: Your Language Model is Secretly a Reward Model | Direct Preference Optimization: Your Language Model is Secretly a Reward Model |
| 623 | Week_40 | Sebastian Raschka: Practical Tips for Fine-Tuning LLMs Using LoRA | Practical Tips for Finetuning LLMs Using LoRA (Low-Rank Adaptation) |
| 624 | Week_40 | Lilian Weng: The Transformer Family Version 2.0 | The Transformer Family Version 2.0 |
| 633 | Week_41 | Spinning Up in Deep RL — Introduction to RL | Part 1: Key Concepts in RL |
| 647 | Week_41 | RL from Human Feedback: From Zero to chatGPT | RLHF: Reinforcement Learning from Human Feedback |
| 651 | Week_41 | Gymnasium CartPole-v1 | Cart Pole |
| 653 | Week_41 | Spinning Up PPO docs | Proximal Policy Optimization |
| 663 | Week_42 | Spinning Up — PPO documentation | Proximal Policy Optimization |
| 664 | Week_42 | RLHF: From Zero to ChatGPT | RLHF: Reinforcement Learning from Human Feedback |
| 677 | Week_43 | RLHF vs DPO vs GRPO: An Overview | Preference Tuning LLMs with Direct Preference Optimization Methods |
| 678 | Week_43 | Understanding DPO | Direct Preference Optimization (DPO) |
| 689 | Week_44 | How to Build a Preference Dataset for DPO | Preference Tuning LLMs with Direct Preference Optimization Methods |
| 690 | Week_44 | Generating Synthetic Data for DPO Fine-Tuning | Synthetic data: save money, time and carbon with open source |
| 695 | Week_44 | psycopg2 documentation | Psycopg 2.9.12 documentation |
| 696 | Week_44 | PostgreSQL statement_timeout | 19.11. Client Connection Defaults |
| 697 | Week_44 | HuggingFace datasets.push_to_hub | Share a dataset to the Hub |
| 726 | Week_46 | GRPO: A Simple Alternative to PPO | GRPO: Building Intuition Through Ablation Studies |
| 727 | Week_46 | Understanding DeepSeek-R1 and its Surprising Emergent Reasoning | Crafting a good (reasoning) model |
| 738 | Week_47 | Reward Hacking in RLHF: How to Detect and Prevent It | Reward Hacking in Reinforcement Learning |
| 739 | Week_47 | Designing Reward Functions for Code Generation | SWE-bench |
| 743 | Week_47 | psycopg2 error codes | psycopg2.errors – Exception classes mapping PostgreSQL errors |
| 744 | Week_47 | PostgreSQL statement_timeout | 19.11. Client Connection Defaults |
| 752 | Week_48 | RunPod documentation | Welcome to Runpod |
| 754 | Week_48 | Training DeepSeek-R1 yourself: practical guide | Open-R1: a fully open reproduction of DeepSeek-R1 |
| 755 | Week_48 | GRPO in Practice: What Works and What Doesn't | Recent reasoning research: GRPO tweaks, base model RL, and data curation |
| 765 | Week_48 | The Arithmetic of Compute for RLVR | Group Relative Policy Optimization (GRPO) |
| 772 | Week_49 | A Survey of Preference-Based Reinforcement Learning Methods | Preference Tuning LLMs with Direct Preference Optimization Methods |
| 773 | Week_49 | DPO vs KTO vs SimPO: Practical Comparison | ORPO: Preference Optimization without the Supervised Fine-tuning (SFT) Step |
| 804 | Week_50 | Debugging Machine Learning Models | Machine Learning Crash Course |
| 805 | Week_50 | The Bitter Lesson: Scaling vs. Algorithmic Improvements | The Bitter Lesson |
| 814 | Week_50 | Reward Hacking in Reinforcement Learning | Faulty reward functions in the wild |
| 815 | Week_50 | SQLGlot | tobymao/sqlglot |
| 816 | Week_51 | Utility is in the Eye of the User: A Critique of NLP Leaderboards (Ethayarajh & Jurafsky) | Utility is in the Eye of the User: A Critique of NLP Leaderboards |
| 827 | Week_51 | The ML Test Score: A Rubric for Production Readiness (Breck et al. 2017) | The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction |
| 828 | Week_51 | Spider 2.0 Leaderboard | Spider 2.0 |
| 845 | Week_52 | HuggingFace — Illustrating Reinforcement Learning from Human Feedback (RLHF) | Illustrating Reinforcement Learning from Human Feedback (RLHF) |
| 846 | Week_52 | HuggingFace — Fine-tune Mistral-7b with DPO | Fine-tune Llama 2 with DPO |
| 847 | Week_52 | HuggingFace — ORPO: An Efficient LLM Alignment Approach | Fine-tune Llama 3 with ORPO |
| 848 | Week_52 | HuggingFace — Open-R1: A Fully Open Reproduction of DeepSeek-R1 | Open-R1: a fully open reproduction of DeepSeek-R1 |
| 872 | Week_53 | Tulu 3 blog post (AllenAI) | Tülu 3 opens language model post-training up to more tasks and more people |
| 874 | Week_53 | Lilian Weng — Prompt Engineering | Prompt Engineering |
| 876 | Week_53 | datasketch | ekzhu/datasketch |
| 877 | Week_53 | sqlglot | tobymao/sqlglot |
| 878 | Week_53 | HuggingFace datasets deduplication | huggingface/datatrove |
| 880 | Week_53 | PostgreSQL SQL syntax reference | Part II. The SQL Language |
| 881 | Week_53 | TimescaleDB documentation — hyperfunctions | Hyperfunctions overview |
| 884 | Week_53 | RedPajama-Data-v2 | togethercomputer/RedPajama-Data |
| 893 | Week_54 | Argilla blog — synthetic data for NLP | Synthetic data for LLM fine-tuning and alignment |
| 894 | Week_54 | Hamel Husain — Generating SQL training data | dbt |
| 900 | Week_54 | Anthropic API rate limits | Rate limits |
| 901 | Week_54 | psycopg2 documentation | Psycopg – PostgreSQL database adapter for Python |
| 902 | Week_54 | TimescaleDB SQL reference | TimescaleDB API reference |
| 905 | Week_54 | sqlglot documentation | SQLGlot |
| 909 | Week_55 | DEITA: What Makes Good Data for Alignment? | What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning |
| 913 | Week_55 | Hamel Husain — Your AI product needs evals | Your AI Product Needs Evals |
| 914 | Week_55 | Eugene Yan — Patterns for LLM evaluation | Patterns for Building LLM-based Systems & Products |
| 916 | Week_55 | fastchat-lmsys/FastChat | lm-sys/FastChat |
| 919 | Week_55 | sklearn.metrics.cohen_kappa_score | cohen_kappa_score |
| 920 | Week_55 | OpenAI API — structured outputs | Structured model outputs |
| 928 | Week_56 | Hugging Face — Training on conversational data with TRL | SFT Trainer |
| 929 | Week_56 | Lilian Weng — Neural machine translation and sequence-to-sequence | Attention? Attention! |
| 930 | Week_56 | Defog blog — multi-turn SQL models | Defog Data |
| 936 | Week_56 | TRL SFTTrainer — dataset_text_field and dataset_kwargs | SFT Trainer |
| 937 | Week_56 | Qwen2.5 chat template | Qwen2.5-Coder-7B-Instruct |
| 938 | Week_56 | TimescaleDB time_bucket_gapfill | time_bucket_gapfill() |
| 948 | Week_57 | Tim Dettmers — GPU memory guide | Which GPU(s) to Get for Deep Learning: My Experience and Advice for Using GPUs in Deep Learning |
| 949 | Week_57 | RunPod — Getting started guide | Deploy your first Pod |
| 953 | Week_57 | PostgreSQL documentation download | Documentation |
| 954 | Week_57 | TimescaleDB documentation source | timescale/docs |
| 956 | Week_57 | Unsloth CPT documentation | unslothai/unsloth |
| 967 | Week_58 | Hugging Face — SFT Trainer docs | SFT Trainer |
| 969 | Week_58 | Unsloth blog — SFT fine-tuning guide | 🧬Fine-tuning LLMs Guide |
| 973 | Week_58 | TRL SFTTrainer | SFT Trainer |
| 974 | Week_58 | Qwen2.5 model card | Qwen2.5-Coder-7B-Instruct |
| 975 | Week_58 | RunPod documentation — PyTorch templates | Overview |
| 984 | Week_59 | Philschmid — DPO practical guide | RLHF in 2024 with DPO & Hugging Face |
| 986 | Week_59 | NousResearch — DPO tips and tricks | Preference Tuning LLMs with Direct Preference Optimization Methods |
| 987 | Week_59 | huggingface/trl — DPOTrainer | huggingface/trl |
| 1006 | Week_6 | Tokenization Tutorial — Stanford CS224N | CS224N: Natural Language Processing with Deep Learning |
| 1007 | Week_6 | What Is ChatGPT Doing and Why Does It Work? | What Is ChatGPT Doing … and Why Does It Work? |
| 1025 | Week_60 | REINFORCE algorithm tutorial | Reinforcement Learning (DQN) Tutorial |
| 1027 | Week_61 | bird-bench.github.io | BIRD-SQL |
| 1028 | Week_61 | yale-lily/spider on GitHub | taoyds/spider |
| 1035 | Week_61 | BIRD-SQL leaderboard | BIRD-SQL |
| 1036 | Week_61 | Defog SQLCoder model card and benchmark | Open-sourcing SQLCoder2-15b and SQLCoder-7b |
| 1042 | Week_61 | BIRD-SQL official download | BIRD-SQL |
| 1043 | Week_61 | Spider evaluation script | taoyds/spider |
| 1045 | Week_61 | Spider 2.0 PostgreSQL subset | Spider 2.0 |
| 1047 | Week_61 | bird-bench.github.io | BIRD-SQL |
| 1048 | Week_61 | yale-lily/spider on GitHub | taoyds/spider |
| 1051 | Week_62 | SQLCoder: Defog's SQL-Specialized LLM | Open-sourcing SQLCoder2-15b and SQLCoder-7b |
| 1053 | Week_62 | mlxtend — McNemar's test for paired model evaluation | McNemar's test for classifier comparisons |
| 1054 | Week_62 | Defog SQL benchmark leaderboard | defog-ai/sql-eval |
| 1058 | Week_62 | OpenAI API pricing | API Pricing |
| 1059 | Week_62 | Anthropic API pricing | Build on the Claude Platform |
| 1060 | Week_62 | Together AI model catalog | Build on the AI Native Cloud |
| 1062 | Week_62 | statsmodels.stats.contingency_tables.mcnemar documentation | statsmodels.stats.contingency_tables.mcnemar¶ |
| 1072 | Week_63 | The Illustrated Quantization | A Visual Guide to Quantization |
| 1073 | Week_63 | Unsloth GGUF Export Guide | Saving to GGUF |
| 1089 | Week_64 | Hugging Face GPTQ Integration Guide | Making LLMs lighter with AutoGPTQ and transformers |
| 1091 | Week_64 | The Illustrated Guide to LLM Quantization (Maarten Grootendorst) | A Visual Guide to Quantization |
| 1101 | Week_64 | GGUF model card template (community) | Tom Jobbins |
| 1107 | Week_65 | Ollama Modelfile Reference | Modelfile Reference |
| 1108 | Week_65 | Ollama OpenAI Compatibility | OpenAI compatibility |
| 1109 | Week_65 | How to Run LLMs on Apple Silicon with llama.cpp (Simon Willison) | llamafile is the new best way to run an LLM on your own computer |
| 1115 | Week_65 | PyInstaller Quickstart | Using PyInstaller |
| 1117 | Week_65 | LM Studio | Run AI models, locally and privately. |
| 1123 | Week_66 | FastAPI Official Tutorial | Tutorial - User Guide |
| 1124 | Week_66 | RunPod Documentation — Expose Ports | Expose ports |
| 1128 | Week_66 | vLLM OpenAI-Compatible Server | OpenAI-Compatible Server |
| 1129 | Week_66 | vLLM Supported Models | Supported Models |
| 1133 | Week_66 | CloudFlare Tunnel | Cloudflare Tunnel |
| 1142 | Week_67 | How to Write the Methods Section of a Machine Learning Paper | Write the Paper First |
| 1147 | Week_67 | Overleaf | Write like a rocket scientist with Overleaf... |
| 1148 | Week_67 | Markdown to PDF via Pandoc | Pandoc User's Guide |
| 1159 | Week_68 | ML Reproducibility Checklist (NeurIPS 2021) | NeurIPS 2021 Paper Checklist Guidelines |
| 1167 | Week_68 | The Illustrated Transformer Training (Jay Alammar) | The Illustrated Transformer |
| 1170 | Week_69 | Defog SQL Eval | defog-ai/sql-eval |
| 1173 | Week_69 | Thinking about High-Quality Human Data (Lilian Weng) | Thinking about High-Quality Human Data |
| 1178 | Week_69 | SciPy stats.binom_test | binomtest |
| 1195 | Week_7 | How to Generate Text: Different Decoding Methods | How to generate text: using different decoding methods for language generation with Transformers |
| 1210 | Week_70 | How to Write a Good Scientific Paper: Conclusion and Future Work (SPIE) | How to Write a Good Scientific Paper |
| 1211 | Week_70 | arXiv Submission Guide for Authors | Submission Guidelines |
| 1217 | Week_70 | Overleaf NeurIPS 2024 Template | Formatting Instructions For NeurIPS 2026 |
| 1218 | Week_70 | Grammarly | You think big. We'll take care of the details. |
| 1219 | Week_70 | Papers With Code — Submit Your Paper | Daily Papers |
| 1220 | Week_70 | HuggingFace Daily Papers Submission | Daily Papers |
| 1221 | Week_70 | Research Debt (Chris Olah and Shan Carter) | Research Debt |
| 1229 | Week_71 | SmolLM2 HuggingFace Blog | SmolLM - blazingly fast and remarkably powerful |
| 1243 | Week_72 | DeepSeek-R1 Zero: Reasoning Through RL Without SFT (DeepSeek blog) | DeepSeek-R1 Release |
| 1244 | Week_72 | Understanding DeepSeek's MoE Architecture (Sebastian Raschka) | From DeepSeek V3 to V3.2: Architecture, Sparse Attention, and RL Updates |
| 1259 | Week_73 | Toy Models of Superposition (Anthropic Transformer Circuits Thread) | Toy Models of Superposition |
| 1260 | Week_73 | Understanding Anthropic's Interpretability Research (Jack Clark) | Import AI 340: Drone VS human... |
| 1267 | Week_73 | TransformerLens Documentation | TransformerLens |
| 1313 | Week_76 | Simon Willison — The ReAct Pattern for LLM Tools | Simon Willison's Weblog |
| 1314 | Week_76 | Modal — Running PostgreSQL in a Serverless Function | Write to Google Sheets from Postgres |
| 1318 | Week_76 | psycopg2 — Exception Classes | psycopg2.errors – Exception classes mapping PostgreSQL errors |
| 1319 | Week_76 | OpenAI — Function Calling Guide | Function calling |
| 1329 | Week_77 | AI4Bharat — Samanantar: The Largest Publicly Available Parallel Corpus for Indic Languages | Dataset Card for Samanantar |
| 1331 | Week_77 | Sebastian Ruder — A Survey of Cross-Lingual Embedding Methods | A survey of cross-lingual word embedding models |
| 1337 | Week_77 | OPUS — Open Parallel Corpus (Tamil-English) | Find your corpora |
| 1338 | Week_77 | Dravidian-CodeMix NLP at FIRE 2021 | Dravidian-CodeMix - FIRE 2021 |
| 1346 | Week_78 | Eugene Yan — Practical MLOps for the Individual Practitioner | A Practical Guide to Maintaining Machine Learning in Production |
| 1347 | Week_78 | Andrej Karpathy — The Unreasonable Effectiveness of Data | (empty — page title not extracted) |
| 1348 | Week_78 | Sebastian Ruder — Tracking Progress in NLP | Tracking the Progress in Natural Language Processing |
| 1349 | Week_78 | Swyx — The AI Engineer (latent.space) | The Rise of the AI Engineer |
| 1354 | Week_78 | HuggingFace Papers | Daily Papers |
| 1356 | Week_78 | Chip Huyen — Designing Machine Learning Systems (O'Reilly, 2022) | Designing Machine Learning Systems |
| 1387 | Week_9 | Effective Attention Visualization with matplotlib | Annotated heatmap |

---

## Category F — Capitalisation / punctuation only (~25 links)

Minor differences where the page heading is identical except for capitalisation, punctuation, or ellipsis style.

| ID | Week | Stored text | Actual H1 | Difference |
|----|------|-------------|-----------|------------|
| 830 | Week_52 | Learning to Summarize from Human Feedback | Learning to summarize from human feedback | Case |
| 944 | Week_57 | StarCoder: May the Source Be with You! | StarCoder: may the source be with you! | Case |
| 1252 | Week_73 | Scaling and Evaluating Sparse Autoencoders | Scaling and evaluating sparse autoencoders | Case |
| 1007 | Week_6 | What Is ChatGPT Doing and Why Does It Work? | What Is ChatGPT Doing … and Why Does It Work? | Ellipsis style |
| 1181 | Week_7 | HuggingFace Transformers: State-of-the-Art NLP | HuggingFace's Transformers: State-of-the-art Natural Language Processing | Apostrophe + case |
| 1136 | Week_67 | LIMA: Less is More for Alignment | LIMA: Less Is More for Alignment | Case |
| 1253 | Week_73 | Interpretability in the Wild: A Circuit for IOI | Interpretability in the Wild: a Circuit for IOI in GPT-2 small | Case + subtitle |
| 146 | Week_17 | Scaling Laws for Fine-Tuning | Scaling Laws for Transfer | Different subtitle |

---

## Category G — Model card name mismatch (~15 links)

HuggingFace model cards where the stored text uses an unofficial/abbreviated name but the actual model card H1 is different.

| ID | Week | Stored text | Actual H1 |
|----|------|-------------|-----------|
| 316 | Week_24 | meta-llama/Llama-3-8B | meta-llama/Meta-Llama-3-8B |
| 317 | Week_24 | deepseek-ai/DeepSeek-Coder-V2-Lite | deepseek-ai/DeepSeek-Coder-V2-Lite-Base |
| 683 | Week_43 | HuggingFaceH4/ultrafeedback_binarized | Dataset Card for UltraFeedback Binarized |
| 693 | Week_44 | HuggingFaceH4/ultrafeedback_binarized | Dataset Card for UltraFeedback Binarized |
| 952 | Week_57 | bigcode/the-stack-v2-train | bigcode/the-stack-v2-train-full-ids |
| 1061 | Week_62 | DeepSeek-Coder-V2-Lite model card | deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct |
| 1101 | Week_64 | GGUF model card template (community) | Tom Jobbins |
| 1196 | Week_7 | distilgpt2 | DistilGPT2 |
| 1197 | Week_7 | Qwen/Qwen2.5-Coder-7B | Qwen2.5-Coder-7B |
| 1198 | Week_7 | Qwen/Qwen2.5-Coder-1.5B | Qwen2.5-Coder-1.5B |
| 1199 | Week_7 | Spider on HuggingFace Hub | Spider Tea |
| 1299 | Week_75 | Llama 3.1 Model Card — Meta AI | meta-llama/Llama-3.1-8B-Instruct |
| 1300 | Week_75 | Gemma 2 9B Instruct — Google DeepMind | Gemma 2 model card |
| 1301 | Week_75 | DeepSeek-R1-Distill-Qwen-7B Model Card | DeepSeek-R1 |

---

## Full per-week mismatch table

| Week | Mismatches | Primary pattern |
|------|:----------:|-----------------|
| Week_1 | 2 | Medium/TDS redirect (Cat C) |
| Week_2 | 2 | yale-lily/spider repo moved (Cat B) |
| Week_3 | 0 | — |
| Week_4 | 1 | Paper capitalisation (Cat F) |
| Week_5 | 9 | PyTorch docs informal labels (Cat E) |
| Week_6 | 12 | HF tokenizer docs informal labels (Cat E) |
| Week_7 | 16 | HF ecosystem docs informal labels (Cat E) |
| Week_8 | 7 | nanoGPT/GitHub fragment links + HF docs (Cat E, B) |
| Week_9 | 3 | Illustrated posts subtitle dropped (Cat D) |
| Week_10 | 0 | — |
| Week_11 | 1 | Blog moved to new domain (Cat C) |
| Week_12 | 1 | Wrong URL (Cat C) |
| Week_13 | 0 | — |
| Week_14 | 1 | Llama 2 title informal (Cat E) |
| Week_15 | 1 | Karpathy recipe page (Cat D) |
| Week_16 | 0 | — |
| Week_17 | 3 | Scaling laws blog posts informal (Cat C, D) |
| Week_18 | 2 | Repo transferred + informal (Cat B, E) |
| Week_19 | 4 | Org renamed + HF docs informal (Cat B, E) |
| Week_20 | 1 | RoPE paper title informal (Cat D) |
| Week_21 | 4 | Blog/docs informal descriptions (Cat C, E) |
| Week_22 | 2 | Repo file renamed + OpenAI blog (Cat B, C) |
| Week_23 | 5 | Spider repo + BIRD title + LLM eval survey (Cat B, D, E) |
| Week_24 | 5 | Model card names + doc informal (Cat G, E) |
| Week_25 | 4 | BIRD title + WizardLM abbreviated (Cat D, E) |
| Week_26 | 1 | WizardLM abbreviated (Cat D) |
| Week_27 | 2 | Blog descriptions informal (Cat E) |
| Week_28 | 2 | Llama 3 report abbreviated (Cat D) |
| Week_29 | 0 | — |
| Week_30 | 3 | PEFT survey + LoRA blog (Cat D, E) |
| Week_31 | 3 | PEFT survey + LoRA blog (Cat D, E) |
| Week_32 | 1 | SqueezeLLM subtitle (Cat D) |
| Week_33 | 1 | Flash Attention blog informal (Cat E) |
| Week_34 | 1 | Unsloth blog informal (Cat E) |
| Week_35 | 1 | HF Ray Tune blog informal (Cat E) |
| Week_36 | 3 | RSLoRA + DoRA + PEFT survey (Cat D, E) |
| Week_37 | 2 | Defog blog + TimescaleDB rebranded (Cat C, E) |
| Week_38 | 3 | LoRA Land + Unsloth + T2SQL survey (Cat D, E) |
| Week_39 | 5 | BIRD/Spider/DAIL-SQL titles (Cat D, E) |
| Week_40 | 7 | PEFT/DPO/GRPO papers informal (Cat D, E) |
| Week_41 | 8 | RL/PPO/RLHF descriptions informal (Cat D, E) |
| Week_42 | 17 | URLs as text + PPO informal + arXiv abbrev (Cat A, D, E) |
| Week_43 | 10 | DPO docs + repo files + HF datasets (Cat D, E, G) |
| Week_44 | 11 | SPIN/Magpie titles + psycopg2/PG docs (Cat D, E, G) |
| Week_45 | 10 | DPO docs + author/year suffix + repos (Cat D, E) |
| Week_46 | 11 | GRPO docs + repo files + paper abbrev (Cat D, E) |
| Week_47 | 10 | Reward/GRPO/scaling paper abbrev (Cat D, E) |
| Week_48 | 13 | GRPO/DeepSeek paper abbrev + repos (Cat D, E) |
| Week_49 | 9 | DPO variants docs + KTO/SimPO abbrev (Cat D, E) |
| Week_50 | 7 | W&B/repos/paper informal (Cat E) |
| Week_51 | 7 | HF docs + W&B + evaluation (Cat E) |
| Week_52 | 16 | RLHF/DPO/GRPO paper abbrev + TRL docs (Cat D, E, F) |
| Week_53 | 12 | Alignment papers + dataset/tool labels (Cat D, E) |
| Week_54 | 12 | Synthetic data papers + tool docs (Cat D, E) |
| Week_55 | 10 | LLM eval papers + tools (Cat D, E) |
| Week_56 | 8 | CoSQL + TRL/SFT docs (Cat D, E) |
| Week_57 | 13 | CPT papers + tool/dataset labels (Cat D, E, G) |
| Week_58 | 6 | TRL SFT + W&B + Unsloth (Cat E) |
| Week_59 | 9 | DPO variants + IPO/RSO abbrev (Cat D, E) |
| Week_60 | 12 | RLHF/GRPO papers + wrong URL (Cat C, D, E) |
| Week_61 | 13 | BIRD/Spider labels + eval paper abbrev (Cat D, E) |
| Week_62 | 11 | LLM judge + pricing + stats tools (Cat D, E) |
| Week_63 | 10 | Quantization papers + tool docs (Cat D, E) |
| Week_64 | 9 | GPTQ/AWQ docs + model card names (Cat E, G) |
| Week_65 | 10 | llama.cpp/Ollama docs informal (Cat E) |
| Week_66 | 8 | vLLM/FastAPI/TGI docs informal (Cat E) |
| Week_67 | 8 | Tulu/LIMA abbrev + wrong URLs (Cat C, D) |
| Week_68 | 7 | HF Trainer + NeurIPS checklist + Chinchilla (Cat D, E) |
| Week_69 | 11 | BIRD/Spider/eval labels (Cat D, E) |
| Week_70 | 9 | Llama/arXiv/NeurIPS informal (Cat D, E) |
| Week_71 | 8 | SmolLM2/OLMo paper abbrev (Cat D) |
| Week_72 | 4 | DeepSeek blog + Sebastian Raschka (Cat E) |
| Week_73 | 8 | SAE/circuits papers + wrong URL (Cat C, D) |
| Week_74 | 4 | YaRN/Lost in Middle/IRNet abbrev (Cat D) |
| Week_75 | 10 | Llama 3/Gemma/DeepSeek model cards + chat templates (Cat D, G) |
| Week_76 | 15 | Agent/ReAct/SQL paper abbrev + wrong URLs (Cat C, D, E) |
| Week_77 | 10 | Multilingual papers abbrev + HF datasets (Cat D, E) |
| Week_78 | 9 | NLP benchmarking papers + HF pages (Cat D, E) |

---

## High-priority fixes (actionable link corrections)

These mismatches indicate the link itself may be pointing to the wrong page or the repo no longer exists at the stored URL:

1. **ID 56 (Week_12):** "Understanding GQA and MQA" → URL leads to KV cache article. Replace URL with the correct GQA/MQA article.
2. **ID 554 (Week_37), 881 (Week_53), 902 (Week_54), 938 (Week_56):** TimescaleDB docs have moved to TigerData (`tigerdata.com`). Update the domain in all TimescaleDB links.
3. **IDs 196, 215, 283, 588 (Weeks 2, 23, 39):** `yale-lily/spider` ↔ `taoyds/spider` inconsistency. Standardise all Spider references to `taoyds/spider` (the current canonical URL).
4. **ID 1016 (Week_60):** "Zephyr: Direct Distillation" → URL goes to HF RLHF blog post. The link and description are mismatched — fix one or the other.
5. **ID 1017 (Week_60):** "willthompson.io — GRPO from scratch" → URL is `abderrahmanskiredj.github.io`. Verify the intended link.
6. **ID 1141 (Week_67):** "Lessons from Thousands of ML Papers (Karpathy)" → URL leads to Karpathy's PhD survival guide. Find the correct ML papers blog URL.
7. **IDs 654, 655, 671, 672 (Week_42), 960 (Week_57):** Raw URLs used as display text. Replace with proper titles.
8. **IDs 185, 186 (Week_19):** GitHub org renames — update `microsoft/DeepSpeed` → `deepspeedai/DeepSpeed` and `pytorch/torchtune` → `meta-pytorch/torchtune`.

---

*Generated 2026-05-01 from `.verify/llm_2026-05-01T09-52-32.json`*
