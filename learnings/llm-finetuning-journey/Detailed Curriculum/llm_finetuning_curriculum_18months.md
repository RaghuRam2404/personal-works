# 18-Month Curriculum: From Deep Learning Beginner to Domain-Expert LLM Fine-Tuner

**Prepared for:** Raghuram (Software Engineer, India)
**Time budget:** 6–8 hrs/week × 78 weeks ≈ **470–620 total hours**
**Compute budget:** Mac/CPU + Google Colab (Free → Pro when needed) + RunPod (~$150–180 reserved for late phases). **Hard ceiling: $200.**
**Framework:** PyTorch, exclusively. No TensorFlow, no Keras, no JAX.
**Domain target:** **Code/SQL domain expert** — specifically, a **Text-to-SQL / PostgreSQL+TimescaleDB query generator** (you already have deep PostgreSQL/TimescaleDB expertise; datasets are massive and free; evaluation is objective — does the SQL run and return correct rows?).
**Final goal:** A 7B-parameter open-source base model (Qwen2.5-Coder-7B or DeepSeek-Coder-V2-Lite) fine-tuned to **beat GPT-4-class generalist models** on a held-out PostgreSQL/TimescaleDB SQL benchmark you build yourself.

---

## How this document is structured

Six phases. Each phase has weekly modules. Each module specifies:

1. **Concepts** — what you learn this week
2. **Required reading** — papers, blog posts, book chapters (linked)
3. **Required watching** — specific YouTube videos (channel + title + length)
4. **Coding project** — what you must build
5. **Deliverable** — concrete artifact you must produce
6. **Acceptance criteria** — how you know you can move on
7. **What I will NOT do for you** — explicit list of things you must do yourself, no shortcuts
8. **Compute** — Mac / Colab Free / Colab Pro / RunPod, with $ estimate

A **Phase Gate** at the end of each phase is a real test. If you cannot pass the gate, repeat weak modules. Do not move forward.

---

## Compute strategy & budget allocation ($200 ceiling)

| Phase | Months | Compute | Estimated $ | Cumulative $ |
|---|---|---|---|---|
| Phase 1 | 1–2 | Mac CPU + Colab Free | $0 | $0 |
| Phase 2 | 3–4 | Mac CPU + Colab Free | $0 | $0 |
| Phase 3 | 5–7 | Colab Free + Colab Pro (1 month, $10) | $10 | $10 |
| Phase 4 | 8–10 | Colab Pro (3 months, $30) + RunPod A100 ~10 hrs ($15) | $45 | $55 |
| Phase 5 | 11–13 | Colab Pro (3 months, $30) + RunPod A100 ~20 hrs ($30) | $60 | $115 |
| Phase 6 | 14–18 | Colab Pro (5 months, $50) + RunPod H100 ~10 hrs + A100 ~15 hrs ($30) | $80 | $195 |

**Buffer: $5 for emergencies.** If you stay disciplined, you finish under budget.

> **What I will NOT do for you:** sign up for Colab Pro or RunPod. **You must create accounts on [colab.research.google.com](https://colab.research.google.com), [huggingface.co](https://huggingface.co), [wandb.ai](https://wandb.ai), [github.com](https://github.com), and [runpod.io](https://runpod.io) before Week 1.** Verify billing limits on RunPod and set a spending cap.

---

## Tooling baseline (set up in Week 0, before Month 1)

You **must** install and verify these before starting Week 1. Do not skip.

1. **Python 3.11+** via `pyenv` or `uv` (NOT system Python)
2. **PyTorch 2.5+** with MPS (Mac) or CUDA (Colab/RunPod): `pip install torch torchvision torchaudio`
3. **Hugging Face stack**: `pip install transformers datasets accelerate peft trl bitsandbytes`
4. **Experiment tracking**: `pip install wandb` and login (`wandb login`)
5. **Git + GitHub**: create a repo `llm-finetuning-journey` — every weekly project goes here as a tagged commit
6. **VSCode** with extensions: Python, Jupyter, GitHub Copilot (which you already use)
7. **A daily lab notebook** — Notion, Obsidian, or a `journal.md` in your repo. Log every session.

> **What I will NOT do for you:** install software, fix CUDA/MPS issues, or debug your environment. Hugging Face's [installation guide](https://huggingface.co/docs/transformers/installation) and PyTorch's [Get Started](https://pytorch.org/get-started/locally/) page are your references.

---

## Dataset strategy — start collecting in Week 1

Your final capstone needs a **PostgreSQL/TimescaleDB-specific SQL dataset**. You will build this **incrementally, alongside the curriculum**, not at the end.

**Public datasets to download in Week 1 and explore:**
- **Spider** ([yale-lily/spider](https://github.com/taoyds/spider)) — 10K text-to-SQL pairs, 200 databases
- **BIRD-SQL** ([bird-bench.github.io](https://bird-bench.github.io/)) — 12K complex queries, harder than Spider
- **WikiSQL** — simpler, good for sanity checks
- **The Stack v2** ([HuggingFaceH4/the-stack-v2](https://huggingface.co/datasets/bigcode/the-stack-v2)) — filter for `.sql` files
- **CoSQL, SParC** — conversational SQL, use later

**Your custom dataset (build during Phases 3–5):**
- Scrape PostgreSQL official docs for query examples
- Scrape TimescaleDB blog + docs
- Hand-write 200–500 hard PostgreSQL/TimescaleDB query/answer pairs from your own work
- Synthesize 5K–20K pairs using a strong teacher model (GPT-4 / Claude API — budget $10–20) in Phase 5

> **What I will NOT do for you:** build the dataset. You must register for the datasets, download them, write the cleaning scripts, and curate the custom set yourself. This is a **non-negotiable, deliberate part of your learning.**

---

# PHASE 1 — PyTorch Fluency & Deep Learning Hands-On (Months 1–2, 8 weeks)

**Phase goal:** Become fluent in PyTorch. Implement classical neural networks from scratch. Build the muscle memory you currently lack.

**Why this phase exists:** You said you've only hand-coded 1–2 layer NNs in plain Python. That is not enough. Before you touch a transformer, you must be able to write a CNN, train it, debug it, and read its loss curves with confidence.

---

### Week 1 — PyTorch tensors, autograd, and the training loop

**Concepts:** Tensors, broadcasting, `requires_grad`, `.backward()`, computational graph, optimizer.step(), loss.backward(), the canonical training loop.

**Required reading:**
- PyTorch official tutorial: [Learn the Basics](https://pytorch.org/tutorials/beginner/basics/intro.html) — all 8 sections
- Karpathy: [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) — read 3 times this phase

**Required watching:**
- Andrej Karpathy, [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) (2h25m). **Watch fully. Code along. Do not skip.**

**Coding project:** Re-implement `micrograd` from scratch following Karpathy's video. Then port your existing 1-2 layer Python NN to PyTorch.

**Deliverable:** GitHub commit `week-01-micrograd` with: (a) your micrograd implementation, (b) your old Python NN re-written in PyTorch, (c) a `journal.md` entry comparing the two.

**Acceptance criteria:** You can explain in writing what `loss.backward()` does to every parameter in the graph. You can write a training loop from memory.

> **What I will NOT do for you:** write the training loop for you. Reproduce Karpathy's exact code by typing it yourself, line by line. No copy-paste.

**Compute:** Mac CPU.

---

### Week 2 — MLPs, activations, initialization, and the bag of tricks

**Concepts:** Multi-layer perceptrons, ReLU/GELU/Tanh, Xavier/Kaiming init, batch norm, dropout, weight decay, learning rate schedules.

**Required reading:**
- Karpathy: [Yes you should understand backprop](https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b)
- [Deep Learning Book](https://www.deeplearningbook.org/), Chapter 6 (Goodfellow et al.) — free online

**Required watching:**
- Andrej Karpathy, [The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo) (1h15m)
- Andrej Karpathy, [Building makemore Part 2: MLP](https://www.youtube.com/watch?v=TCH_1BHY58I) (1h15m)
- Andrej Karpathy, [Building makemore Part 3: Activations & Gradients, BatchNorm](https://www.youtube.com/watch?v=P6sfmUTpUmc) (1h55m)

**Coding project:** Build the bigram and MLP language models from Karpathy's makemore videos on a names dataset. Then **swap the dataset** to SQL keyword sequences (extract from Spider).

**Deliverable:** GitHub commit `week-02-makemore` with bigram + MLP models trained on names AND on SQL tokens. Loss curves logged to W&B.

**Acceptance criteria:** You can plot training/validation loss in W&B, identify overfitting, explain why batch norm helps, and re-derive Kaiming init on paper.

> **What I will NOT do for you:** explain why your model is overfitting. You must diagnose it by reading the curves yourself. If stuck >2 hours, search PyTorch forums or ask GPT — but write down what you learned.

**Compute:** Mac CPU.

---

### Week 3 — Convolutional Neural Networks (yes, you need this)

**Concepts:** Convolutions, pooling, receptive field, parameter sharing, why CNNs matter even for an LLM career (FlashAttention uses CUDA tricks descended from CNN optimization).

**Required reading:**
- [CS231n notes: ConvNets](https://cs231n.github.io/convolutional-networks/)
- [An Intuitive Explanation of Convolutional Neural Networks](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/)

**Required watching:**
- Karpathy, [Building makemore Part 4: Becoming a Backprop Ninja](https://www.youtube.com/watch?v=q8SA3rM6ckI) (1h55m) — **critical**, do not skip
- Karpathy, [Building makemore Part 5: Building a WaveNet](https://www.youtube.com/watch?v=t3YJ5hKiMQ0) (1h56m)

**Coding project:** Train a small CNN on CIFAR-10 in PyTorch. Use `torchvision.datasets.CIFAR10`. Achieve ≥75% test accuracy.

**Deliverable:** GitHub commit `week-03-cnn-cifar10` with model code, training script, W&B run link, confusion matrix plot.

**Acceptance criteria:** ≥75% test accuracy on CIFAR-10. You can manually compute the output shape of any conv layer given input shape, kernel size, stride, padding.

> **What I will NOT do for you:** tune your hyperparameters. Read [Recipe for Training NNs](https://karpathy.github.io/2019/04/25/recipe/) and follow Karpathy's debugging discipline.

**Compute:** Colab Free (T4 GPU). ~1 hour total training.

---

### Week 4 — RNNs, LSTMs, and why we abandoned them

**Concepts:** Sequence modeling, hidden state, BPTT, vanishing gradients, LSTM/GRU gates, sequence-to-sequence with attention (the original 2014 idea, not transformer attention).

**Required reading:**
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — Chris Olah
- [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)

**Required watching:**
- StatQuest: [Recurrent Neural Networks Clearly Explained](https://www.youtube.com/watch?v=AsNTP8Kwu80) (16m)
- StatQuest: [Long Short-Term Memory (LSTM) Clearly Explained](https://www.youtube.com/watch?v=YCzL96nL7j0) (20m)

**Coding project:** Implement a character-level LSTM in PyTorch. Train it on a corpus of SQL queries (extracted from Spider). Generate samples.

**Deliverable:** GitHub commit `week-04-char-lstm` with model, samples of generated SQL (mostly garbage is fine — that's the point), W&B link.

**Acceptance criteria:** You can explain why LSTMs were superseded by transformers (parallelization, long-range dependencies). You have generated plausible-looking SQL gibberish.

> **What I will NOT do for you:** implement teacher forcing for you. Read the PyTorch docs and figure it out.

**Compute:** Colab Free. ~30 min training.

---

### Week 5 — Optimization, regularization, and how to read loss curves

**Concepts:** SGD vs Adam vs AdamW, learning rate warmup, cosine schedules, gradient clipping, label smoothing, mixed precision.

**Required reading:**
- [An overview of gradient descent optimization algorithms](https://www.ruder.io/optimizing-gradient-descent/) — Sebastian Ruder
- [Decoupled Weight Decay Regularization (AdamW paper)](https://arxiv.org/abs/1711.05101) — read intro + algorithm box

**Required watching:**
- Yannic Kilcher, [AdamW paper explained](https://www.youtube.com/watch?v=oWZbcq_figk) (~25m)

**Coding project:** Take your Week 3 CNN. Add: AdamW, learning rate warmup, cosine schedule, gradient clipping, mixed precision (`torch.cuda.amp`). Re-train. Compare loss curves.

**Deliverable:** Side-by-side W&B report comparing baseline vs. optimized run.

**Acceptance criteria:** Optimized run reaches same accuracy in fewer steps OR achieves higher final accuracy. You can write 2 paragraphs explaining why.

> **What I will NOT do for you:** explain why your training is unstable. Read the curves, form hypotheses, test them.

**Compute:** Colab Free.

---

### Week 6 — Tokenization deep dive (the most underrated topic)

**Concepts:** Why tokenization exists, byte-pair encoding (BPE), WordPiece, SentencePiece, tiktoken, vocabulary trade-offs, how tokenization affects model behavior on numbers, code, non-English text.

**Required reading:**
- Karpathy: [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) (companion repo: [minbpe](https://github.com/karpathy/minbpe))
- HuggingFace [Tokenizers tutorial](https://huggingface.co/learn/llm-course/chapter6/1)

**Required watching:**
- Andrej Karpathy, [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) (2h13m). **Code along.**

**Coding project:** Implement BPE from scratch following Karpathy's video. Train it on a SQL corpus. Compare your vocab to GPT-4's tokenization of the same SQL.

**Deliverable:** GitHub commit `week-06-bpe` with your BPE implementation, trained tokenizer on SQL, comparison report.

**Acceptance criteria:** You can explain why `"PostgreSQL"` might tokenize differently in GPT-4 vs. your SQL-trained tokenizer, and why this matters for fine-tuning.

> **What I will NOT do for you:** debug regex patterns. Karpathy's video shows the GPT-2/4 splitting regexes — type them yourself.

**Compute:** Mac CPU.

---

### Week 7 — HuggingFace ecosystem onboarding

**Concepts:** `transformers`, `datasets`, `tokenizers`, `accelerate`, model hub, dataset hub, AutoClass pattern.

**Required reading:**
- [HuggingFace LLM Course Chapters 1–3](https://huggingface.co/learn/llm-course/chapter1/1) — do every exercise

**Coding project:**
1. Load `distilgpt2`. Generate text. Inspect logits.
2. Load `Spider` dataset via `datasets`. Tokenize it with `Qwen2.5-Coder-7B`'s tokenizer.
3. Push a tokenized version of Spider to your HuggingFace account as a private dataset.

**Deliverable:** Public HuggingFace profile with at least 1 dataset uploaded. GitHub commit `week-07-hf-onboarding`.

**Acceptance criteria:** You can load any model from the hub, run inference, inspect attention masks, push artifacts back.

> **What I will NOT do for you:** create your HuggingFace account, generate your access token, or set up `huggingface-cli login`. Do this in Week 0.

**Compute:** Colab Free.

---

### Week 8 — Phase 1 Gate: Capstone mini-project

**Project:** End-to-end PyTorch project. Your choice between:
- **Option A (recommended):** Train a small char-level transformer (you'll learn what this means in Phase 2 — for now, copy [nanoGPT](https://github.com/karpathy/nanoGPT)'s architecture but use your own data loader and training loop) on Spider's SQL queries. Generate SQL samples.
- **Option B:** Fine-tune `distilgpt2` for 100 steps on a small corpus using HuggingFace `Trainer`. Just to feel the API.

**Deliverable:** A clean, well-documented GitHub repo with a README that explains: dataset, model, training, results, what you learned, what surprised you.

**Phase Gate (you must pass these to advance):**
- [ ] You can write a PyTorch training loop from memory in <10 minutes
- [ ] You can read a W&B loss curve and diagnose: overfitting, underfitting, LR too high, LR too low
- [ ] You can explain backprop using a 2-layer network on paper, by hand
- [ ] You have committed code for every week
- [ ] You have a HuggingFace account with at least 1 artifact

If you fail any item, **repeat the relevant week. Do not advance.**

> **What I will NOT do for you:** judge whether you've passed. You must be honest with yourself. The cost of advancing while weak is brutal — Phase 2 builds directly on these foundations.

---

# PHASE 2 — Attention, Transformers, and Hand-Coding GPT (Months 3–4, 8 weeks)

**Phase goal:** Read the "Attention Is All You Need" paper and understand every line. Hand-code a GPT from scratch. Train it on tiny data. Understand every detail of the modern decoder-only architecture.

**Why this phase exists:** You said you learned attention 2 years ago but never coded it. After this phase, you will have hand-typed a working transformer at least three times. There is no substitute.

---

### Week 9 — The original attention mechanism (Bahdanau, 2014)

**Concepts:** Why attention was invented (machine translation), additive vs. multiplicative attention, the seq2seq + attention architecture.

**Required reading:**
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — Bahdanau et al. **Read fully. Take notes.**
- [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/) — Lilian Weng's blog

**Required watching:**
- Yannic Kilcher, [Attention Is All You Need](https://www.youtube.com/watch?v=iDulhoQ2pro) (28m) — **only the first 10 min, on history**

**Coding project:** Implement Bahdanau attention on top of a tiny seq2seq LSTM (use PyTorch). Train it to reverse short strings. Plot attention weights.

**Deliverable:** GitHub commit `week-09-bahdanau-attn` with attention heatmap visualization.

**Acceptance criteria:** You can draw the seq2seq+attention architecture on a whiteboard from memory.

> **What I will NOT do for you:** explain attention to you. You must derive the score function on paper.

**Compute:** Mac CPU.

---

### Week 10 — "Attention Is All You Need" — the paper, line by line

**Concepts:** Self-attention, multi-head attention, positional encoding, encoder-decoder transformer, the entire 2017 architecture.

**Required reading:**
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017. **Read 3 times this week. Take handwritten notes.**
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Jay Alammar
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) — Sasha Rush. **This is your coding bible this week.**

**Required watching:**
- Yannic Kilcher, [Attention Is All You Need explained](https://www.youtube.com/watch?v=iDulhoQ2pro) (28m, full)
- 3Blue1Brown, [But what is a GPT? Visual intro to transformers](https://www.youtube.com/watch?v=wjZofJX0v4M) (27m)
- 3Blue1Brown, [Attention in transformers, visually explained](https://www.youtube.com/watch?v=eMlx5fFNoYc) (26m)

**Coding project:** Implement the encoder-decoder transformer from "The Annotated Transformer" notebook. Type every line yourself. Train on a tiny English→French toy task.

**Deliverable:** GitHub commit `week-10-annotated-transformer` with your full implementation and a working translation example.

**Acceptance criteria:** You can derive the formula for scaled dot-product attention on paper and explain the `1/sqrt(d_k)` term.

> **What I will NOT do for you:** explain why we use `1/sqrt(d_k)`. The paper explains it. Read until you understand.

**Compute:** Colab Free.

---

### Week 11 — Decoder-only transformers and the GPT family

**Concepts:** Causal masking, decoder-only architecture, why GPT-2/3/4 dropped the encoder, weight tying, residual stream view.

**Required reading:**
- [Improving Language Understanding by Generative Pre-Training (GPT-1)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) — Radford et al. 2018
- [Language Models are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — Radford et al. 2019
- [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165) — Brown et al. 2020 (skim — focus on architecture section + scaling laws)

**Required watching:**
- Andrej Karpathy, [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) (1h56m). **Code along. Do not skip a single second.**

**Coding project:** Type out Karpathy's `nanoGPT`-style implementation from his video, training on Tiny Shakespeare. Then re-train it on a SQL corpus.

**Deliverable:** GitHub commit `week-11-nanogpt` with two trained models (Shakespeare + SQL). Generated samples from each.

**Acceptance criteria:** You understand every line of `model.py` from [nanoGPT](https://github.com/karpathy/nanoGPT). You can answer: why does `nn.Embedding` and the LM head share weights?

> **What I will NOT do for you:** debug your training when loss diverges. Look at the gradients in W&B. The answer is almost always: bad init, bad LR, or no warmup.

**Compute:** Colab Free.

---

### Week 12 — Modern architectural improvements (2019–2024)

**Concepts:** Pre-LN vs Post-LN, RMSNorm, SwiGLU activation, Rotary Position Embeddings (RoPE), Grouped-Query Attention (GQA), Sliding Window Attention.

**Required reading:**
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) — Pre-LN vs Post-LN
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) — RMSNorm
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — SwiGLU
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — RoPE
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)

**Required watching:**
- Umar Jamil, [LLaMA explained: KV-Cache, Rotary Positional Embedding, RMS Norm, Grouped Query Attention, SwiGLU](https://www.youtube.com/watch?v=Mn_9W1nCFLo) (1h10m). **Excellent.**

**Coding project:** Modify your nanoGPT from Week 11. Replace LayerNorm → RMSNorm. Replace learned positional embeddings → RoPE. Replace ReLU → SwiGLU. Re-train. Compare.

**Deliverable:** GitHub commit `week-12-modern-arch` with side-by-side comparison.

**Acceptance criteria:** You can explain RoPE on paper using complex number rotation.

> **What I will NOT do for you:** implement RoPE for you. Use the original paper's pseudocode + Umar Jamil's video. Type it yourself.

**Compute:** Colab Free.

---

### Week 13 — KV cache, inference optimization, sampling

**Concepts:** Why inference is different from training, KV cache, greedy/beam/top-k/top-p/temperature sampling, speculative decoding (intro).

**Required reading:**
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) — focus on KV cache section
- [How to generate text](https://huggingface.co/blog/how-to-generate) — HuggingFace blog
- HuggingFace [Generation strategies](https://huggingface.co/docs/transformers/main/en/generation_strategies)

**Required watching:**
- Umar Jamil, [LLaMA 2 explained](https://www.youtube.com/watch?v=Mn_9W1nCFLo) (revisit KV cache section)

**Coding project:** Add a working KV cache to your nanoGPT. Implement temperature, top-k, and top-p sampling from scratch. Benchmark inference speed with vs. without KV cache.

**Deliverable:** GitHub commit `week-13-kv-cache-sampling` with benchmark numbers.

**Acceptance criteria:** Your KV-cached inference is at least 5x faster than recomputing. You can explain why.

> **What I will NOT do for you:** explain why beam search is rarely used in modern LLM inference. Read the HF blog.

**Compute:** Colab Free.

---

### Week 14 — Reading and understanding the LLaMA paper(s)

**Concepts:** Production-grade open transformer architecture, design choices, dataset composition.

**Required reading:**
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) — Touvron et al. 2023. **Read fully.**
- [Llama 2](https://arxiv.org/abs/2307.09288) — section on architecture changes
- [Llama 3](https://arxiv.org/abs/2407.21783) — Herd of Models paper. Read sections 1, 3, 5.

**Coding project:** Read the `transformers` library's [`modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py). Print it. Annotate every line on paper. Write a 1-page summary of how it differs from your Week 12 nanoGPT.

**Deliverable:** A scanned/photographed annotated PDF or detailed Markdown notes in `journal.md`.

**Acceptance criteria:** You can answer: what is GQA's `num_key_value_heads`? Why does Llama 3 use 8?

> **What I will NOT do for you:** read the paper for you. This is the most important reading week of the entire curriculum. **Spend 6+ hours on it.**

**Compute:** None.

---

### Week 15 — From-scratch GPT-2 reproduction (Karpathy build-the-GPT-2 video)

**Concepts:** Production-grade training, mixed precision, distributed training basics, gradient accumulation.

**Required watching:**
- Andrej Karpathy, [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU) (4h01m). **The single most valuable video in this curriculum. Code along across the entire week.**

**Coding project:** Reproduce GPT-2 124M training following Karpathy's video. Use Colab Pro for the actual training run (1 month subscription = $10, this is where you start spending).

**Deliverable:** GitHub commit `week-15-gpt2-repro` with: training script, W&B run, generated samples, val loss vs. GPT-2 124M's known val loss on OpenWebText.

**Acceptance criteria:** Your val loss is within 5% of GPT-2 124M's published val loss.

> **What I will NOT do for you:** subscribe to Colab Pro for you. **Buy 1 month of Colab Pro this week.** Use the A100 runtime when available.

**Compute:** Colab Pro ($10). Budget cumulative: $10.

---

### Week 16 — Phase 2 Gate

**Project:** Without looking at any reference, implement a decoder-only transformer with: RMSNorm, RoPE, SwiGLU, GQA, KV cache, and top-p sampling. Train it on Spider SQL. Generate plausible SQL.

**Phase Gate:**
- [ ] You can implement scaled dot-product attention from memory
- [ ] You can implement multi-head attention from memory
- [ ] You can read `modeling_llama.py` and explain every block
- [ ] You have reproduced GPT-2 124M
- [ ] You can explain RoPE, GQA, RMSNorm, SwiGLU on a whiteboard

If you fail, **stop. Re-do the failing weeks.** This phase is the hinge of the entire curriculum.

---

# PHASE 3 — Pretraining Mechanics & a Tiny LM From Scratch (Months 5–7, 12 weeks)

**Phase goal:** Understand pretraining at a deep level. Train a 50M-parameter language model from scratch (this is your 1 dirty-hands pretraining experience). Then commit to fine-tuning as your main path forward.

**Why this phase exists:** You said you don't want to focus heavily on pretraining, but you do want hands-on experience. This phase delivers that — bounded, focused, and budget-respecting.

---

### Week 17 — Scaling laws

**Concepts:** Compute-optimal training, Chinchilla scaling, parameters vs. tokens trade-off.

**Required reading:**
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — Kaplan et al. 2020
- [Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556) — Hoffmann et al. 2022

**Required watching:**
- Yannic Kilcher, [Chinchilla paper](https://www.youtube.com/watch?v=PZXN7jTLnso) (~30m)

**Deliverable:** A 1-page Markdown writeup: given X dollars of compute, what's the compute-optimal model size and dataset size? Apply this to your $50 remaining Phase 6 budget.

> **What I will NOT do for you:** do the math. Use the Chinchilla formulas yourself.

**Compute:** None.

---

### Week 18 — Data: where pretraining data comes from and why it matters

**Concepts:** Common Crawl, C4, RefinedWeb, FineWeb, The Pile, deduplication, quality filtering, data mixing.

**Required reading:**
- [The Pile: An 800GB Dataset of Diverse Text](https://arxiv.org/abs/2101.00027)
- [The RefinedWeb Dataset for Falcon LLM](https://arxiv.org/abs/2306.01116)
- [FineWeb: decanting the web for the finest text data](https://huggingface.co/datasets/HuggingFaceFW/fineweb) — read the dataset card AND the [technical blog](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)

**Coding project:** Download a 1GB shard of FineWeb-Edu. Run basic statistics: doc length, language distribution, quality filtering with `fasttext`. Implement MinHash deduplication on a small sample.

**Deliverable:** Notebook with data analysis. GitHub commit `week-18-pretraining-data`.

**Acceptance criteria:** You can explain why "pretraining data quality" is the most important variable in modern LLM training.

> **What I will NOT do for you:** explain MinHash. Read [datasketch docs](https://github.com/ekzhu/datasketch).

**Compute:** Colab Free.

---

### Week 19 — Distributed training conceptually (DDP, FSDP, ZeRO)

**Concepts:** Data parallelism, model parallelism, pipeline parallelism, ZeRO stages, FSDP. **Conceptual only — you won't run multi-GPU.**

**Required reading:**
- [PyTorch DDP tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- HuggingFace [Accelerate concepts](https://huggingface.co/docs/accelerate/concept_guides/big_model_inference)

**Coding project:** Use HuggingFace `accelerate` to train your nanoGPT on a single Colab GPU. Then read (don't run) a multi-GPU FSDP config and write notes on what each line does.

**Deliverable:** Notes in `journal.md`. GitHub commit `week-19-distributed-concepts`.

> **What I will NOT do for you:** rent multi-GPU instances. You don't need to run distributed training to be hireable for fine-tuning roles.

**Compute:** Colab Free.

---

### Weeks 20–22 — Train your own 50M-parameter LM from scratch

**Concepts:** End-to-end pretraining run, monitoring training health, recovering from divergence, evaluation perplexity.

**Required reading:**
- nanoGPT [README](https://github.com/karpathy/nanoGPT) and [GPT-2 reproduction guide](https://github.com/karpathy/llm.c/discussions/677)
- [Cramming: Training a Language Model on a Single GPU in One Day](https://arxiv.org/abs/2212.14034)

**Required watching:**
- Re-watch the relevant sections of Karpathy's [Reproduce GPT-2 video](https://www.youtube.com/watch?v=l8pRSuU81PU)

**Coding project (3 weeks):**
- **Week 20:** Set up codebase. Use `nanoGPT` as starting point but modify to your own structure. Choose model size: ~50M params. Train tokenizer on FineWeb-Edu sample.
- **Week 21:** Run pretraining for ~24 hours total Colab Pro A100 time across the week. Target ~2B tokens. Log everything to W&B.
- **Week 22:** Evaluate. Compute val perplexity. Generate samples. Write up findings.

**Deliverable:** GitHub commit `weeks-20-22-50M-pretrain` with full training run, W&B link, model checkpoint pushed to your HuggingFace account.

**Acceptance criteria:** You have a working 50M LM with sensible perplexity (<25 on FineWeb val). You have generated coherent (if simple) English.

> **What I will NOT do for you:** save you from a divergent loss. If the loss spikes, **debug it.** Lower LR, add warmup, lower batch size. Document the failure in `journal.md`.

**Compute:** Colab Pro ($10/month, this is month 2 of Pro). Budget cumulative: $20.

---

### Week 23 — Evaluation 101: perplexity, downstream benchmarks

**Concepts:** Perplexity, log-likelihood, LM Evaluation Harness, MMLU, HellaSwag, ARC.

**Required reading:**
- [EleutherAI lm-evaluation-harness README](https://github.com/EleutherAI/lm-evaluation-harness)
- [HELM: Holistic Evaluation of Language Models](https://crfm.stanford.edu/helm/latest/) — read methodology

**Coding project:** Run `lm-eval-harness` on your 50M model from Weeks 20–22. Run it on `gpt2` (124M) for comparison. Write up.

**Deliverable:** Eval report comparing your model to GPT-2-small. GitHub commit `week-23-eval-harness`.

> **What I will NOT do for you:** install lm-eval-harness — you must `pip install lm-eval` and read the docs yourself.

**Compute:** Colab Free.

---

### Week 24 — Reading week: SOTA pretraining recipes (2024–2026)

**Required reading (skim widely, deep-read 2):**
- [Llama 3 paper](https://arxiv.org/abs/2407.21783)
- [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)
- [Qwen2.5-Coder Technical Report](https://arxiv.org/abs/2409.12186)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [DeepSeek-Coder paper](https://arxiv.org/abs/2401.14196)

**Deliverable:** A 3-page comparison Markdown: how do Llama 3, Qwen2.5, DeepSeek-V3 differ in: data, architecture, training recipe, post-training?

> **What I will NOT do for you:** summarize papers. Reading and synthesizing is the skill.

**Compute:** None.

---

### Week 25–26 — Domain dataset construction kickoff

**Concepts:** Dataset curation for fine-tuning, instruction format, conversation format, ShareGPT vs. Alpaca format.

**Required reading:**
- [Alpaca dataset format](https://github.com/tatsu-lab/stanford_alpaca)
- [ShareGPT format](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)
- [Self-Instruct paper](https://arxiv.org/abs/2212.10560)

**Coding project (2 weeks):** Build v1 of your **PostgreSQL/TimescaleDB SQL dataset**:
- Parse Spider, BIRD, WikiSQL into ChatML format
- Hand-write 100 PostgreSQL/TimescaleDB-specific Q→SQL pairs from your own work
- Filter The Stack v2 for `.sql` files (you can extract NL→SQL pairs from comments)
- Aim for 5,000 high-quality pairs in v1

**Deliverable:** Private HuggingFace dataset `<your-handle>/postgres-sql-v1` with at least 5K examples. Schema documented. GitHub commit `weeks-25-26-dataset-v1`.

> **What I will NOT do for you:** write the curation scripts. This is **your** dataset; ownership is the point.

**Compute:** Mac/Colab Free.

---

### Week 27 — Phase 3 Gate

**Phase Gate:**
- [ ] You have trained a 50M-param LM from scratch
- [ ] You can compute perplexity yourself
- [ ] You have a v1 5K-example domain dataset
- [ ] You can read any modern LLM technical report and identify the architectural and training choices

---

# PHASE 4 — The Fine-Tuning Stack: SFT, LoRA, QLoRA (Months 8–10, 12 weeks)

**Phase goal:** Master full SFT, then parameter-efficient methods (LoRA, QLoRA, DoRA). Be able to fine-tune any 7B model on a single 24GB GPU.

---

### Week 28 — What is fine-tuning, really?

**Concepts:** Continued pretraining vs. supervised fine-tuning vs. instruction tuning, when to use which, the post-training pipeline (SFT → DPO → GRPO).

**Required reading:**
- HuggingFace [LLM Course Chapter 11: Supervised Fine-Tuning](https://huggingface.co/learn/llm-course/chapter11/1)
- [InstructGPT paper](https://arxiv.org/abs/2203.02155) — read sections 1–3
- [The 2025 LLM Year in Review](https://karpathy.bearblog.dev/year-in-review-2025/) — Karpathy

**Required watching:**
- Karpathy, [Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI) (3h31m). **Watch over the week.**

**Deliverable:** A diagram (hand-drawn or in Excalidraw) of the modern post-training pipeline. GitHub commit.

> **What I will NOT do for you:** explain when to do continued pretraining vs. SFT. Read InstructGPT until you understand the difference.

**Compute:** None.

---

### Week 29 — Full SFT on a tiny model

**Concepts:** Hands-on with HuggingFace `Trainer` and `SFTTrainer` from `trl`.

**Required reading:**
- [TRL SFTTrainer docs](https://huggingface.co/docs/trl/sft_trainer)
- [HuggingFace fine-tuning tutorial](https://huggingface.co/docs/transformers/training)

**Coding project:** Full SFT (no LoRA) of `Qwen2.5-0.5B` on a 1K subset of your domain dataset. Use `SFTTrainer`.

**Deliverable:** GitHub commit `week-29-sft-tiny`. Model pushed to HuggingFace. W&B run linked.

**Acceptance criteria:** Loss decreases over training. Model produces outputs that look slightly more SQL-like than the base.

> **What I will NOT do for you:** debug your `chat_template`. Read the tokenizer config of Qwen2.5-0.5B.

**Compute:** Colab Pro ($10, month 1 of Phase 4 Pro). Budget cumulative: $30.

---

### Week 30 — LoRA: the math and the intuition

**Concepts:** Low-rank adaptation, why fine-tuning is intrinsically low-rank, LoRA hyperparameters (r, alpha, dropout, target_modules).

**Required reading:**
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Hu et al. 2021. **Read fully.**
- [Parameter-Efficient Fine-Tuning Methods Survey](https://arxiv.org/abs/2403.14608)

**Required watching:**
- Yannic Kilcher, [LoRA explained](https://www.youtube.com/watch?v=DhRoTONcyZE) (~25m)
- Sebastian Raschka, [Practical Tips for Finetuning LLMs Using LoRA](https://www.youtube.com/watch?v=YVU5wAA6Txo) (~1h)

**Coding project:** Implement LoRA from scratch in PyTorch (a `LoraLinear` module). Apply it to your nanoGPT from Phase 2. Verify it works.

**Deliverable:** GitHub commit `week-30-lora-from-scratch`. A blog-post-style writeup in your repo.

**Acceptance criteria:** You can derive on paper why a rank-r LoRA has `r * (d_in + d_out)` trainable params.

> **What I will NOT do for you:** explain matrix decomposition. Brush up on linear algebra if needed.

**Compute:** Colab Free.

---

### Week 31 — LoRA via `peft` library, and choosing target_modules

**Required reading:**
- [PEFT documentation](https://huggingface.co/docs/peft/index)
- Sebastian Raschka, [Practical Tips for Finetuning LLMs Using LoRA](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms)

**Coding project:** LoRA-fine-tune `Qwen2.5-Coder-1.5B` on your 5K domain dataset using `peft` + `SFTTrainer`. Sweep r ∈ {8, 16, 32, 64}. Compare.

**Deliverable:** W&B sweep report. GitHub commit `week-31-lora-sweep`.

**Acceptance criteria:** You have empirical loss curves for 4 different ranks and can explain trade-offs.

> **What I will NOT do for you:** pick the right target_modules. Read [Sebastian Raschka's article](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms) — the answer is "all linear layers."

**Compute:** Colab Pro. Budget cumulative: $40.

---

### Week 32 — Quantization fundamentals

**Concepts:** FP32, FP16, BF16, INT8, INT4, NF4, calibration, weight-only vs. activation quantization.

**Required reading:**
- [LLM.int8(): 8-bit Matrix Multiplication for Transformers](https://arxiv.org/abs/2208.07339) — Dettmers
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)
- [AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978)
- [The from-scratch quantization lesson covering FP8, GPTQ, AWQ, GGUF](https://github.com/huggingface/quantization-tutorial) (search "from-scratch quantization lesson 2026" if link moves)

**Required watching:**
- Maarten Grootendorst, [A Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization)

**Coding project:** Quantize `Qwen2.5-Coder-1.5B` to 4-bit using `bitsandbytes`. Measure: model size on disk, VRAM usage, perplexity drop, inference speed. Repeat with GPTQ via `auto-gptq`.

**Deliverable:** GitHub commit `week-32-quantization`. Comparison table.

**Acceptance criteria:** You can explain NF4 vs. INT4 vs. FP4 in writing.

> **What I will NOT do for you:** install CUDA drivers. Use Colab.

**Compute:** Colab Pro.

---

### Week 33 — QLoRA: the killer combo for your hardware

**Concepts:** 4-bit base model + LoRA adapters, double quantization, paged optimizers. **This is what makes 7B fine-tuning possible on a 24GB GPU.**

**Required reading:**
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — Dettmers 2023. **Read fully.**

**Coding project:** QLoRA-fine-tune `Qwen2.5-Coder-7B` on your 5K dataset using Colab Pro A100. **This is your first 7B fine-tune.** Use HuggingFace `peft` + `bitsandbytes` + `SFTTrainer`.

**Deliverable:** Pushed model on HuggingFace: `<your-handle>/qwen-coder-7b-postgres-v1`. W&B run. GitHub commit `week-33-qlora-7b`.

**Acceptance criteria:** Training completes. The resulting model produces correct PostgreSQL on at least 30% of a held-out 100-example test set.

> **What I will NOT do for you:** build the held-out test set. Build it in Week 32 from your own data.

**Compute:** Colab Pro. Budget cumulative: $50.

---

### Week 34 — Unsloth: the speed unlock

**Concepts:** Unsloth's optimized kernels, why it's 2–5x faster than vanilla HF, single-GPU focus.

**Required reading:**
- [Unsloth GitHub](https://github.com/unslothai/unsloth) — README + notebooks
- [Unsloth blog posts](https://unsloth.ai/blog) — read the most recent 3

**Required watching:**
- Daniel Han (Unsloth founder), [GPU Mode talk](https://www.youtube.com/results?search_query=daniel+han+unsloth) — find latest

**Coding project:** Re-run your Week 33 QLoRA fine-tune using Unsloth's [Qwen2.5-Coder Colab notebook](https://github.com/unslothai/unsloth/blob/main/README.md#-finetune-for-free). Measure: time, VRAM, final loss vs. vanilla.

**Deliverable:** Comparison report. GitHub commit `week-34-unsloth`.

**Acceptance criteria:** You confirm the 2–5x speedup empirically.

> **What I will NOT do for you:** decide whether to use Unsloth, vanilla HF, or Axolotl going forward. **My recommendation: Unsloth for all single-GPU runs in this curriculum.** Stick with it.

**Compute:** Colab Pro.

---

### Week 35 — Hyperparameter tuning for SFT/LoRA

**Concepts:** Learning rate, batch size, epochs, LoRA rank/alpha, packing, gradient checkpointing.

**Required reading:**
- Sebastian Raschka, [LoRA insights from hundreds of experiments](https://lightning.ai/pages/community/lora-insights/)

**Coding project:** Set up a small W&B sweep over: LR ∈ {1e-5, 5e-5, 2e-4}, rank ∈ {16, 32}, alpha ∈ {16, 32, 64}, on a 1K subset (for speed). Find best config.

**Deliverable:** W&B sweep report.

**Acceptance criteria:** You can articulate, in writing, what each hyperparameter does to the loss curve.

> **What I will NOT do for you:** tell you what hyperparameters to use. Run the sweep.

**Compute:** Colab Pro.

---

### Week 36 — DoRA, RSLoRA, LoftQ, and other LoRA variants

**Required reading:**
- [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353)
- [RSLoRA: Rank-Stabilized LoRA](https://arxiv.org/abs/2312.03732)
- [LoftQ: LoRA-Fine-Tuning-Aware Quantization](https://arxiv.org/abs/2310.08659)

**Coding project:** Try DoRA via `peft`. Compare to vanilla LoRA on your dataset.

**Deliverable:** Short comparison. GitHub commit `week-36-dora`.

> **What I will NOT do for you:** decide whether DoRA helps your task. Measure.

**Compute:** Colab Pro. Budget cumulative: $60.

---

### Weeks 37–39 — Domain-tuning sprint #1: build your v1 SQL expert

**Project (3 weeks):**
- **Week 37:** Expand dataset to 15K examples (use synthetic generation via Claude API or GPT-4 API — budget $10–20 from outside the $200, optional).
- **Week 38:** QLoRA fine-tune Qwen2.5-Coder-7B on 15K examples. Use Unsloth. Run on Colab Pro A100.
- **Week 39:** Evaluate. Build a custom **execution-based eval**: spin up a Postgres Docker container in Colab, run model-generated SQL against it, measure exact-match + execution-correctness.

**Deliverable:** `<your-handle>/postgres-sqlcoder-7b-v1` on HuggingFace. Eval report. **Compare to Qwen2.5-Coder-7B base, GPT-4o (via API), and Claude 3.5 Sonnet (via API).**

**Acceptance criteria:** Your fine-tuned 7B beats the base Qwen2.5-Coder-7B on your held-out PostgreSQL test set. (Beating GPT-4o is the Phase 6 goal — not yet.)

> **What I will NOT do for you:** set up the Docker Postgres eval harness. This is **a critical engineering project** — search for `sql-eval` (defog-ai's repo) for inspiration.

**Compute:** Colab Pro. Budget cumulative: $70.

---

### Week 40 — Phase 4 Gate

**Phase Gate:**
- [ ] You have done at least 4 successful fine-tuning runs (full SFT, LoRA, QLoRA, Unsloth-QLoRA)
- [ ] You can fine-tune a 7B model on Colab Pro
- [ ] You have a working execution-based eval harness
- [ ] You have a v1 model on HuggingFace
- [ ] You have a 15K-example domain dataset

---

# PHASE 5 — Preference Optimization: RLHF, DPO, GRPO (Months 11–13, 12 weeks)

**Phase goal:** Learn the post-SFT alignment stage. Apply DPO to your domain model. Then learn GRPO (the 2025 RLVR breakthrough) and apply it to make your model better at executable SQL via verifiable rewards.

---

### Week 41 — RL primer (the minimum you need for LLMs)

**Concepts:** MDPs, policy/value, on-policy vs. off-policy, REINFORCE, policy gradient theorem.

**Required reading:**
- [Spinning Up in Deep RL — Intro](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) — OpenAI, by Josh Achiam
- [Policy Gradient Algorithms](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/) — Lilian Weng

**Required watching:**
- David Silver's RL course is overkill. Instead: HuggingFace [Deep RL Course](https://huggingface.co/learn/deep-rl-course/unit0/introduction) — Units 1, 2, 4, 8. **That's it. Skip the rest.**

**Deliverable:** Notes on policy gradient. Solve OpenAI Gym CartPole with REINFORCE in PyTorch.

**Acceptance criteria:** You can derive the policy gradient on paper.

> **What I will NOT do for you:** make you an RL expert. We need only what serves LLMs.

**Compute:** Colab Free.

---

### Week 42 — PPO and the original RLHF stack

**Concepts:** PPO clipping objective, advantage estimation (GAE), reference model + KL penalty, the InstructGPT pipeline.

**Required reading:**
- [Proximal Policy Optimization Algorithms (PPO)](https://arxiv.org/abs/1707.06347) — Schulman et al.
- [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) — Christiano et al. 2017
- [InstructGPT paper](https://arxiv.org/abs/2203.02155) — re-read sections 3 and 4

**Required watching:**
- Yannic Kilcher, [InstructGPT explained](https://www.youtube.com/watch?v=VPRSBzXzavo) (~30m)

**Coding project:** Read TRL's [PPOTrainer source code](https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py). Annotate.

**Deliverable:** Annotated source notes.

> **What I will NOT do for you:** explain GAE. Read [Spinning Up's GAE section](https://spinningup.openai.com/en/latest/algorithms/ppo.html).

**Compute:** None.

---

### Week 43 — DPO: skipping the reward model

**Concepts:** Direct Preference Optimization, the math (closed-form solution to KL-constrained RL), why DPO often replaces PPO in 2024+.

**Required reading:**
- [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290) — Rafailov et al. 2023. **Read fully, including appendix A.1 derivation.**
- HuggingFace [DPO Trainer docs](https://huggingface.co/docs/trl/dpo_trainer)

**Required watching:**
- Umar Jamil, [DPO Direct Preference Optimization explained](https://www.youtube.com/watch?v=hvGa5Mba4c8) (~50m)

**Coding project:** Read [philschmid's DPO notebook](https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/rl-with-llms-in-2025-dpo.ipynb). Run it end-to-end on a small open-source preference dataset (`HuggingFaceH4/ultrafeedback_binarized`).

**Deliverable:** Trained DPO model. GitHub commit `week-43-dpo`.

**Acceptance criteria:** You can derive the DPO loss on paper.

> **What I will NOT do for you:** the derivation. The paper's appendix walks through it.

**Compute:** Colab Pro. Budget cumulative: $80.

---

### Week 44 — Building a preference dataset for SQL

**Concepts:** Preference data sources, AI feedback (RLAIF), constitutional AI, synthetic preferences.

**Required reading:**
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — Anthropic
- [RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback](https://arxiv.org/abs/2309.00267)

**Coding project:** Build a 2K-pair preference dataset for SQL:
- For each prompt, generate 2 SQL queries (one from your SFT model, one from base Qwen2.5-Coder-7B)
- Run both against a real Postgres DB
- The one that executes correctly + matches expected output is "chosen"; other is "rejected"
- This is **execution-based preference data** — clean, objective, no humans needed

**Deliverable:** HF dataset `<your-handle>/postgres-sql-preferences-v1`.

> **What I will NOT do for you:** decide which preferences matter. You decide what "good SQL" means and codify it.

**Compute:** Colab Pro.

---

### Week 45 — DPO on your domain model

**Coding project:** Apply DPO to your `postgres-sqlcoder-7b-v1` from Phase 4 using your Week 44 preference data. Use Unsloth's DPO trainer.

**Deliverable:** `<your-handle>/postgres-sqlcoder-7b-v2-dpo`. Eval report comparing v1 (SFT only) vs. v2 (SFT+DPO).

**Acceptance criteria:** v2 outperforms v1 on execution-correctness eval.

> **What I will NOT do for you:** debug your DPO loss going negative. Read the TRL forum.

**Compute:** Colab Pro. Budget cumulative: $90.

---

### Week 46 — GRPO and RLVR (the 2025 breakthrough)

**Concepts:** Group Relative Policy Optimization, RL with Verifiable Rewards, why DeepSeek-R1 changed everything.

**Required reading:**
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning](https://arxiv.org/abs/2402.03300) — introduces GRPO
- [DeepSeek-R1: Incentivizing Reasoning Capability via RL](https://arxiv.org/abs/2501.12948) — **read fully, this is THE paper of 2025**
- HuggingFace [GRPO Trainer docs](https://huggingface.co/docs/trl/grpo_trainer)
- HuggingFace [LLM Course Chapter 12: Implementing GRPO](https://huggingface.co/learn/llm-course/chapter12/4)

**Required watching:**
- Karpathy [LLM Year in Review 2025 talk](https://www.youtube.com/watch?v=Zaf218pEb-8) — the RLVR section
- Yannic Kilcher [DeepSeek-R1 explained](https://www.youtube.com/results?search_query=yannic+kilcher+deepseek+r1)

**Coding project:** Read and annotate TRL's `GRPOTrainer` source.

**Deliverable:** Annotated source notes + a 2-page Markdown explainer in your own words.

> **What I will NOT do for you:** explain why GRPO works without a critic network. Read DeepSeekMath section 3.

**Compute:** None.

---

### Weeks 47–48 — GRPO with executable rewards on SQL

**Coding project (2 weeks):** This is the **most novel part** of your curriculum. Apply GRPO to your SQL model with a reward function that:
- Generates K candidate SQL queries (group size = 8 or 16)
- Executes each against Postgres
- Reward = 1 if execution-correct, 0 if syntax error, partial credit for type-correct-but-wrong-row-count
- Uses Unsloth's GRPO (5GB VRAM!)

**Deliverable:** `<your-handle>/postgres-sqlcoder-7b-v3-grpo`. **This is the model that will likely beat GPT-4 on your domain.**

**Acceptance criteria:** Execution-correctness on held-out test set increases meaningfully (>5pp) over v2.

> **What I will NOT do for you:** design the reward function. **You** are the domain expert. Decide what makes good SQL.

**Compute:** Colab Pro + RunPod A100 (~5 hours, $10). Budget cumulative: $115.

---

### Week 49 — KTO, ORPO, and the alignment zoo

**Required reading (skim):**
- [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306)
- [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691)
- [SimPO](https://arxiv.org/abs/2405.14734)

**Deliverable:** A 1-page comparison table.

> **What I will NOT do for you:** make you try every method. Use this week to consolidate.

**Compute:** None.

---

### Week 50–51 — Iteration: improving v3

Run experiments. Try things. Use these 2 weeks to fix bugs, expand the dataset, retry with different reward shapings.

**Deliverable:** Your best model so far. Eval report.

**Compute:** Colab Pro + RunPod (~5 hours, $10). Budget cumulative: $135.

---

### Week 52 — Phase 5 Gate

**Phase Gate:**
- [ ] You can explain DPO, PPO, and GRPO mathematically
- [ ] You have applied SFT, DPO, and GRPO to your domain model in sequence
- [ ] Your v3 model beats your v1 model on held-out eval
- [ ] You have run at least one GRPO run with executable rewards

---

# PHASE 6 — Capstone, Quantization, Deployment, Paper (Months 14–18, 22 weeks)

**Phase goal:** Take your fine-tuned model, polish it to "small model beats trillion-param generalist on this niche" quality, quantize it for deployment, deploy it, and write up the entire project as a public technical report.

---

### Weeks 53–56 — Dataset v3: scale to 50K high-quality examples

**Coding project (4 weeks):**
- **Week 53:** Read [Less is More for Alignment (LIMA)](https://arxiv.org/abs/2305.11206), [Tülu 3 paper](https://arxiv.org/abs/2411.15124). Decide on data quality strategy.
- **Week 54:** Synthesize 30K SQL pairs using a strong teacher (GPT-4o or Claude API, ~$30 outside the $200 budget — optional but recommended). Use [Magpie-style self-instruct](https://arxiv.org/abs/2406.08464) or [Genie-style](https://arxiv.org/abs/2401.14367) approaches.
- **Week 55:** Filter aggressively. Run LLM-as-judge on every example. Reject anything that doesn't execute correctly.
- **Week 56:** Add 5K **conversational multi-turn** SQL examples (CoSQL/SParC style). Build domain-specific evals: TimescaleDB time-series queries, hyperfunctions, continuous aggregates.

**Deliverable:** `<your-handle>/postgres-sql-v3` — 50K cleaned examples.

> **What I will NOT do for you:** spend your money. Synthetic data generation is the optional upsell — you can also achieve a strong result with 15K hand-curated examples.

**Compute:** Colab Pro. Budget cumulative: $135.

---

### Weeks 57–60 — Final training pipeline: SFT → DPO → GRPO

**Coding project (4 weeks):**
- **Week 57:** Continued pretraining on a 100M-token corpus of PostgreSQL/TimescaleDB docs and Stack Overflow. Use Unsloth on RunPod H100 (~3 hrs, $5).
- **Week 58:** Full SFT on 50K v3 dataset.
- **Week 59:** DPO on 5K refreshed preference dataset.
- **Week 60:** GRPO with executable rewards.

**Deliverable:** `<your-handle>/postgres-sqlcoder-7b-final` — your capstone model.

**Compute:** RunPod H100 (~10 hrs, $20) + Colab Pro. Budget cumulative: $165.

---

### Week 61–62 — Comprehensive evaluation harness

**Concepts:** SQL-specific evaluation, BIRD-SQL benchmark, hand-built TimescaleDB eval, head-to-head vs. frontier models.

**Required reading:**
- [BIRD: A Big Bench for Large-Scale Database Grounded Text-to-SQL](https://arxiv.org/abs/2305.03111)
- [Spider 2.0 benchmark](https://spider2-sql.github.io/)
- [Defog SQLCoder eval framework](https://github.com/defog-ai/sql-eval)

**Coding project:** Build a comprehensive eval suite:
- BIRD-SQL execution accuracy
- Spider execution accuracy
- Your custom 200-example PostgreSQL/TimescaleDB benchmark
- Head-to-head: your model vs. base Qwen2.5-Coder-7B vs. GPT-4o vs. Claude 3.5 vs. SQLCoder vs. DeepSeek-Coder-V2

**Deliverable:** Public eval results. GitHub repo with reproducible eval scripts.

> **What I will NOT do for you:** pay for GPT-4o API calls. Budget ~$10–20 for the eval comparison (outside the $200 ceiling).

**Compute:** Colab Pro. Budget cumulative: $175.

---

### Week 63–64 — Quantization deep dive for deployment

**Concepts:** GGUF format, llama.cpp, GPTQ, AWQ, choosing the right quantization for the right use case.

**Required reading:**
- [Complete LLM Quantization Comparison: GPTQ, AWQ, GGUF](https://www.youngju.dev/blog/llm/2026-03-06-llm-quantization-gptq-awq-gguf-comparison.en)
- [llama.cpp documentation](https://github.com/ggerganov/llama.cpp)
- [Unsloth saving notebooks (GGUF export)](https://github.com/unslothai/unsloth/wiki)

**Coding project:** Quantize your final model to:
- GGUF Q4_K_M (for local Mac/CPU inference via llama.cpp/Ollama)
- AWQ INT4 (for vLLM/cloud GPU)
- GPTQ INT4 (alternative GPU)

Measure perplexity, exact-match accuracy, and inference throughput at each quantization.

**Deliverable:** Three quantized variants pushed to HuggingFace. Comparison table.

**Acceptance criteria:** GGUF Q4_K_M version runs on your Mac via Ollama at ≥15 tok/sec.

> **What I will NOT do for you:** install llama.cpp. Build it from source on your Mac yourself.

**Compute:** Mac + Colab Pro.

---

### Week 65–66 — Deployment: local + cloud

**Coding project:**
- Deploy GGUF version locally via Ollama. Build a CLI tool that lets you ask your model PostgreSQL questions from your terminal.
- Deploy AWQ version on vLLM (RunPod, $5 of testing time). Expose as OpenAI-compatible API.
- Optional: build a small Streamlit/Gradio web app on HuggingFace Spaces (free).

**Deliverable:** Working local CLI + working public Spaces demo.

**Compute:** Mac + RunPod (~3 hrs, $5). Budget cumulative: $180.

---

### Weeks 67–70 — Write the technical report

**Project (4 weeks):** Write a 15–25 page technical report (Markdown → PDF or arXiv-style LaTeX) covering:
- Problem statement
- Dataset construction methodology
- Training pipeline (continued pretraining → SFT → DPO → GRPO)
- Architecture choices (why Qwen2.5-Coder-7B as base)
- Evaluation methodology + results table
- Ablations (what helped, what didn't)
- Limitations + future work
- Reproducibility appendix

**Deliverable:** `report.pdf` published on your GitHub. **Optional but encouraged: post on HuggingFace blog, LinkedIn, X/Twitter.**

> **What I will NOT do for you:** write the report. This is the artifact that makes you hireable.

---

### Weeks 71–74 — Reading frontier research & directions

Use these 4 weeks to read ahead and stay current. Recommended reading list:
- [Tülu 3](https://arxiv.org/abs/2411.15124) — fully open post-training recipe
- [SmolLM2](https://huggingface.co/blog/smollm) — small models done right
- [OLMo 2](https://arxiv.org/abs/2501.00656) — fully open frontier
- Latest DeepSeek papers
- Latest Qwen technical reports
- [Anthropic's interpretability work](https://transformer-circuits.pub/) — for breadth
- [LongRoPE](https://arxiv.org/abs/2402.13753), [YaRN](https://arxiv.org/abs/2309.00071) — context extension, if relevant

**Deliverable:** A weekly reading log. By Week 74, you should have read ~20 frontier papers across the curriculum.

> **What I will NOT do for you:** keep your reading list current. Subscribe to [AlphaSignal](https://alphasignal.ai/), [The Batch](https://www.deeplearning.ai/the-batch/), [Sebastian Raschka's Substack](https://magazine.sebastianraschka.com/).

---

### Weeks 75–78 — Iteration, polish, and v2 capstone

Optional fourth-month iteration window. Use it to:
- Try a different base model (Llama 3.1 8B, Gemma 2 9B, DeepSeek-Coder-V2-Lite)
- Try multi-turn / agentic SQL (model + tool use)
- Add a Tamil-language layer (your dataset becomes bilingual NL→SQL)
- Build a real product around it

**Final Phase 6 Gate:**
- [ ] You have a final model that beats GPT-4o on your held-out PostgreSQL/TimescaleDB benchmark
- [ ] Quantized versions (GGUF, AWQ, GPTQ) pushed to HuggingFace
- [ ] Local Ollama deployment working
- [ ] Public Spaces demo working
- [ ] 15–25 page technical report published
- [ ] You have read and can discuss ≥30 papers
- [ ] You can implement DPO and GRPO from scratch

---

## Final budget recap

| Category | Spend |
|---|---|
| Colab Pro (10 months @ $10) | $100 |
| RunPod A100/H100 (~30 hrs total) | $80 |
| **Total** | **$180** |
| Buffer remaining | **$20** |

Optional outside-budget spend for synthetic data generation (GPT-4o/Claude API): ~$30–50. This is the highest-ROI add-on if you have it.

---

## Skills you will have at Month 18

- Implement a transformer (with RoPE, GQA, RMSNorm, SwiGLU, KV cache) from scratch in PyTorch
- Pretrain a small LM end-to-end
- Fine-tune any 7B–9B HuggingFace model with QLoRA on a single 24GB GPU
- Apply SFT → DPO → GRPO pipelines
- Build domain datasets (curated + synthetic + preference)
- Build execution-based eval harnesses
- Quantize and deploy models for production
- Read and reproduce modern LLM papers
- A public capstone project that proves all of the above

---

## Things explicitly **NOT** covered (and why)

- **Multimodal models (vision-language)** — out of scope, would require 6+ extra weeks
- **MoE architecture training** — read about it (in DeepSeek/Qwen papers) but don't train one ($$$)
- **Distributed multi-node training** — conceptual only; you don't need it for fine-tuning roles
- **Agent frameworks (LangChain, LlamaIndex, etc.)** — different skill set; learn separately
- **CUDA kernel writing** — Triton/CUDA programming is a 6-month curriculum on its own
- **Full RL course** — only what serves LLMs

If after Month 18 you want any of these, you'll have the foundation to self-teach in 2–4 weeks each.

---

## How to use this document

1. **Print Phase 1.** Stick it on your wall.
2. Each Sunday: review the upcoming week's module. Block 6–8 hrs in your calendar.
3. Each weekday morning (you wake at 5 AM): 30–60 minutes of reading.
4. Each weekend: bigger coding/training session.
5. Every Sunday evening: update `journal.md`, commit code, push to GitHub.
6. Every Phase Gate: be honest. If you fail, repeat. Don't move forward weak.
7. Once a month: write a public LinkedIn/Twitter post on what you learned. **This is non-negotiable.** Public learning compounds.

---

## Single most important advice

The temptation will be to skip the foundational weeks (Phase 1 PyTorch, Phase 2 hand-coded transformer) and jump to "the cool stuff" (Phase 4–5 fine-tuning). **Resist this with every fiber of your being.**

People who skip foundations end up as "Hugging Face script kiddies" — they can copy a fine-tuning notebook but can't debug when it breaks, can't read papers, and can't innovate. People who do the foundations become the engineers labs hire to actually build models.

You said Naval said "fine-tuning a model is the new way of coding." He's right. But just as great coders know assembly even if they write Python, great fine-tuners know the transformer down to the matmul. Don't skip the matmul.

Good luck. See you in 18 months with a model that beats GPT-4 on PostgreSQL.

---

*Curriculum prepared April 26, 2026. Last verified resources April 2026. If a link breaks, search the title — these are durable resources.*
