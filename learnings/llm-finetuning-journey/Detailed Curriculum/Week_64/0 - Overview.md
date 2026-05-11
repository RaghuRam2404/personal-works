# Week 64 Overview — Quantization Part 2: Quantize Your Final Model

This file is your map for Week 64. Read it first; everything else fits inside it.

## The story this week

Last week you compared GGUF, GPTQ, and AWQ theoretically. This week you execute the full pipeline on `postgres-sqlcoder-7b-final`, the model you trained over the previous 12 weeks. There is a specific order of operations that matters.

## What you need to do

- [ ] `postgres-sqlcoder-7b-final` merged BF16 checkpoint available locally (14–15 GB)
- [ ] llama.cpp cloned and built: `cmake -B build && cmake --build build -j8 --config Release`
- [ ] `autoawq` installed: `pip install autoawq`
- [ ] `auto-gptq` installed: `pip install auto-gptq optimum`
- [ ] Hugging Face account with `huggingface-cli login` completed
- [ ] Calibration dataset prepared: 512 SQL instruction strings from your v3 training set
- [ ] 50 GB free disk space (F16 GGUF + three quantized variants)

Concretely, by the end of the week you should be able to:

- Export `postgres-sqlcoder-7b-final` to three quantized formats: Q4_K_M GGUF, AWQ INT4, and GPTQ INT4
- Run perplexity evaluation on each variant using llama.cpp's `perplexity` binary and a held-out SQL corpus
- Measure end-to-end inference throughput (tokens/second) and memory footprint for each format
- Diagnose accuracy degradation by running your 200-example custom benchmark on each quantized model
- Push all three variants to Hugging Face Hub under a consistent naming convention

## Suggested order through the files

1. **1 - Curriculum.md** — read first. Concepts, connections, common pitfalls.
2. **2 - Resources.md** — papers, videos, blog posts, repos, docs to consult while studying.
3. **3 - Assignment.md** — the hands-on task you must complete this week.
4. **4 - AssignmentSolutions.md** — only after you have attempted the assignment yourself.
5. **5 - Quiz.md** — self-test that you have absorbed the week's material.
6. **6 - Answers.md** — quiz answers and explanations; check after attempting.
7. **7 - Glossary.md** — terminology used this week, indexed for fast lookup.
8. **8 - TakeAway.md** — the one-page summary you keep for the rest of the course.

## Time budget

- 0.5h: Merge adapter, verify BF16 model directory is complete
- 1.5h: GGUF conversion and Q4_K_M quantization + perplexity check
- 1.5h: AWQ quantization with SQL calibration data
- 1.5h: GPTQ quantization
- 1.0h: Evaluation harness across all three variants, build comparison table
- 0.5h: Write model cards, push all three to Hub

## Why this week matters

Merge first, quantize second, benchmark third, then push — in that exact order, every time.
