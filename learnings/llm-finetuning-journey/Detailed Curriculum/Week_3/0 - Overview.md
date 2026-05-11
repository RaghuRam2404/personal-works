# Week 3 Overview — Convolutional Neural Networks, Backprop Ninja, and WaveNet

This file is your map for Week 3. Read it first; everything else fits inside it.

## The story this week

A convolution applies a filter (kernel) of shape `(C_out, C_in, kH, kW)` to an input of shape `(N, C_in, H, W)` to produce output `(N, C_out, H_out, W_out)`. The key idea is **parameter sharing**: one kernel is slid across the entire spatial dimension, so `C_out * C_in * kH * kW` parameters cover an arbitrarily large input.

## What you need to do

- [ ] Google Colab account active. Open a new notebook, switch runtime to T4 GPU (Runtime → Change runtime type → T4 GPU). Verify: `import torch; print(torch.cuda.is_available())` → `True`.
- [ ] `torchvision` available on Colab: `import torchvision` — pre-installed, should work.
- [ ] W&B set up on Colab: `!pip install wandb -q` and `wandb.login()` with your API key.
- [ ] Karpathy makemore Parts 4 and 5 queued.

Concretely, by the end of the week you should be able to:

- Compute the output shape of any convolutional layer from input shape, kernel size, stride, and padding — in your head, without code.
- Explain parameter sharing in CNNs and why it is the right inductive bias for spatially-structured data.
- Train a CNN on CIFAR-10 in PyTorch and achieve at least 75% test accuracy using Colab's T4 GPU.
- Derive the backpropagation rules for a convolution and a max-pooling layer by hand (the "backprop ninja" skill).
- Explain why WaveNet's dilated causal convolutions solved the long-range dependency problem for audio.
- Recognize the connection from WaveNet's hierarchical dilated convolutions to modern transformer architectures.

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

| Activity | Time |
|---|---|
| Read CS231n ConvNets notes | 1 h |
| Watch makemore Part 4 (Backprop Ninja) — code along | 2 h |
| Watch makemore Part 5 (WaveNet) — code along | 2 h |
| Build CNN on CIFAR-10, train on Colab T4 | 2 h |
| Journal entry + GitHub commit | 30 min |

## Why this week matters

**This week in 15 words:** Conv = parameter sharing over space; backprop ninja; WaveNet = dilated causal hierarchy.
