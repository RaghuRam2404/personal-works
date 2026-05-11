# Week 73 Overview — Frontier Reading 3: Anthropic Interpretability

This file is your map for Week 73. Read it first; everything else fits inside it.

## The story this week

At the end of 60 weeks of training, you have a model that achieves 83.1% accuracy. The 16.9% of failures are not random — there are systematic patterns (35% involve `time_bucket`, 26% involve wrong JOIN types). Interpretability research asks: can we look inside the model's weights and activations to understand why these failures occur? Can we find the internal representations that cause schema hallucination?

## What you need to do

- [ ] Read access to transformer-circuits.pub (browser, no login required)
- [ ] `pip install transformer_lens circuitsvis` in your environment
- [ ] Your merged BF16 model loaded locally (or a small proxy model like GPT-2 for fast experimentation)
- [ ] One failed SQL example selected from your Custom-200 benchmark (specifically a schema hallucination failure)

Concretely, by the end of the week you should be able to:

- Explain the Superposition Hypothesis and why it predicts that neural networks represent more features than they have neurons
- Describe how sparse autoencoders (SAEs) are used to find interpretable features in LLM activations
- Relate interpretability findings to your SQL model's failure modes (schema hallucination, wrong JOIN type)
- Evaluate one concrete interpretability technique you could apply to debug your model's behavior
- Assess where interpretability is currently useful vs where it is still too immature for practical debugging

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

- 1.5h: Read "Toy Models of Superposition" (transformer-circuits.pub)
- 1.5h: Read "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning"
- 1.0h: Read "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet"
- 1.0h: Explore TransformerLens: run a simple attention visualization on one of your SQL failures
- 1.0h: Write synthesis notes and applicability assessment for your SQL model

## Why this week matters

Interpretability tools can tell you where a failure happens; they cannot yet reliably tell you how to fix it. Use them for diagnosis.
