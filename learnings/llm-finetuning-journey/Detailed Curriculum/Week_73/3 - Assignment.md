# Week 73 Assignment — Frontier Reading: Anthropic Interpretability

## Setup Checklist

- [ ] Read access to transformer-circuits.pub (browser, no login required)
- [ ] `pip install transformer_lens circuitsvis` in your environment
- [ ] Your merged BF16 model loaded locally (or a small proxy model like GPT-2 for fast experimentation)
- [ ] One failed SQL example selected from your Custom-200 benchmark (specifically a schema hallucination failure)

## Task 1: Superposition and SAE Reading

**Goal:** Understand the theoretical foundation of modern interpretability.

**Requirements:**
- [ ] Read "Toy Models of Superposition" (transformer-circuits.pub/2022/toy_model): focus on Sections 1–3 (superposition, sparsity, geometry)
- [ ] Read "Towards Monosemanticity" (transformer-circuits.pub/2023/monosemanticity): focus on the method (SAE training) and examples (Section 3)
- [ ] Write `reading_notes/interpretability_notes.md`:
  - In 3 sentences: what is superposition and why does it make neurons uninterpretable?
  - In 3 sentences: how does an SAE recover interpretable features?
  - In 2 sentences: what is "monosemanticity" and why is it the goal?
- [ ] Identify one SAE feature from "Towards Monosemanticity" that has a plausible analogue in a SQL model (e.g., "this feature activates on function call syntax" → could apply to SQL function calls)

**Deliverable:** `reading_notes/interpretability_notes.md` (300–500 words)

## Task 2: Attention Visualization on a SQL Failure

**Goal:** Apply a practical interpretability tool to understand one SQL failure.

**Requirements:**
- [ ] Select one of your Custom-200 failures where the model generated a wrong table name or hallucinated a column
- [ ] Load the model in TransformerLens (or use GPT-2 as a proxy if your model is too large):
  ```python
  from transformer_lens import HookedTransformer
  model = HookedTransformer.from_pretrained("gpt2")  # or your model
  logits, cache = model.run_with_cache(your_prompt)
  ```
- [ ] Visualize the attention pattern at the last layer when generating the hallucinated token
- [ ] Identify: which input tokens does the model attend to when generating the wrong output?
- [ ] Save the visualization as `reading_notes/attention_viz.png`
- [ ] Write 3 sentences in your notes: what does the attention pattern show, and what does it suggest about the failure mechanism?

**Deliverable:** `reading_notes/attention_viz.png` + analysis in `reading_notes/interpretability_notes.md`

## Task 3: Logit Lens Analysis

**Goal:** Understand at which layer the model "commits" to a wrong prediction.

**Requirements:**
- [ ] Using your TransformerLens cache from Task 2, apply the logit lens:
  ```python
  # Project residual stream at each layer through the unembedding matrix
  for layer in range(model.cfg.n_layers):
      residual = cache[f"blocks.{layer}.hook_resid_post"]
      logits = model.unembed(model.ln_final(residual))
      top_token = logits[0, -1].argmax()
      print(f"Layer {layer}: {model.tokenizer.decode(top_token)}")
  ```
- [ ] Record: at which layer does the model first predict the wrong token (hallucinated column name or wrong JOIN)?
- [ ] Compare: at which layer does a correct example's prediction stabilize?
- [ ] Write 2–3 sentences in your notes interpreting the layer attribution

**Deliverable:** Layer attribution analysis in `reading_notes/interpretability_notes.md`

## Task 4: Applicability Assessment

**Goal:** A clear-eyed assessment of what interpretability can and cannot currently do for your SQL model.

**Requirements:**
- [ ] Write `reading_notes/interpretability_assessment.md` (300–400 words) covering:
  - What interpretability can currently tell you about your model's SQL failures (be specific about which techniques and what findings they would produce)
  - What interpretability cannot currently do (what questions remain unanswerable)
  - One interpretability finding from "Scaling Monosemanticity" (Claude 3 Sonnet) that you would most want to apply to your model, and why
  - Whether you would recommend running a full SAE on your model given your compute budget and the current maturity of the technique

**Deliverable:** `reading_notes/interpretability_assessment.md`

## Stretch Goals

- Train a tiny SAE on GPT-2's residual stream activations on a SQL dataset using the open-source SAE training code (EleutherAI/sae); visualize the top-10 activating examples for 5 learned features
- Read "In-context Learning and Induction Heads" and explain in writing whether your model's schema-reading behavior is implemented by induction heads
- Write a 200-word section for your technical report Appendix describing what attention visualization reveals about your model's schema-reading mechanism
