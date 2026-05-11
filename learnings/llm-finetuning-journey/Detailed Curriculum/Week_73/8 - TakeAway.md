# Week 73 TakeAway — Anthropic Interpretability

Interpretability tools can tell you where a failure happens; they cannot yet reliably tell you how to fix it. Use them for diagnosis.

## Core Concepts

```
Superposition: d neurons represent f >> d features because data is sparse
SAE: train a wide (k >> d) autoencoder with L1 sparsity → monosemantic features
Logit lens: project residual stream at each layer → see when prediction commits
Attention patching: replace head activations from one example → establish causality
Activation steering: add a direction to residual stream to shift model behavior
```

## Practical Tools

```python
# TransformerLens — run with cache
from transformer_lens import HookedTransformer
model = HookedTransformer.from_pretrained("gpt2")
logits, cache = model.run_with_cache(prompt)

# Logit lens at each layer
for layer in range(model.cfg.n_layers):
    resid = cache[f"blocks.{layer}.hook_resid_post"][:, -1, :]
    top = (model.unembed(model.ln_final(resid)))[0].argmax()
    print(f"L{layer}: {model.tokenizer.decode(top)}")

# Attention pattern at layer L, head H
pattern = cache[f"blocks.{L}.attn.hook_pattern"][0, H]  # [seq, seq]
```

## Decision Rules

- If hallucination: visualize attention → check which tokens are attended to at wrong prediction
- If wrong logic (JOIN type, aggregation): apply logit lens → find the layer where prediction diverges
- If you want causal evidence: use activation patching, not attention weight analysis
- Activation steering: use only for research/debugging, not production — too unreliable
- Best fix for failures found via interpretability: add targeted training examples (option a), not surgery

## Numbers to Remember

- Superposition: a model with d=4096 neurons can represent O(d × log(1/sparsity)) features
- SAE dictionary size: typically 4x–64x the model's hidden dimension (e.g., d=4096 → 16K–256K features)
- Logit lens commits to final prediction: typically at 60–80% of total layers for GPT-class models
- Monosemantic threshold: a feature is considered monosemantic if it activates on a coherent concept across ≥80% of its top-activating examples

## Red Flags

- Using attention weights to claim causal understanding: correlation only — use activation patching
- Claiming a feature is monosemantic from 3 examples: inspect the top 20 activating examples
- Trusting activation steering in production: side effects are unpredictable at current maturity
- Running a full SAE before simplifying your hypothesis: start with attention visualization and logit lens
