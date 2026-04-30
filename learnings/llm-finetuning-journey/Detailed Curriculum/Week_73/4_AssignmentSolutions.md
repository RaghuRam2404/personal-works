# Week 73 Assignment Solutions

## Task 1: Superposition and SAE — Key Definitions

Superposition in 3 sentences: Neural networks have fewer neurons than features they need to represent. Because real data is sparse (most features are inactive most of the time), the network can store multiple features in the same neuron direction without severe interference. The result is that single neuron activations are "polysemantic" — they respond to multiple unrelated features — making them uninterpretable in isolation.

SAE recovery in 3 sentences: A sparse autoencoder trains a wider encoder (e.g., 16K directions) to reconstruct the model's d-dimensional activation with a sparsity penalty. The sparsity penalty forces each reconstruction to use only a few dictionary elements, encouraging the dictionary to learn directions that correspond to individual features rather than superpositions. After training, each dictionary element (a direction in the model's activation space) activates on a specific, human-interpretable pattern.

Monosemanticity in 2 sentences: A monosemantic feature is one where a single direction in the SAE dictionary activates strongly and consistently on a specific, interpretable concept (e.g., "base64 encoded text" or "Python function definitions"). The goal of interpretability is to decompose polysemantic neurons into monosemantic features because monosemantic features can be understood, tracked, and potentially manipulated.

## Task 2: TransformerLens Attention Visualization

```python
from transformer_lens import HookedTransformer
import circuitsvis as cv

# Use GPT-2 if your 7B model is too slow
model = HookedTransformer.from_pretrained("gpt2")

# Your SQL prompt (or a proxy)
prompt = """CREATE TABLE orders (id INT, customer_id INT, amount DECIMAL, ts TIMESTAMP);
Question: Find total revenue per customer.
SQL: SELECT customer_id, SUM("""

logits, cache = model.run_with_cache(prompt)

# Visualize last layer attention
attn = cache["blocks.11.attn.hook_pattern"][0]  # [heads, seq, seq]
tokens = model.to_str_tokens(prompt)

# Display with circuitsvis
cv.attention.attention_patterns(tokens=tokens, attention=attn)
```

What to look for: when the model is about to generate the wrong column name, does it attend strongly to the wrong position in the schema? Hallucination often correlates with the model attending to the question tokens (which contain related words) instead of the schema tokens (which contain the actual column names).

## Task 3: Logit Lens

```python
import torch

prompt_tokens = model.to_tokens(prompt)
logits, cache = model.run_with_cache(prompt_tokens)

for layer in range(model.cfg.n_layers):
    resid = cache[f"blocks.{layer}.hook_resid_post"][:, -1, :]  # last position
    # Apply final layer norm and unembed
    normalized = model.ln_final(resid)
    layer_logits = model.unembed(normalized)
    top_k = layer_logits[0].topk(3)
    tokens = [model.tokenizer.decode([t]) for t in top_k.indices]
    print(f"Layer {layer:2d}: {tokens}")
```

Typical finding: early layers predict generic tokens (common SQL keywords). The prediction stabilizes to the correct or wrong specific token (column name, function name) around layers 60–80% of the way through the model (layer 8–10 for GPT-2, layer 20–25 for a 32-layer 7B model). If the prediction commits to the wrong token early and does not change, the failure was established early in the forward pass.

## Task 4: Applicability Assessment Template

What interpretability currently can tell you:
- Attention visualization reveals whether your model reads schema tokens correctly during generation
- Logit lens reveals at which layer specific predictions stabilize
- Activation patching can localize which specific attention heads or MLP layers cause wrong JOIN types

What it cannot currently do:
- Tell you how to fix a failure — knowing that layer 24 MLPs contribute to wrong JOIN selection does not tell you how to change those weights
- Scale to a full interpretability analysis of all failures on a production timeline

Best finding to apply from Scaling Monosemanticity: the "safety-relevant features" finding — Anthropic found SAE features corresponding to deceptive and harmful concepts and showed these could be suppressed via activation steering. The SQL analogue: find the SAE feature corresponding to "hallucinating a column name" and suppress it during inference. This is aspirational for current compute budgets but is the right direction.

## Common Gotchas

- TransformerLens requires the model in its own format; for your Qwen model, you may need to convert via `model = HookedTransformer.from_pretrained("Qwen/Qwen2.5-Coder-7B")` — check if Qwen2.5 is supported.
- Attention visualization alone is insufficient evidence for causal claims. High attention to a token shows correlation; use activation patching for causality.
- Logit lens works best at the final token position; predictions at middle positions are less meaningful.

## How to Verify You Did It Right

Your attention visualization analysis is complete if you can answer: "In the failed example, when generating the hallucinated column name, which 3 input tokens had the highest attention weight in the final layer?" If you can answer this with specific token strings and attention values, you have done the analysis correctly.
