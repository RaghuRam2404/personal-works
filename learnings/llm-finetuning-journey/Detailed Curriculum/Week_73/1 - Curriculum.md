# Week 73 — Frontier Reading 3: Anthropic Interpretability

## Learning Objectives

By the end of this week, you will be able to:

- Explain the Superposition Hypothesis and why it predicts that neural networks represent more features than they have neurons
- Describe how sparse autoencoders (SAEs) are used to find interpretable features in LLM activations
- Relate interpretability findings to your SQL model's failure modes (schema hallucination, wrong JOIN type)
- Evaluate one concrete interpretability technique you could apply to debug your model's behavior
- Assess where interpretability is currently useful vs where it is still too immature for practical debugging

## Concepts

### Why Interpretability Matters for Your SQL Model

At the end of 60 weeks of training, you have a model that achieves 83.1% accuracy. The 16.9% of failures are not random — there are systematic patterns (35% involve `time_bucket`, 26% involve wrong JOIN types). Interpretability research asks: can we look inside the model's weights and activations to understand why these failures occur? Can we find the internal representations that cause schema hallucination?

This is no longer a purely academic question. Anthropic's interpretability team has found that LLMs represent human-interpretable features (emotions, facts, syntax) in their activations, and that manipulating these representations produces predictable behavioral changes. For your SQL model, the relevant question is: do specific features correspond to "which table is being joined" or "this is a time-bucket query"?

### The Superposition Hypothesis

The Superposition Hypothesis (Elman 1990, revived by Anthropic 2022) states that neural networks learn to represent more features than they have neurons because natural data is sparse — in any given input, only a small fraction of all possible features are active. The model can pack multiple features into each neuron direction by exploiting sparsity: as long as two features are rarely active simultaneously, they can share neuron space without much interference.

Mathematically: if a model has d neurons and represents f features where f >> d, it stores each feature as a direction in d-dimensional space (not aligned with any single neuron). Each direction is a superposition of many features. This explains why individual neuron activation is hard to interpret — a single neuron participates in hundreds of features.

For your SQL model: the attention heads and FFN neurons are not "the JOIN head" or "the table-name neuron." The representation of JOIN knowledge is distributed across many directions in many layers.

### Sparse Autoencoders (SAEs) for Feature Discovery

Sparse Autoencoders (SAEs) are a practical tool for interpreting LLM internals. The idea: train a separate autoencoder with a much larger dictionary (e.g., 16K or 65K directions) to reconstruct the model's activations, with a sparsity penalty that forces each reconstruction to use only a few dictionary elements at a time.

```
h = model_activation(x)      # d-dimensional activation
f = ReLU(W_enc @ h + b_enc)  # k-dimensional sparse feature vector (k >> d)
h_hat = W_dec @ f + b_dec    # reconstructed activation
Loss = ||h - h_hat||^2 + λ * ||f||_1  # reconstruction + sparsity
```

The resulting dictionary elements W_dec[:, i] are interpretable directions in activation space — each one activates on a specific, human-interpretable pattern. Anthropic's research found SAE features corresponding to: names of specific people, programming language constructs, emotional valence, and geographical concepts.

For SQL: you could train an SAE on your model's residual stream activations over SQL examples and look for features that activate on `time_bucket` queries, JOIN clauses, or schema names. This could explain which model components are responsible for your 35% `time_bucket` failure rate.

### Circuits: How Information Flows

The Transformer Circuits framework (Elhage et al. 2021) analyzes how information moves through a transformer via "circuits" — patterns of attention heads and MLPs that compute specific functions. The key findings:

Induction heads: a specific two-head circuit that implements in-context learning (pattern copying from earlier in the context). These emerge early in training and are universal across models.

Copy suppression: a mechanism that prevents models from copying input tokens too literally, enabling generalization.

For SQL: the attention mechanism that reads the schema (column names, table names) from the context and uses them to generate SQL is likely implemented by induction-like heads. If the model hallucinates a column name, it is because the schema-reading circuit is failing — either the column name is too far back in the context (context window limitation), or the attention pattern did not attend to the right schema position.

### Practical Interpretability for SQL Model Debugging

Even without running SAEs yourself (which requires significant compute), you can apply lightweight interpretability tools:

Attention visualization: use `bertviz` or `TransformerLens` to visualize which schema tokens your model attends to when generating a wrong SQL query.

```python
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("your-model")
logits, cache = model.run_with_cache(prompt)
# cache["blocks.15.attn.hook_pattern"] — attention patterns at layer 15
```

Activation patching: replace the activations at a specific position with those from a different example to test counterfactuals. If patching the attention values at the table-name position from a correct example to a wrong-table example causes schema hallucination, you have localized the failure mechanism.

Logit lens: examine what the model predicts at each layer by projecting residual stream states through the unembedding matrix. This shows which layer "commits" to the wrong JOIN type.

## Connections

Interpretability is not directly in your training pipeline, but it connects to failure mode analysis (Week 69) and informs what to fix in iteration weeks (75–77). If interpretability reveals that `time_bucket` failures are due to attention not reaching the schema tokens when the schema is long, the fix is a schema compression preprocessing step (not more training). If the failure is due to a wrong internal feature representation, the fix is more training data.

## Common Misconceptions / Pitfalls

The most common mistake is expecting interpretability to provide causal control — "I found the feature that causes hallucination, so I can delete it." Feature suppression in SAEs is a promising research direction but is not reliable enough for production use. Interpretability is currently most useful for diagnosis, not treatment.

Do not confuse attention weights with attention contribution. High attention weight on a token does not mean that token is causally important — it just means the attention pattern is high. Use activation patching or attribution methods to establish causality.

## Time Allocation (6–8 hours)

- 1.5h: Read "Toy Models of Superposition" (transformer-circuits.pub)
- 1.5h: Read "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning"
- 1.0h: Read "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet"
- 1.0h: Explore TransformerLens: run a simple attention visualization on one of your SQL failures
- 1.0h: Write synthesis notes and applicability assessment for your SQL model
