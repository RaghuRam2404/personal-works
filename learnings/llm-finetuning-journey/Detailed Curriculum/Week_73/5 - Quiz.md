# Week 73 Quiz — Anthropic Interpretability

## Multiple Choice

**Q1.** The Superposition Hypothesis predicts that a neural network with 4,096 neurons can represent significantly more than 4,096 features. What key property of natural data enables this?

A. Natural data is redundant — most inputs are near-duplicates of other inputs
B. Natural data is sparse — in any given input, only a small fraction of all possible features are simultaneously active
C. Natural data is low-rank — the covariance matrix of features is approximately rank-512
D. Natural data is normalized — all feature activations are bounded in [0, 1]

**Q2.** A sparse autoencoder (SAE) is trained on a language model's residual stream activations with a sparsity penalty (L1 on the feature activations). After training, a researcher finds that feature 3,247 activates strongly on: Python `def` keywords, SQL `CREATE FUNCTION` statements, and JavaScript `function` declarations. This feature is described as:

A. Polysemantic — it responds to multiple unrelated concepts and should be split
B. Monosemantic — it represents the abstract concept of "function definition" across programming languages
C. Superposed — it is a linear combination of three separate features and the SAE failed to separate them
D. A noise feature — activating on multiple languages indicates the SAE was undertrained

**Q3.** You use activation patching to test whether a specific attention head (layer 15, head 4) is responsible for your model's schema hallucination. You patch the attention values at the schema position from a correct example into a wrong example and find the hallucination disappears. What does this result establish?

A. Head 15.4 is sufficient for correct schema reading
B. Head 15.4 is necessary and sufficient for correct schema reading
C. Patching the activations from a correct example establishes that head 15.4 causally contributes to schema reading in this example
D. The model has memorized the correct schema from the training data

**Q4.** The "logit lens" technique projects the residual stream at each layer through the unembedding matrix to see intermediate predictions. You apply it to a hallucination failure and find the model predicts the correct column name through layers 1–18 but switches to the wrong name at layer 19. What does this suggest?

A. Layers 1–18 are irrelevant; layer 19 contains all SQL knowledge
B. The MLP or attention mechanism at layer 19 overwrites the correct prediction; investigate layer 19's contribution to this failure
C. The residual stream is corrupted at layer 19 due to a training bug
D. The model needs more layers to correctly predict column names

## Short Answer

**Q5.** Explain why high attention weight on a schema token does not prove that the attention head causally uses that token to generate the next SQL token. What technique establishes causality instead?

**Q6.** Your model hallucinates column names 21% of the time on your failure analysis. Using the interpretability concepts from this week, propose a two-step investigation to understand this failure mechanically.

**Q7.** Anthropic found that some SAE features in Claude 3 Sonnet correspond to the concept of "deception." Analogously, what features would you most want to find and suppress in a SQL model to improve safety for automated SQL execution?

## Deep Scenario

**Q8.** You have identified (via logit lens) that your SQL model commits to a wrong JOIN type at layer 22 of 32. Your options are: (a) add more JOIN examples to training data and re-train, (b) use activation steering to suppress the wrong JOIN feature at inference time, or (c) add a post-processing SQL validator that catches wrong JOIN types and re-runs generation.

Write a 200-word analysis: (a) evaluate each option's reliability, implementation complexity, and compute cost, and (b) recommend one primary approach for your specific situation (7B model, production SQL assistant, limited compute budget for retraining).
