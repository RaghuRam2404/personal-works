# Week 71 Quiz — Frontier Reading: Tulu 3, SmolLM2, OLMo 2

## Multiple Choice

**Q1.** Tulu 3 introduces RLVR (Reinforcement Learning with Verifiable Rewards). How does RLVR differ from classical RLHF (Reinforcement Learning from Human Feedback)?

A. RLVR uses human preference labels; RLHF uses automated rewards
B. RLVR uses automated verification (code execution, math equality) as rewards; RLHF uses a learned reward model trained on human preferences
C. RLVR is only applicable to math tasks; RLHF is domain-general
D. RLVR eliminates the need for any preference data; RLHF requires millions of preference labels

**Q2.** SmolLM2-1.7B is reported to outperform several 3B models on targeted benchmarks despite being half the parameter count. What is the most important factor driving this result?

A. SmolLM2 uses a deeper but narrower architecture that is more efficient
B. SmolLM2 uses flash attention 3.0 which is not available to 3B models
C. SmolLM2 was trained on higher-quality curated data (FineWeb-Edu, DCLM) resulting in better sample efficiency than the comparison models' noisier training data
D. SmolLM2 uses a longer context window that allows it to process more information per inference step

**Q3.** OLMo 2 uses a mid-training phase that upweights high-quality data sources (StackExchange, code, Wikipedia) after the main pretraining phase. What does this strategy most directly parallel in your postgres-sqlcoder-7b pipeline?

A. Your GRPO training stage, which uses verifiable rewards
B. Your CPT stage, which trains on PostgreSQL-domain text after the base model's original pretraining
C. Your DPO stage, which uses preference data
D. Your LLM-as-judge filtering step

**Q4.** All three papers (Tulu 3, SmolLM2, OLMo 2) publish intermediate training checkpoints. From a scientific standpoint, what is the primary benefit of publishing intermediate checkpoints that is NOT provided by publishing only the final model?

A. Users can choose a smaller model if they have less compute
B. Researchers can study capability emergence and attribute specific abilities to specific training stages
C. Intermediate checkpoints are faster to download
D. Legal compliance requires multiple checkpoint versions

## Short Answer

**Q5.** Tulu 3 reports that 10K high-quality preference pairs outperform 100K noisy preference pairs in DPO. How does this finding validate or challenge the choices you made in Week 59 (DPO on 5K curated pairs)?

**Q6.** OLMo 2's fully open stack includes training code, training data, intermediate checkpoints, and evaluation code. Your postgres-sqlcoder-7b release includes model weights and training code. What is missing from your release that OLMo 2 includes, and what scientific benefit would it provide?

**Q7.** SmolLM2 demonstrates that tokenizer efficiency matters for small models. Explain why a SQL model's tokenizer efficiency specifically affects inference speed on production SQL queries.

## Deep Scenario

**Q8.** You read Tulu 3 and want to apply on-policy data generation to improve postgres-sqlcoder-7b. Specifically, you plan to:
1. Use the current model to generate 10K SQL completions for new schema+question pairs
2. Execute each completion against a test database
3. Add the passing completions back to the training set and re-run SFT

A skeptical colleague says: "This will cause your model to become more confident in its own errors and collapse." Write a 150-word response that (a) names the phenomenon your colleague is describing, (b) proposes two specific safeguards to prevent it, and (c) cites evidence from Tulu 3 or another paper that on-policy generation can be done safely.
