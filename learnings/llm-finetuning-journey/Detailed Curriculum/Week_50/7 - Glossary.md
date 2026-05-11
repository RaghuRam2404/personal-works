# Week 50 Glossary

**Iteration log**: A structured document recording hypothesis, experiment design, results, and analysis for each training experiment; the primary artifact of iteration weeks.

**Hypothesis-driven iteration**: The practice of always formulating a specific, falsifiable hypothesis before running a new experiment; prevents wasted compute and provides insight even when the experiment fails.

**Zone of proximal development (GRPO)**: Prompts where the current model succeeds 20–60% of the time; these produce the most useful gradient signal because the group of K completions has genuine variance.

**Controlled experiment**: An experiment that changes exactly one variable at a time, allowing attribution of any change in metrics to the specific change made.

**Patch from checkpoint**: Resuming GRPO training from an existing model checkpoint (e.g., v3) rather than starting from v1 or v2; preserves improvements already made and focuses the new training on specific gaps.

**Ablation study**: Running an experiment where you revert one change and keep another, to isolate which change caused an observed effect.

**Aggregation query**: A SQL query that uses GROUP BY, COUNT, SUM, AVG, MIN, MAX, or similar aggregate functions; often requires the model to understand that SELECT * is wrong and a specific projection is needed.

**Reward level demotion**: Reducing the reward value for a specific outcome to make it less attractive as a reward hacking target (e.g., changing "executes unverified" from 0.2 to 0.05).

**Entropy bonus**: A regularization term in the RL loss that rewards policy entropy (diversity of outputs), preventing mode collapse; used in DAPO and other production RL systems.

**clip-higher trick (DAPO)**: Asymmetric PPO clipping where the upper clip bound is larger than the lower clip bound (e.g., 1+3ε vs 1-ε), allowing larger probability increases for high-advantage actions.
