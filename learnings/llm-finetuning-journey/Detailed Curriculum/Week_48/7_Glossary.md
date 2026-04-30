# Week 48 Glossary

**RunPod**: Cloud GPU provider; A100 instances are the recommended compute for this week's GRPO training (~$2.50–4/hour).

**mean_reward**: Average reward across all completions in the current GRPO batch; the primary metric for tracking training progress.

**reward_std**: Standard deviation of rewards within each group of K completions; near-zero means all completions score similarly, producing zero gradient.

**grad_norm**: The L2 norm of the gradient vector; should stay below 1.0. Values above 5.0 indicate instability.

**max_grad_norm**: A gradient clipping hyperparameter; setting this to 0.5 prevents single-batch gradient explosions from corrupting the model.

**Checkpoint**: A saved state of the model and optimizer at a specific training step; allows training to resume after disconnection.

**screen/tmux**: Unix terminal multiplexers that keep processes running after SSH disconnection; essential for long training runs on cloud instances.

**Mode collapse (GRPO)**: When the model converges to generating nearly identical completions for every prompt, causing reward_std → 0 and stopping learning.

**semantic accuracy gap**: When a model has higher execution accuracy but lower semantic accuracy than a baseline; indicates the model executes SQL but with wrong logic.

**Three-way comparison**: Eval comparing v1 (SFT), v2 (DPO), and v3 (GRPO) on the same held-out test set; the definitive measure of Phase 5 progress.

**Resume from checkpoint**: Restarting GRPO training from a saved checkpoint when a session is interrupted; requires saving both model weights and optimizer state.

**Temperature (GRPO generation)**: Controls diversity of completions during rollout. Higher temperature (0.8–1.0) increases reward_std but may reduce mean_reward; lower temperature (0.5–0.7) produces more consistent but potentially less exploratory outputs.
