# Week 49 Assignment Solutions

## Task 1 — Complete Comparison Table

| Method | Paper/Year | Ref model | Reward model | Data format | Policy | Memory | Core loss | Best use | Worst case |
|---|---|---|---|---|---|---|---|---|---|
| PPO | Schulman 2017 | Yes (frozen) | Yes | Online rollouts | On-policy | 4× | PPO clip + KL penalty | Any reward, flexible | Expensive; 4 model forward passes |
| DPO | Rafailov 2023 | Yes (frozen) | No | Paired prefs | Off-policy | 2× | -log σ(β·(log_ratio_w − log_ratio_l)) | Human/execution preferences | Stale data; fails on distribution shift |
| GRPO | Shao 2024 | Yes (frozen) | No | Online rollouts + verifier | On-policy | 2× | PPO clip with group-norm advantage | Verifiable rewards (code, SQL, math) | High K cost; bad when all rewards equal |
| KTO | Ethayarajh 2024 | Yes (frozen) | No | Unpaired good/bad | Off-policy | 2× | Prospect theory loss on unpaired labels | Unpaired annotation datasets | Weaker signal than paired DPO |
| ORPO | Hong 2024 | No | No | Paired prefs | Off-policy | 1× | SFT loss + log odds ratio penalty | Memory-constrained, 1-stage training | Weak regularization; model drift |
| SimPO | Meng 2024 | No | No | Paired prefs | Off-policy | 1× | -log σ((avg_logp_w − avg_logp_l) − γ) | Length normalization needed | No KL anchor; may drift from pretrained |

---

## Task 3 — TRL Trainer Survey (Reference Answers)

| Method | TRL Trainer | Key Unique Param | Default |
|---|---|---|---|
| PPO | PPOTrainer | kl_coef | 0.05 |
| DPO | DPOTrainer | beta | 0.1 |
| GRPO | GRPOTrainer | num_generations | 8 |
| KTO | KTOTrainer | desirable_weight | 1.0 |
| ORPO | ORPOTrainer | lambda_orpo | 0.1 |
| SimPO | SimPOTrainer | gamma (margin) | 0.5 |

Note: KTO's `desirable_weight` controls the relative loss weighting for desirable vs. undesirable examples (analogous to how DPO treats chosen vs. rejected). ORPO's `lambda_orpo` balances the SFT loss against the preference loss. SimPO's `gamma` is the minimum margin the model must maintain between chosen and rejected log-probabilities.

---

## Task 2 — SQL Domain Analysis (Key Points)

**Why DPO for Week 45 (not KTO or SimPO)?**
DPO was the right choice because: (1) we had execution-labeled preference pairs — the data format exactly matches DPO's input (paired chosen/rejected), (2) the reference model (v1) was available and stable, and (3) the offline nature of DPO was appropriate since we had a fixed preference dataset from Week 44. KTO would require converting pairs into unpaired format (discarding the pairing information), losing statistical efficiency. SimPO lacks a reference model, which risks losing the schema-specific behavior v1 learned during SFT.

**Why GRPO for Weeks 47–48 (not PPO or DPO)?**
GRPO was correct because: (1) SQL execution is a verifiable reward — no reward model needed, (2) we wanted online training (fresh rollouts at each step, not stale preference data), and (3) we needed to avoid the critic (value function) instability of PPO on sparse rewards. PPO would have required a separate value network that struggles with sparse end-of-episode rewards. DPO was offline — it could not adapt to new SQL patterns the model discovered during training. GRPO's on-policy group normalization was the right fit.

**When would you switch away from GRPO?**
Switch to PPO if: you add a learned reward model that goes beyond binary execution success (e.g., a quality scorer trained on human ratings of SQL). Switch to DPO if: you accumulate a large offline preference dataset (100K+ pairs) that is fresher than your training distribution. Switch to KTO if: you have production SQL logs that are unlabeled pairs (just "good SQL" from expert users and "bad SQL" from error logs). Never switch from GRPO for verifiable rewards unless the reward function is broken.

---

## How to Verify You Did It Right

1. Every cell in the comparison table is filled. No "?" or "TBD" remaining.
2. The TRL trainer survey shows different key parameters for each method (not all the same).
3. The SQL domain analysis uses specific references to your Phase 5 experience (not generic platitudes).
4. You can explain DPO vs. KTO vs. SimPO differences in 30 seconds each without looking at your notes.
