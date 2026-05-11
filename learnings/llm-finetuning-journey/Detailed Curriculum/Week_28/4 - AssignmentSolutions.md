# Week 28 Assignment Solutions

## Task 1 — Karpathy Video Notes

No code to verify here, but your notes should contain at minimum:

**Expected content:**
- Base model = document completer, not an assistant
- RLHF emerged from the observation that generation quality is hard to supervise directly
- The "helpful, harmless, honest" framing came from Anthropic's Constitutional AI
- Token generation is fundamentally discrete sampling with temperature control
- Tool use (browsing, code execution) is a post-SFT capability added via specialized SFT data

**How to verify:** Re-read your notes the next day without the video. Can you reconstruct the argument for why SFT alone is insufficient for alignment? If yes, you understood it.

---

## Task 2 — InstructGPT Summary: Key Answers

**SFT dataset:** ~13K prompts with human-written responses (contractor-written). Small but high-quality.

**Reward model:** Trained on pairs (prompt, response_A, response_B) → scalar preference score. Input: prompt + response. Output: scalar reward.

**Why not SFT alone?** SFT teaches format and tone, but cannot directly optimize for human preference rankings. The reward model learns a finer-grained signal from pairwise comparisons. Then PPO maximizes the learned reward while staying close to the SFT policy (KL penalty prevents reward hacking).

**Pipeline mapping:**
```
InstructGPT                Your Phase 4-5 plan
SFT on 13K pairs     →     SFT on 5K–15K SQL pairs (Weeks 29, 33, 38)
Reward model         →     Execution correctness as explicit reward (Week 39+)
PPO / RLHF           →     DPO (Week 43) + GRPO (Week 46)
```

---

## Task 3 — Pipeline Diagram: What It Should Contain

Your diagram is correct if it shows:

```
[Pretrained Base Model]
        |
  (optional) [Continued Pretraining]
        |          on domain text
        |
  [SFT Stage]
  Input: (instruction, response) pairs
  Loss: cross-entropy on response tokens only
  Data: 1K–100K examples
        |
  [DPO Stage]
  Input: (prompt, chosen, rejected) triples
  Loss: DPO objective (implicit reward maximization)
  Data: 1K–50K preference pairs
        |
  [GRPO/RLVR Stage]
  Input: prompt → model generates → verifier checks correctness
  Reward: binary (SQL runs and returns correct rows)
        |
  [Deployed Model]
```

---

## Task 4 — Comparison Table: Reference Answers

| | Continued Pretraining | SFT | Instruction Tuning |
|---|---|---|---|
| Loss | Next-token prediction (CLM) | Cross-entropy on output tokens only | Cross-entropy on response tokens only |
| Data format | Raw domain text | (input, output) pairs | (instruction, response) pairs |
| Typical data size | 1B–100B tokens | 1K–100K pairs | 10K–1M pairs |
| Compute cost | High (same as pretraining, scaled down) | Low–Medium | Medium |
| What model learns | Domain vocabulary, syntax, knowledge | Task format and output style | General instruction-following |
| When to use | Domain vocab absent from pretrain data | Task behavior shift needed | General-purpose assistant |
| Catastrophic forgetting risk | Low (same objective) | Medium (high LR = high risk) | Medium |

**Common gotchas:**
- Students often say "SFT changes the loss function" — it does not. The loss is still cross-entropy. What changes is which tokens are masked (only output tokens contribute to the loss in SFT).
- "Instruction tuning needs millions of examples" — InstructGPT used 13K. Quality > quantity.
- The input tokens during SFT are included in the forward pass but excluded from the loss. This is often called "input masking" in code. In `SFTTrainer`, this is handled automatically via `dataset_text_field`.

---

## How to Verify You Did It Right

- You can explain continued pretraining vs. SFT vs. instruction tuning to someone who asks, without notes
- Your pipeline diagram has at least 4 distinct stages with labeled inputs and losses
- Your InstructGPT summary correctly identifies the 13K SFT data size and the pairwise preference structure of the reward model
- You have all 4 files committed: `week28_karpathy_notes.md`, `week28_instructgpt_summary.md`, `week28_pipeline_diagram.png`, `week28_comparison.md`
