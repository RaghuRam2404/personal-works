[← Month 4 overview](<../Month4.md>) · [← Session 15](<Month4CurrSession15.md>) · [Session 17 (optional) →](<Month4CurrSession17.md>)

## Session 16 — Train, evaluate, and defend

### Objective

Train the Month 4 character GPT, preserve a useful Month 5 baseline, and prove that the implementation is understood rather than copied.

### Training protocol

Use the character tokenizer and TinyShakespeare pipeline from Session 10.

Record:

- Device and dtype.
- Vocabulary size.
- Block size.
- Batch size.
- Model dimension.
- Head count and head dimension.
- Layer count.
- Dropout.
- Parameter count.
- Optimizer and learning rate.
- Evaluation interval and number of evaluation batches.
- Random seed.
- Total updates and wall-clock time.

Use the mixed-precision and profiling skills from Month 2 only after the FP32 training path is correct. If AMP is used, state the dtype and hardware support explicitly.

### Evaluation

At fixed intervals:

- Switch to evaluation mode.
- Estimate training loss across multiple mini-batches.
- Estimate validation loss across multiple mini-batches.
- Restore training mode.
- Save the best validation checkpoint and a resumable latest checkpoint.

Stop when one of these occurs:

- Validation loss has clearly plateaued under your defined criterion.
- The allocated compute budget is exhausted.
- Repeated validation degradation indicates overfitting.

Do not report a single favorable mini-batch loss as validation performance.

### Month 4 minimum versus Phase B target

#### Required Month 4 minimum

- Small GPT trains with decreasing train and validation loss.
- Approximately 10M model is instantiated, counted, and passes forward/backward.
- At least one meaningful full-stream training run is completed.
- A 200-character sample is generated from the best checkpoint.
- All attention correctness tests pass.

#### Preferred Month 4 completion

- Train the project-shaped character GPT until validation loss plateaus.
- Reach or approach the Phase B character-level target of 1.6 nats/token.
- Generate locally English-like text.

#### Not a Month 4 failure

If compute or time prevents the final loss target but the architecture, tests, training curve, checkpointing, and causal attention are correct, carry the checkpoint into Month 5. The final Project B1 acceptance remains at the end of Phase B.

### Sampling experiment

Generate from the same prompt with several temperatures, including a low, medium, and high value. Optionally add top-k sampling only after basic multinomial sampling works.

For each setting, describe:

- Repetition.
- Diversity.
- Local spelling/punctuation structure.
- Failure patterns.

Do not select a sample and call the model better based only on aesthetics; retain validation loss as the main training metric.

### Timed memory implementation

In a blank file and without references, implement causal multi-head self-attention. Include:

- Q/K/V projections.
- Head split.
- Scaled score matrix.
- Causal mask.
- Softmax over keys.
- Value aggregation.
- Head merge.
- Output projection.

Month 4 exit target: correct code in 15 minutes, then tests. Continue practicing in Month 5 until the Phase B under-10-minute target is met.

### Oral examination

Answer without notes:

1. Why should initial logits be low-confidence?
2. Why can tanh saturation suppress gradients?
3. What does fan-in scaling try to preserve?
4. Which axes does BatchNorm normalize in the MLP?
5. Why does BatchNorm need running statistics?
6. Why does LayerNorm not need running batch statistics?
7. Why must broadcast gradients be reduced over expanded axes?
8. Why do repeated embedding IDs require gradient accumulation?
9. Derive the cross-entropy gradient with respect to logits conceptually.
10. How does a hierarchical model grow its receptive field?
11. What makes attention content-dependent?
12. Distinguish queries, keys, and values.
13. Why scale query-key scores?
14. Why mask before softmax?
15. What does each axis in `[B, H, T, T]` mean?
16. Why can Transformer training process every position in parallel despite causality?
17. Why is autoregressive generation sequential?
18. Why are residual connections useful for deep optimization?
19. What separate roles do attention and the feed-forward network play?
20. Why are raw nats/token not comparable across different tokenizers?
21. What computation will KV caching avoid next month?

### Final report

Write `outputs/month4-report.md`:

```text
1. Month objective
2. Initialization experiments
3. BatchNorm implementation and train/eval behavior
4. Activation/gradient/update diagnostics
5. Manual backward parity
6. Hierarchical model and shape analysis
7. Attention v1/v2/v3 design
8. Forward and gradient parity results
9. Causality tests
10. GPT architecture and parameter count
11. Training configuration and curves
12. Validation result and samples
13. Bugs and root causes
14. Remaining risks before BPE and KV caching
```

### Exit criteria

- All tests pass from a clean checkout.
- Best and latest checkpoints are loadable.
- A 200-character sample is reproducible from a saved checkpoint and seed.
- Attention can be reconstructed without reference material.
- At least 17 of the 21 oral questions can be answered clearly; misses become Month 5 review items.
