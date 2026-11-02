# Month 4 Notes — Index

Session-wise notes for [Month 4 curriculum](<../Curriculum/Monthly Curriculum/Month4.md>). Each session's notes live in [Month4Notes/](<Month4Notes/>).

| Session | Notes | What's captured |
|---:|---|---|
| 2 | [Month4NotesSession2.md](<Month4Notes/Month4NotesSession2.md>) | Initialization experiments: output confidence (first loss 23.52 → 3.32 vs `log 27 = 3.2958`), tanh saturation (68.42 % → 0 % / 18.5 %), fan-in scaling (`gain/√fan_in`, tanh gain 5/3), Assignment M4.2 answers, evidence vs. explanation, the zero-init break, and the "remember always" takeaways |
| 3–4 | [Month4NotesSession3-4.md](<Month4Notes/Month4NotesSession3-4.md>) | How to think about init in a deep net (keep per-layer activation/gradient std stable), diagnosing with per-layer tools rather than final loss; links to the deep-net runs and the inferences decision tree |
| 5–7 | [Month4NotesSession5-7.md](<Month4Notes/Month4NotesSession5-7.md>) | Manual backprop implemented; compact loss and BatchNorm backward derivations skipped |
| 8–9 | [Month4NotesSession8-9.md](<Month4Notes/Month4NotesSession8-9.md>) | WaveNet hierarchy: 3 aggregation levels, receptive field 2 → 4 → 8, weight sharing across positions, why it is not attention, time axis traced through shapes |
| 10 | [Month4NotesSession10.md](<Month4Notes/Month4NotesSession10.md>) | TinyShakespeare pipeline: char tokenizer, contiguous 90/10 split, shifted batches, bigram model, loss reshaping, training (train 2.464 / test 2.492), generation |
| 11 | [Month4NotesSession11.md](<Month4Notes/Month4NotesSession11.md>) | Raschka single-head walkthrough: meaning of each step, convention rewrite, differences from the curriculum, reproduced numbers, `randn` vs `rand` scale lesson, open items for Parts 3–8 |

## Month 4 progress at a glance

- **Initialization:** first loss should be ≈ `log V`; scale with `gain/√fan_in`; randomness breaks symmetry, scale controls saturation and confidence.
- **Deep-network diagnostics:** judge init by per-layer saturation %, gradient std and grad/param ratio, not by the final loss.
- **Manual backprop:** done by hand; compact-form derivations still pending.
- **Hierarchy:** fixed tree of shared-weight local mixers; it is not content-addressed like attention.
- **GPT pipeline:** data → loss → train → eval → generate loop works with a bigram baseline (~2.46 loss).
- **Attention:** single-head walkthrough reproduced; Session 11 Parts 3–8 still open.
