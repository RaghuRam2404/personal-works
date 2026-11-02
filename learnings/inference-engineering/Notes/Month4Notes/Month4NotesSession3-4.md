[← Month 4 notes](<../Month4Notes.md>) · [← Session 2](<Month4NotesSession2.md>) · [Session 5-7 →](<Month4NotesSession5-7.md>)

## Session 3,4

**How to think about init (from the [deep net runs](../../Month%204/3%20deep%20nn%20runs.md)):** in a deep net, init is about keeping every layer's activation std and gradient std the same from layer to layer. `gain/√fan_in` does that for the hidden layers, and a small last layer gives a near-uniform first prediction. Diagnose it with the per-layer tools (saturation %, gradient std, grad/param ratio), not the final loss: in Part A the bad inits wrecked training (dev 4.57 for std-1) while every reasonable init ended within 0.015 of the others.

Other important files 
- [Inferences decision tree.png](../../Notes/Month%204/Inferences%20decision%20tree.png)
- [3 deep nn runs.md](../../Month%204/3%20deep%20nn%20runs.md)
