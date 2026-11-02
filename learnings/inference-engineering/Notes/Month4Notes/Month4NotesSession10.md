[← Month 4 notes](<../Month4Notes.md>) · [← Session 8-9](<Month4NotesSession8-9.md>) · [Session 11 →](<Month4NotesSession11.md>)

## Session 10

Built the character-level pipeline on TinyShakespeare and trained a bigram baseline on it. Code is in [6 shakespeare.py](../../Month%204/6%20shakespeare.py) and the [gpt package](../../Month%204/gpt/): [tokenizer.py](../../Month%204/gpt/tokenizer.py), [data.py](../../Month%204/gpt/data.py), [bigram.py](../../Month%204/gpt/bigram.py).

The goal is not a good model. It is a working data → loss → train → eval → generate loop, so that when attention comes in Session 11, any new bug can only be in the attention code.

**Pipeline, end to end**

```text
text → CharTokenizer.encode → 1-D long tensor → contiguous 90/10 split
     → get_batch → x [B,T], y [B,T]
     → BigramLanguageModel → logits [B,T,V]
     → reshape to [B*T,V] and [B*T] → cross_entropy → backward → Adam step
```

**Tokenizer** ([tokenizer.py](../../Month%204/gpt/tokenizer.py))
- One token is one character. Capitalization, spaces and newlines are all kept, and the file is a single continuous stream with no word splitting or end-of-name token (this differs from the names dataset).
- The vocabulary is `sorted(set(text))`, with IDs `0..V-1`, stored in both `stoi` and `itos`. Sorting makes the mapping deterministic: the same text always gives the same IDs. A Python `set` has no stable order, so an unsorted vocab would give different IDs on different runs and break saved models.
- An unknown character raises `ValueError` by default. With `handle_unknown=True` it maps to `<UNK>`, which adds one entry to the vocab.
- `V` is built once from the full text and reused for both splits.

**Split and batching** ([data.py](../../Month%204/gpt/data.py))
- `encode_and_split` cuts the encoded stream at `int(0.9 * N)`: the first 90% is train and the last 10% is validation. There is no shuffling. A random per-character split would put almost-identical overlapping windows in both sets and leak context into validation.
- `get_batch` picks `B` random start positions `i` and builds `x = data[i:i+T]`, `y = data[i+1:i+T+1]`. So `y` is `x` shifted by one, and `x[:, 1:] == y[:, :-1]`.
- `torch.randint(high=len(data) - T)` excludes `high`, so the largest start is `len - T - 1`. Then `y`'s slice ends exactly at `len`, and nothing runs off the end.
- A window needs `T+1` source characters, but `x` and `y` hold `T` each. Each row gives `T` prediction tasks, not one: `x[t] → y[t]` for every `t`.
- Sampling windows inside an already-separated region is fine. The leak is only when overlapping windows are assigned to train or validation at random.

**Model** ([bigram.py](../../Month%204/gpt/bigram.py))
- `nn.Embedding(V, V)` is the entire model. Row `c` of the table is the next-character logits for input `c`. It is not a feature vector.
- Shapes: `[B,T]` long IDs → `[B,T,V]` logits. `nn.Embedding` needs integer IDs, so `torch.randn` input fails. Logits go out, never back in.
- Even with `T = 8`, each prediction uses one character. All `T` positions are computed in parallel, but they never see each other. The same input character gives the same logits in any sentence. `[B,T]` does not give sequence context by itself.

**Loss**
- `loss_fn(logits.view(-1, V), y.view(-1))`: `[B,T,V] → [B*T,V]` and `[B,T] → [B*T]`.
- With `B=32, T=8` that is 256 independent prediction tasks per step. `CrossEntropyLoss` wants the class (vocab) axis last in 2D (or at dim 1), so flatten `B` and `T` together and keep `V` as the last axis. The vocab axis must not be mixed into the batch axis, or the scores will be paired with the wrong targets.
- A fresh model should start at about `ln(V)`, the uniform guess. For 65 characters that is ≈ 4.17.

**Training and evaluation** (settings in [6 shakespeare.py](../../Month%204/6%20shakespeare.py))
- seed 1337, `B=32`, `T=8`, Adam with `lr=1e-2`, 2000 steps, evaluating every 200.
- `estimate_loss` averages the loss over `eval_batches=20` fresh batches of `B=32`, with no gradients, then restores `train()`. A single batch would be noisy, so this gives a steadier number. `learning_rate` was originally defined but not passed to the optimizer, so it ran at Adam's default 1e-3. It is now passed in.
- The whole-stream check at the end runs `model(train[:-1])` against `train[1:]`. With 1-D input the embedding returns `[N,V]` directly, so the loss over all N tokens needs no batching.
- Final losses: **train 2.464, test 2.492**. That is down from ≈ 4.17, and validation is close to train, which is expected for a model with only `V×V` parameters. It is too small to overfit the training text.

**Generation**
- Loop: take the logits of the last position → softmax over the vocab axis → `torch.multinomial` → append the sampled ID → repeat. The output is `[B, T + max_new_tokens]`.
- It is sequential even though training is parallel: the next input does not exist until the previous one is sampled. A cache would not remove that dependency; it would only avoid recomputing old positions.
- The bigram model only uses the last token, so `generate` feeds just `idx[:, -1]`. That is shape `[B]`, the embedding returns `[B,V]`, and so softmax uses `dim=1`. Feeding the full prefix would work too but recomputes `T` positions it never uses.
- The method must call `self(...)`, not the global `model`. It only worked at first because the script's variable happened to be named `model`.
- Samples of 300 characters from a zero start (ID 0 is the first sorted character):

  | | Sample |
  |---|---|
  | Before training | `\nSKIcLT;AcELMoTbvZv C?nq-QE33:CJqkOKH-q;:la!oiywkHj...` (random characters, including rare ones like `$`, `3`, `&`) |
  | After training | `Thand, sowht ClanBy-r har / KINRD 'swind / We, Palion, s w pro y ben dis ounerrofrssple tim; ...` |

  After training you see speaker-label-like capitals, line breaks, common letter pairs (`th`, `ou`, `er`) and short word-like chunks. The words are not real, because each character depends only on the one before it.

**Takeaways**
- Check each piece alone before combining: tokenizer round-trip, shift (`y` is `x` moved by one), shapes, first loss ≈ `ln V`, then train.
- Random windows from a contiguous train region are good. Random assignment of overlapping windows to train and val is leakage.
- Bigram loss bottoms out around 2.4–2.5 for this data because the model sees one character. Lower needs context, so the next session adds communication between positions (attention). Everything built here is reused as is.
