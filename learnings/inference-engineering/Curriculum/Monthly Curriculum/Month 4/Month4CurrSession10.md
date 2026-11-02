[← Month 4 overview](<../Month4.md>) · [← Session 9](<Month4CurrSession9.md>) · [Session 11 →](<Month4CurrSession11.md>)

## Session 10 — Build and test the TinyShakespeare pipeline

### What you should have working at the end

A runnable program that:

1. Reads Shakespeare text.
2. Converts every character into an integer ID.
3. Samples batches of text with correctly shifted targets.
4. Uses each input character to predict the next character.
5. Trains, reports training and validation losses, and generates a text sample.

Keep attention, positional embeddings, Transformer blocks, and KV caching out of this session. Your curriculum introduces those later. 

### Before coding: predict these three things

Write your answers in `notes/session-10.md`, without running anything:

1. For the text `"hello"` and `T = 4`, what are `x` and `y`?
2. If the input character is `"l"` in two different sentences, should a bigram model produce the same next-character distribution in both places?
3. Does providing a batch with shape `[B, T]` automatically mean the model uses `T` characters of context for each prediction?

These are checkpoints, not prerequisites that should stop you from starting. Revisit your answers after implementing the model.

### Part 1 — Download and inspect the text

Save the [tinyshakespeare dataset](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) locally as:

```text
data/tinyshakespeare.txt
```

Use the URL in your assignment. The source is plain text containing dialogue, speaker labels, punctuation, spaces, and line breaks; it begins with `"First Citizen:\nBefore we proceed..."`.[3]

Write code that reads the file as UTF-8 and prints:

```python
len(text)
repr(text[:200])
len(set(text))
```

For this assignment:

- Preserve capitalization.
- Preserve spaces and newlines.
- Do not split into words or names.
- Do not add `"."` padding or end-of-name tokens.

The last point is a deliberate change from your names dataset: here, treat the file as one continuous character stream.

Done when: you can inspect the text and explain what counts as one token.

### Part 2 — Implement the character tokenizer

Create `gpt/tokenizer.py`.

Use this interface:

```python
class CharTokenizer:
    def __init__(self, text: str):
        ...

    @property
    def vocab_size(self) -> int:
        ...

    def encode(self, text: str) -> list[int]:
        ...

    def decode(self, ids: list[int]) -> str:
        ...
```

Implement these rules:

1. Build the vocabulary once from the complete dataset text for this lecture-style exercise.
2. Sort the unique characters before assigning IDs.
3. Assign consecutive IDs starting at `0`.
4. Store both character-to-ID and ID-to-character mappings.
5. Raise a clear exception when encoding an unknown character.
6. Raise a clear exception when decoding an invalid ID.
7. Do not silently remove or replace unsupported characters.

Do not rebuild the vocabulary separately for training and validation.

### Write these tests

Create `tests/test_tokenizer.py`:

- Round-trip a string containing letters, spaces, punctuation, and a newline.
- Check every encoded ID satisfies `0 <= id < vocab_size`.
- Construct two tokenizers from the same text and check their mappings match.
- Check unknown-character encoding raises an exception.
- Check negative and out-of-range IDs raise an exception.

Use a tiny synthetic text in these tests so they do not require downloading the dataset.

Done when: the tests pass and you can explain why sorting the vocabulary matters.

### Part 3 — Encode and split the stream

Create `gpt/data.py`.

Implement:

```python
def encode_and_split(
    text: str,
    tokenizer: CharTokenizer,
    train_fraction: float = 0.9,
) -> tuple[torch.Tensor, torch.Tensor]:
    ...
```

Required behavior:

1. Encode the complete text.
2. Convert the IDs into one 1-D `torch.long` tensor.
3. Compute the split index:

   ```python
   split_index = int(train_fraction * len(data))
   ```

4. Return:

   ```python
   train_data = data[:split_index]
   val_data = data[split_index:]
   ```

5. Do not shuffle the stream before splitting.
6. Do not allow sampled windows to cross the split boundary.

This preserves the curriculum’s requirement to create separate contiguous training and validation regions before sampling windows. 

### Write these tests

- Both outputs are 1-D `torch.long` tensors.
- Their lengths add up to the encoded text length.
- Concatenating them reconstructs the original encoded stream.
- Training contains the prefix and validation contains the suffix—not shuffled characters.

Done when: you can point to exactly which part of the original file belongs to each split.

### Part 4 — Implement the random-window batch sampler

In `gpt/data.py`, implement:

```python
def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    ...
```

Use these initial settings:

```python
batch_size = 4
block_size = 8
```

For each row, choose a start position `i`, then construct:

```python
x_row = data[i : i + block_size]
y_row = data[i + 1 : i + block_size + 1]
```

Stack the rows to return:

```text
x: [batch_size, block_size]
y: [batch_size, block_size]
```

Requirements:

- Sample start positions independently; repeated positions are allowed.
- The largest valid start index is `len(data) - block_size - 1`.
- Remember that `torch.randint` excludes its upper bound.
- Reject a stream shorter than `block_size + 1`.
- Keep this function CPU-only initially; move batches to the model’s device in the training script.

#### Prove the shift manually

First use synthetic data, not Shakespeare:

```text
stream = "hello"
T = 4

x = "hell"
y = "ello"
```

Write the four prediction tasks:

```text
h → e
e → l
l → l
l → o
```

Then sample one Shakespeare batch and print:

```python
repr(tokenizer.decode(x[0].tolist()))
repr(tokenizer.decode(y[0].tolist()))
```

Use `repr` so spaces and newlines are visible.

#### Write these tests

Create `tests/test_data.py`:

- Shapes are exactly `[B, T]`.
- Both tensors have dtype `torch.long`.
- For `T > 1`:

  ```python
  assert torch.equal(x[:, 1:], y[:, :-1])
  ```

- On a synthetic stream with unique integer values, verify every complete row corresponds to a valid length-`T + 1` slice of the source. This also checks the final target, which the assertion above does not check.
- A stream of exactly `T + 1` tokens works.
- A stream with fewer than `T + 1` tokens raises an exception.

Done when: every target—including the final target in each row—is demonstrably the next character after its input.

### Part 5 — Implement only the bigram baseline

Create `gpt/model.py`.

Use this interface:

```python
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        ...

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        ...
```

Use one trainable table:

```python
nn.Embedding(vocab_size, vocab_size)
```

For this model, interpret each table row as next-character logits, not as an intermediate feature vector.

Required shape flow:

```text
input IDs              [B, T]
lookup table           [V, V]
output logits          [B, T, V]
```

Do not add an MLP, BatchNorm, time-position information, or attention. The curriculum explicitly calls for a token-transition baseline before attention. 

#### Model checks

Write tests that confirm:

- `[B, T]` input produces `[B, T, V]` output.
- Two occurrences of the same input token produce identical logits.
- Backward produces a finite gradient for the embedding table.

Checkpoint question: what information can distinguish the two `"l"` inputs in the `"hello"` example? Compare your answer with your pre-coding prediction.

### Part 6 — Connect the logits to the loss

In your training code, use:

```python
logits = model(x)

loss = F.cross_entropy(
    logits.reshape(-1, tokenizer.vocab_size),
    y.reshape(-1),
)
```

Before running it, write the shapes for:

```text
B = 4
T = 8
V = tokenizer.vocab_size
```

Your notes should identify:

- The number of prediction tasks in this batch.
- Which axis contains candidate next-character scores.
- Which target belongs to each score vector.

For an additional correctness check, compare the flattened loss against the mean of individually computed position losses on a tiny batch.

Done when: the losses agree and `loss.backward()` works.

### Part 7 — Run a controlled tiny-data learning check

There is an important ambiguity in the original exit criterion: “overfits a tiny batch” should not automatically mean “reaches nearly zero loss.” In your own `"hello"` example, the same input `"l"` has two different targets; your model check above requires the same logits for both occurrences.

Use two separate checks.

#### Check A: an unambiguous synthetic stream

Use a repeating stream such as:

```text
"abcabcabcabcabc..."
```

1. Build a tokenizer for this synthetic experiment.
2. Create a fresh model.
3. Sample one batch containing all three transitions.
4. Keep that batch fixed.
5. Train repeatedly on that exact batch.
6. Record initial and final loss.

Acceptance criterion: loss drops substantially toward zero and the highest-scoring next token is correct for all three transitions.

#### Check B: one fixed Shakespeare batch

1. Create the real Shakespeare tokenizer and a fresh model.
2. Sample one training batch.
3. Keep it fixed.
4. Train on it repeatedly.
5. Record initial and final loss.
6. Inspect any input characters associated with multiple different targets.

Acceptance criterion: loss decreases; do not require near-zero loss when the batch contains conflicting next-character targets.

Checkpoint question: why would making the batch smaller sometimes make a bigram model appear much better at memorization?

### Part 8 — Train on random Shakespeare batches

Create `gpt/train.py`.

Use these starting settings:

```python
seed = 1337
batch_size = 32
block_size = 8
learning_rate = 1e-2
max_steps = 2000
eval_interval = 200
eval_batches = 20
```

These are starting choices, not performance targets.

Implement this sequence:

1. Set the random seed.
2. Load the text and construct the tokenizer.
3. Create contiguous training and validation tensors.
4. Create a fresh bigram model.
5. Move the model to your chosen device.
6. Create an AdamW optimizer.
7. Evaluate training and validation loss before the first update.
8. For each training step:
   - Sample a fresh training batch.
   - Move `x` and `y` to the model’s device.
   - Compute logits and loss.
   - Clear gradients.
   - Run backward.
   - Update parameters.
9. Every `eval_interval` steps, estimate both losses.
10. Evaluate again after the final update.

For each evaluation:

- Use `model.eval()` and `torch.no_grad()`.
- Average loss across `eval_batches` batches.
- Do not call backward or update parameters.
- Restore training mode afterward.

Save a CSV with:

```text
step,train_loss,val_loss
```

Done when: you have a reproducible run, finite losses, and a lower final training-stream loss than at initialization. Do not use the later GPT loss target as a requirement for this baseline.

### Part 9 — Generate a sample

Create `gpt/generate.py` with:

```python
def generate(
    model,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
) -> torch.Tensor:
    ...
```

For each new character:

1. Obtain logits from the model.
2. Select the final time position.
3. Apply softmax over the vocabulary axis.
4. Sample one token using `torch.multinomial`.
5. Append the sampled ID to the sequence.
6. Repeat.

Use evaluation mode and disable gradient tracking.

Generate 300 new characters before training and another 300 afterward. Decode and save both samples.

For this baseline, feeding just the latest token is sufficient because that is the only input its prediction uses. Keeping a full-prefix implementation is acceptable for practicing the interface you will later reuse.

Do not implement a KV cache.

Done when: generation returns the prompt plus exactly the requested number of new tokens, and every generated ID decodes successfully.

### Part 10 — Answer the revised first-principles questions

Use these in place of the original questions:

1. A sampled window requires `T + 1` source characters. Why do its input and target tensors each contain only `T` characters?
2. Why does one row provide `T` prediction tasks, rather than just one?
3. Why can the bigram model compute all positions’ logits and losses in parallel during training?
4. Why does ordinary autoregressive sampling generate one new character at a time? Would adding a cache remove the dependency on the previously sampled character?
5. Why is randomly sampling windows within an already-separated training region different from randomly assigning overlapping windows to training and validation?
6. Why does this baseline use only one character of predictive context even when `block_size = 8`?
7. What does flattening `[B, T, V]` into `[B*T, V]` preserve? What mistake would mix up the vocabulary axis?
8. Is flattening the only valid way to arrange these tensors for cross-entropy, or is it simply the convention chosen here?
9. Why can the same model nearly memorize the synthetic `"abc"` transitions but fail to achieve zero loss on a Shakespeare batch?

The revised generation question deliberately separates sequential dependence from caching, and the revised loss question avoids treating one tensor arrangement as the only possible API usage.

### Final deliverables and stopping point

```text
gpt/
    tokenizer.py
    data.py
    model.py
    train.py
    generate.py

tests/
    test_tokenizer.py
    test_data.py
    test_bigram.py

notes/
    session-10.md

reports/
    bigram_training.csv
    bigram_samples.txt
```

Stop this session when:

- Tokenizer tests pass.
- Split and batch tests pass.
- You have manually verified a decoded shifted example.
- Model shapes and repeated-token behavior pass.
- The synthetic fixed-batch experiment learns its transitions.
- The Shakespeare baseline learns and generates a sample.
- You can explain why a sequence-shaped input does not, by itself, give this model sequence context.

That last distinction is the bridge into Session 11: the current model processes multiple positions, but the next session introduces communication between positions. Your curriculum places explicit loop-based attention immediately after this baseline. 
