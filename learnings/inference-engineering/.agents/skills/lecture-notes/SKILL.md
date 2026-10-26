---
name: lecture-notes
description: Compile a month's (or a range of sessions') work (code, run outputs, notes, daily logs, curriculum) into one top-to-bottom revision "lecture" markdown file like Notes/lectures/Lecture3.md. Use whenever the user asks for a lecture, revision notes, a consolidated write-up, "Lecture N", or to turn Month N files, .py scripts, notes, logs or outputs into readable study material, even if they don't say "lecture".
---

# Lecture notes

Turn what the user has **already done** into one document they can open later and read from top to bottom to revise it. The reader is the user, a few weeks from now, who remembers the sessions vaguely and wants to rebuild the ideas and the code from memory.

The reference for structure and tone is `Notes/lectures/Lecture3.md` (and `Lecture2.md`). Read one of them before writing, to match the voice and depth. Copy its *shape*, not every habit of it. The rules under "What to leave out" override anything in those files that breaks them.

## The one rule that shapes everything

**Write only about what was done.** The document is a record of understanding, not a gap analysis. Nothing in it should make the reader think about work they skipped, code that is missing, or a better version they did not build.

### What to leave out

- Anything the daily logs mark as skipped, postponed, "didn't do", or not understood yet.
- Curriculum items, assignments, or checkpoints that have no matching code, output, notes or log entry. The curriculum says what was *planned*. It is never evidence that something happened.
- Sentences of the form "this script does not…", "no X is used", "a fuller version would…", "the curriculum asks for…", "not yet covered", "left for later", "this proves nothing about…".
- Critiques of the user's own code for what it lacks (unseeded RNG, missing tests, no shuffle, etc.). If the lack caused something the user *observed* (a run-to-run difference, a bug they hit and fixed), describe the observation and fix, not the omission.
- Forward references to future sessions or months beyond a short "what this sets up" clause, and only when the logs or notes themselves say so.

When in doubt, leave it out. A missing detail costs the reader little; a paragraph about something they never did is noise that pulls them out of revision.

### What the sources are for

| Source | Use it to |
|---|---|
| `.py` / `.ipynb` files | The ground truth of what was built. Quote the real code, in an order that teaches. |
| Run outputs (`*.md`, `*.txt`, printed output, plots, images) | Real numbers and results. Never invent or round a number that is not recorded. |
| Existing notes (`Notes/*.md`, transcripts) | The user's own explanations and mental models. Reuse their framing and fold their insights in. |
| Daily logs | **Scope filter and memory aid.** Decide what was done, in what order, what confused the user, what they understood. Mistakes the user hit and fixed are worth teaching points. |
| Curriculum | **Vocabulary and session boundaries only.** Use it to name sessions and concepts and to see how files group. Never to add content. |

## Workflow

1. **Pin down the scope.** The user usually names a month, sessions, or files. If not, infer from the folder (`Month N/`, `Notes/Notes N.md`, the matching section of the daily logs). Pick the next lecture number from `Notes/lectures/`. Ask only if the scope is truly ambiguous.
2. **Read everything in scope**, completely: every script, every output file, the notes, the relevant stretch of the daily logs, and the matching curriculum month. Look at images that carry results (loss curves, graphs) and reference them. Do not run long training jobs. Use recorded outputs. A quick run is fine only when a short, deterministic output is needed and none is recorded.
3. **Build an inventory** (in your head or scratch): for each file, the concepts it demonstrates, the numbers it produced, the bugs or changes the user went through, and whether the logs confirm it was done. Drop anything that fails the "was done" test.
4. **Choose the order by the learning chain, not the file order.** Arrange parts so each one needs only what came before: the idea, then the smallest code that shows it, then the result. Several files may feed one part and one file may feed several.
5. **Write the lecture** using the layout below.
6. **Verify** (see checklist) and fix before reporting.

## Layout

Use this skeleton, as in Lecture 3. Every section is optional except the first four; include a section only if there is real content for it.

```markdown
# Lecture N — <the through-line of the lecture, as a title>

<2–3 sentences: which month/sessions this covers, what chain of ideas it rebuilds, and that the numbers are real outputs.>

**How to use these notes:** <one line on how to read it: Parts a–b for ideas, the check section for revision.>

| Part | Topic | Source files |
|---|---|---|
| 1 | ... | [file.py](../../Month%20N/file.py) |

<One paragraph: the thread through all of it, in bold for the key sentence.>

---

## Part 1 — <Topic>
...
## Part K — Results (real numbers, from the recorded outputs)
## Part K+1 — Mistakes hit and fixed (only ones that actually happened)
## Part K+2 — Self-check (answer aloud, then compare)
```

Links to source files are relative from `Notes/lectures/` and URL-encode spaces (`../../Month%203/3%20mlp.py`). Images go in `Notes/images/` and are linked as `../images/<name>.png`. If the original image lives elsewhere (e.g. `Notes/Month 4/`), link it where it is rather than moving it.

## How to write each part

Write like a patient tutor explaining to the user's future self.

- **Idea first, then code, then what it printed.** Start with the question the part answers in one or two sentences. Show the real code, trimmed to the lines that carry the idea (`...` for the rest), with short inline comments. Then the actual output and how to read it.
- **Explain the why.** Every non-obvious line gets its reason: why this axis, why `keepdim`, why this order, why this learning rate worked. The aim is that the reader could rewrite it.
- **Use the user's own discoveries.** If the logs or notes record a confusion, a bug, or an "oh, that's why" moment, give it a short callout in the part where it belongs. These are the most memorable parts of a revision.
- **Make it scannable.** Tables for shapes, parameters, comparisons and run results. Numbered or bulleted lists for steps. Math in `$$ $$` when a formula is the point. ASCII diagrams where a picture of the flow helps. A boxed `> **Rule to remember:**` for the one-line takeaways.
- **Numbers must trace to a source.** Quote outputs exactly. If two runs differ, say which run is which. Do not compute new results to make a table look complete.
- **Shape ledgers and counts.** For model code, include the tensor-shape table and parameter count when the code makes them derivable, and double-check the arithmetic.
- **Keep the voice calm and direct.** No hype, no filler, no second-guessing the user's work.

### Mistakes section

Include only mistakes that were made and resolved, or common traps that the code visibly handles (for example "normalize along the right axis" where the code does it and the logs show the bug was met). Phrase each as a recognition rule: what it looks like, why it happens, how it was fixed. Do not list generic best practices the code did not touch.

### Self-check section

Write 15–30 question-and-answer pairs that cover every part, ordered as the parts are. Questions are the "why" and "what would happen if" type that test understanding, not trivia. Answers are 1–3 sentences, correct against the code and numbers in the document. End with a "what you should be able to do" table mapping each skill to its part if the lecture is long enough to benefit.

## Verification checklist

Before finishing, check each item and fix what fails:

1. **Every claim is backed** by a file, an output, a note, or a log entry. Remove the rest.
2. **Scan for gap language.** Search the draft for: `does not`, `not used`, `no `, `lacks`, `missing`, `would`, `could`, `should have`, `curriculum`, `assignment`, `skipped`, `later`, `not yet`, `yet`. Each hit is either rewritten as a statement about what was done, or deleted. Plain factual uses ("`.grad` does not reset itself", a rule the user learned) are fine when they describe how the thing works rather than what the user left out.
3. **Numbers match** the recorded outputs exactly, and the arithmetic (parameter counts, shapes, sums) is right.
4. **Code snippets match** the real files (names, shapes, hyperparameters). Do not tidy the code into something the user never wrote.
5. **Links resolve.** Every relative link and image path points at a file that exists (check with `ls`).
6. **Reads top to bottom.** Each part depends only on earlier parts, the table of contents matches the headings, and "Part N" cross-references point at the right parts.
7. **Self-check answers** agree with the body.

## Output and hand-off

Save to `Notes/lectures/Lecture<N>.md`. Do not edit the source scripts, notes, logs or curriculum. Reply with: the path, the parts it contains, which source files fed it, and anything you left out because the sources did not support it (so the user can say if they actually did it). Keep that reply short.
