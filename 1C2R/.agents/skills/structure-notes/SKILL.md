---
name: structure-notes
description: 'Reads a text source file (video/podcast transcript, article, meeting notes, raw notes) and produces precise, detailed structured notes for future reference — preserving every concrete point, number, name, step, and framework without paraphrasing away detail. Use when asked to: structure notes from a file, extract key points, break down a transcript, organize raw notes, turn a transcript into reference notes, or create a detailed breakdown of a document. Non-coding content task — do not use for source code files or coding/refactoring requests.'
argument-hint: 'path/to/source-file.md'
---

# Structure Notes

Turns a raw, unstructured source file into precise, detailed, well-organized notes for future reference — without losing any concrete information from the original.

## When to Use
- User points at a file (transcript, article, raw notes, saved post) and asks for it to be "structured," "broken down," "organized," or "turned into notes"
- Content is dense or unstructured (e.g. a wall-of-text transcript) and needs a scannable, hierarchical reference version
- Not for source code, config files, or coding tasks — this skill is for reference/knowledge content only

## Procedure

1. **Read the full file.** Use the exact path given, or ask for one if missing. Read the entire file — for long files, keep reading until the whole thing is covered. Never structure notes from a partial read.
2. **Identify the natural structure.** Look for how the content is already divided: numbered lists ("secret #3", "step 2"), chronological segments (e.g. multiple videos/chapters in one file), or topical sections. This determines the headings in the output.
3. **Extract every substantive point.** Favor completeness over brevity, with no length cap — a long, dense source should produce long, dense notes. Preserve exact numbers, names, proper nouns, acronyms, frameworks, and terminology — don't paraphrase specifics into vague language.
4. **Organize using the template below.** Mirror the source's own order and grouping; don't reorganize or editorialize beyond what's needed for clarity. Omit any template section that doesn't apply to this particular source.
5. **Self-check against the Quality Checklist** before presenting.
6. **Present the structured notes directly in the chat reply.** Do not create, overwrite, or save any file unless the user explicitly asks you to.

## Output Template

```markdown
# Structured Notes: <Title>

**Source:** <relative file path>
**Type:** <transcript / article / raw notes / other>

## Overview
<2-4 sentence plain-language summary of what this file covers and why it matters>

## Detailed Breakdown
<Mirror the source's own structure. If the file contains multiple sub-parts (e.g. several videos in one file), use one top-level section per sub-part.>

### <Section / Secret / Topic name>
- <Core idea, one line>
- <Supporting detail, kept concrete — numbers, names, examples>
- <Nested sub-points as needed>

## Frameworks & Named Systems
<Only if the source names a specific model, acronym, or system>
- **<Name>** — components in order, each briefly explained

## Actionable Takeaways
- <Concrete, checklist-style actions distilled from the content>

## Notable Quotes
<Only if the source has standout lines worth preserving verbatim>
- <Exact quote>

## Gaps / Ambiguities
<Only if something in the source was unclear, inaudible, or contradictory — flag it here rather than guessing>
```

## Quality Checklist
- [ ] Every numbered/named item in the source (e.g. "secret #5", chapter 3) has its own heading in the output — none merged, skipped, or renumbered
- [ ] Concrete details are preserved exactly: numbers, names, proper nouns, frameworks, quotes — never replaced with vague language
- [ ] Output order/hierarchy matches the source's own order/hierarchy
- [ ] Nothing is invented — unclear or ambiguous parts of the source are flagged in "Gaps," not guessed at
- [ ] Output is self-contained — someone who never reads the original should fully understand it from the notes alone
