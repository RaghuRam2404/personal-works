---
name: x-post-suggester
description: 'Plans next week''s X (Twitter) post topics for DadFit by mapping the day-of-week content rotation in Daily Strategy.md to audience pain-point keywords/traits in SubNiche-Keyword-Traits.md, avoiding repeated angles, then appends the plan to PostsTitle.md. Use for: weekly X content planning, generating next week X post topics, updating PostsTitle.md, X post ideation, XPosts topic planning.'
---

# X Post Suggester

Plans the **topics** for the next 7 days of X posts (not the post copy itself) and appends them to [PostsTitle.md](../../../XPosts/PostsTitle.md) as a new, numbered week block.

## Inputs

- [Daily Strategy.md](../../../XPosts/Daily%20Strategy.md) — day-of-week → content format rotation.
- [SubNiche-Keyword-Traits.md](../../../Resources/SubNiche-Keyword-Traits.md) — subniche → keyword → trait mapping with content-allocation %. **Only use the Physical Fitness (Workout, Diet) and Mental Fitness sections. Ignore Financial Fitness entirely** (Family risk management, Automated wealth building) — it's out of scope for X posts right now.
- [PostsTitle.md](../../../XPosts/PostsTitle.md) — running history of previously planned posts. May be empty on first run.

## Step 1 — Read history

Read all of `PostsTitle.md`. For every past entry, note its `Problem to address` keyword, `Perspective`, `Format`, and date. This is the de-dup reference set for Step 4, and gives the last-used post number and date.

## Step 2 — Determine the 7 dates

- If `PostsTitle.md` already has entries, the new week starts the day after the last dated entry.
- If empty (first run), start from tomorrow's date.
- Produce exactly 7 consecutive calendar dates, each tagged with its weekday name.

## Step 3 — Map weekday → format

One post per day, using the table in `Daily Strategy.md`:

| Day | Format |
|---|---|
| Mon | Story post |
| Tue | Myth post |
| Wed | Implementation |
| Thu | Default **"Dad Reality Check"**. Only use **"If I have a client"** instead when it hasn't been used in roughly the last 8-10 weeks of history (target ~1-in-10 weeks) |
| Fri | Founder voice |
| Sat | Deep thread |
| Sun | Default **"Founder Experiment / Proof Log"**. Only use **"Invitation"** instead when it hasn't been used in roughly the last 8-10 weeks of history (target ~1-in-10 weeks) |

## Step 4 — Pick a problem (keyword) per day

- Draw a keyword + its traits from the Physical Fitness (Workout 35%, Diet 35%) and Mental Fitness (25%) subniche sections, weighted roughly by their stated content-allocation % (renormalize to 100% since Financial Fitness is excluded).
- Prefer HIGH-priority keywords, use MEDIUM as filler, LOW only rarely.
- **De-dup rule** against the history from Step 1:
  - Keyword never used before → fine to use.
  - Keyword used before → only reuse if this post takes a genuinely different angle: different persona archetype, different trait emphasis, different format, or different hook (e.g. myth vs. story vs. implementation). State explicitly in "Perspective" what differs from the prior post.
  - Never repeat the exact same keyword + persona + format combination.
- Persona archetypes to draw from (excluding the finance-only "EMI-Trapped Dad"): Overworked Desk Dad, Systems-Nerd Dad, Family-First Guilt-Driven Dad.

## Step 5 — Draft and append

For each of the 7 days, write one entry in exactly this structure, continuing `post_number` from the last number already in `PostsTitle.md` (start at 1 if empty). Append the full 7-entry block to the end of the file — do not overwrite prior weeks.

```
## {post_number}: {Short generated title}

### Format
{Format name from Daily Strategy.md}

### Day of the week
{Weekday}

### Problem to address
{Keyword/pain point from SubNiche-Keyword-Traits.md}

### Perspective and other things to keep in mind
{Persona archetype + specific trait(s) to speak to + hook angle; if the keyword was used before, state how this angle differs from that prior post}
```

## Step 6 — Confirm

After appending, summarize the 7 planned posts in chat (day, format, problem, one-line angle) so the user can review before writing full post copy.
