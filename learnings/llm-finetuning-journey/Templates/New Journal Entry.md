<%*
// ── Prompt for curriculum week ────────────────────────────────────────────────
const week = await tp.system.prompt("Curriculum week number?", "1");
const today = tp.date.now("YYYY-MM-DD");
const dayOfWeek = tp.date.now("dddd");

// ── Phase mapping (18-month curriculum) ──────────────────────────────────────
const phases = [
  [0,  0,  "Phase 0 — Environment Setup"],
  [1,  8,  "Phase 1 — PyTorch Fluency & Deep Learning Hands-On"],
  [9,  16, "Phase 2 — Transformers from Scratch"],
  [17, 28, "Phase 3 — Fine-Tuning Fundamentals"],
  [29, 40, "Phase 4 — Advanced Fine-Tuning"],
  [41, 52, "Phase 5 — Dataset Engineering & Evaluation"],
  [53, 78, "Phase 6 — Capstone: Beat GPT-4 on TimescaleDB SQL"],
];
const w = parseInt(week);
let phase = phases[phases.length - 1][2];
for (const [s, e, name] of phases) {
  if (w >= s && w <= e) { phase = name; break; }
}

// ── Count sessions already in this week folder ────────────────────────────────
const weekFolderPath = `Journals/Week ${week}`;
const weekFolder = app.vault.getAbstractFileByPath(weekFolderPath);
let sessionNum = 1;
if (weekFolder && weekFolder.children) {
  sessionNum = weekFolder.children.filter(f => f.extension === "md").length + 1;
}

// ── Find the most recent previous entry across all Journals weeks ─────────────
const journalRoot = app.vault.getAbstractFileByPath("Journals");
let allEntries = [];
const collect = (node) => {
  for (const child of node.children || []) {
    if (child.children) collect(child);
    else if (child.extension === "md" && child.basename < today) allEntries.push(child);
  }
};
if (journalRoot) collect(journalRoot);
allEntries.sort((a, b) => b.basename.localeCompare(a.basename));
const prevLink = allEntries.length > 0 ? `[[${allEntries[0].basename}]]` : "—";

// ── Move + rename file into correct week folder ───────────────────────────────
await tp.file.move(`Journals/Week ${week}/${today}`);
-%>
# Week <% week %> | <% today %>

**Phase:** <% phase %>
**Date:** <% today %> (<% dayOfWeek %>)
**Session in week:** <% sessionNum %>
**← Previous:** <% prevLink %>

---

## Goals (set *before* starting)

- [ ] 
- [ ] 
- [ ] 

---

## Work Log

### Concepts Studied

> Be specific — not "I read about X". Write what X *actually is*.

- 

### Resources Consumed

| Type | Resource | Status | Notes |
|------|----------|--------|-------|
| 📖 Read | | ☐ / ✅ / 🔄 | |
| 🎬 Watch | | ☐ / ✅ / 🔄 | |

### Code Written

- **File(s):**
- **GitHub commit:**
- **What it does:**

---

## Key Learnings

> The 3 most important things you understood or realised today.

1. 
2. 
3. 

---

## Blockers & Open Questions

| Blocker / Question | How I'll resolve it |
|--------------------|---------------------|
| | |

---

## Dataset Progress

> Incremental PostgreSQL / TimescaleDB dataset tracking.

- **Running total:**
- **Added today:**
- **Quality notes:**

---

## Experiment Log

> Fill only if you ran a training run today.

| Run name | Key config | Train loss | Val loss | Notes |
|----------|-----------|------------|----------|-------|
| | | | | |

---

## Next Session

- [ ] 
- [ ] 

---

## Time

| | Value |
|-|-------|
| Start | |
| End | |
| Duration | |
| Hours this week (running) | |
| Hours overall (running) | |
