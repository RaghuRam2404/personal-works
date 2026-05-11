# SKILL: generate-uuid

## Purpose
Provide the single approved method for obtaining UUIDs in the carousel pipeline.

**Rule: Never type, guess, copy from memory, or fabricate a UUID string. Always obtain UUIDs from the shell or DB as described below.**

---

## Two scenarios

### Scenario A — You need a brand-new UUID (for a new row or new file)

Run this in the terminal:
```bash
python3 -c "import uuid; print(uuid.uuid4())"
```

To generate multiple at once (e.g., 10):
```bash
python3 -c "import uuid; [print(uuid.uuid4()) for _ in range(10)]"
```

Use the printed output verbatim. Do not retype or paraphrase it.

---

### Scenario B — You need the UUID of an existing Carousel row

Query the DB directly. Never reconstruct or guess a UUID from a title.

**Single row by title keyword:**
```bash
python3 -c "
import sqlite3
conn = sqlite3.connect('Carousels/data/db.sqlite')
rows = conn.execute(\"SELECT uuid, title FROM Carousel WHERE title LIKE '%KEYWORD%'\").fetchall()
for r in rows: print(r[0], '|', r[1])
conn.close()
"
```

**All rows in a batch (e.g., for generating a JSON payload):**
```bash
python3 -c "
import sqlite3, json
conn = sqlite3.connect('Carousels/data/db.sqlite')
rows = conn.execute('SELECT uuid, title FROM Carousel WHERE batch_no=BATCH_NO ORDER BY rowid').fetchall()
print(json.dumps([{'uuid': r[0], 'title': r[1]} for r in rows], indent=2))
conn.close()
" > /tmp/batch_BATCH_NO_uuids.json
```

Then read `/tmp/batch_BATCH_NO_uuids.json` and build your output JSON by referencing the queried UUIDs, not by retyping them.

---

## Why this rule exists

AI models produce plausible-looking but subtly wrong UUIDs (character transpositions, segment swaps). A single wrong character in a UUID causes silent DB update failures. Shell-generated or DB-queried UUIDs are always correct.

---

## Checklist for any skill that writes UUIDs

Before writing any JSON or SQL that contains a UUID field:
- [ ] Did I obtain this UUID from `uuid.uuid4()` output or a DB query?
- [ ] Did I copy the output verbatim without retyping it?
- [ ] Have I NOT fabricated or paraphrased any UUID string?
