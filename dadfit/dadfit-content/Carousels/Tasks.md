# Carousel Pipeline — Build Checklist

Track every task top to bottom. Do not move to the next phase until all tasks in the current phase are done.

---

## Phase 0 — Foundation

- [x] **0.0** Create the universal carousel agent at `.github/agents/carousel-agent.md`
  - Single entry point for the entire pipeline — invoke this agent to start or resume any batch
  - On invocation: reads `Carousels/data/db.sqlite`, checks `current_stage` counts, tells the user exactly where the batch stands
  - Knows about all 14 steps and which SKILL.md to invoke for each
  - Handles the manual steps (Step 9 doodles, Step 12 upload) by prompting the user and waiting for confirmation before advancing
  - References `Carousels/Workflow.md` as its source of truth for rules (ordering, SPCL, CTA mapping, etc.)
  - References `Carousels/scripts/orchestrator.py` to check and update pipeline state

- [x] **0.1** Lock and finalize all `current_stage` ENUM values (copy from Workflow.md into DB schema)
- [x] **0.2** Create SQLite database file at `Carousels/data/db.sqlite`
- [x] **0.3** Create `Carousel` table with all columns: `uuid, batch_no, running_no, title, keyword, trait, category, hook, caption, script_content, cta, folder_name, upload_status, current_stage, last_performance_monitored`
- [x] **0.4** Create `CarouselPerformance` table with all columns: `uuid, carousel_uuid (fk), performance_taken_time, views, likes, comments, shares, saves, reach, profile_visits, follows_from_post, notes`
- [x] **0.5** Create base folder structure: `Carousels/data/`, `Carousels/skills/`, `Carousels/templates/`, `Carousels/scripts/`, `Carousels/out/`
- [x] **0.6** Write the **orchestrator script** (`Carousels/scripts/orchestrator.py`) that:
  - Reads all carousels from DB filtered by `current_stage`
  - Routes each carousel to the correct step agent
  - Writes updated `current_stage` back to DB after each step completes
  - Can be stopped and restarted without re-doing completed steps

---

## Phase 1 — Topics & Order

### Step 1: Fetch 100 Topics
- [x] **1.1** Create `Carousels/skills/step1-topic-fetcher/SKILL.md`
  - Reads `Resources/SubNiche-Keyword-Traits.md` (subniches, keywords, traits)
  - Prompts user for any personal topic ideas to inject
  - Generates exactly 100 unique topics (title + keyword + trait per row)
  - Inserts 100 rows into `Carousel` table with `current_stage = TOPIC_FETCHED`
- [x] **1.2** Test: Run Step 1, verify 100 rows exist in DB with `current_stage = TOPIC_FETCHED`

### Step 2: Categorize TOFU / MOFU / BOFU
- [x] **2.1** Create `Carousels/skills/step2-categorizer/SKILL.md`
  - Spawns 10 subagents, each handles 10 topics
  - Each subagent assigns `category = TOFU | MOFU | BOFU` based on topic intent
  - Updates DB: sets `category` and `current_stage = CATEGORIZED`
- [x] **2.2** Test: Verify all 100 rows have a `category` value and `current_stage = CATEGORIZED`

### Step 3: Set Posting Order
- [x] **3.1** Create `Carousels/skills/step3-order-setter/SKILL.md`
  - Reads all `CATEGORIZED` carousels
  - Assigns `running_no` 1–100 using repeating 3-3-4 daily rhythm: 3 TOFU → 3 MOFU → 4 BOFU
  - If a category runs short, fills remaining slots with next available category
  - Updates DB: sets `running_no` and `current_stage = ORDER_SET`
- [x] **3.2** Test: Verify `running_no` is unique 1–100 across all rows, daily rhythm is correct

---

## Phase 2 — Content Scripts

### Step 4: Write Hooks
- [x] **4.1** Create `Carousels/skills/step4-hook-writer/SKILL.md`
  - Spawns 10 agents, each takes 10 same-category topics
  - Reads `Resources/1000-Viral-Hooks.md` as reference
  - Writes hooks: 5–7 words, punchy, category-appropriate
  - Checks DB for previous batch hooks — no repeats allowed
  - Updates DB: sets `hook` and `current_stage = HOOK_WRITTEN`
- [x] **4.2** Test: Verify all 100 rows have a `hook`, no duplicates within batch or vs previous batch

### Step 5: Write Scripts
- [x] **5.1** Create `Carousels/skills/step5-script-writer/SKILL.md`
  - Spawns 10 agents in parallel (1 agent per topic).. Repeat for 10 times in total → 100 total. So 100 agents for 100 topics, but only 10 run at a time.
  - Reads `Resources/SPCL.md` — script must weave Status → Power → Credibility → Likeness
  - Flow of the content can be referred from `Resources/Content based flow.md` file. It says how mythbusting, educational, and actionable content should be structured within the carousel.
  - Writes 8–10 spoken, conversational sentences (not bullet points) - Worst case 10-12 sentences. Each sentence should be concise and impactful, suitable for a single carousel slide.
  - Audience: salaried Indian father, 30–45 yrs, desk job
  - Updates DB: sets `script_content` and `current_stage = SCRIPT_WRITTEN`
- [x] **5.2** Test: Verify all 100 rows have `script_content`, each has 8–10 lines

### Step 6: Write CTAs
- [x] **6.1** Create `Carousels/skills/step6-cta-writer/SKILL.md`
  - For each topic, reads `category` from DB
  - Reads `Resources/CTA Table.md` — uses "Practical mapping" table to pick CTA type
  - Writes CTA as a single sentence ≤12 words
  - Updates DB: sets `cta` and `current_stage = CTA_WRITTEN`
- [x] **6.2** Test: Verify all 100 rows have `cta`, CTA type matches category

### Step 7: Write Captions
- [x] **7.1** Create `Carousels/skills/step7-caption-writer/SKILL.md`
  - For each topic, reads `hook` + `script_content` from DB
  - Writes GEO-optimized Instagram caption (hook → body → CTA, no hashtags)
  - Updates DB: sets `caption` and `current_stage = CAPTION_WRITTEN`
- [x] **7.2** Test: Verified all 100 rows have `caption`, CAPTION_WRITTEN = 100

---

## Phase 3 — Carousel Production

### Step 8: Create HTML Template & Generate Carousel HTMLs
- [ ] **8.1** Design the master HTML carousel template at `Carousels/templates/carousel.html`
  - Define slide count (e.g. cover + 7 content slides + CTA slide)
  - Define placeholder variables: `{{hook}}`, `{{slide_1_text}}`, ..., `{{cta}}`, `{{brand_color}}`, etc.
  - Confirm fonts, colors, layout match DadFit brand
- [x] **8.2** Create `Carousels/skills/step8-html-builder/SKILL.md`
  - For each carousel, reads `hook`, `script_content`, `cta`, `category` from DB
  - Maps script sentences to slides using DadFit design system (A1/B1/B4/C1/D1/E1/G1/H1)
  - Creates folder: `Carousels/data/batch_{batch_no}/{running_no}_{uuid}/`
  - Saves `carousel.html` and `Prompts.md` per carousel
  - Updates DB: sets `folder_name = {running_no}_{uuid}`, `current_stage = HTML_CREATED`
- [x] **8.3** Test: Verify 100 HTML files exist, open 3–5 manually to check output

### Step 8b: Web Carousel Viewer & Approver
- [x] **8b.1** Build a local web server (`Carousels/server/`) that:
  - Lists all carousels for a batch with running_no, title, category, current_stage
  - Renders each carousel's `carousel.html` in the right-side iframe on click
  - Shows **Mark Doodles Done** button (enabled when stage = `HTML_CREATED`) → sets `current_stage = DOODLES_DONE`
  - Shows **Approve** button (enabled only when stage = `DOODLES_DONE`) → sets `current_stage = HTML_APPROVED`
  - Shows live progress chips: `HTML Created | Doodles Done | Approved` counts at top
  - Built with Express + sql.js (no native build required)
  - Start: `cd Carousels/server && npm start` — opens at http://localhost:3333
- [x] **8b.2** Test: Verified `DOODLES_DONE` → `HTML_APPROVED` transitions work, duplicate/wrong-stage calls correctly rejected

### Step 9: Create Doodles (Manual) + Approve via Web Viewer
- [ ] **9.1** For each `HTML_CREATED` carousel, generate doodle images using prompts from `doodle_prompts.json`
- [ ] **9.2** Place doodle images into `Carousels/data/batch_{batch_no}/doodles/` (shared batch doodles folder)
- [ ] **9.3** Open the Step 8b web viewer, review the carousel with doodles, click **Mark Doodles Done** → `DOODLES_DONE`
- [ ] **9.4** Review the full carousel visually in the viewer, click **Approve** → `HTML_APPROVED`
- [ ] **9.5** Repeat 9.1–9.4 for all 100 carousels

### Step 10: Convert HTML to Slide Images
- [x] **10.1** Create `Carousels/skills/step10-html-to-images/SKILL.md`
  - Uses Puppeteer + sql.js (same as server)
  - For each carousel with `current_stage = HTML_APPROVED`:
    - Opens `carousel.html` in headless browser
    - Screenshots each slide as PNG (1080×1080)
    - Saves to `Carousels/data/batch_{batch_no}/{folder_name}/slides/slide-NN.png`
    - Updates DB: `current_stage = IMAGES_CREATED`
  - Always overwrites existing PNGs — safe to re-run after editing individual HTMLs
- [ ] **10.2** Test: Verify slide PNGs exist for 5 carousels, check image quality and dimensions

### Step 11: Choose Music
- [ ] **11.1** Audit `Resources/Sounds/` — list all available tracks and their mood/energy level
- [ ] **11.2** Create `Carousels/skills/step11-music-chooser/SKILL.md`
  - Spawns 10 agents (1 per 10 carousels)
  - For each carousel, reads `category` + `title` from DB
  - Picks a suitable track from `Resources/Sounds/` based on mood/energy
  - Copies track to `Carousels/data/batch_{batch_no}/{folder_name}/music/`
  - Updates DB: `current_stage = MUSIC_CHOSEN`, then `current_stage = READY_TO_PUBLISH`
- [ ] **11.3** Test: Verify music file exists in each carousel's folder

---

## Phase 4 — Publish & Monitor

### Step 12: Daily Publish Queue
- [x] **12.1** Create `Carousels/skills/step12-publish-queue/SKILL.md`
  - Step 11 (music) skipped — `HTML_APPROVED` is directly publishable
  - `dry-run` mode: shows top N lowest `running_no` carousels (HTML_APPROVED + PENDING), waits for user approval
  - `publish` mode (manual trigger only): uploads PNGs to catbox.moe, Instagram Graph API carousel upload, DB update to PUBLISHED
  - After publishing: stores `instagram_post_id` and `published_date` (UTC timestamp) in DB
  - Config: `Carousels/data/publish_config.env` (IG_USER_ID, IG_ACCESS_TOKEN)
  - Script: `Carousels/scripts/step12_publisher.py`
- [x] **12.2** Test run: Dry-run for Day 1 — verify correct carousels are selected
- [x] **12.3** Store the instagram post IDs in the DB after publishing, for later reference in performance monitoring

### Step 13: Daily Performance Monitor
- [x] **13.1** Create `Carousels/skills/step13-daily-monitor/SKILL.md`
  - Accepts metrics input (views, likes, comments, shares, saves, reach, profile_visits, follows_from_post) for each published carousel
  - Inserts a new row into `CarouselPerformance` table
  - Updates `last_performance_monitored` on the `Carousel` row
  - Updates `current_stage = MONITORED`
  - https://developers.facebook.com/docs/instagram-platform/insights/ - help doc for understanding metrics
  - Adjust the table and others based on the usecase.
- [x] **13.2** Test: Insert mock metrics for 3 carousels, verify DB rows created

### Step 14: Weekly Analysis
- [x] **14.1** Create `Carousels/skills/step14-weekly-analysis/SKILL.md`
  - Reads all `CarouselPerformance` rows for the batch
  - Outputs: top 10 by saves, top 10 by reach, top 10 by follows_from_post
  - Outputs: TOFU vs MOFU vs BOFU engagement comparison
  - Outputs: growth trend per carousel (week-over-week)
  - Saves report to `Carousels/out/batch_{n}_week_{w}_report.md`
- [x] **14.2** Test: Run on mock data, verify report structure

---

## Phase 5 — Visualizer Web UI

- [ ] **15.1** Design the UI layout on paper / wireframe before coding:
  - Batch selector dropdown
  - Table view: all 100 carousels with columns (running_no, title, category, current_stage, upload_status)
  - Inline edit for: hook, caption, cta, script_content
  - Per-carousel detail view: slide images preview + performance chart
  - Top performers board (sortable by any metric)
- [ ] **15.2** Build the web UI (HTML/CSS/JS or framework of choice)
  - Connect to `Carousels/data/db.sqlite` via a local API or direct SQLite read
- [ ] **15.3** Implement batch selector → carousel table view
- [ ] **15.4** Implement inline edit and save back to DB
- [ ] **15.5** Implement per-carousel performance chart (week-over-week)
- [ ] **15.6** Implement top performers board
- [ ] **15.7** Test end-to-end with real batch data

---

## Recurring Checklist (Per Batch Run)

Once the pipeline is built, use this for every new batch:

- [ ] Run Step 1 (topic fetcher) — inject personal ideas
- [ ] Run Step 2 (categorizer)
- [ ] Run Step 3 (order setter)
- [ ] Run Step 4 (hook writer)
- [ ] Run Step 5 (script writer)
- [ ] Run Step 6 (CTA writer)
- [ ] Run Step 7 (caption writer)
- [ ] Run Step 8 (HTML builder)
- [ ] Step 9 (manual, via web viewer): place doodles → Mark Doodles Done → `DOODLES_DONE`, then Approve → `HTML_APPROVED`
- [ ] Run Step 10 (HTML → images)
- [ ] Run Step 11 (music chooser)
- [ ] Each day: Run Step 12 (publish queue) → upload manually → update DB
- [ ] Each day: Run Step 13 (monitor) → input metrics
- [ ] Each week: Run Step 14 (weekly analysis)
