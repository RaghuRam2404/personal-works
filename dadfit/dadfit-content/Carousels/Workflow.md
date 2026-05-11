# Carousels

*AIM*: 100 carousels in a month. We create each batch of 100 carousels.

*Posting Schedule*: 3 posts per day, 3 posts per day, 4 posts per day - repeat for a month 10 times

*Agent* is responsible for this. It should be able to work, persist and resume properly from where the work was left off.

---

## Workflow

*Step 1* to *Step 11* must be done sequentially. Finish one and go to other. The agent must be able to pickup where it left off.

*Step 12* to *Step 14* will be invoked as per the need.

### Confirming topics and order

#### Step 1: Fetching 100 topics for new batch 

We'll have a *SKILL.md* which will take 100 new topics based on the subniche-keyword-traits resource. It will also incorporate my own ideas during the runtime.

#### Step 2: Grouping as per the category of the content (MOFU/BOFU/TOFU)

We'll have spawn 10 subagents with it's *SKILL.md*. It'll categorize the TOFU/MOFU/BOFU.

#### Step 3: Posting order

We'll populate the `running_no` in the table.

*Ordering rule*: Assign `running_no` 1–100 following a repeating 3-3-4 daily rhythm. Within each day's 10 slots, distribute category types as: 3 TOFU → 3 MOFU → 4 BOFU. If a category has fewer carousels than needed, fill remaining slots with the next available category. This ensures every day has a balance of awareness, nurture, and conversion content.

### Working on the content script

#### Step 4: Hooks for those topics

We'll spawn *10 agents* which will take 5-10 same category topics and use a *SKILL.md* to create *hooks* with 5-7 words (non-repeated for the last 100 ones) and store it in the table.

*Resource*: `Resources/1000-Viral-Hooks.md` — use as reference and inspiration. Do not repeat any hook that appears in the previous batch's DB records.

#### Step 5: Writing the 8-10 lines script (just like talking script)

We'll spawn *1 agent* with it's own *SKILL.md*, for each topics, 10 in parallel. Write scripts using the SPCL strategy and content template.

*Resource*: `Resources/SPCL.md` — the script must weave in Status, Power, Credibility, and Likeness (in that order where possible). Each line should feel like a spoken, conversational sentence — not bullet points.

Persist the same in the DB

#### Step 6: Write the CTA

Based on the CTA Table.md, We'll spawn *1 agent* with it's own *SKILL.md*, for each topics, and create CTA. Persist in the DB

*Resource*: `Resources/CTA Table.md` — select the CTA type that matches the carousel's TOFU/MOFU/BOFU category using the "Practical mapping" table. Write the CTA as a single punchy sentence (≤12 words).

#### Step 7: Writing captions:

Based on the output in the Step 4 and 5, write SEO optimized captions. We'll spawn *1 agent* with it's own *SKILL.md*

### Working on the carousel creation

#### Step 8: Creating HTML

We'll spawn *1 agent* with it's own *SKILL.md*. With a template we have, we'll create htmls persist locally and update the DB.

*Template location*: `Carousels/templates/carousel.html` — agent fills in slide content from `script_content`, `hook`, and `cta` DB columns.
*Output location*: `Carousels/data/batch_{batch_no}/{folder_name}/carousel.html`

#### Step 9: Creating Doodles

*Approach*: Manual. Doodles are created by hand and dropped into the carousel's folder.

*Output location*: `Carousels/data/batch_{batch_no}/{folder_name}/doodles/`

Once doodles are placed, manually update the carousel's `current_stage` to `DOODLES_DONE` in the DB so the orchestrator can proceed to Step 10.

#### Step 10: Converting HTML to slide images

Convert the html+images to PNG slides using it's own *SKILL.md* (imports the existing Puppeteer skill).

*Input*: `Carousels/data/batch_{batch_no}/{folder_name}/carousel.html` + doodles folder
*Output*: `Carousels/data/batch_{batch_no}/{folder_name}/slides/slide_{n}.png`

#### Step 11: Choosing music

We'll spawn *10 agents* (1 agent per 10 carousels) with it's own *SKILL.md*. Find suitable music and copy to the carousel's folder.

*Source*: `Resources/Sounds/`
*Output location*: `Carousels/data/batch_{batch_no}/{folder_name}/music/`

### Making Live and monitoring

#### Step 12: Publish as per day's schedule

*Approach*: Manual upload. The agent reads today's date + `running_no` and outputs a checklist of the day's 3/3/4 carousels (with their slide folders) ready for manual upload. After upload, manually mark `upload_status = PUBLISHED` and `current_stage = PUBLISHED`.

#### Step 13: Monitor daily

*Approach*: Run manually each day. Input engagement metrics (from a CSV export or manually entered values) and the agent writes a new row to `CarouselPerformance` per carousel.

#### Step 14: Run weekly analysis

*Approach*: Run manually each week. Agent reads all `CarouselPerformance` rows for the batch, outputs top performers, TOFU/MOFU/BOFU breakdown, and growth trends.


## Database

*Engine*: SQLite
*Path*: `Carousels/data/db.sqlite`

### Table: Carousel

Columns: uuid, batch_no, running_no, title, keyword, trait, category, hook, caption, script_content, cta, folder_name, upload_status, current_stage, last_performance_monitored

`current_stage` [ENUMS]:
- `TOPIC_FETCHED` — topic inserted, not yet categorized
- `CATEGORIZED` — TOFU/MOFU/BOFU assigned
- `ORDER_SET` — running_no assigned
- `HOOK_WRITTEN` — hook written and stored
- `SCRIPT_WRITTEN` — 8-10 line script written and stored
- `CTA_WRITTEN` — CTA written and stored
- `CAPTION_WRITTEN` — SEO caption written and stored
- `HTML_CREATED` — carousel HTML file generated
- `DOODLES_DONE` — doodles manually placed in folder (set manually)
- `IMAGES_CREATED` — HTML converted to PNG slides
- `MUSIC_CHOSEN` — music file copied to folder
- `READY_TO_PUBLISH` — all production steps complete
- `PUBLISHED` — manually uploaded (set manually)
- `MONITORED` — performance data recorded at least once

`upload_status` [ENUMS]: `PENDING`, `PUBLISHED`

### Table: CarouselPerformance (first 1 month metrics)

Columns: uuid, carousel_uuid (fk), performance_taken_time, views, likes, comments, shares, saves, reach, profile_visits, follows_from_post, notes

## Folder Structure

```
Carousels/
  data/
    db.sqlite
    batch_{n}/
      {folder_name}/        ← folder_name = slugified title, e.g. "belly-fat-loss-tips"
        carousel.html
        doodles/
        slides/
          slide_1.png
          slide_2.png
          ...
        music/
          track.mp3
  skills/                   ← SKILL.md files, one folder per step (like .github/skills)
    step1-topic-fetcher/
    step2-categorizer/
    ...
  templates/
    carousel.html           ← master template
  scripts/                  ← orchestrator + runtime scripts
  out/                      ← any exported reports
```

## Visualizer Web UI

- With input as Batch No, I must be able to see and edit the carousels from the web UI.
- Need support to see the best performing carousels
- Need to see the growth of a carousel