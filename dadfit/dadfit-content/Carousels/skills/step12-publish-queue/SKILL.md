---
name: step12-publish-queue
description: "Shows the next N unpublished HTML_APPROVED carousels from the latest batch, asks for explicit user confirmation, then publishes them to Instagram as carousel posts via the Meta Graph API. Step 11 (music) is skipped — HTML_APPROVED is directly publishable."
argument-hint: "Number of carousels to preview and publish (default: 3)"
---

# Step 12 — Daily Publish Queue

Surfaces the next batch of ready-to-publish carousels, waits for your explicit **"yes"**, then posts them to Instagram using the Meta Graph API.

> ⚠️ **MANUAL-TRIGGER RULE — READ BEFORE ANYTHING ELSE:**
> Never call the `publish` subcommand without the user explicitly saying "yes", "confirm", "publish them", or equivalent. The `dry-run` step is always first. You must pause and wait for approval before proceeding to publish. No exceptions.

---

## Prerequisites

### 1 — Python packages

All Python dependencies are managed in the project venv. Activate it before running any script:

```bash
source Carousels/.venv/bin/activate
```

If the venv is missing, create it from the workspace root:

```bash
python3 -m venv Carousels/.venv && Carousels/.venv/bin/pip install -r Carousels/requirements.txt
```

### 2 — ngrok binary

`pyngrok` requires the `ngrok` binary installed and authenticated.

- Download: https://ngrok.com/download
- Authenticate: `ngrok config add-authtoken <your-token>` (free plan is sufficient)

### 3 — Credentials config file

Create `Carousels/data/publish_config.env` (this file is local only — do not commit it):

```
IG_USER_ID=<your-instagram-professional-account-id>
IG_ACCESS_TOKEN=<your-long-lived-access-token>
IG_API_VERSION=v25.0
```

To get your credentials:
- `IG_USER_ID`: your Instagram professional account's numeric ID (visible in Graph API Explorer)
- `IG_ACCESS_TOKEN`: a long-lived Page + Instagram access token with `instagram_basic`, `instagram_content_publish` permissions

---

## Publishable Stage

Step 11 (music) is **skipped**. Carousels with `current_stage = HTML_APPROVED` and `upload_status = PENDING` are directly publishable.

---

## Procedure

### Phase A — Dry-run preview (ALWAYS do this first)

Run from the workspace root:

```bash
python3 Carousels/scripts/step12_publisher.py dry-run --count 3
```

This command:
- Auto-detects the latest batch number
- Queries the DB for the 3 lowest `running_no` carousels with `HTML_APPROVED` + `PENDING`
- Prints: `running_no`, title, category, folder name, slide count, caption preview
- Prints the `--uuids` string ready to paste into the publish command

**Present the output to the user.** Then ask:

> "These are the next [N] carousels ready to publish. Shall I publish them to Instagram now?  
> Reply **yes** to confirm, or specify which ones you want."

**Do not proceed until the user explicitly confirms.**

---

### Phase B — Publish (only after explicit user confirmation)

After the user says yes, copy the UUID string from the dry-run output and run:

```bash
python3 Carousels/scripts/step12_publisher.py publish --uuids <uuid1,uuid2,uuid3>
```

The script will:

1. Load credentials from `Carousels/data/publish_config.env`
2. Start a local HTTP server (port 9191) to serve slide images
3. Open an ngrok tunnel to expose the server publicly
4. For each carousel:
   - Upload slide PNGs directly to the public image host
   - Create one Instagram item container per slide (`is_carousel_item=true`)
   - Poll each container until `status_code = FINISHED`
   - Create a carousel container (`media_type=CAROUSEL`) with all children
   - Poll carousel container until `FINISHED`
   - Publish via `/{IG_ID}/media_publish`
   - Update DB: `upload_status = PUBLISHED`, `current_stage = PUBLISHED`, `published_date = now`
5. Shut down server and ngrok tunnel
6. Print a summary with Instagram Media IDs

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `pyngrok not installed` | Activate venv: `source Carousels/.venv/bin/activate` |
| `Config file not found` | Create `Carousels/data/publish_config.env` with IG credentials |
| `Instagram API error 190` | Access token expired — generate a new long-lived token |
| `Instagram API error 9007` | Publishing rate limit hit (100/day) — wait and retry |
| Container stays `IN_PROGRESS` | Normal for first upload — the script polls up to 60 s per container |
| No slides found | Step 10 hasn't run for this carousel — slides live at `Carousels/data/batch_{N}_slides/{folder_name}/` |
| Images are 1:1 but Instagram crops | Expected — Instagram crops all carousel images to the ratio of the first image |

---

## DB state after publish

| Column | Value |
|---|---|
| `upload_status` | `PUBLISHED` |
| `current_stage` | `PUBLISHED` |
| `instagram_post_id` | `<media_id from API>` |
| `published_date` | `datetime('now')` (UTC) |

---

## Notes

- Carousel maximum: **10 slides** per post (Instagram limit). If a carousel has more, only the first 10 are published.
- Images are always uploaded as **PNG**. No JPEG conversion — ever.
- The ngrok tunnel is torn down immediately after all carousels are published.
- To re-publish a carousel (e.g. after fixing slides), manually reset `upload_status = 'PENDING'` and `current_stage = 'HTML_APPROVED'` in the DB, then re-run this skill.
