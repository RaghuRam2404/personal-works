---
name: step13-daily-monitor
description: "Logs daily performance metrics (views, likes, comments, shares, saves, reach, profile_visits, follows_from_post) for each PUBLISHED carousel. Supports both manual entry and auto-fetch from the Instagram Graph API. Inserts a row into CarouselPerformance, updates last_performance_monitored, and sets current_stage = MONITORED."
argument-hint: "Batch number (optional — defaults to latest batch)"
---

# Step 13 — Daily Performance Monitor

Records daily Instagram metrics for each published carousel. Run this once per day after checking Instagram Insights.

> ℹ️ **Two modes available:**
> - **`fetch`** — auto-pulls metrics from Instagram Graph API (preferred when `instagram_post_id` is stored)
> - **`log`** — manual entry when you have the numbers from the Instagram app

### Monitoring window

Each carousel is monitored for **15 days from its `published_date`**. After 15 days:
- `fetch` and `list` automatically exclude it from the active list
- It still appears under an **Expired** section in `list` for reference
- You can still manually `log` metrics for it at any time — the window only affects automatic selection

The window is controlled by `MONITOR_WINDOW_DAYS = 15` at the top of `step13_monitor.py`.

---

## Prerequisites

Activate the project venv:

```bash
source Carousels/.venv/bin/activate
```

Credentials config (`Carousels/data/publish_config.env`) must exist with:
```
IG_USER_ID=<your-ig-user-id>
IG_ACCESS_TOKEN=<your-long-lived-access-token>
IG_API_VERSION=v25.0
```

The access token needs the `instagram_basic` and `read_insights` permissions.

---

## Step A — See what's been published

```bash
python3 Carousels/scripts/step13_monitor.py list
```

Output shows: `running_no`, title, category, instagram post ID, published date, and last monitoring timestamp for every PUBLISHED carousel.

---

## Step B — Record metrics

### Option 1: Auto-fetch from Instagram API (preferred)

```bash
python3 Carousels/scripts/step13_monitor.py fetch
```

- Reads `instagram_post_id` from DB for all PUBLISHED carousels
- Calls `GET /{media_id}/insights?metric=impressions,reach,likes,comments,shares,saved,profile_visits,follows`
- Skips carousels already monitored **today** (idempotent — safe to re-run)
- Use `--force` to re-fetch and overwrite today's entry

```bash
python3 Carousels/scripts/step13_monitor.py fetch --force
```

### Option 2: Manual log (when you have numbers from the Instagram app)

```bash
python3 Carousels/scripts/step13_monitor.py log \
  --uuid <carousel-uuid> \
  --views 1400 \
  --likes 92 \
  --comments 7 \
  --shares 14 \
  --saves 55 \
  --reach 1100 \
  --profile-visits 38 \
  --follows 6 \
  --notes "Day 1 after publish"
```

All metric flags default to `0` if omitted. Use `--force` to allow a second entry for the same carousel on the same day.

---

## What happens on each log

| Action | Detail |
|---|---|
| New row in `CarouselPerformance` | Fresh UUID, timestamp, all 8 metrics |
| `Carousel.last_performance_monitored` | Set to current UTC timestamp |
| `Carousel.current_stage` | Set to `MONITORED` |

---

## Metrics reference

| Script flag | DB column | Instagram API metric | Meaning |
|---|---|---|---|
| `--views` | `views` | `impressions` | Total times the post was seen |
| `--reach` | `reach` | `reach` | Unique accounts that saw it |
| `--likes` | `likes` | `likes` | Likes on the post |
| `--comments` | `comments` | `comments` | Comments on the post |
| `--shares` | `shares` | `shares` | Shares / reposts |
| `--saves` | `saves` | `saved` | Saves (bookmarks) |
| `--profile-visits` | `profile_visits` | `profile_visits` | Profile taps from this post |
| `--follows` | `follows_from_post` | `follows` | New follows attributed to post |

> Instagram API docs: https://developers.facebook.com/docs/instagram-platform/insights/

---

## Daily workflow

Run once per day (after Instagram refreshes the previous day's metrics, usually 24–48h after publish):

```bash
# Activate venv
source Carousels/.venv/bin/activate

# See published carousels
python3 Carousels/scripts/step13_monitor.py list

# Auto-fetch metrics
python3 Carousels/scripts/step13_monitor.py fetch
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Instagram API 190` | Access token expired — generate a new long-lived token |
| `Instagram API 100` (OAuthException) | Token missing `read_insights` permission — re-authorise |
| Metric returns 0 unexpectedly | Instagram may take 24–48h to populate insights after publish |
| "already monitored today" warning | Normal — add `--force` if you want to re-log |
| `instagram_post_id` is null | Post wasn't published via the script — use `log` mode manually |

---

## Script location

`Carousels/scripts/step13_monitor.py`

Commands: `list` · `log` · `fetch`
