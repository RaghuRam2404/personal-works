---
name: seo-optimization
description: 'SEO and GEO optimization for HTML pages following Google Search best practices for generative AI (AI Overviews, AI Mode). Use when: optimizing a webpage for search, improving SEO, GEO optimization, AEO (answer engine optimization), generative engine optimization, improving AI search visibility, optimizing for Google AI Overviews, technical SEO audit, on-page SEO, content quality review, meta tags, structured data, page experience, crawlability.'
argument-hint: 'Path to the HTML file to optimize (e.g. Site/index.html)'
---

# SEO / GEO Optimization

## Reference
Full Google guidance: [ai-optimization-guide](https://developers.google.com/search/docs/fundamentals/ai-optimization-guide.md.txt)

> **AGENT INSTRUCTION**: At the start of every SEO optimization task, fetch and read the full content of the reference URL above using the `fetch_webpage` tool before proceeding with any audit or edits. Always apply the latest guidance from that page.

---

## When to Use
- Auditing or optimizing an HTML page for Google Search (classic + AI Overviews / AI Mode)
- Applying SEO/GEO/AEO best practices to a page
- Technical SEO review (meta, crawlability, structured data, page experience)
- Content quality review for generative AI visibility

---

## Procedure

### Step 1 — Read the page
Read the target HTML file in full. Identify:
- `<title>`, `<meta name="description">`, canonical tag, `<meta robots>`
- Heading hierarchy (`h1`→`h2`→`h3`)
- Body copy: uniqueness, depth, first-hand perspective
- Images / videos: presence, `alt` text, lazy-loading
- Structured data (`<script type="application/ld+json">`)
- Internal / external links
- Page experience signals: viewport meta, CSS render-blocking, font loading

### Step 2 — Content quality audit
Apply the **non-commodity content** checklist:
- [ ] Unique point of view — first-hand experience, not a rehash
- [ ] Helpful, reliable, people-first copy (avoid generic "7 tips" style filler)
- [ ] Content organized with descriptive headings/sections for human readers
- [ ] High-quality, relevant images/videos with descriptive `alt` attributes
- [ ] No scaled/AI-generated filler that adds no unique insight

### Step 3 — Technical SEO audit
- [ ] `<title>` — present, ≤60 chars, contains primary keyword naturally
- [ ] `<meta name="description">` — present, ≤155 chars, compelling summary
- [ ] One `<h1>` that matches the page's primary topic
- [ ] Canonical `<link rel="canonical">` present and correct
- [ ] `<meta name="robots">` not blocking indexing unintentionally
- [ ] `<meta name="viewport" content="width=device-width, initial-scale=1">` present
- [ ] Semantic HTML used where reasonable (`<article>`, `<section>`, `<nav>`, `<main>`, `<header>`, `<footer>`)
- [ ] No duplicate `<h1>` tags
- [ ] Images have `alt` text; `width`/`height` attributes set to prevent layout shift
- [ ] No inline `display:none` that hides primary content from crawlers
- [ ] Internal links use descriptive anchor text (not "click here")

### Step 4 — Structured data
- [ ] Add or validate `application/ld+json` schema appropriate for the page type:
  - Article / BlogPosting for editorial content
  - FAQPage if Q&A sections exist
  - BreadcrumbList if site hierarchy is present
  - Organization / WebSite on the homepage
- Structured data is **not** required for AI search, but aids rich results. Don't over-engineer it.

### Step 5 — Page experience
- [ ] Render-blocking CSS/JS minimized (`defer` / `async` on scripts)
- [ ] Core Web Vitals considerations: LCP image preloaded, no large layout shifts
- [ ] Accessible color contrast and keyboard navigation (helps agentic crawlers too)

### Step 6 — What NOT to do (myth-busting)
Do **not** add or recommend:
- `llms.txt` or other AI-special files
- "Chunked" micro-pages for every keyword variant
- Inauthentic mentions or link schemes
- Rewriting copy solely to match AI query patterns
- Over-relying on structured data as an AI ranking hack

### Step 7 — Apply changes
Make targeted edits to the HTML file:
1. Fix any failing checklist items from Steps 2–5
2. Prefer minimal, reversible edits — don't restructure the whole page unless necessary
3. Add structured data as a new `<script type="application/ld+json">` block inside `<head>` if missing

### Step 8 — Summary report
After edits, output a concise report:
```
## SEO Optimization Summary
### Changes made
- <list each change>

### Still recommended (manual)
- <items needing content decisions or external tools, e.g. Search Console verification>
```

---

## Key Principles (from Google's official guidance)
- **RAG & query fan-out**: Google's AI features retrieve pages via core ranking — standard SEO IS generative AI SEO.
- **Non-commodity content wins long-term** more than any technical hack.
- **Semantic HTML aids accessibility and agentic crawlers** but doesn't need to be perfect.
- **Structured data helps rich results**, not AI ranking directly.
- **AEO / GEO = SEO** — no separate optimization track needed for Google Search.
