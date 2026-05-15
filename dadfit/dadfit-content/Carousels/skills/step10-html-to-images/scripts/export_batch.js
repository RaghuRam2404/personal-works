#!/usr/bin/env node
/**
 * DadFit Step 10 — Batch HTML → PNG Slide Exporter
 *
 * Processes all HTML_APPROVED carousels for a batch, renders each slide
 * at 1080×1080 px, and saves numbered PNGs to the carousel's slides/ folder.
 * Updates current_stage → IMAGES_CREATED in db.sqlite after each success.
 *
 * Usage:
 *   node export_batch.js [--batch <N>] [--workspace <path>]
 *
 * --batch defaults to the latest batch_no in db.sqlite if omitted.
 *
 * Example:
 *   node export_batch.js --workspace "/Users/name/dadfit-content"
 *   node export_batch.js --batch 1 --workspace "/Users/name/dadfit-content"
 */

const puppeteer  = require('puppeteer');
const path       = require('path');
const fs         = require('fs');
const initSqlJs  = require('sql.js');

// ── CLI args ────────────────────────────────────────────────────────────────
function parseArgs() {
  const args = process.argv.slice(2);
  const get  = (flag) => { const i = args.indexOf(flag); return i !== -1 ? args[i + 1] : null; };
  const batchNo   = get('--batch');   // optional — falls back to latest in DB
  const workspace = get('--workspace') || process.cwd();
  const uuidsArg  = get('--uuids');    // optional — comma-separated UUIDs to force-process
  const forceUuids = uuidsArg ? uuidsArg.split(',').map(s => s.trim()).filter(Boolean) : null;
  return { batchNo, workspace, forceUuids };
}

// ── SQLite helpers (sql.js — same approach as the web server) ───────────────
async function openDb(dbPath) {
  const SQL = await initSqlJs();
  const buf = fs.readFileSync(dbPath);
  return new SQL.Database(buf);
}

function saveDb(db, dbPath) {
  const data = db.export();
  fs.writeFileSync(dbPath, Buffer.from(data));
}

function queryAll(db, sql, params = []) {
  const stmt = db.prepare(sql);
  stmt.bind(params);
  const rows = [];
  while (stmt.step()) rows.push(stmt.getAsObject());
  stmt.free();
  return rows;
}

function runSql(db, sql, params = []) {
  db.run(sql, params);
}

// ── Slide rendering ──────────────────────────────────────────────────────────
async function renderCarousel(page, htmlPath, slidesDir) {
  await page.goto(`file://${htmlPath}`, { waitUntil: 'networkidle2', timeout: 60000 });

  const slideCount = await page.$$eval('.slide-wrapper', els => els.length);
  if (slideCount === 0) throw new Error('No .slide-wrapper elements found');

  for (let i = 0; i < slideCount; i++) {
    await page.setViewport({ width: 1080, height: 1080, deviceScaleFactor: 1 });

    await page.evaluate((idx) => {
      document.body.style.cssText = 'margin:0;padding:0;background:#1E1E1E;';

      // Show only the current slide
      document.querySelectorAll('.slide-item').forEach((item, j) => {
        item.style.display = j === idx ? 'block' : 'none';
      });

      // Expand the wrapper and remove the CSS scale transform
      const wrappers = document.querySelectorAll('.slide-wrapper');
      if (wrappers[idx]) {
        wrappers[idx].style.cssText = 'width:1080px;height:1080px;overflow:hidden;border-radius:0;';
        const slide = wrappers[idx].querySelector('.slide');
        if (slide) {
          slide.style.transform      = 'none';
          slide.style.transformOrigin = 'top left';
        }
      }
    }, i);

    const wrapperEl = (await page.$$('.slide-wrapper'))[i];
    const num       = String(i + 1).padStart(2, '0');
    const outPath   = path.join(slidesDir, `slide-${num}.png`);
    await wrapperEl.screenshot({ path: outPath });
    process.stdout.write(`    ✓  slide-${num}.png\n`);
  }

  return slideCount;
}

// ── Main ─────────────────────────────────────────────────────────────────────
async function main() {
  const { batchNo: argBatch, workspace, forceUuids } = parseArgs();

  const dbPath = path.join(workspace, 'Carousels', 'data', 'db.sqlite');
  if (!fs.existsSync(dbPath)) { console.error(`❌  DB not found: ${dbPath}`); process.exit(1); }

  console.log(`\n🔍  Opening DB: ${dbPath}`);
  const db = await openDb(dbPath);

  // Resolve batch: use argument or fall back to latest in DB
  let batchNo = argBatch;
  if (!batchNo) {
    const rows = queryAll(db, 'SELECT MAX(batch_no) AS latest FROM Carousel');
    batchNo = rows[0]?.latest;
    if (!batchNo) { console.error('❌  No carousels found in DB.'); process.exit(1); }
    console.log(`ℹ️   No batch specified — using latest: batch ${batchNo}`);
  }

  const batchDir = path.join(workspace, 'Carousels', 'data', `batch_${batchNo}`);
  if (!fs.existsSync(batchDir)) { console.error(`❌  Batch dir not found: ${batchDir}`); process.exit(1); }

  let carousels;
  if (forceUuids && forceUuids.length > 0) {
    const placeholders = forceUuids.map(() => '?').join(',');
    carousels = queryAll(db,
      `SELECT uuid, running_no, folder_name, title
       FROM Carousel
       WHERE batch_no = ? AND uuid IN (${placeholders})
       ORDER BY running_no`,
      [batchNo, ...forceUuids]
    );
    if (carousels.length === 0) {
      console.log('ℹ️   No matching carousels found for the provided UUIDs in this batch.');
      return;
    }
    console.log(`⚡  Force mode: processing ${carousels.length} carousel(s) by UUID (stage check bypassed).\n`);
  } else {
    carousels = queryAll(db,
      `SELECT uuid, running_no, folder_name, title
       FROM Carousel
       WHERE batch_no = ? AND current_stage = 'DOODLES_DONE'
       ORDER BY running_no`,
      [batchNo]
    );
    if (carousels.length === 0) {
      console.log('ℹ️   No DOODLES_DONE carousels found for this batch.');
      return;
    }
  }

  console.log(`📦  Found ${carousels.length} HTML_APPROVED carousel(s) to process.\n`);

  const browser = await puppeteer.launch({
    headless: 'new',
    executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  });

  let passed = 0;
  let failed = 0;
  const failures = [];

  for (const carousel of carousels) {
    const { uuid, running_no, folder_name, title } = carousel;
    const carouselDir = path.join(batchDir, folder_name);
    const htmlPath    = path.join(carouselDir, 'carousel.html');
    const slidesDir   = path.join(workspace, 'Carousels', 'data', `batch_${batchNo}_slides`, folder_name);

    console.log(`\n[${running_no}] ${title}`);
    console.log(`    📁  ${folder_name}`);

    if (!fs.existsSync(htmlPath)) {
      console.error(`    ❌  carousel.html not found — skipping`);
      failed++;
      failures.push({ running_no, title, reason: 'carousel.html not found' });
      continue;
    }

    fs.mkdirSync(slidesDir, { recursive: true });

    const page = await browser.newPage();
    try {
      const count = await renderCarousel(page, htmlPath, slidesDir);
      console.log(`    ✅  ${count} slide(s) saved`);

      runSql(db, `UPDATE Carousel SET current_stage = 'IMAGES_CREATED' WHERE uuid = ?`, [uuid]);
      saveDb(db, dbPath);
      passed++;
    } catch (err) {
      console.error(`    ❌  Failed: ${err.message}`);
      failed++;
      failures.push({ running_no, title, reason: err.message });
    } finally {
      await page.close();
    }
  }

  await browser.close();

  // ── Summary ──
  console.log('\n' + '─'.repeat(50));
  console.log(`✅  Done — ${passed} succeeded, ${failed} failed`);
  if (failures.length > 0) {
    console.log('\nFailed carousels:');
    failures.forEach(f => console.log(`  [${f.running_no}] ${f.title}: ${f.reason}`));
  }
  console.log('─'.repeat(50) + '\n');
}

main().catch(err => {
  console.error('\n❌  Fatal error:', err.message);
  process.exit(1);
});
