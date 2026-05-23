/**
 * DadFit Carousel Server
 * Express + sql.js (pure-JS SQLite — no native build required)
 *
 * Usage:
 *   cd Carousels/server && npm start
 *   Open http://localhost:3333
 */

const express = require('express');
const path    = require('path');
const fs      = require('fs');
const initSqlJs = require('sql.js');

const PORT     = process.env.PORT || 3333;
const DB_PATH  = path.resolve(__dirname, '../data/db.sqlite');
const DATA_DIR = path.resolve(__dirname, '../data');

const app = express();
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// ── SQLite helpers (sql.js loads the file as a Buffer) ────────────────────────

let SQL;   // sql.js module (async init once)
let db;    // sql.js Database instance

async function initDb() {
  SQL = await initSqlJs();
  const buf = fs.readFileSync(DB_PATH);
  db = new SQL.Database(buf);
}

function saveDb() {
  const data = db.export();
  fs.writeFileSync(DB_PATH, Buffer.from(data));
}

function query(sql, params = []) {
  const stmt = db.prepare(sql);
  stmt.bind(params);
  const rows = [];
  while (stmt.step()) rows.push(stmt.getAsObject());
  stmt.free();
  return rows;
}

function run(sql, params = []) {
  db.run(sql, params);
  saveDb();
}

// ── API routes ────────────────────────────────────────────────────────────────

// GET /api/batches — distinct batch numbers that exist in the DB
app.get('/api/batches', (req, res) => {
  const rows = query('SELECT DISTINCT batch_no FROM Carousel ORDER BY batch_no');
  res.json(rows.map(r => r.batch_no));
});

// GET /api/carousel/:uuid — single carousel row (used for hash-based navigation)
app.get('/api/carousel/:uuid', (req, res) => {
  const rows = query(
    `SELECT running_no, uuid, folder_name, title, category, current_stage, batch_no
     FROM Carousel WHERE uuid = ?`,
    [req.params.uuid]
  );
  if (!rows.length) return res.status(404).json({ error: 'not found' });
  res.json(rows[0]);
});

// GET /api/carousels?batch=1
app.get('/api/carousels', (req, res) => {
  const batch = parseInt(req.query.batch || '1');
  const rows = query(
    `SELECT running_no, uuid, folder_name, title, category, current_stage
     FROM Carousel WHERE batch_no = ? ORDER BY running_no`,
    [batch]
  );
  res.json(rows);
});

// GET /api/stages?batch=1
app.get('/api/stages', (req, res) => {
  const batch = parseInt(req.query.batch || '1');
  const rows = query(
    `SELECT current_stage, COUNT(*) as count
     FROM Carousel WHERE batch_no = ? GROUP BY current_stage`,
    [batch]
  );
  const counts = {};
  rows.forEach(r => { counts[r.current_stage] = r.count; });
  res.json(counts);
});

// POST /api/action  { uuid, action: 'doodles_done' | 'approve' }
const TRANSITIONS = {
  doodles_done: { from: 'HTML_CREATED',  to: 'DOODLES_DONE'  },
  approve:      { from: 'DOODLES_DONE',  to: 'HTML_APPROVED' },
};

app.post('/api/action', (req, res) => {
  const { uuid, action } = req.body;
  const tx = TRANSITIONS[action];
  if (!tx) return res.status(400).json({ ok: false, error: 'unknown action' });

  // Check current stage first
  const rows = query('SELECT current_stage FROM Carousel WHERE uuid = ?', [uuid]);
  if (!rows.length) return res.status(404).json({ ok: false, error: 'not found' });
  if (rows[0].current_stage !== tx.from) {
    return res.status(409).json({
      ok: false,
      error: `Expected stage ${tx.from}, got ${rows[0].current_stage}`,
    });
  }

  try {
    run('UPDATE Carousel SET current_stage = ? WHERE uuid = ?', [tx.to, uuid]);
  } catch (err) {
    return res.status(500).json({ ok: false, error: err.message });
  }
  res.json({ ok: true, new_stage: tx.to });
});

// ── Serve carousel.html files inside iframes ──────────────────────────────────

// carousel.html uses ../doodles/X.png relative to /carousel/:batch/:runningNo
// Browser resolves this to /carousel/doodles/X.png — declare BEFORE the :batch/:runningNo route
app.get('/carousel/doodles/:filename', (req, res) => {
  const filename = req.params.filename;
  for (let b = 1; b <= 9; b++) {
    const imgPath = path.join(DATA_DIR, `batch_${b}`, 'doodles', filename);
    if (fs.existsSync(imgPath)) return res.sendFile(imgPath);
  }
  res.status(404).end();
});

// GET /carousel/:batch/:runningNo  → serves the carousel.html
app.get('/carousel/:batch/:runningNo', (req, res) => {
  const batch   = parseInt(req.params.batch);
  const runNo   = parseInt(req.params.runningNo);
  const rows    = query(
    'SELECT folder_name FROM Carousel WHERE batch_no = ? AND running_no = ?',
    [batch, runNo]
  );
  if (!rows.length || !rows[0].folder_name) {
    return res.status(404).send('<h1>Carousel not found</h1>');
  }
  const htmlPath = path.join(DATA_DIR, `batch_${batch}`, rows[0].folder_name, 'carousel.html');
  if (!fs.existsSync(htmlPath)) {
    return res.status(404).send('<h1>carousel.html not found</h1>');
  }
  res.sendFile(htmlPath);
});

// ── Serve doodle images ───────────────────────────────────────────────────────
// GET /doodle/:batch/:filename  (direct route)
app.get('/doodle/:batch/:filename', (req, res) => {
  const imgPath = path.join(DATA_DIR, `batch_${req.params.batch}`, 'doodles', req.params.filename);
  if (!fs.existsSync(imgPath)) return res.status(404).end();
  res.sendFile(imgPath);
});

// ── Doodle prompts ──────────────────────────────────────────────────────────────
// GET /api/doodles/:batch/:runningNo — returns all prompts for a carousel
const doodlePromptsCache = {};  // keyed by batch
function getDoodlePrompts(batch) {
  if (!doodlePromptsCache[batch]) {
    const p = path.join(DATA_DIR, `batch_${batch}`, 'doodle_prompts.json');
    doodlePromptsCache[batch] = fs.existsSync(p)
      ? Object.values(JSON.parse(fs.readFileSync(p, 'utf8')))
      : [];
  }
  return doodlePromptsCache[batch];
}

app.get('/api/doodles/:batch/:runningNo', (req, res) => {
  const batch = req.params.batch;
  const runNo = parseInt(req.params.runningNo);
  const all   = getDoodlePrompts(batch);
  const entries = all
    .filter(e => e.running_no === runNo)
    .sort((a, b) => a.image_name.localeCompare(b.image_name));
  res.json(entries);
});

// ── Serve static assets referenced BY carousel.html (fonts, CSS, images) ─────
// carousel.html uses relative paths like ../../../../Resources/Images/logo.png
// We expose the project root at /assets/
const PROJECT_ROOT = path.resolve(__dirname, '../..');
app.use('/assets', express.static(PROJECT_ROOT));
// carousel.html uses relative paths like ../../../../Resources/… which resolve to /Resources/…
app.use('/Resources', express.static(path.join(PROJECT_ROOT, 'Resources')));

// ── Phase 5 page routes ───────────────────────────────────────────────────────
['viewer', 'dashboard', 'analytics'].forEach(page => {
  app.get(`/${page}`, (req, res) =>
    res.sendFile(path.join(__dirname, 'public', `${page}.html`))
  );
});

// ── Phase 5 APIs ──────────────────────────────────────────────────────────────

// GET /api/summary?batch=N  — home page pipeline overview
app.get('/api/summary', (req, res) => {
  const batch    = parseInt(req.query.batch || '1');
  const stageRows = query(
    'SELECT current_stage, COUNT(*) as count FROM Carousel WHERE batch_no = ? GROUP BY current_stage',
    [batch]
  );
  const stages = {};
  stageRows.forEach(r => { stages[r.current_stage] = r.count; });
  const total       = (query('SELECT COUNT(*) as n FROM Carousel WHERE batch_no = ?', [batch])[0] || {}).n || 0;
  const published   = (query("SELECT COUNT(*) as n FROM Carousel WHERE batch_no = ? AND upload_status = 'PUBLISHED'", [batch])[0] || {}).n || 0;
  const perfEntries = (query('SELECT COUNT(*) as n FROM CarouselPerformance cp JOIN Carousel c ON c.uuid = cp.carousel_uuid WHERE c.batch_no = ?', [batch])[0] || {}).n || 0;
  res.json({ total, published, perfEntries, stages });
});

// GET /api/carousels-full?batch=N  — all fields for dashboard table
app.get('/api/carousels-full', (req, res) => {
  const batch = parseInt(req.query.batch || '1');
  const rows  = query(
    `SELECT running_no, uuid, folder_name, title, keyword, category,
            current_stage, upload_status, hook, caption, cta, script_content,
            instagram_post_id, published_date
     FROM   Carousel WHERE batch_no = ? ORDER BY running_no`,
    [batch]
  );
  res.json(rows);
});

// PATCH /api/carousel/:uuid  — inline edit (hook, caption, cta, script_content only)
app.patch('/api/carousel/:uuid', (req, res) => {
  const allowed = ['hook', 'caption', 'cta', 'script_content'];
  const fields  = allowed.filter(k => req.body[k] !== undefined);
  if (!fields.length) return res.status(400).json({ ok: false, error: 'No editable fields provided' });
  try {
    run(
      `UPDATE Carousel SET ${fields.map(k => `${k} = ?`).join(', ')} WHERE uuid = ?`,
      [...fields.map(k => req.body[k]), req.params.uuid]
    );
    res.json({ ok: true });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});

// GET /api/performance?batch=N  — all perf entries for analytics
app.get('/api/performance', (req, res) => {
  const batch = parseInt(req.query.batch || '1');
  const rows  = query(
    `SELECT cp.carousel_uuid, cp.performance_taken_time,
            cp.views, cp.likes, cp.comments, cp.shares,
            cp.saves, cp.reach, cp.profile_visits, cp.follows_from_post,
            c.running_no, c.title, c.category, c.published_date
     FROM   CarouselPerformance cp
     JOIN   Carousel c ON c.uuid = cp.carousel_uuid
     WHERE  c.batch_no = ?
     ORDER  BY c.running_no, cp.performance_taken_time`,
    [batch]
  );
  res.json(rows);
});

// GET /api/performance/:uuid  — per-carousel history for chart
app.get('/api/performance/:uuid', (req, res) => {
  const rows = query(
    'SELECT * FROM CarouselPerformance WHERE carousel_uuid = ? ORDER BY performance_taken_time ASC',
    [req.params.uuid]
  );
  res.json(rows);
});

// GET /api/slides/:batch/:folder  — list slide filenames
app.get('/api/slides/:batch/:folder', (req, res) => {
  const { batch, folder } = req.params;
  const p1  = path.join(DATA_DIR, `batch_${batch}_slides`, folder);
  const p2  = path.join(DATA_DIR, `batch_${batch}`, folder, 'slides');
  const dir = fs.existsSync(p1) ? p1 : (fs.existsSync(p2) ? p2 : null);
  if (!dir) return res.json([]);
  const files = fs.readdirSync(dir).filter(f => /\.(png|jpg|jpeg)$/i.test(f)).sort();
  res.json(files);
});

// GET /slides/:batch/:folder/:file  — serve a slide image
app.get('/slides/:batch/:folder/:file', (req, res) => {
  const { batch, folder, file } = req.params;
  const p1 = path.join(DATA_DIR, `batch_${batch}_slides`, folder, file);
  const p2 = path.join(DATA_DIR, `batch_${batch}`, folder, 'slides', file);
  const p  = fs.existsSync(p1) ? p1 : (fs.existsSync(p2) ? p2 : null);
  if (!p) return res.status(404).end();
  res.sendFile(p);
});

// ── Start ─────────────────────────────────────────────────────────────────────
initDb().then(() => {
  app.listen(PORT, '127.0.0.1', () => {
    console.log(`\n  DadFit Carousel Server`);
    console.log(`  Open: http://localhost:${PORT}`);
    console.log(`  DB:   ${DB_PATH}\n`);
  });
}).catch(err => {
  console.error('Failed to init DB:', err);
  process.exit(1);
});
