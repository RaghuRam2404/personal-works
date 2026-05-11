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
