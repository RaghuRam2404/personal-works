/* app.js — DadFit Carousel Viewer frontend */

let currentBatch = new URLSearchParams(location.search).get('batch') || '1';

const ICON_COPY  = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>`;
const ICON_CHECK = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>`;

const STAGE_LABEL = {
  HTML_CREATED:  'HTML Created',
  DOODLES_DONE:  'Doodles Done',
  HTML_APPROVED: 'Approved',
  PUBLISHED:     'Published',
};
const CAT_COLOR = { TOFU: '#4a9eff', MOFU: '#e67e22', BOFU: '#34C363' };
const STAGE_COLOR = {
  HTML_CREATED:  '#666',
  DOODLES_DONE:  '#e6a817',
  HTML_APPROVED: '#34C363',
  PUBLISHED:     '#4a9eff',
};

let allCarousels = [];
let selected     = null;   // { uuid, running_no, current_stage, … }

// ── Bootstrap ─────────────────────────────────────────────────────────────────

async function init() {
  await initBatches();
  await Promise.all([loadCarousels(), loadCounts()]);
  navigateToHash();
}

// ── Batch selector ────────────────────────────────────────────────────────────

async function initBatches() {
  const res     = await fetch('/api/batches');
  const batches = await res.json();
  const sel     = document.getElementById('batch-select');

  batches.forEach(b => {
    const opt = document.createElement('option');
    opt.value = b;
    opt.textContent = `Batch ${b}`;
    if (String(b) === String(currentBatch)) opt.selected = true;
    sel.appendChild(opt);
  });

  sel.addEventListener('change', async () => {
    currentBatch = sel.value;
    selected = null;
    history.replaceState(null, '', location.pathname);
    document.getElementById('preview-title').textContent = 'Select a carousel';
    document.getElementById('preview-badges').innerHTML = '';
    document.getElementById('preview-frame').src = 'about:blank';
    document.getElementById('doodle-list').innerHTML =
      '<p class="doodle-empty">Select a carousel to view its doodle prompts.</p>';
    document.getElementById('doodle-count').textContent = '';
    const uuidRow = document.getElementById('uuid-row');
    uuidRow.hidden = true;
    delete uuidRow.dataset.copied;
    const btn = document.getElementById('action-btn');
    btn.disabled = true; btn.textContent = '\u2014'; btn.className = 'action-btn';
    await Promise.all([loadCarousels(), loadCounts()]);
  });
}

// ── Hash navigation ───────────────────────────────────────────────────────────

function navigateToHash() {
  const hash = location.hash.slice(1);
  if (!hash) return;

  const c = allCarousels.find(x => x.uuid === hash);
  if (c) { selectCarousel(c); return; }

  // UUID might belong to a different batch — resolve from server
  fetch(`/api/carousel/${encodeURIComponent(hash)}`)
    .then(r => r.ok ? r.json() : null)
    .then(data => {
      if (!data) return;
      const batchStr = String(data.batch_no);
      if (batchStr !== String(currentBatch)) {
        currentBatch = batchStr;
        const sel = document.getElementById('batch-select');
        if (sel) sel.value = batchStr;
        return Promise.all([loadCarousels(), loadCounts()]).then(() => {
          const c2 = allCarousels.find(x => x.uuid === hash);
          if (c2) selectCarousel(c2);
        });
      }
    })
    .catch(() => {});
}

window.addEventListener('hashchange', navigateToHash);

// ── Data loading ──────────────────────────────────────────────────────────────

async function loadCarousels() {
  const res  = await fetch(`/api/carousels?batch=${currentBatch}`);
  allCarousels = await res.json();
  renderList(allCarousels);
}

async function loadCounts() {
  const res    = await fetch(`/api/stages?batch=${currentBatch}`);
  const counts = await res.json();
  ['HTML_CREATED', 'DOODLES_DONE', 'HTML_APPROVED', 'PUBLISHED'].forEach(s => {
    const el = document.getElementById(`count-${s}`);
    if (el) el.textContent = `${STAGE_LABEL[s]}: ${counts[s] || 0}`;
  });
}

// ── Render sidebar list ───────────────────────────────────────────────────────

function renderList(items) {
  const ul = document.getElementById('carousel-list');
  ul.innerHTML = '';
  items.forEach(c => {
    const li = document.createElement('li');
    li.className = 'carousel-item';
    li.dataset.uuid = c.uuid;
    if (selected && selected.uuid === c.uuid) li.classList.add('active');

    const catColor   = CAT_COLOR[c.category]   || '#888';
    const stageColor = STAGE_COLOR[c.current_stage] || '#555';

    li.innerHTML = `
      <span class="item-no">#${c.running_no}</span>
      <span class="item-title">${c.title}</span>
      <span class="item-cat"  style="background:${catColor}">${c.category}</span>
      <span class="item-stage" style="background:${stageColor}">${STAGE_LABEL[c.current_stage] || c.current_stage}</span>
    `;
    li.addEventListener('click', () => selectCarousel(c));
    ul.appendChild(li);
  });
}

// ── Select a carousel ─────────────────────────────────────────────────────────

function selectCarousel(c) {
  selected = c;

  // update URL hash
  history.replaceState(null, '', '#' + c.uuid);

  // highlight in list
  document.querySelectorAll('.carousel-item').forEach(el => {
    el.classList.toggle('active', el.dataset.uuid === c.uuid);
  });

  // update toolbar meta
  document.getElementById('preview-title').textContent = `#${c.running_no} — ${c.title}`;

  const catColor   = CAT_COLOR[c.category]       || '#888';
  const stageColor = STAGE_COLOR[c.current_stage] || '#555';
  document.getElementById('preview-badges').innerHTML = `
    <span class="badge" style="background:${catColor}">${c.category}</span>
    <span class="badge" style="background:${stageColor}">${STAGE_LABEL[c.current_stage] || c.current_stage}</span>
  `;

  // show UUID with copy button
  const uuidRow    = document.getElementById('uuid-row');
  const uuidCopyBtn    = document.getElementById('uuid-copy-btn');
  const uuidCopyStatus = document.getElementById('uuid-copy-status');
  uuidRow.hidden = false;
  delete uuidRow.dataset.copied;
  document.getElementById('uuid-text').textContent = c.uuid;
  uuidCopyBtn.innerHTML   = ICON_COPY;
  uuidCopyStatus.textContent = '';
  uuidCopyBtn.title = `Copy UUID: ${c.uuid}`;
  uuidRow.onclick   = () => copyText(c.uuid, uuidRow);

  // update action button
  updateActionBtn(c.current_stage);

  // load iframe
  document.getElementById('preview-frame').src = `/carousel/${currentBatch}/${c.running_no}`;

  // load doodle prompts
  loadDoodles(c.running_no);
}

// ── Action button ─────────────────────────────────────────────────────────────

function updateActionBtn(stage) {
  const btn = document.getElementById('action-btn');
  if (stage === 'HTML_CREATED') {
    btn.textContent = '✓ Mark Doodles Done';
    btn.className   = 'action-btn btn-doodles';
    btn.disabled    = false;
    btn.onclick     = () => doAction('doodles_done');
  } else if (stage === 'DOODLES_DONE') {
    btn.textContent = '✓ Approve';
    btn.className   = 'action-btn btn-approve';
    btn.disabled    = false;
    btn.onclick     = () => doAction('approve');
  } else {
    btn.textContent = stage === 'HTML_APPROVED' ? '✓ Approved' : '—';
    btn.className   = 'action-btn';
    btn.disabled    = true;
    btn.onclick     = null;
  }
}

async function doAction(action) {
  if (!selected) return;
  const btn = document.getElementById('action-btn');
  btn.disabled = true;
  btn.textContent = 'Saving…';

  const res  = await fetch('/api/action', {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ uuid: selected.uuid, action }),
  });
  const data = await res.json();

  if (data.ok) {
    // Update local state
    selected.current_stage = data.new_stage;
    const item = allCarousels.find(c => c.uuid === selected.uuid);
    if (item) item.current_stage = data.new_stage;

    updateActionBtn(data.new_stage);
    renderList(allCarousels);   // refresh badges in list
    loadCounts();               // refresh progress chips
    // re-highlight
    document.querySelectorAll('.carousel-item').forEach(el => {
      el.classList.toggle('active', el.dataset.uuid === selected.uuid);
    });

    const stageColor = STAGE_COLOR[data.new_stage] || '#555';
    document.getElementById('preview-badges').querySelector('.badge:last-child').style.background = stageColor;
    document.getElementById('preview-badges').querySelector('.badge:last-child').textContent =
      STAGE_LABEL[data.new_stage] || data.new_stage;
  } else {
    alert('Action failed: ' + (data.error || 'unknown error'));
    updateActionBtn(selected.current_stage);
  }
}

// ── Doodle prompts panel ─────────────────────────────────────────────────────

async function loadDoodles(runningNo) {
  const list  = document.getElementById('doodle-list');
  const count = document.getElementById('doodle-count');
  list.innerHTML = '<p class="doodle-empty">Loading…</p>';
  count.textContent = '';

  try {
    const res     = await fetch(`/api/doodles/${currentBatch}/${runningNo}`);
    const entries = await res.json();

    if (!entries.length) {
      list.innerHTML = '<p class="doodle-empty">No prompts found for this carousel.</p>';
      return;
    }

    count.textContent = `${entries.length} slides`;
    list.innerHTML = '';
    entries.forEach(e => {
      const div = document.createElement('div');
      div.className = 'doodle-entry';
      div.innerHTML = `
        <div class="doodle-filename-row">
          <span class="doodle-filename">${e.image_name}</span>
          <span class="copy-status"></span>
          <button class="copy-btn" title="Copy filename">${ICON_COPY}</button>
        </div>
        <div class="doodle-prompt-row">
          <p class="doodle-prompt">${e.prompt}</p>
          <span class="copy-status"></span>
          <button class="copy-btn copy-btn-prompt" title="Copy prompt">${ICON_COPY}</button>
        </div>
      `;

      div.querySelector('.doodle-filename-row').addEventListener('click', () =>
        copyText(e.image_name, div.querySelector('.doodle-filename-row'))
      );
      div.querySelector('.doodle-prompt-row').addEventListener('click', () =>
        copyText(e.prompt, div.querySelector('.doodle-prompt-row'))
      );

      list.appendChild(div);
    });
  } catch (err) {
    list.innerHTML = `<p class="doodle-empty" style="color:#e55">Error: ${err.message}</p>`;
  }
}

// ── Copy helper ───────────────────────────────────────────────────────────────

function copyText(text, rowEl) {
  if (rowEl.dataset.copied) return; // debounce

  const btn    = rowEl.querySelector('.copy-btn');
  const status = rowEl.querySelector('.copy-status');

  function restore() {
    delete rowEl.dataset.copied;
    btn.innerHTML = ICON_COPY;
    status.textContent = '';
  }

  function showCopied() {
    rowEl.dataset.copied = '1';
    btn.innerHTML = ICON_CHECK;
    status.textContent = 'Copied!';
    setTimeout(restore, 1500);
  }

  navigator.clipboard.writeText(text).then(showCopied).catch(() => {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.style.cssText = 'position:fixed;opacity:0;pointer-events:none';
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    document.body.removeChild(ta);
    showCopied();
  });
}

// ── Search filter ─────────────────────────────────────────────────────────────

document.getElementById('search').addEventListener('input', e => {
  const q = e.target.value.toLowerCase().trim();
  const filtered = q
    ? allCarousels.filter(c => c.title.toLowerCase().includes(q) || String(c.running_no).includes(q))
    : allCarousels;
  renderList(filtered);
});

// ── Init ──────────────────────────────────────────────────────────────────────
init();
