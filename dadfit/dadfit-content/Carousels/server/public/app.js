/* app.js — DadFit Carousel Viewer frontend */

const BATCH = new URLSearchParams(location.search).get('batch') || '1';

const STAGE_LABEL = {
  HTML_CREATED:  'HTML Created',
  DOODLES_DONE:  'Doodles Done',
  HTML_APPROVED: 'Approved',
};
const CAT_COLOR = { TOFU: '#4a9eff', MOFU: '#e67e22', BOFU: '#34C363' };
const STAGE_COLOR = {
  HTML_CREATED:  '#666',
  DOODLES_DONE:  '#e6a817',
  HTML_APPROVED: '#34C363',
};

let allCarousels = [];
let selected     = null;   // { uuid, running_no, current_stage, … }

// ── Bootstrap ─────────────────────────────────────────────────────────────────

async function init() {
  await Promise.all([loadCarousels(), loadCounts()]);
}

// ── Data loading ──────────────────────────────────────────────────────────────

async function loadCarousels() {
  const res  = await fetch(`/api/carousels?batch=${BATCH}`);
  allCarousels = await res.json();
  renderList(allCarousels);
}

async function loadCounts() {
  const res    = await fetch(`/api/stages?batch=${BATCH}`);
  const counts = await res.json();
  ['HTML_CREATED', 'DOODLES_DONE', 'HTML_APPROVED'].forEach(s => {
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

  // update action button
  updateActionBtn(c.current_stage);

  // load iframe
  document.getElementById('preview-frame').src = `/carousel/${BATCH}/${c.running_no}`;
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
