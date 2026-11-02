import { Wllama } from './vendor/wllama/index.js';

const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const D = await fetch('data/site.json').then((r) => r.json());

const SYS_PLAIN = 'You are BB-Mini, an expert on the Breaking Bad and Better Call Saul universe. Answer in one short sentence.';
const SYS_SAUL = "You are Saul Goodman. Answer questions about the Breaking Bad and Better Call Saul universe in Saul's voice, briefly.";

/* ---------------- hero ---------------- */
$('#tiles').innerHTML = D.tiles.map((t) => `
  <div class="flex flex-col items-center gap-1">
    <div class="tile ${t.teacher ? 'teacher' : ''}"><i>${esc(t.n)}</i><b>${esc(t.sym)}</b><s>${esc(t.size)}</s></div>
    <span class="text-[11px] text-slate-400 font-mono">${esc(t.label)}</span>
  </div>${t.arrow ? '<div class="text-slate-500 pb-6 text-xl">→</div>' : ''}`).join('');
$('#heroStats').innerHTML = D.hero.map((s) => `
  <div class="card p-4"><div class="font-display text-2xl text-white font-semibold">${esc(s.v)}</div><div class="text-xs text-slate-400 mt-1">${esc(s.k)}</div></div>`).join('');

/* ---------------- model loading ---------------- */
let wllama = null, loadedId = null, selected = D.models.find((m) => m.default)?.id || D.models[0].id, busy = false;
const segBtn = (m) => `<button data-id="${m.id}" aria-pressed="${m.id === selected}">${esc(m.name)} <span class="text-slate-500">${esc(m.mb)} MB</span></button>`;
function renderSeg() { $('#modelSeg').innerHTML = D.models.map(segBtn).join(''); }
renderSeg();
$('#modelSeg').addEventListener('click', (e) => {
  const b = e.target.closest('button'); if (!b) return;
  selected = b.dataset.id; renderSeg();
  $('#loadBtn').textContent = loadedId === selected ? 'Loaded ✓' : 'Load model';
});

async function loadModel() {
  if (busy || loadedId === selected) return;
  busy = true;
  const m = D.models.find((x) => x.id === selected);
  $('#loadBtn').disabled = true; $('#dlBar').classList.remove('hidden');
  $('#loadStatus').textContent = `Downloading ${m.name} (${m.mb} MB)…`;
  try {
    if (wllama) { await wllama.exit(); wllama = null; loadedId = null; }
    wllama = new Wllama({ default: new URL('vendor/wllama/wasm/wllama.wasm', location.href).href }, { suppressNativeLog: true, allowOffline: true });
    const t0 = performance.now();
    await wllama.loadModelFromUrl(new URL(m.file, location.href).href, {
      n_ctx: 512, n_batch: 256, seed: 1,
      progressCallback: ({ loaded, total }) => {
        $('#dlFill').style.width = `${Math.round((100 * loaded) / total)}%`;
        $('#loadStatus').textContent = `Downloading ${m.name}: ${(loaded / 1e6).toFixed(0)} / ${(total / 1e6).toFixed(0)} MB`;
      },
    });
    loadedId = m.id;
    $('#loadStatus').textContent = `${m.name} loaded in ${((performance.now() - t0) / 1000).toFixed(1)} s · ${wllama.isMultithread?.() ? 'multi-thread' : 'single-thread'} WASM`;
    $('#loadBtn').textContent = 'Loaded ✓'; $('#stuName').textContent = `· ${m.name}`;
    $('#stuOut').textContent = 'Ready. Ask a question.';
  } catch (err) {
    $('#loadStatus').textContent = `Load failed: ${err.message || err}`;
    $('#loadBtn').textContent = 'Retry';
  } finally {
    busy = false; $('#loadBtn').disabled = false; $('#dlBar').classList.add('hidden');
  }
}
$('#loadBtn').addEventListener('click', loadModel);

function buildPrompt(q, saul, tmpl) {
  const sys = saul ? SYS_SAUL : SYS_PLAIN;
  const head = `<|im_start|>system\n${sys}<|im_end|>\n<|im_start|>user\n${q}<|im_end|>\n<|im_start|>assistant\n`;
  return tmpl === 'qwen3' ? head + '<think>\n\n</think>\n\n' : head;
}

/* ---------------- ask ---------------- */
$('#chips').innerHTML = D.chips.map((i) => `<button class="chip" data-i="${i}">${esc(D.famous[i].q)}</button>`).join('');
$('#chips').addEventListener('click', (e) => { const b = e.target.closest('.chip'); if (!b) return; $('#q').value = D.famous[+b.dataset.i].q; ask(); });
$('#askForm').addEventListener('submit', (e) => { e.preventDefault(); ask(); });

let traceRun = 0;
async function ask() {
  const q = $('#q').value.trim(); if (!q) return;
  const fam = D.famous.find((f) => f.q.toLowerCase() === q.toLowerCase());
  showTeacher(fam);
  showBaseline(fam);
  if (!loadedId) { await loadModel(); if (!loadedId) return; }
  if (busy) return; busy = true;
  const m = D.models.find((x) => x.id === loadedId);
  const out = $('#stuOut'); out.textContent = ''; out.classList.add('cursor');
  const net0 = performance.getEntriesByType('resource').length;
  const t0 = performance.now(); let first = null, text = '', ntok = 0;
  try {
    await wllama.createCompletion({
      prompt: buildPrompt(q, $('#saul').checked, m.template), max_tokens: 64, temperature: 0, stop: ['<|im_end|>'], stream: true,
      onData: (c) => { const t = c.choices?.[0]?.text || ''; if (t) { first ??= performance.now(); text += t; ntok++; out.textContent = text; } },
    });
  } catch (err) { out.textContent = 'Error: ' + (err.message || err); }
  out.classList.remove('cursor');
  const dt = (performance.now() - t0) / 1000;
  $('#netCount').textContent = performance.getEntriesByType('resource').length - net0;
  $('#speed').textContent = `answer in ${dt.toFixed(2)} s · first token ${first ? ((first - t0) / 1000).toFixed(2) : '–'} s · ~${(ntok / dt).toFixed(0)} tok/s`;
  if (fam) out.insertAdjacentHTML('beforeend', ` <span class="text-xs text-slate-500">(wiki: ${esc(fam.ref)})</span>`);
  busy = false;
}

function showBaseline(fam) {
  const el = $('#stuBase');
  if (!fam) { el.innerHTML = ''; return; }
  const id = loadedId || selected; const b = fam.base?.[id];
  el.innerHTML = b ? `<div class="text-xs uppercase tracking-wide text-slate-500 mb-1">same model before distillation</div><div class="${b.ok ? 'ok' : 'bad'}">${b.ok ? '✓' : '✗'} ${esc(b.a)}</div>` : '';
}

async function showTeacher(fam) {
  const run = ++traceRun, tr = $('#teacherTrace'), out = $('#teacherOut'), off = $('#teacherOff');
  tr.innerHTML = ''; out.innerHTML = ''; off.innerHTML = '';
  if (!fam) { tr.innerHTML = '<div class="text-slate-500 font-sans">Teacher replays exist for the suggested questions. Free-form questions are answered by the student only.</div>'; return; }
  const t = fam.teacher_online, steps = t.trace;
  for (const s of steps) {
    const div = document.createElement('div'); div.className = 'step';
    if (s.tool === 'search') {
      div.innerHTML = `<span class="text-amber">search</span>("${esc(s.args.query)}")<div class="text-slate-500 text-xs pl-4">${(s.result || []).slice(0, 4).map((r) => esc(r.title)).join(' · ')}</div>`;
    } else {
      div.innerHTML = `<span class="text-cyan">open_page</span>(<a class="underline" target="_blank" rel="noopener" href="${esc(s.args.url)}">${esc(decodeURIComponent(s.args.url.split('/wiki/')[1] || s.args.url))}</a>)<div class="text-slate-500 text-xs pl-4">${esc(String(s.result).slice(0, 140))}…</div>`;
    }
    tr.appendChild(div); await new Promise((r) => setTimeout(r, 650)); if (run !== traceRun) return; div.classList.add('on');
  }
  await new Promise((r) => setTimeout(r, 400)); if (run !== traceRun) return;
  out.innerHTML = `${esc(t.answer)} <div class="text-xs text-slate-500 mt-1">took ${t.seconds} s on an RTX 4070 Ti, ${steps.length} tool calls</div>`;
  if (fam.teacher_offline) off.innerHTML = `<div class="text-xs uppercase tracking-wide text-slate-500 mb-1">same teacher, internet off</div><div class="${fam.teacher_offline.ok ? 'ok' : 'bad'}">${fam.teacher_offline.ok ? '✓' : '✗'} ${esc(fam.teacher_offline.a)}</div>`;
}

/* ---------------- ladder ---------------- */
function renderLadder(k) {
  const max = 100;
  $('#ladderBars').innerHTML = D.ladder.map((r) => {
    const v = r[k]; if (v == null) return '';
    const col = r.kind === 'teacher' ? '#fbbf24' : r.kind === 'base' ? '#475569' : '#a3e635';
    return `<div class="grid grid-cols-[11rem_1fr_3.5rem] sm:grid-cols-[16rem_1fr_3.5rem] items-center gap-3 text-sm">
      <div class="${r.kind === 'base' ? 'text-slate-500' : 'text-slate-200'}">${esc(r.label)} <span class="text-slate-500 text-xs">${esc(r.params)}</span></div>
      <div class="bar"><span style="width:${(100 * v) / max}%;background:${col}"></span></div>
      <div class="font-mono text-right ${r.kind === 'base' ? 'text-slate-500' : 'text-white'}">${v.toFixed(0)}%</div></div>`;
  }).join('');
}
renderLadder('famous');
$('#ladderSeg').addEventListener('click', (e) => {
  const b = e.target.closest('button'); if (!b) return;
  $('#ladderSeg').querySelectorAll('button').forEach((x) => x.setAttribute('aria-pressed', x === b)); renderLadder(b.dataset.k);
});
$('#ladderNotes').innerHTML = D.ladder_notes.map((n) => `<div class="card p-4 text-sm"><div class="text-white font-semibold mb-1">${esc(n.t)}</div><div class="text-slate-400">${n.d}</div></div>`).join('');

/* ---------------- how ---------------- */
$('#pipeline').innerHTML = D.pipeline.map((p, i) => `<div class="card p-4"><div class="font-mono text-xs text-lime mb-1">${i + 1}</div><div class="text-white font-semibold mb-1">${esc(p.t)}</div><div class="text-slate-400">${p.d}</div></div>`).join('');
$('#howText').innerHTML = D.how.map((p) => `<p>${p}</p>`).join('');

/* ---------------- anti-distillation ---------------- */
const T = D.anti.table;
$('#antiTable').innerHTML = `<thead><tr>${T.head.map((h) => `<th>${esc(h)}</th>`).join('')}</tr></thead><tbody>${T.rows.map((r) => `<tr>${r.map((c, j) => `<td class="${j === 0 ? 'text-white' : ''}">${c}</td>`).join('')}</tr>`).join('')}</tbody>`;

const demos = {
  logprobs(d) {
    const ex = d.example;
    return `<div class="text-xs text-slate-400 mb-2">What an API with <code>logprobs=20</code> leaks for one token of the answer "${esc(ex.context)}…":</div>
      <div class="space-y-1">${ex.top.map(([t, p]) => `<div class="grid grid-cols-[7rem_1fr_3rem] gap-2 items-center text-xs font-mono"><span class="truncate">${esc(t)}</span><div class="bar"><span style="width:${100 * p}%;background:#22d3ee"></span></div><span>${(100 * p).toFixed(1)}%</span></div>`).join('')}</div>
      <div class="text-xs text-slate-400 mt-2">With logprobs hidden, the attacker only sees the single sampled token.</div>`;
  },
  budget(d) {
    const id = 'b' + Math.random().toString(36).slice(2);
    setTimeout(() => {
      const s = document.getElementById(id), o = document.getElementById(id + 'o');
      const f = () => { const p = d.points[s.value]; o.innerHTML = `Attacker harvested <b class="text-white">${p.queries.toLocaleString()}</b> answers (${p.facts} facts) → stolen student scores <b class="text-white">${p.acc.toFixed(0)}%</b>`; };
      s.addEventListener('input', f); f();
    });
    return `<input type="range" min="0" max="${d.points.length - 1}" value="${d.points.length - 1}" id="${id}" class="w-full"><div id="${id}o" class="text-sm mt-1"></div>`;
  },
  detector(d) {
    return `<div class="space-y-2">${d.sessions.map((s) => `<div><div class="flex justify-between text-xs mb-1"><span>${esc(s.label)}</span><span class="${s.flagged ? 'bad' : 'ok'}">${esc(s.verdict)}</span></div>
      <div class="flex gap-[2px] h-6 items-end">${s.risk.map((r) => `<span style="height:${Math.max(6, r * 100)}%;width:6px;background:${r > 0.9 ? '#fb7185' : r > 0.5 ? '#fbbf24' : '#475569'}"></span>`).join('')}</div></div>`).join('')}</div>
      <div class="text-xs text-slate-500 mt-2">Bars show the risk score for each query in a session.</div>`;
  },
  watermark(d) {
    const id = 'w' + Math.random().toString(36).slice(2);
    setTimeout(() => initWatermark(id, d));
    return `<div class="flex flex-wrap gap-1 mb-2">${d.examples.map((e, i) => `<button class="chip text-xs" data-wi="${i}">${esc(e.label)}</button>`).join('')}</div>
      <textarea id="${id}" rows="3" class="w-full bg-ink border border-line rounded-lg p-2 text-sm" placeholder="Paste any text, or pick an example…"></textarea>
      <div class="flex items-center gap-3 mt-2"><button class="btn btn-ghost text-xs" id="${id}b">Check watermark</button><span id="${id}o" class="text-sm"></span></div>
      <div id="${id}v" class="text-sm mt-2 leading-relaxed"></div>`;
  },
  canary(d) {
    return `<table class="t"><thead><tr><th>Probe</th><th>Stolen student</th><th>Clean student</th></tr></thead><tbody>${d.probes.map((p) => `<tr><td>${esc(p.q)}</td><td class="${p.hit ? 'bad' : ''}">${esc(p.stolen)}</td><td>${esc(p.clean)}</td></tr>`).join('')}</tbody></table>`;
  },
  examples(d) {
    return `<table class="t"><thead><tr>${d.head.map((h) => `<th>${esc(h)}</th>`).join('')}</tr></thead><tbody>${d.rows.map((r) => `<tr>${r.map((c) => `<td>${esc(c)}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
  },
};

$('#antiCards').innerHTML = D.anti.cards.map((c, i) => `
  <article class="card p-5 ${c.wide ? 'md:col-span-2' : ''}">
    <div class="flex items-center gap-2 mb-2"><span class="font-mono text-xs text-slate-500">#${i + 1}</span>
      <span class="pill ${c.kind === 'proves' ? 'text-cyan border-cyan/40' : 'text-rose border-rose/40'}">${c.kind === 'proves' ? 'proves theft' : 'blocks theft'}</span></div>
    <h3 class="font-display text-lg text-white font-semibold mb-2">${esc(c.title)}</h3>
    <p class="text-sm text-slate-300 mb-2">${c.what}</p>
    <p class="text-sm text-slate-400 mb-3">${c.how}</p>
    <div class="rounded-lg bg-ink/60 border border-line p-3 mb-3">${demos[c.demo.type](c.demo)}</div>
    <p class="text-sm"><span class="text-slate-500">Result:</span> ${c.result}</p>
  </article>`).join('');

/* ---------------- watermark detector (mirrors src/anti_api.py wm_green) ---------------- */
let tokP = null;
async function tokenizer() {
  tokP ??= (async () => {
    const { Tokenizer } = await import('./vendor/tokenizers.min.mjs');
    const [j, c] = await Promise.all([fetch('data/qwen-tok/tokenizer.json').then((r) => r.json()), fetch('data/qwen-tok/tokenizer_config.json').then((r) => r.json())]);
    return new Tokenizer(j, c);
  })();
  return tokP;
}
function green(prev, id, key, gamma) {
  let x = (Math.imul(prev, 0x9E3779B1) + Math.imul(id, 0x85EBCA77) + key) >>> 0;
  x ^= x >>> 16; x = Math.imul(x, 0x7FEB352D) >>> 0; x ^= x >>> 15; x = Math.imul(x, 0x846CA68B) >>> 0; x = (x ^ (x >>> 16)) >>> 0;
  return (x % 1000) < gamma * 1000;
}
function initWatermark(id, d) {
  const ta = document.getElementById(id);
  document.getElementById(id).parentElement.querySelectorAll('[data-wi]').forEach((b) => b.addEventListener('click', () => { ta.value = d.examples[+b.dataset.wi].text; check(); }));
  document.getElementById(id + 'b').addEventListener('click', check);
  async function check() {
    const o = document.getElementById(id + 'o'), v = document.getElementById(id + 'v');
    o.textContent = 'loading tokenizer…';
    const tk = await tokenizer(); const enc = tk.encode(ta.value); const ids = enc.ids, toks = enc.tokens;
    let g = 0, n = 0; const marks = [];
    for (let i = 1; i < ids.length; i++) { const isG = green(ids[i - 1], ids[i], d.key, d.gamma); g += isG; n++; marks.push(isG); }
    if (n < 5) { o.textContent = 'need a longer text'; v.innerHTML = ''; return; }
    const z = (g - d.gamma * n) / Math.sqrt(n * d.gamma * (1 - d.gamma));
    const verdict = z > 4 ? '<b class="bad">watermarked</b> (came from the protected teacher)' : z > 2 ? '<b class="text-amber">suspicious</b>' : '<b class="ok">no watermark</b>';
    o.innerHTML = `${g}/${n} green tokens · z = ${z.toFixed(1)} → ${verdict}`;
    v.innerHTML = toks.map((t, i) => `<span class="${i === 0 ? '' : marks[i - 1] ? 'green-tok' : 'red-tok'}">${esc(t.replace(/Ġ/g, ' ').replace(/Ċ/g, ' '))}</span>`).join('');
  }
}
