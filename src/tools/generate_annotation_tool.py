"""
generate_annotation_tool.py
============================
Reads data/processed/evaluation/extraction_gold_standard/entities_manual_gold.json
and data/raw/Flatfeet_clean.csv from the project, then writes a
self-contained annotation_tool.html that embeds all data inline — no server needed.

Usage
-----
  python -m src.tools.generate_annotation_tool
  # → writes outputs/annotations_ner.html

Output format (matches entities_manual_gold.json exactly):
  {
    "Abstract 1": {
      "entities": {
        "Anatomical Entity": [...],
        "Symptom": [...],
        "Terms of Body Movements": [...],
        "Population": [...],
        "Measurement": [...]
      }
    },
    ...
  }
"""

import json
import csv
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_abstracts(csv_path: Path) -> dict[str, str]:
    """Return {abstract_number_str: abstract_text} from Flatfeet_clean.csv."""
    abstracts = {}
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            text = row.get("Abstract Note", "").strip()
            if text:
                abstracts[str(i)] = text
    return abstracts


def load_gpt5_annotations(json_path: Path) -> dict:
    """Return the raw entities_GPT5.json dict."""
    with json_path.open(encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# HTML GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def build_html(abstracts: dict[str, str], gpt5: dict) -> str:
    abstracts_json = json.dumps(abstracts, ensure_ascii=False)
    gpt5_json      = json.dumps(gpt5,      ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>NER Annotation Tool</title>
<style>
/* ── RESET & BASE ─────────────────────────────────────────────────────────── */
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
:root {{
  --bg:        #0f1117;
  --surface:   #1a1d27;
  --surface2:  #22263a;
  --border:    #2e3348;
  --text:      #e2e4f0;
  --muted:     #6b7099;
  --accent:    #7c6af7;

  --c-anat:    #3b82f6;  /* blue   – Anatomical Entity        */
  --c-sym:     #ef4444;  /* red    – Symptom                  */
  --c-mov:     #a855f7;  /* purple – Terms of Body Movements  */
  --c-pop:     #22c55e;  /* green  – Population               */
  --c-meas:    #f59e0b;  /* amber  – Measurement              */

  --font: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
}}
html, body {{ height: 100%; background: var(--bg); color: var(--text);
             font-family: var(--font); font-size: 13px; }}

/* ── LAYOUT ──────────────────────────────────────────────────────────────── */
#app {{ display: grid; grid-template-columns: 260px 1fr 300px;
       grid-template-rows: 48px 1fr; height: 100vh; overflow: hidden; }}

/* ── TOPBAR ──────────────────────────────────────────────────────────────── */
#topbar {{
  grid-column: 1 / -1;
  display: flex; align-items: center; gap: 12px;
  padding: 0 16px;
  background: var(--surface); border-bottom: 1px solid var(--border);
}}
#topbar h1 {{ font-size: 13px; font-weight: 600; letter-spacing: .05em;
             color: var(--accent); flex: 1; }}
.nav-btn {{
  background: var(--surface2); border: 1px solid var(--border);
  color: var(--text); padding: 4px 12px; border-radius: 4px;
  cursor: pointer; font-family: var(--font); font-size: 12px;
}}
.nav-btn:hover {{ border-color: var(--accent); color: var(--accent); }}
#abs-counter {{ color: var(--muted); font-size: 12px; min-width: 70px; text-align: center; }}
#export-btn {{
  background: var(--accent); border: none; color: #fff;
  padding: 5px 14px; border-radius: 4px; cursor: pointer;
  font-family: var(--font); font-size: 12px; font-weight: 600;
}}
#export-btn:hover {{ opacity: .85; }}
#copy-btn {{
  background: var(--surface2); border: 1px solid var(--border);
  color: var(--text); padding: 5px 14px; border-radius: 4px;
  cursor: pointer; font-family: var(--font); font-size: 12px;
}}
#copy-btn:hover {{ border-color: var(--accent); color: var(--accent); }}
#undo-btn {{
  background: var(--surface2); border: 1px solid var(--border);
  color: var(--muted); padding: 5px 10px; border-radius: 4px;
  cursor: pointer; font-family: var(--font); font-size: 12px;
}}
#undo-btn:hover {{ border-color: #ef4444; color: #ef4444; }}

/* ── LEFT PANEL – category picker ───────────────────────────────────────── */
#left-panel {{
  background: var(--surface); border-right: 1px solid var(--border);
  padding: 16px 12px; overflow-y: auto;
  display: flex; flex-direction: column; gap: 8px;
}}
#left-panel h2 {{ font-size: 10px; letter-spacing: .12em; color: var(--muted);
                 text-transform: uppercase; margin-bottom: 4px; }}
.cat-btn {{
  display: flex; align-items: center; gap: 8px;
  padding: 7px 10px; border-radius: 6px;
  border: 1.5px solid transparent; cursor: pointer;
  background: var(--surface2); color: var(--text);
  font-family: var(--font); font-size: 11px; text-align: left;
  transition: border-color .15s, background .15s;
}}
.cat-btn .dot {{ width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }}
.cat-btn.active {{ background: var(--surface); }}
.cat-btn[data-cat="Anatomical Entity"]        .dot {{ background: var(--c-anat); }}
.cat-btn[data-cat="Symptom"]                  .dot {{ background: var(--c-sym);  }}
.cat-btn[data-cat="Terms of Body Movements"]  .dot {{ background: var(--c-mov);  }}
.cat-btn[data-cat="Population"]               .dot {{ background: var(--c-pop);  }}
.cat-btn[data-cat="Measurement"]              .dot {{ background: var(--c-meas); }}
.cat-btn[data-cat="Anatomical Entity"].active        {{ border-color: var(--c-anat); }}
.cat-btn[data-cat="Symptom"].active                  {{ border-color: var(--c-sym);  }}
.cat-btn[data-cat="Terms of Body Movements"].active  {{ border-color: var(--c-mov);  }}
.cat-btn[data-cat="Population"].active               {{ border-color: var(--c-pop);  }}
.cat-btn[data-cat="Measurement"].active              {{ border-color: var(--c-meas); }}
.cat-hint {{ font-size: 10px; color: var(--muted); line-height: 1.5;
            padding: 8px 4px; border-top: 1px solid var(--border); margin-top: 4px; }}
#progress-section {{ margin-top: 12px; border-top: 1px solid var(--border); padding-top: 12px; }}
#progress-section h2 {{ font-size: 10px; letter-spacing: .12em; color: var(--muted);
                        text-transform: uppercase; margin-bottom: 8px; }}
.progress-row {{ display: flex; justify-content: space-between;
                align-items: center; margin-bottom: 4px; font-size: 11px; }}
.progress-row .label {{ color: var(--muted); }}
.progress-row .count {{ font-weight: 600; }}

/* ── MAIN PANEL – abstract text ─────────────────────────────────────────── */
#main-panel {{
  background: var(--bg); padding: 24px 28px; overflow-y: auto;
  display: flex; flex-direction: column; gap: 16px;
}}
#abs-id {{ font-size: 11px; color: var(--muted); letter-spacing: .08em; }}
#abs-text {{
  line-height: 2.4; font-size: 14px; color: var(--text);
  user-select: none; cursor: text;
}}
.tok {{ display: inline; padding: 1px 0; cursor: pointer; border-radius: 2px; }}
.tok:hover {{ background: rgba(124,106,247,.15); }}
.tok.selecting {{ background: rgba(124,106,247,.25) !important; }}

/* annotated token styles */
.tok[data-ann="Anatomical Entity"]       {{ background: rgba(59,130,246,.25);
  border-bottom: 2px solid var(--c-anat); }}
.tok[data-ann="Symptom"]                 {{ background: rgba(239,68,68,.2);
  border-bottom: 2px solid var(--c-sym);  }}
.tok[data-ann="Terms of Body Movements"] {{ background: rgba(168,85,247,.22);
  border-bottom: 2px solid var(--c-mov);  }}
.tok[data-ann="Population"]              {{ background: rgba(34,197,94,.2);
  border-bottom: 2px solid var(--c-pop);  }}
.tok[data-ann="Measurement"]             {{ background: rgba(245,158,11,.22);
  border-bottom: 2px solid var(--c-meas); }}

/* ── RIGHT PANEL – annotation list ─────────────────────────────────────── */
#right-panel {{
  background: var(--surface); border-left: 1px solid var(--border);
  padding: 16px 12px; overflow-y: auto;
  display: flex; flex-direction: column; gap: 6px;
}}
#right-panel h2 {{ font-size: 10px; letter-spacing: .12em; color: var(--muted);
                  text-transform: uppercase; margin-bottom: 4px; }}
.ann-group {{ margin-bottom: 8px; }}
.ann-group-header {{
  font-size: 10px; font-weight: 600; letter-spacing: .06em;
  padding: 3px 6px; border-radius: 3px; margin-bottom: 4px;
  display: flex; align-items: center; gap: 6px;
}}
.ann-group-header .dot {{ width: 6px; height: 6px; border-radius: 50%; }}
.ann-item {{
  display: flex; align-items: center; justify-content: space-between;
  padding: 4px 8px; border-radius: 4px; background: var(--surface2);
  margin-bottom: 2px; font-size: 11px; gap: 6px;
}}
.ann-item .text {{ flex: 1; color: var(--text); word-break: break-word; }}
.ann-item .del {{
  background: none; border: none; color: var(--muted); cursor: pointer;
  font-size: 14px; line-height: 1; padding: 0 2px; flex-shrink: 0;
}}
.ann-item .del:hover {{ color: #ef4444; }}
.empty-hint {{ color: var(--muted); font-size: 11px; padding: 8px 4px; }}

/* ── TOAST ───────────────────────────────────────────────────────────────── */
#toast {{
  position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%);
  background: var(--accent); color: #fff; padding: 8px 18px;
  border-radius: 6px; font-size: 12px; pointer-events: none;
  opacity: 0; transition: opacity .2s;
}}
#toast.show {{ opacity: 1; }}
</style>
</head>
<body>
<div id="app">

  <!-- TOPBAR -->
  <header id="topbar">
    <h1>NER Annotation Tool</h1>
    <button class="nav-btn" id="prev-btn">← Prev</button>
    <span id="abs-counter"></span>
    <button class="nav-btn" id="next-btn">Next →</button>
    <button id="undo-btn" title="Undo last annotation">↩ Undo</button>
    <button id="copy-btn">Copy JSON</button>
    <button id="export-btn">↓ Export JSON</button>
  </header>

  <!-- LEFT: category picker -->
  <aside id="left-panel">
    <h2>Active Category</h2>
    <button class="cat-btn active" data-cat="Anatomical Entity">
      <span class="dot"></span>Anatomical Entity
    </button>
    <button class="cat-btn" data-cat="Symptom">
      <span class="dot"></span>Symptom
    </button>
    <button class="cat-btn" data-cat="Terms of Body Movements">
      <span class="dot"></span>Terms of Body Movements
    </button>
    <button class="cat-btn" data-cat="Population">
      <span class="dot"></span>Population
    </button>
    <button class="cat-btn" data-cat="Measurement">
      <span class="dot"></span>Measurement
    </button>
    <p class="cat-hint">
      Select a category, then drag across tokens to annotate.<br><br>
      Click an annotated span to <strong>remove</strong> it.
    </p>
    <div id="progress-section">
      <h2>This abstract</h2>
      <div id="progress-rows"></div>
    </div>
  </aside>

  <!-- MAIN: abstract text -->
  <main id="main-panel">
    <div id="abs-id"></div>
    <div id="abs-text"></div>
  </main>

  <!-- RIGHT: annotation list -->
  <aside id="right-panel">
    <h2>Annotations</h2>
    <div id="ann-list"></div>
  </aside>

</div>
<div id="toast"></div>

<script>
// ═══════════════════════════════════════════════════════════════════════════
// DATA  — embedded project data
// ═══════════════════════════════════════════════════════════════════════════
const ABSTRACTS = {abstracts_json};
const GPT5_RAW  = {gpt5_json};

const CATEGORIES = [
  "Anatomical Entity",
  "Terms of Body Movements",
  "Symptom",
  "Population",
  "Measurement",
];

// Normalise GPT5 keys: "Abstract 1" → "1"
function normaliseGPT5(raw) {{
  const out = {{}};
  for (const [k, v] of Object.entries(raw)) {{
    const num = k.replace(/^Abstract\\s*/i, "").trim();
    const ents = v?.entities ?? {{}};
    out[num] = {{}};
    for (const cat of CATEGORIES) {{
      // deduplicate, filter empty strings
      out[num][cat] = [...new Set((ents[cat] ?? []).filter(s => s && s.trim()))];
    }}
  }}
  return out;
}}

// ═══════════════════════════════════════════════════════════════════════════
// STATE  — single source of truth; mutated only by STATE functions
// ═══════════════════════════════════════════════════════════════════════════
const STATE = {{
  absKeys:     Object.keys(ABSTRACTS).sort((a,b) => +a - +b),
  absIndex:    0,
  activeCategory: "Anatomical Entity",
  annotations: normaliseGPT5(GPT5_RAW),  // {{absKey: {{cat: [strings]}}}}
  history:     [],   // stack of {{absKey, cat, text}} for undo
  selecting:   false,
  selectStart: null,
  selectEnd:   null,
}};

function currentKey()  {{ return STATE.absKeys[STATE.absIndex]; }}
function currentText() {{ return ABSTRACTS[currentKey()] ?? ""; }}
function currentAnns() {{
  const k = currentKey();
  if (!STATE.annotations[k]) {{
    STATE.annotations[k] = Object.fromEntries(CATEGORIES.map(c => [c, []]));
  }}
  return STATE.annotations[k];
}}

function stateSetCategory(cat) {{
  STATE.activeCategory = cat;
}}

function stateNavigate(delta) {{
  STATE.absIndex = Math.max(0, Math.min(STATE.absKeys.length - 1,
                                        STATE.absIndex + delta));
  STATE.selecting = false;
  STATE.selectStart = null;
  STATE.selectEnd   = null;
}}

function stateAddAnnotation(text) {{
  const cat  = STATE.activeCategory;
  const anns = currentAnns();
  const t    = text.trim();
  if (!t || anns[cat].includes(t)) return false;
  anns[cat].push(t);
  STATE.history.push({{ key: currentKey(), cat, text: t }});
  return true;
}}

function stateRemoveAnnotation(cat, text) {{
  const anns = currentAnns();
  anns[cat] = anns[cat].filter(t => t !== text);
}}

function stateUndo() {{
  const last = STATE.history.pop();
  if (!last) return null;
  const anns = STATE.annotations[last.key];
  if (anns && anns[last.cat]) {{
    anns[last.cat] = anns[last.cat].filter(t => t !== last.text);
  }}
  return last;
}}

// ═══════════════════════════════════════════════════════════════════════════
// RENDER  — pure DOM updates from state; no state mutations here
// ═══════════════════════════════════════════════════════════════════════════
const CAT_COLORS = {{
  "Anatomical Entity":       "var(--c-anat)",
  "Symptom":                 "var(--c-sym)",
  "Terms of Body Movements": "var(--c-mov)",
  "Population":              "var(--c-pop)",
  "Measurement":             "var(--c-meas)",
}};

function renderAll() {{
  renderTopbar();
  renderCategoryPicker();
  renderAbstractText();
  renderAnnotationList();
  renderProgress();
}}

function renderTopbar() {{
  const k = STATE.absIndex;
  const n = STATE.absKeys.length;
  document.getElementById("abs-counter").textContent = `${{k+1}} / ${{n}}`;
  document.getElementById("abs-id").textContent = `abs_${{String(+currentKey()).padStart(3,"0")}}`;
  document.getElementById("prev-btn").disabled = k === 0;
  document.getElementById("next-btn").disabled = k === n - 1;
}}

function renderCategoryPicker() {{
  document.querySelectorAll(".cat-btn").forEach(btn => {{
    btn.classList.toggle("active", btn.dataset.cat === STATE.activeCategory);
  }});
}}

function renderAbstractText() {{
  const text   = currentText();
  const anns   = currentAnns();
  const tokens = tokenize(text);

  // Build a set of annotated spans: for each annotation, find all token ranges
  // that match the phrase, record which tokens are covered and by which cat.
  const tokenCats = new Array(tokens.length).fill(null);
  for (const cat of CATEGORIES) {{
    for (const phrase of (anns[cat] ?? [])) {{
      markPhrase(tokens, phrase, cat, tokenCats);
    }}
  }}

  const el = document.getElementById("abs-text");
  el.innerHTML = "";
  tokens.forEach((tok, i) => {{
    const span = document.createElement("span");
    span.className = "tok";
    span.dataset.i = i;
    span.textContent = tok.text;
    if (tokenCats[i]) span.dataset.ann = tokenCats[i];
    el.appendChild(span);
    // preserve whitespace between tokens
    if (i < tokens.length - 1 && tok.end < tokens[i+1].start) {{
      el.appendChild(document.createTextNode(
        text.slice(tok.end, tokens[i+1].start)
      ));
    }}
  }});
}}

function renderAnnotationList() {{
  const anns = currentAnns();
  const el   = document.getElementById("ann-list");
  el.innerHTML = "";

  let total = 0;
  for (const cat of CATEGORIES) {{
    const items = anns[cat] ?? [];
    total += items.length;
    const group = document.createElement("div");
    group.className = "ann-group";

    const header = document.createElement("div");
    header.className = "ann-group-header";
    header.innerHTML = `<span class="dot" style="background:${{CAT_COLORS[cat]}}"></span>${{cat}}`;
    group.appendChild(header);

    if (items.length === 0) {{
      const hint = document.createElement("div");
      hint.className = "empty-hint";
      hint.textContent = "—";
      group.appendChild(hint);
    }} else {{
      items.forEach(text => {{
        const row = document.createElement("div");
        row.className = "ann-item";
        row.innerHTML = `<span class="text">${{escHtml(text)}}</span>
          <button class="del" data-cat="${{escHtml(cat)}}" data-text="${{escHtml(text)}}" title="Remove">×</button>`;
        group.appendChild(row);
      }});
    }}
    el.appendChild(group);
  }}
}}

function renderProgress() {{
  const anns = currentAnns();
  const el   = document.getElementById("progress-rows");
  el.innerHTML = "";
  for (const cat of CATEGORIES) {{
    const count = (anns[cat] ?? []).length;
    const row   = document.createElement("div");
    row.className = "progress-row";
    row.innerHTML = `<span class="label">${{cat.split(" ")[0]}}</span>
                     <span class="count" style="color:${{CAT_COLORS[cat]}}">${{count}}</span>`;
    el.appendChild(row);
  }}
}}

// ═══════════════════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════════════════
function tokenize(text) {{
  const tokens = [];
  const re = /\\S+/g;
  let m;
  while ((m = re.exec(text)) !== null) {{
    tokens.push({{ text: m[0], start: m.index, end: m.index + m[0].length }});
  }}
  return tokens;
}}

function markPhrase(tokens, phrase, cat, tokenCats) {{
  const pToks = phrase.trim().split(/\\s+/);
  for (let i = 0; i <= tokens.length - pToks.length; i++) {{
    const match = pToks.every((p, j) =>
      tokens[i+j].text.toLowerCase().replace(/[.,;:!?()"']/g,"") ===
      p.toLowerCase().replace(/[.,;:!?()"']/g,"")
    );
    if (match) {{
      for (let j = 0; j < pToks.length; j++) {{
        if (!tokenCats[i+j]) tokenCats[i+j] = cat;
      }}
    }}
  }}
}}

function getSelectedText() {{
  const s = Math.min(STATE.selectStart, STATE.selectEnd);
  const e = Math.max(STATE.selectStart, STATE.selectEnd);
  const tokens = tokenize(currentText());
  return tokens.slice(s, e+1).map(t => t.text).join(" ");
}}

function clearSelectionHighlight() {{
  document.querySelectorAll(".tok.selecting").forEach(t =>
    t.classList.remove("selecting")
  );
}}

function escHtml(s) {{
  return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")
          .replace(/"/g,"&quot;").replace(/'/g,"&#39;");
}}

function showToast(msg) {{
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.classList.add("show");
  setTimeout(() => el.classList.remove("show"), 1800);
}}

function buildExportJSON() {{
  const out = {{}};
  for (const [k, cats] of Object.entries(STATE.annotations)) {{
    out[`Abstract ${{k}}`] = {{ entities: cats }};
  }}
  return JSON.stringify(out, null, 2);
}}

// ═══════════════════════════════════════════════════════════════════════════
// EVENTS  — the only place that calls both STATE and RENDER
// ═══════════════════════════════════════════════════════════════════════════

// Navigation
document.getElementById("prev-btn").addEventListener("click", () => {{
  stateNavigate(-1); renderAll();
}});
document.getElementById("next-btn").addEventListener("click", () => {{
  stateNavigate(+1); renderAll();
}});

// Keyboard navigation
document.addEventListener("keydown", e => {{
  if (e.target.tagName === "BUTTON") return;
  if (e.key === "ArrowLeft")  {{ stateNavigate(-1); renderAll(); }}
  if (e.key === "ArrowRight") {{ stateNavigate(+1); renderAll(); }}
  if (e.key === "z" && (e.ctrlKey || e.metaKey)) {{
    stateUndo(); renderAll(); showToast("Undone");
  }}
}});

// Category picker
document.querySelectorAll(".cat-btn").forEach(btn => {{
  btn.addEventListener("click", () => {{
    stateSetCategory(btn.dataset.cat);
    renderCategoryPicker();
  }});
}});

// Token selection (mousedown → mouseover → mouseup)
const absTextEl = document.getElementById("abs-text");

absTextEl.addEventListener("mousedown", e => {{
  const tok = e.target.closest(".tok");
  if (!tok) return;
  // clicking an already-annotated token removes it
  if (tok.dataset.ann) {{
    stateRemoveAnnotation(tok.dataset.ann, getTokenPhrase(tok));
    renderAbstractText();
    renderAnnotationList();
    renderProgress();
    return;
  }}
  STATE.selecting   = true;
  STATE.selectStart = +tok.dataset.i;
  STATE.selectEnd   = +tok.dataset.i;
  tok.classList.add("selecting");
  e.preventDefault();
}});

absTextEl.addEventListener("mouseover", e => {{
  if (!STATE.selecting) return;
  const tok = e.target.closest(".tok");
  if (!tok) return;
  STATE.selectEnd = +tok.dataset.i;
  clearSelectionHighlight();
  const s = Math.min(STATE.selectStart, STATE.selectEnd);
  const en = Math.max(STATE.selectStart, STATE.selectEnd);
  document.querySelectorAll(".tok").forEach(t => {{
    const i = +t.dataset.i;
    if (i >= s && i <= en) t.classList.add("selecting");
  }});
}});

document.addEventListener("mouseup", () => {{
  if (!STATE.selecting) return;
  STATE.selecting = false;
  const text = getSelectedText();
  clearSelectionHighlight();
  if (text.trim()) {{
    const added = stateAddAnnotation(text);
    if (added) {{
      showToast(`+ "${{text.trim()}}" → ${{STATE.activeCategory}}`);
    }}
  }}
  renderAbstractText();
  renderAnnotationList();
  renderProgress();
}});

// Helper: find the full phrase for a token from current annotations
function getTokenPhrase(tok) {{
  const anns = currentAnns();
  const cat  = tok.dataset.ann;
  const i    = +tok.dataset.i;
  const tokens = tokenize(currentText());
  for (const phrase of (anns[cat] ?? [])) {{
    const pToks = phrase.trim().split(/\\s+/);
    for (let start = 0; start <= tokens.length - pToks.length; start++) {{
      const match = pToks.every((p, j) =>
        tokens[start+j].text.toLowerCase().replace(/[.,;:!?()"']/g,"") ===
        p.toLowerCase().replace(/[.,;:!?()"']/g,"")
      );
      if (match && i >= start && i < start + pToks.length) return phrase;
    }}
  }}
  return "";
}}

// Delete button in annotation list
document.getElementById("ann-list").addEventListener("click", e => {{
  const btn = e.target.closest(".del");
  if (!btn) return;
  stateRemoveAnnotation(btn.dataset.cat, btn.dataset.text);
  renderAbstractText();
  renderAnnotationList();
  renderProgress();
}});

// Undo
document.getElementById("undo-btn").addEventListener("click", () => {{
  stateUndo(); renderAll(); showToast("Undone");
}});

// Export
document.getElementById("export-btn").addEventListener("click", () => {{
  const blob = new Blob([buildExportJSON()], {{type: "application/json"}});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "entities_manual_gold.json";
  a.click();
  showToast("Downloaded entities_manual_gold.json");
}});

// Copy
document.getElementById("copy-btn").addEventListener("click", () => {{
  navigator.clipboard.writeText(buildExportJSON()).then(() =>
    showToast("Copied to clipboard")
  );
}});

// ═══════════════════════════════════════════════════════════════════════════
// BOOT
// ═══════════════════════════════════════════════════════════════════════════
renderAll();
</script>
</body>
</html>"""

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    project_root = Path(__file__).resolve().parent.parent.parent  # src/tools → project root

    csv_path  = project_root / "data" / "raw" / "Flatfeet_clean.csv"
    json_path = project_root / "data" / "processed" / "evaluation" / "extraction_gold_standard" / "entities_manual_gold.json"
    out_path  = project_root / "outputs" / "annotations_ner.html"

    if not csv_path.exists():
        sys.exit(f"ERROR: cannot find {csv_path}")
    if not json_path.exists():
        sys.exit(f"ERROR: cannot find {json_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading abstracts from {csv_path} ...")
    abstracts = load_abstracts(csv_path)
    print(f"  → {len(abstracts)} abstracts loaded")

    print(f"Loading annotations from {json_path} ...")
    gpt5 = load_gpt5_annotations(json_path)
    print(f"  → {len(gpt5)} abstract annotations loaded")

    print(f"Writing annotation tool to {out_path} ...")
    html = build_html(abstracts, gpt5)
    out_path.write_text(html, encoding="utf-8")
    print(f"  Done! Open {out_path.name} in your browser.")


if __name__ == "__main__":
    main()
