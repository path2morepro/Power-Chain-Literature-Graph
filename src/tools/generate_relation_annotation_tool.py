"""
generate_re_annotation_tool.py
================================
Reads normalized_entities.json and relations_open.json, then writes a
self-contained re_annotation_tool.html for manual review and correction
of relation triples.

Usage
-----
  python generate_re_annotation_tool.py
  # → writes re_annotation_tool.html in the project root

Workflow per abstract
---------------------
  - Abstract text shown at top with entities highlighted by category
  - Each extracted triple shown as a card: Subject — Relation → Object + evidence
  - Per triple: Accept / Edit / Delete
  - Add new triples manually via entity dropdowns + free-text relation field
  - Export produces relations_manual_gold.json in the same format as
    relations_open.json / relations_GPT5.json

Output format:
  {
    "abs_001": [
      {"subject": "...", "relation": "...", "object": "...", "evidence": "..."},
      ...
    ],
    ...
  }
"""

import json
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def extract_records(normalized: dict) -> dict[str, dict]:
    """Return {abstract_id: {text, entity_forms: [{form, field}]}}."""
    records = {}
    for entry in normalized["abstracts"]:
        abs_id = entry["abstract"]["abstract_id"]
        text   = entry["abstract"].get("text", "").strip()
        seen   = set()
        forms  = []
        for ent in entry.get("entities", []):
            f = ent.get("entity_form", "").strip()
            field = ent.get("field", "")
            if f and f not in seen:
                seen.add(f)
                forms.append({"form": f, "field": field})
        records[abs_id] = {"text": text, "entities": forms}
    return records


# ─────────────────────────────────────────────────────────────────────────────
# HTML GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def build_html(records: dict, relations: dict) -> str:
    records_json   = json.dumps(records,   ensure_ascii=False)
    relations_json = json.dumps(relations, ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>RE Annotation Tool</title>
<style>
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
:root {{
  --bg:       #0f1117;
  --surface:  #1a1d27;
  --surface2: #22263a;
  --surface3: #2a2f45;
  --border:   #2e3348;
  --text:     #e2e4f0;
  --muted:    #6b7099;
  --accent:   #7c6af7;
  --green:    #22c55e;
  --red:      #ef4444;
  --amber:    #f59e0b;

  --c-anat:  #3b82f6;
  --c-sym:   #ef4444;
  --c-mov:   #a855f7;
  --c-pop:   #22c55e;
  --c-meas:  #f59e0b;

  --font: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
}}
html, body {{ height: 100%; background: var(--bg); color: var(--text);
             font-family: var(--font); font-size: 13px; }}

/* ── LAYOUT ──────────────────────────────────────────────────────────────── */
#app {{ display: grid; grid-template-columns: 1fr 380px;
       grid-template-rows: 48px 1fr; height: 100vh; overflow: hidden; }}

/* ── TOPBAR ──────────────────────────────────────────────────────────────── */
#topbar {{
  grid-column: 1 / -1;
  display: flex; align-items: center; gap: 10px;
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
#stats-badge {{
  font-size: 11px; color: var(--muted);
  background: var(--surface2); border: 1px solid var(--border);
  padding: 3px 10px; border-radius: 10px;
}}

/* ── MAIN PANEL ──────────────────────────────────────────────────────────── */
#main-panel {{
  overflow-y: auto; padding: 20px 24px;
  display: flex; flex-direction: column; gap: 16px;
}}
#abs-header {{
  display: flex; align-items: baseline; gap: 10px;
}}
#abs-id {{ font-size: 11px; color: var(--accent); letter-spacing: .1em; font-weight: 600; }}
#triple-count {{ font-size: 11px; color: var(--muted); }}

/* abstract text with entity highlights */
#abs-text {{
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 14px 16px;
  line-height: 2.2; font-size: 13px; color: var(--text);
}}
.ent-anat  {{ background: rgba(59,130,246,.2);  border-bottom: 2px solid var(--c-anat);
             border-radius: 2px; padding: 1px 2px; }}
.ent-sym   {{ background: rgba(239,68,68,.18);  border-bottom: 2px solid var(--c-sym);
             border-radius: 2px; padding: 1px 2px; }}
.ent-mov   {{ background: rgba(168,85,247,.2);  border-bottom: 2px solid var(--c-mov);
             border-radius: 2px; padding: 1px 2px; }}
.ent-pop   {{ background: rgba(34,197,94,.18);  border-bottom: 2px solid var(--c-pop);
             border-radius: 2px; padding: 1px 2px; }}
.ent-meas  {{ background: rgba(245,158,11,.18); border-bottom: 2px solid var(--c-meas);
             border-radius: 2px; padding: 1px 2px; }}

/* ── TRIPLE CARDS ────────────────────────────────────────────────────────── */
#triples-section h2 {{
  font-size: 10px; letter-spacing: .12em; color: var(--muted);
  text-transform: uppercase; margin-bottom: 10px;
}}
.triple-card {{
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 12px 14px; margin-bottom: 8px;
  transition: border-color .15s;
}}
.triple-card.accepted {{ border-color: var(--green); }}
.triple-card.deleted  {{ opacity: .35; border-color: var(--red); }}
.triple-card.edited   {{ border-color: var(--amber); }}

.triple-main {{
  display: flex; align-items: center; gap: 6px;
  flex-wrap: wrap; margin-bottom: 8px;
}}
.triple-subj, .triple-obj {{
  background: var(--surface2); border: 1px solid var(--border);
  padding: 3px 8px; border-radius: 4px; font-size: 12px; font-weight: 600;
}}
.triple-rel {{
  color: var(--accent); font-size: 11px; font-style: italic;
  padding: 0 4px;
}}
.triple-arrow {{ color: var(--muted); font-size: 12px; }}

.triple-evidence {{
  font-size: 11px; color: var(--muted); font-style: italic;
  border-left: 2px solid var(--border); padding-left: 8px;
  margin-bottom: 8px; line-height: 1.5;
}}

.triple-actions {{
  display: flex; gap: 6px; align-items: center;
}}
.btn-accept {{
  background: rgba(34,197,94,.15); border: 1px solid var(--green);
  color: var(--green); padding: 3px 10px; border-radius: 4px;
  cursor: pointer; font-family: var(--font); font-size: 11px;
}}
.btn-accept:hover {{ background: rgba(34,197,94,.25); }}
.btn-delete {{
  background: rgba(239,68,68,.12); border: 1px solid var(--red);
  color: var(--red); padding: 3px 10px; border-radius: 4px;
  cursor: pointer; font-family: var(--font); font-size: 11px;
}}
.btn-delete:hover {{ background: rgba(239,68,68,.22); }}
.btn-edit {{
  background: rgba(245,158,11,.12); border: 1px solid var(--amber);
  color: var(--amber); padding: 3px 10px; border-radius: 4px;
  cursor: pointer; font-family: var(--font); font-size: 11px;
}}
.btn-edit:hover {{ background: rgba(245,158,11,.22); }}
.status-tag {{
  margin-left: auto; font-size: 10px; padding: 2px 7px;
  border-radius: 10px; letter-spacing: .05em;
}}
.tag-accepted {{ background: rgba(34,197,94,.15);  color: var(--green); }}
.tag-deleted  {{ background: rgba(239,68,68,.12);  color: var(--red);   }}
.tag-edited   {{ background: rgba(245,158,11,.12); color: var(--amber); }}
.tag-pending  {{ background: var(--surface2);      color: var(--muted); }}

/* inline edit form */
.edit-form {{
  background: var(--surface2); border: 1px solid var(--border);
  border-radius: 6px; padding: 10px 12px; margin-top: 8px;
  display: flex; flex-direction: column; gap: 8px;
}}
.edit-row {{ display: flex; gap: 8px; align-items: center; }}
.edit-row label {{ font-size: 10px; color: var(--muted); min-width: 60px; }}
.edit-row select, .edit-row input {{
  flex: 1; background: var(--surface); border: 1px solid var(--border);
  color: var(--text); padding: 4px 8px; border-radius: 4px;
  font-family: var(--font); font-size: 12px;
}}
.edit-row select:focus, .edit-row input:focus {{
  outline: none; border-color: var(--accent);
}}
.btn-save-edit {{
  background: var(--accent); border: none; color: #fff;
  padding: 4px 12px; border-radius: 4px; cursor: pointer;
  font-family: var(--font); font-size: 11px; align-self: flex-end;
}}

/* ── RIGHT PANEL – add new triple ───────────────────────────────────────── */
#right-panel {{
  background: var(--surface); border-left: 1px solid var(--border);
  padding: 16px 14px; overflow-y: auto;
  display: flex; flex-direction: column; gap: 12px;
}}
#right-panel h2 {{
  font-size: 10px; letter-spacing: .12em; color: var(--muted);
  text-transform: uppercase;
}}
.form-field {{ display: flex; flex-direction: column; gap: 4px; }}
.form-field label {{ font-size: 10px; color: var(--muted); }}
.form-field select, .form-field input, .form-field textarea {{
  background: var(--surface2); border: 1px solid var(--border);
  color: var(--text); padding: 6px 8px; border-radius: 4px;
  font-family: var(--font); font-size: 12px;
}}
.form-field select:focus, .form-field input:focus, .form-field textarea:focus {{
  outline: none; border-color: var(--accent);
}}
.form-field textarea {{ resize: vertical; min-height: 60px; }}
#add-triple-btn {{
  background: var(--accent); border: none; color: #fff;
  padding: 7px 14px; border-radius: 4px; cursor: pointer;
  font-family: var(--font); font-size: 12px; font-weight: 600;
}}
#add-triple-btn:hover {{ opacity: .85; }}
.divider {{ border: none; border-top: 1px solid var(--border); }}

/* legend */
.legend {{ display: flex; flex-direction: column; gap: 4px; }}
.legend-row {{ display: flex; align-items: center; gap: 6px; font-size: 11px; }}
.legend-dot {{ width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }}

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

  <header id="topbar">
    <h1>RE Annotation Tool</h1>
    <button class="nav-btn" id="prev-btn">← Prev</button>
    <span id="abs-counter"></span>
    <button class="nav-btn" id="next-btn">Next →</button>
    <span id="stats-badge"></span>
    <button id="export-btn">↓ Export JSON</button>
  </header>

  <main id="main-panel">
    <div id="abs-header">
      <span id="abs-id"></span>
      <span id="triple-count"></span>
    </div>
    <div id="abs-text"></div>
    <div id="triples-section">
      <h2>Extracted Triples</h2>
      <div id="triple-list"></div>
    </div>
  </main>

  <aside id="right-panel">
    <h2>Add New Triple</h2>
    <div class="form-field">
      <label>Subject</label>
      <select id="new-subj"></select>
    </div>
    <div class="form-field">
      <label>Relation (free text)</label>
      <input id="new-rel" type="text" placeholder="e.g. is a risk factor for">
    </div>
    <div class="form-field">
      <label>Object</label>
      <select id="new-obj"></select>
    </div>
    <div class="form-field">
      <label>Evidence (optional)</label>
      <textarea id="new-ev" placeholder="Paste the relevant sentence..."></textarea>
    </div>
    <button id="add-triple-btn">+ Add Triple</button>

    <hr class="divider">

    <h2>Entity Legend</h2>
    <div class="legend">
      <div class="legend-row"><span class="legend-dot" style="background:var(--c-anat)"></span>Anatomical Entity</div>
      <div class="legend-row"><span class="legend-dot" style="background:var(--c-sym)"></span>Symptom</div>
      <div class="legend-row"><span class="legend-dot" style="background:var(--c-mov)"></span>Terms of Body Movements</div>
      <div class="legend-row"><span class="legend-dot" style="background:var(--c-pop)"></span>Population</div>
      <div class="legend-row"><span class="legend-dot" style="background:var(--c-meas)"></span>Measurement</div>
    </div>

    <hr class="divider">
    <h2>Progress</h2>
    <div id="progress-detail" style="font-size:11px;color:var(--muted);line-height:1.8;"></div>
  </aside>

</div>
<div id="toast"></div>

<script>
// ═══════════════════════════════════════════════════════════════════════════
// DATA
// ═══════════════════════════════════════════════════════════════════════════
const RECORDS   = {records_json};
const RELATIONS = {relations_json};

const FIELD_CLASS = {{
  "Anatomical Entity":       "ent-anat",
  "Symptom":                 "ent-sym",
  "Terms of Body Movements": "ent-mov",
  "Population":              "ent-pop",
  "Measurement":             "ent-meas",
}};

// ═══════════════════════════════════════════════════════════════════════════
// STATE
// ═══════════════════════════════════════════════════════════════════════════
const ABS_KEYS = Object.keys(RECORDS).sort();

const STATE = {{
  absIndex:    0,
  // annotations: {{abs_id: [{{subject, relation, object, evidence, status}}]}}
  // status: "pending" | "accepted" | "deleted" | "edited"
  annotations: {{}},
}};

// Initialise annotations from RELATIONS
for (const absId of ABS_KEYS) {{
  const triples = RELATIONS[absId] ?? [];
  STATE.annotations[absId] = triples.map(t => ({{
    subject:  t.subject  ?? "",
    relation: t.relation ?? "",
    object:   t.object   ?? "",
    evidence: t.evidence ?? "",
    status:   "pending",
  }}));
}}

function currentKey()  {{ return ABS_KEYS[STATE.absIndex]; }}
function currentRec()  {{ return RECORDS[currentKey()];     }}
function currentAnns() {{ return STATE.annotations[currentKey()] ?? []; }}

function stateNavigate(delta) {{
  STATE.absIndex = Math.max(0, Math.min(ABS_KEYS.length - 1, STATE.absIndex + delta));
}}

function stateSetStatus(absId, idx, status) {{
  if (STATE.annotations[absId]?.[idx]) {{
    STATE.annotations[absId][idx].status = status;
  }}
}}

function stateEditTriple(absId, idx, subj, rel, obj, ev) {{
  const t = STATE.annotations[absId]?.[idx];
  if (!t) return;
  t.subject  = subj;
  t.relation = rel;
  t.object   = obj;
  t.evidence = ev;
  t.status   = "edited";
}}

function stateAddTriple(absId, subj, rel, obj, ev) {{
  if (!STATE.annotations[absId]) STATE.annotations[absId] = [];
  STATE.annotations[absId].push({{
    subject: subj, relation: rel, object: obj, evidence: ev, status: "accepted"
  }});
}}

// ═══════════════════════════════════════════════════════════════════════════
// RENDER
// ═══════════════════════════════════════════════════════════════════════════
function renderAll() {{
  renderTopbar();
  renderAbstractText();
  renderTriples();
  renderAddForm();
  renderProgress();
}}

function renderTopbar() {{
  const i = STATE.absIndex, n = ABS_KEYS.length;
  document.getElementById("abs-counter").textContent = `${{i+1}} / ${{n}}`;
  document.getElementById("abs-id").textContent = currentKey();
  document.getElementById("prev-btn").disabled = i === 0;
  document.getElementById("next-btn").disabled = i === n - 1;

  // global stats
  let total = 0, accepted = 0, deleted = 0;
  for (const anns of Object.values(STATE.annotations)) {{
    for (const t of anns) {{
      total++;
      if (t.status === "accepted" || t.status === "edited") accepted++;
      if (t.status === "deleted") deleted++;
    }}
  }}
  document.getElementById("stats-badge").textContent =
    `${{accepted}} accepted · ${{deleted}} deleted · ${{total - accepted - deleted}} pending`;

  const anns = currentAnns();
  const active = anns.filter(t => t.status !== "deleted").length;
  document.getElementById("triple-count").textContent =
    `${{active}} triple${{active !== 1 ? "s" : ""}}`;
}}

function highlightEntities(text, entities) {{
  if (!entities?.length) return escHtml(text);
  // Sort by length descending so longer phrases matched first
  const sorted = [...entities].sort((a,b) => b.form.length - a.form.length);
  let result = escHtml(text);
  for (const ent of sorted) {{
    const cls = FIELD_CLASS[ent.field] ?? "";
    if (!cls) continue;
    const escaped = escHtml(ent.form);
    const re = new RegExp(escapeRegex(escaped), "gi");
    result = result.replace(re, m => `<span class="${{cls}}">${{m}}</span>`);
  }}
  return result;
}}

function renderAbstractText() {{
  const rec = currentRec();
  const el  = document.getElementById("abs-text");
  el.innerHTML = highlightEntities(rec.text, rec.entities);
}}

function renderTriples() {{
  const anns = currentAnns();
  const absId = currentKey();
  const el   = document.getElementById("triple-list");
  el.innerHTML = "";

  if (anns.length === 0) {{
    el.innerHTML = `<div style="color:var(--muted);font-size:12px;padding:8px 0;">
      No triples extracted for this abstract.</div>`;
    return;
  }}

  anns.forEach((t, i) => {{
    const card = document.createElement("div");
    card.className = `triple-card ${{t.status !== "pending" ? t.status : ""}}`;
    card.dataset.idx = i;

    const statusLabels = {{
      accepted: '<span class="status-tag tag-accepted">✓ accepted</span>',
      deleted:  '<span class="status-tag tag-deleted">✕ deleted</span>',
      edited:   '<span class="status-tag tag-edited">✎ edited</span>',
      pending:  '<span class="status-tag tag-pending">pending</span>',
    }};

    card.innerHTML = `
      <div class="triple-main">
        <span class="triple-subj">${{escHtml(t.subject)}}</span>
        <span class="triple-rel">${{escHtml(t.relation)}}</span>
        <span class="triple-arrow">→</span>
        <span class="triple-obj">${{escHtml(t.object)}}</span>
        ${{statusLabels[t.status] ?? ""}}
      </div>
      ${{t.evidence ? `<div class="triple-evidence">"${{escHtml(t.evidence)}}"</div>` : ""}}
      <div class="triple-actions">
        <button class="btn-accept" data-action="accept" data-idx="${{i}}">✓ Accept</button>
        <button class="btn-edit"   data-action="edit"   data-idx="${{i}}">✎ Edit</button>
        <button class="btn-delete" data-action="delete" data-idx="${{i}}">✕ Delete</button>
      </div>
    `;
    el.appendChild(card);
  }});
}}

function renderAddForm() {{
  const entities = currentRec().entities ?? [];
  const opts = entities.map(e =>
    `<option value="${{escHtml(e.form)}}">${{escHtml(e.form)}} ([${{e.field.split(" ")[0]}}])</option>`
  ).join("");
  document.getElementById("new-subj").innerHTML = `<option value="">— select —</option>` + opts;
  document.getElementById("new-obj").innerHTML  = `<option value="">— select —</option>` + opts;
  document.getElementById("new-rel").value = "";
  document.getElementById("new-ev").value  = "";
}}

function renderProgress() {{
  const anns   = currentAnns();
  const total   = anns.length;
  const accepted = anns.filter(t => t.status === "accepted" || t.status === "edited").length;
  const deleted  = anns.filter(t => t.status === "deleted").length;
  const pending  = total - accepted - deleted;
  document.getElementById("progress-detail").innerHTML =
    `Total: ${{total}}<br>Accepted/Edited: <span style="color:var(--green)">${{accepted}}</span><br>` +
    `Deleted: <span style="color:var(--red)">${{deleted}}</span><br>` +
    `Pending: <span style="color:var(--amber)">${{pending}}</span>`;
}}

// ═══════════════════════════════════════════════════════════════════════════
// INLINE EDIT FORM
// ═══════════════════════════════════════════════════════════════════════════
function showEditForm(card, idx) {{
  // Remove any existing edit form
  card.querySelector(".edit-form")?.remove();

  const t       = currentAnns()[idx];
  const entities = currentRec().entities ?? [];
  const opts    = entities.map(e =>
    `<option value="${{escHtml(e.form)}}" ${{e.form === t.subject ? "selected" : ""}}>${{escHtml(e.form)}}</option>`
  ).join("");
  const optsObj = entities.map(e =>
    `<option value="${{escHtml(e.form)}}" ${{e.form === t.object ? "selected" : ""}}>${{escHtml(e.form)}}</option>`
  ).join("");

  const form = document.createElement("div");
  form.className = "edit-form";
  form.innerHTML = `
    <div class="edit-row">
      <label>Subject</label>
      <select class="ef-subj"><option value="">—</option>${{opts}}</select>
    </div>
    <div class="edit-row">
      <label>Relation</label>
      <input class="ef-rel" type="text" value="${{escHtml(t.relation)}}">
    </div>
    <div class="edit-row">
      <label>Object</label>
      <select class="ef-obj"><option value="">—</option>${{optsObj}}</select>
    </div>
    <div class="edit-row">
      <label>Evidence</label>
      <input class="ef-ev" type="text" value="${{escHtml(t.evidence)}}">
    </div>
    <button class="btn-save-edit" data-idx="${{idx}}">Save</button>
  `;
  card.appendChild(form);
}}

// ═══════════════════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════════════════
function escHtml(s) {{
  return String(s ?? "")
    .replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")
    .replace(/"/g,"&quot;").replace(/'/g,"&#39;");
}}
function escapeRegex(s) {{
  return s.replace(/[.*+?^${{}}()|[\\]\\\\]/g, "\\\\$&");
}}
function showToast(msg) {{
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.classList.add("show");
  setTimeout(() => el.classList.remove("show"), 1800);
}}
function buildExportJSON() {{
  const out = {{}};
  for (const [absId, anns] of Object.entries(STATE.annotations)) {{
    out[absId] = anns
      .filter(t => t.status !== "deleted")
      .map(t => ({{ subject: t.subject, relation: t.relation,
                    object: t.object,  evidence: t.evidence }}));
  }}
  return JSON.stringify(out, null, 2);
}}

// ═══════════════════════════════════════════════════════════════════════════
// EVENTS
// ═══════════════════════════════════════════════════════════════════════════

// Navigation
document.getElementById("prev-btn").addEventListener("click", () => {{
  stateNavigate(-1); renderAll();
}});
document.getElementById("next-btn").addEventListener("click", () => {{
  stateNavigate(+1); renderAll();
}});
document.addEventListener("keydown", e => {{
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA" ||
      e.target.tagName === "SELECT") return;
  if (e.key === "ArrowLeft")  {{ stateNavigate(-1); renderAll(); }}
  if (e.key === "ArrowRight") {{ stateNavigate(+1); renderAll(); }}
}});

// Triple actions (accept / edit / delete)
document.getElementById("triple-list").addEventListener("click", e => {{
  const btn = e.target.closest("[data-action]");
  if (!btn) {{
    // check for save-edit
    const saveBtn = e.target.closest(".btn-save-edit");
    if (saveBtn) {{
      const idx  = +saveBtn.dataset.idx;
      const form = saveBtn.closest(".edit-form");
      const subj = form.querySelector(".ef-subj").value;
      const rel  = form.querySelector(".ef-rel").value.trim();
      const obj  = form.querySelector(".ef-obj").value;
      const ev   = form.querySelector(".ef-ev").value.trim();
      if (!subj || !rel || !obj) {{ showToast("Subject, relation and object are required"); return; }}
      stateEditTriple(currentKey(), idx, subj, rel, obj, ev);
      renderAll();
      showToast("Triple updated");
    }}
    return;
  }}

  const idx    = +btn.dataset.idx;
  const absId  = currentKey();
  const action = btn.dataset.action;

  if (action === "accept") {{
    stateSetStatus(absId, idx, "accepted");
    renderAll();
    showToast("Accepted");
  }} else if (action === "delete") {{
    stateSetStatus(absId, idx, "deleted");
    renderAll();
    showToast("Deleted");
  }} else if (action === "edit") {{
    const card = btn.closest(".triple-card");
    // toggle: if already open, close it
    if (card.querySelector(".edit-form")) {{
      card.querySelector(".edit-form").remove();
    }} else {{
      showEditForm(card, idx);
    }}
  }}
}});

// Add new triple
document.getElementById("add-triple-btn").addEventListener("click", () => {{
  const subj = document.getElementById("new-subj").value;
  const rel  = document.getElementById("new-rel").value.trim();
  const obj  = document.getElementById("new-obj").value;
  const ev   = document.getElementById("new-ev").value.trim();

  if (!subj || !rel || !obj) {{
    showToast("Subject, relation and object are required");
    return;
  }}
  if (subj === obj) {{
    showToast("Subject and object must be different");
    return;
  }}

  stateAddTriple(currentKey(), subj, rel, obj, ev);
  renderAll();
  showToast(`+ "${{subj}} — ${{rel}} → ${{obj}}"`);
}});

// Export
document.getElementById("export-btn").addEventListener("click", () => {{
  const blob = new Blob([buildExportJSON()], {{type: "application/json"}});
  const a    = document.createElement("a");
  a.href     = URL.createObjectURL(blob);
  a.download = "relations_manual_gold.json";
  a.click();
  showToast("Downloaded relations_manual_gold.json");
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
    project_root   = Path(__file__).resolve().parent.parent.parent  # src/tools → project root
    norm_path      = project_root / "data" / "processed" / "01_ner_normalized" / "normalized_entities.json"
    relations_path = project_root / "data" / "processed" / "03_relations_extracted" / "relations_open.json"
    out_path       = project_root / "outputs" / "annotations_relations.html"

    if not norm_path.exists():
        sys.exit(f"ERROR: cannot find {norm_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # relations_open.json is optional — tool works with empty relations too
    relations = {}
    if relations_path.exists():
        print(f"Loading relations from {relations_path} ...")
        relations = load_json(relations_path)
        total = sum(len(v) for v in relations.values())
        print(f"  → {total} triples across {len(relations)} abstracts")
    else:
        print(f"WARNING: {relations_path} not found — tool will start with no pre-filled triples.")
        print(f"  Run relation_extraction.py first, or annotate from scratch.")

    print(f"Loading normalized entities from {norm_path} ...")
    normalized = load_json(norm_path)
    records    = extract_records(normalized)
    print(f"  → {len(records)} abstracts loaded")

    print(f"Writing annotation tool to {out_path} ...")
    html = build_html(records, relations)
    out_path.write_text(html, encoding="utf-8")
    print(f"  Done!")
    print(f"  Open with: xdg-open re_annotation_tool.html")


if __name__ == "__main__":
    main()
