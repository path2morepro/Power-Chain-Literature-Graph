# Musculoskeletal Knowledge Graph

A pipeline that turns biomedical abstracts about musculoskeletal conditions (flat foot, patellofemoral pain, low back pain, etc.) into an interactive knowledge graph. The graph links anatomical structures, symptoms, and body movements through extracted relations, and is intended for hypothesis generation via path traversal.

## What it does

1. **NER** — extracts entities (anatomical structures, symptoms, movements, population, measurements) from abstracts using GLiNER2, with GPT/Claude as the gold standard
2. **Normalization** — deduplicates and canonicalizes entity surface forms (abbreviation expansion, singularization, laterality stripping)
3. **Entity enrichment** — links symptoms and movement terms to their anatomical locations
4. **Relation extraction** — extracts closed-schema relation triples (associated_with, contributes_to, located_in, prevalence_in) using Claude
5. **Graph construction** — assembles everything into a vis.js interactive knowledge graph
6. **Evaluation** — precision/recall/F1 for NER and relation extraction against manually annotated gold standards

---

## Project layout

```
data/
  raw/                          source abstracts and golden standards
  processed/
    01_ner_normalized/          normalized_entities.json
    02_entity_enrichment/       anatomical_entities_enriched.json
    03_relations_extracted/     relations_Claude.json
    04_knowledge_graph/         graph_ES.json, graph_LI.json
    evaluation/                 metrics, baselines, gold standards

src/
  pipeline/
    ner.py                      NER extraction + normalization
    ner_evaluation.py           NER evaluation (round 1 + round 2)
    knowledge_graph.py          graph construction
    relation_extraction_evaluation.py

  tools/
    generate_annotation_tool.py     NER annotation tool generator
    generate_relation_annotation_tool.py  RE annotation tool generator

outputs/
  visualizations/               HTML graph + NER visualization
  annotations_ner.html
  annotations_relations.html
```

---

## Setup

```bash
# clone and enter the project
cd /path/to/project

# create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# install dependencies
pip install requirements.txt

```


## Running the pipeline from scratch

All commands are run from the project root.

### Step 1 — NER extraction

Runs GLiNER2 on the abstracts and produces `entities_pretrainedmodel.json`, then normalizes into `normalized_entities.json`.

```bash
python -m src.pipeline.ner
```

### Step 2 — NER annotation (manual gold standard)

Generate the annotation tool, open it in your browser, annotate, and export `entities_manual_gold.json` into `data/processed/evaluation/extraction_gold_standard/`.

```bash
python src/tools/generate_annotation_tool.py
xdg-open outputs/annotations_ner.html
```

In the tool: select a category on the left, drag across tokens to annotate, click an annotated span to remove it. Use `←` `→` to navigate abstracts, `Ctrl+Z` to undo. Export when done.

### Step 3 — NER evaluation

```bash
# round 1 only (NER comparison, fast)
python -m src.pipeline.ner_evaluation --mode round1

# round 2 only (anatomy linking, requires BERT models, slow)
python -m src.pipeline.ner_evaluation --mode round2

# both
python -m src.pipeline.ner_evaluation --mode all
```

Results go to `data/processed/evaluation/ner_round1/metrics.csv`.

### Step 4 — Entity enrichment

Links anatomical entities to related symptoms and movements.

```bash
python src/pipeline/task3_anatomical_entities.py
```

### Step 5 — Relation extraction

Uses Claude to extract relation triples from `normalized_entities.json`.

```bash
python src/pipeline/re_extract_open.py
```

Output: `data/processed/03_relations_extracted/relations_Claude.json`

### Step 6 — RE annotation (manual gold standard)

Generate the RE annotation tool, review and correct the extracted triples, export `relations_manual_gold.json`.

```bash
python src/tools/generate_relation_annotation_tool.py
xdg-open outputs/annotations_relations.html
```

In the tool: Accept, Edit, or Delete each triple. Add missing triples via the right panel dropdowns. Export when done.

### Step 7 — RE evaluation

```bash
python -m src.pipeline.relation_extraction_evaluation
```

Results go to `data/processed/evaluation/relation_extraction/`.

### Step 8 — Graph construction

Builds two graph variants: one using entity enrichment (ES), one using located_in triples (LI).

```bash
python -m src.pipeline.knowledge_graph
```

Output: `data/processed/04_knowledge_graph/graph_ES.json` and `graph_LI.json`

### Step 9 — Visualization

Open the interactive graph in your browser:

```bash
python -m http.server 8000
xdg-open outputs/visualizations/graph_visualization.html
```

Click nodes to see entity details. Click edges to see supporting evidence and population context. Use the filter buttons to show specific edge or node types.

---

## Key data files

| File | Description |
|---|---|
| `data/raw/Flatfeet_clean.csv` | Source abstracts |
| `data/raw/golden_standard.csv` | Manual gold for anatomy linking (round 2) |
| `data/processed/01_ner_normalized/normalized_entities.json` | Central hub — feeds everything downstream |
| `data/processed/evaluation/extraction_gold_standard/entities_manual_gold.json` | NER gold standard |
| `data/processed/evaluation/extraction_gold_standard/relations_manual_gold.json` | RE gold standard |
| `data/processed/03_relations_extracted/relations_Claude.json` | Extracted relation triples |
| `data/processed/04_knowledge_graph/graph_LI.json` | Final knowledge graph |

---

