"""
build_graph_v2.py
=================
Constructs a biomedical knowledge graph from:
  • normalized_entities.json
  • relations_GPT5.json

Differences from build_graph.py (v1)
--------------------------------------
  v1  Anatomical ↔ Symptom/Movement edges were derived from the
      pre-computed `related_symptom_entity_ids` / `related_movement_entity_ids`
      lists inside anatomical_entities_enriched.json.

  v2  Anatomical ↔ Symptom/Movement edges are derived directly from
      **located_in** triples in relations_GPT5.json.
      A triple  { subject: X, relation: located_in, object: Y }
      is accepted when
        • X resolves to a Symptom or Movement entity_id, AND
        • Y resolves to an Anatomical Entity entity_id.
      The resulting edge runs  X → Y  (symptom/movement → anatomy).
      Evidence text from the triple and population entities from that
      abstract are attached to the edge exactly as in v1.

      Relation edges (contributes_to / associated_with) between
      Symptom ↔ Symptom / Symptom ↔ Movement / Movement ↔ Movement
      are built identically to v1.

DATA STRUCTURE  (graph_v2.json)
--------------------------------
{
  "metadata": { … },
  "nodes": {                          # keyed by entity_id
    "ent_031": {
      "id":       "ent_031",
      "label":    "knee",
      "type":     "Anatomical Entity" | "Symptom" | "Movement",
      "mentions": [{ "abstract_id", "original_text", "position" }, …]
    }, …
  },
  "edges": [
    {
      "id":          "e_0000",
      "source":      str,             # entity_id of subject
      "target":      str,             # entity_id of object
      "relation":    "anatomical_link" | "contributes_to" | "associated_with",
      "edge_class":  "anatomy" | "relation",
      "abstracts": [
        { "abstract_id", "abstract_text", "evidence" }, …
      ],
      "population_entities": [
        { "entity_id", "entity_form" }, …
      ]
    }, …
  ]
}

Usage
-----
  python build_graph_v2.py

Output
------
  graph_v2.json
"""

import json
import re
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────

def load(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def save(data, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    log.info("Saved → %s", path)

# ─────────────────────────────────────────────────────────────
# Normalisation
# ─────────────────────────────────────────────────────────────

def _norm(text: str) -> str:
    """Lowercase, collapse whitespace, strip trailing 's' (light plural-stem)."""
    t = re.sub(r"\s+", " ", text.lower().strip())
    return t.rstrip("s")

# ─────────────────────────────────────────────────────────────
# Step 1 – collect nodes + abstract index
# ─────────────────────────────────────────────────────────────

TARGET_FIELDS = {"Anatomical Entity", "Symptom", "Terms of Body Movements"}
FIELD_SHORT   = {
    "Anatomical Entity":       "Anatomical Entity",
    "Symptom":                 "Symptom",
    "Terms of Body Movements": "Movement",
}

def collect_nodes(normalized: dict) -> tuple[dict, dict]:
    """
    Returns
    -------
    nodes     : {entity_id → node_dict}
    abs_index : {abstract_id → {text, population_entities}}
    """
    nodes: dict[str, dict] = {}
    abs_index: dict[str, dict] = {}

    for abs_entry in normalized["abstracts"]:
        ab          = abs_entry["abstract"]
        abstract_id = ab["abstract_id"]
        abs_index[abstract_id] = {
            "text":               ab["text"],
            "population_entities": [],
        }
        pop_seen: set = set()

        for ent in abs_entry["entities"]:
            field = ent["field"]

            if field == "Population":
                k = ent["entity_id"]
                if k not in pop_seen:
                    pop_seen.add(k)
                    abs_index[abstract_id]["population_entities"].append({
                        "entity_id":   ent["entity_id"],
                        "entity_form": ent["entity_form"],
                    })
                continue

            if field not in TARGET_FIELDS:
                continue

            eid     = ent["entity_id"]
            mention = {
                "abstract_id":   abstract_id,
                "original_text": ent["mention"]["original_text"],
                "position":      ent["mention"]["position"],
            }
            if eid not in nodes:
                nodes[eid] = {
                    "id":       eid,
                    "label":    ent["entity_form"],
                    "type":     FIELD_SHORT[field],
                    "mentions": [],
                }
            nodes[eid]["mentions"].append(mention)

    log.info(
        "Nodes collected: %d  (Anatomical=%d  Symptom=%d  Movement=%d)",
        len(nodes),
        sum(1 for n in nodes.values() if n["type"] == "Anatomical Entity"),
        sum(1 for n in nodes.values() if n["type"] == "Symptom"),
        sum(1 for n in nodes.values() if n["type"] == "Movement"),
    )
    return nodes, abs_index

# ─────────────────────────────────────────────────────────────
# Step 2 – form lookup  (normalised_form → [entity_id])
# ─────────────────────────────────────────────────────────────

def build_form_lookup(nodes: dict) -> dict[str, list[str]]:
    lookup: dict[str, list[str]] = defaultdict(list)
    for eid, node in nodes.items():
        lookup[_norm(node["label"])].append(eid)
    log.info("Form-lookup entries: %d", len(lookup))
    return lookup

# ─────────────────────────────────────────────────────────────
# Step 3 – resolve a raw form string → list[entity_id]
#
# Two-pass strategy:
#   Pass 1 – exact normalised match  (fast, precise)
#   Pass 2 – substring match: the normalised lookup key is a
#             sub-phrase of the normalised triple-form, or vice-versa.
#             This recovers things like
#               "peak hip adduction" → "hip adduction"  (key is shorter)
#               "stiffness"          → "ankle stiffness" (key is longer,
#                                                          form is shorter)
# ─────────────────────────────────────────────────────────────

def resolve_form(raw: str, form_lookup: dict,
                 required_types: set | None, nodes: dict) -> list[str]:
    """Return entity_ids matching `raw`, optionally filtered by type."""
    n = _norm(raw)

    # Pass 1 – exact
    candidates = list(form_lookup.get(n, []))

    # Pass 2 – substring fallback
    if not candidates:
        for key, eids in form_lookup.items():
            if key in n or n in key:
                candidates.extend(eids)

    if required_types:
        candidates = [eid for eid in candidates
                      if nodes[eid]["type"] in required_types]

    return candidates

# ─────────────────────────────────────────────────────────────
# Step 4a – anatomical edges via located_in
# ─────────────────────────────────────────────────────────────

def build_located_in_edges(
    relations:   dict,
    nodes:       dict,
    form_lookup: dict,
    abs_index:   dict,
) -> tuple[list[dict], int]:
    """
    Scan every located_in triple.
    Accept only:   subject ∈ {Symptom, Movement}
                   object  ∈ {Anatomical Entity}
    Merge duplicate (src, tgt) pairs across abstracts into one edge.
    Direction: src=symptom/movement  →  tgt=anatomy
    """
    SYM_MOV = {"Symptom", "Movement"}
    ANAT    = {"Anatomical Entity"}

    edge_acc: dict[tuple, dict] = {}
    edge_id  = 0

    for abstract_id, triple_list in relations.items():
        ab_data = abs_index.get(abstract_id, {})

        for triple in triple_list:
            if triple["relation"] != "located_in":
                continue

            src_ids = resolve_form(triple["subject"], form_lookup, SYM_MOV, nodes)
            tgt_ids = resolve_form(triple["object"],  form_lookup, ANAT,    nodes)

            if not src_ids:
                log.debug("located_in – no Symptom/Movement match for subject '%s' (abs=%s)",
                          triple["subject"], abstract_id)
                continue
            if not tgt_ids:
                log.debug("located_in – no Anatomical match for object '%s' (abs=%s)",
                          triple["object"], abstract_id)
                continue

            for src_eid in src_ids:
                for tgt_eid in tgt_ids:
                    if src_eid == tgt_eid:
                        continue
                    key = (src_eid, tgt_eid)
                    if key not in edge_acc:
                        edge_acc[key] = {
                            "id":                  f"e_{edge_id:04d}",
                            "source":              src_eid,
                            "target":              tgt_eid,
                            "relation":            "anatomical_link",
                            "edge_class":          "anatomy",
                            "abstracts":           [],
                            "population_entities": [],
                        }
                        edge_id += 1

                    edge = edge_acc[key]

                    # Accumulate abstract entry
                    seen_abs = {a["abstract_id"] for a in edge["abstracts"]}
                    if abstract_id not in seen_abs:
                        edge["abstracts"].append({
                            "abstract_id":   abstract_id,
                            "abstract_text": ab_data.get("text", ""),
                            "evidence":      triple.get("evidence", ""),
                        })

                    # Accumulate population
                    seen_pop = {p["entity_id"] for p in edge["population_entities"]}
                    for pop in ab_data.get("population_entities", []):
                        if pop["entity_id"] not in seen_pop:
                            edge["population_entities"].append(pop)
                            seen_pop.add(pop["entity_id"])

    anat_edges = list(edge_acc.values())
    log.info("Anatomical (located_in) edges: %d", len(anat_edges))
    return anat_edges, edge_id

# ─────────────────────────────────────────────────────────────
# Step 4b – relation edges  (contributes_to / associated_with)
#           identical logic to v1
# ─────────────────────────────────────────────────────────────

ALLOWED_RELATIONS = {"contributes_to", "associated_with"}

def build_relation_edges(
    relations:       dict,
    nodes:           dict,
    form_lookup:     dict,
    abs_index:       dict,
    edge_id_start:   int,
) -> list[dict]:
    SYM_MOV = {"Symptom", "Movement"}
    edge_acc: dict[tuple, dict] = {}
    edge_id  = edge_id_start

    for abstract_id, triple_list in relations.items():
        ab_data = abs_index.get(abstract_id, {})

        for triple in triple_list:
            if triple["relation"] not in ALLOWED_RELATIONS:
                continue

            src_ids = resolve_form(triple["subject"], form_lookup, SYM_MOV, nodes)
            tgt_ids = resolve_form(triple["object"],  form_lookup, SYM_MOV, nodes)

            if not src_ids or not tgt_ids:
                continue

            for src_eid in src_ids:
                for tgt_eid in tgt_ids:
                    if src_eid == tgt_eid:
                        continue

                    key = (src_eid, tgt_eid, triple["relation"])
                    if key not in edge_acc:
                        edge_acc[key] = {
                            "id":                  f"e_{edge_id:04d}",
                            "source":              src_eid,
                            "target":              tgt_eid,
                            "relation":            triple["relation"],
                            "edge_class":          "relation",
                            "abstracts":           [],
                            "population_entities": [],
                        }
                        edge_id += 1

                    edge = edge_acc[key]

                    seen_abs = {a["abstract_id"] for a in edge["abstracts"]}
                    if abstract_id not in seen_abs:
                        edge["abstracts"].append({
                            "abstract_id":   abstract_id,
                            "abstract_text": ab_data.get("text", ""),
                            "evidence":      triple.get("evidence", ""),
                        })

                    seen_pop = {p["entity_id"] for p in edge["population_entities"]}
                    for pop in ab_data.get("population_entities", []):
                        if pop["entity_id"] not in seen_pop:
                            edge["population_entities"].append(pop)
                            seen_pop.add(pop["entity_id"])

    rel_edges = list(edge_acc.values())
    log.info("Relation edges: %d  (contributes_to + associated_with)", len(rel_edges))
    return rel_edges

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    log.info("Loading source files …")
    normalized = load("normalized_entities.json")
    relations  = load("relations_GPT5.json")

    nodes, abs_index = collect_nodes(normalized)
    form_lookup      = build_form_lookup(nodes)

    anat_edges, next_id = build_located_in_edges(
        relations, nodes, form_lookup, abs_index
    )
    rel_edges = build_relation_edges(
        relations, nodes, form_lookup, abs_index, next_id
    )

    all_edges = anat_edges + rel_edges

    # Retain only nodes that appear in at least one edge
    used_node_ids = set()
    for e in all_edges:
        used_node_ids.add(e["source"])
        used_node_ids.add(e["target"])
    active_nodes = {eid: n for eid, n in nodes.items() if eid in used_node_ids}
    log.info("Active nodes (connected): %d / %d", len(active_nodes), len(nodes))

    graph = {
        "metadata": {
            "description": (
                "Biomedical knowledge graph (v2). "
                "Anatomical Entity ↔ Symptom/Movement edges are derived from "
                "'located_in' triples in relations_GPT5.json "
                "(subject=Symptom/Movement, object=Anatomical Entity). "
                "Symptom ↔ Symptom/Movement edges use 'contributes_to' and "
                "'associated_with' triples. "
                "All edges carry supporting abstract evidence and population context."
            ),
            "node_types":   ["Anatomical Entity", "Symptom", "Movement"],
            "edge_types":   ["anatomical_link", "contributes_to", "associated_with"],
            "node_count":   len(active_nodes),
            "edge_count":   len(all_edges),
            "edge_breakdown": {
                "anatomical_link": len(anat_edges),
                "relation":        len(rel_edges),
            },
        },
        "nodes": active_nodes,
        "edges": all_edges,
    }

    save(graph, "graph_v2.json")
    log.info(
        "Graph v2 summary: %d nodes, %d edges (%d anatomy / %d relation)",
        len(active_nodes), len(all_edges), len(anat_edges), len(rel_edges),
    )

if __name__ == "__main__":
    main()
