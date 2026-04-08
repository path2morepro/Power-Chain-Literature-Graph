"""
knowledge_graph.py
==================
Constructs a biomedical knowledge graph from:
  • data/processed/02_entity_enrichment/anatomical_entities_enriched.json
  • data/processed/01_ner_normalized/normalized_entities.json
  • data/processed/03_relations_extracted/relations_claude.json

DATA STRUCTURE
--------------
The graph is stored as a plain JSON object with two top-level keys:

  {
    "nodes": {                          # keyed by entity_id
      "ent_031": {
        "id":          "ent_031",       # = entity_id
        "label":       "knee",          # = entity_form (display text)
        "type":        "Anatomical Entity" | "Symptom" | "Movement",
        "mentions": [                   # all occurrences across abstracts
          { "abstract_id": "abs_001",
            "original_text": "knee",
            "position": 216 }
        ]
      },
      ...
    },

    "edges": [                          # list of directed edge objects
      {
        "id":        "e_000",           # sequential id
        "source":    "ent_114",         # subject entity_id
        "target":    "ent_109",         # object entity_id
        "relation":  "contributes_to" | "associated_with" | "anatomical_link",
        "edge_class":"relation"         # "relation" or "anatomy"
        "abstracts": [                  # one entry per supporting abstract
          {
            "abstract_id": "abs_001",
            "abstract_text": "Background ...",
            "evidence": "Flat foot is one of the contributing factors..."
          }
        ],
        "population_entities": [        # unique population entities from those abstracts
          { "entity_id": "ent_098",
            "entity_form": "nonathletic population" }
        ]
      },
      ...
    ]
  }

EDGE TYPES
----------
1. "anatomical_link"  – Anatomical Entity ↔ Symptom or Anatomical Entity ↔ Movement
   Source: anatomical_entities_enriched.json (related_symptom_entity_ids /
           related_movement_entity_documentationids).
   These edges represent "this anatomy is involved in this symptom/movement".

2. "contributes_to" / "associated_with"  – Symptom ↔ Symptom, Symptom ↔ Movement,
   or Movement ↔ Movement.
   Source: relations_claude.json (only the two named relation types).
   Subject/object strings are matched to entity_ids via a normalised-form lookup.

Usage
-----
  python -m src.pipeline.knowledge_graph

Output
------
  data/processed/04_knowledge_graph/graph_using_relation.json
"""

import json
import re
import logging
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Path definitions
# ─────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # src/pipeline → project root
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"

NORMALIZED_ENTITIES_PATH = DATA_PROCESSED_DIR / "01_ner_normalized" / "normalized_entities.json"
ANATOMICAL_ENTITIES_PATH = DATA_PROCESSED_DIR / "02_entity_enrichment" / "anatomical_entities_enriched.json"
RELATIONS_PATH = DATA_PROCESSED_DIR / "03_relations_extracted" / "relations_Claude.json"
ENTITY_SPECIFICATION_PATH = DATA_PROCESSED_DIR / "evaluation" / "baselines" / "entity_specification.json"

# Output graphs for both methods
GRAPH_METHOD1_PATH = DATA_PROCESSED_DIR / "04_knowledge_graph" / "graph_ES.json"
GRAPH_METHOD2_PATH = DATA_PROCESSED_DIR / "04_knowledge_graph" / "graph_LI.json"
GRAPH_OUTPUT_PATH = DATA_PROCESSED_DIR / "04_knowledge_graph" / "graph_using_relation.json"

# ─────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────

def load(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def save(data, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    log.info("Saved → %s", path)

# ─────────────────────────────────────────────────────────────
# Normalisation helper for form matching
# ─────────────────────────────────────────────────────────────

def _norm(text: str) -> str:
    """Lowercase, collapse whitespace, remove trailing 's' for plural."""
    t = re.sub(r"\s+", " ", text.lower().strip())
    # light stemming: drop trailing 's' (fasciitis→fasciiti already in data)
    return t.rstrip("s")

# ─────────────────────────────────────────────────────────────
# Step 1 – collect nodes
# ─────────────────────────────────────────────────────────────

TARGET_FIELDS = {"Anatomical Entity", "Symptom", "Terms of Body Movements"}
FIELD_SHORT   = {
    "Anatomical Entity":      "Anatomical Entity",
    "Symptom":                "Symptom",
    "Terms of Body Movements":"Movement",
}

def collect_nodes(normalized: dict, anatomical: list) -> tuple[dict, dict]:
    """
    Returns
    -------
    nodes      : {entity_id → node_dict}
    abs_index  : {abstract_id → {text, population_entities: [{entity_id, entity_form}]}}
    """
    nodes: dict[str, dict] = {}
    abs_index: dict[str, dict] = {}

    for abs_entry in normalized["abstracts"]:
        ab        = abs_entry["abstract"]
        abstract_id = ab["abstract_id"]
        abs_index[abstract_id] = {
            "text":               ab["text"],
            "population_entities": [],
        }
        pop_seen = set()

        for ent in abs_entry["entities"]:
            field = ent["field"]

            if field == "Population":
                key = ent["entity_id"]
                if key not in pop_seen:
                    pop_seen.add(key)
                    abs_index[abstract_id]["population_entities"].append({
                        "entity_id":   ent["entity_id"],
                        "entity_form": ent["entity_form"],
                    })
                continue

            if field not in TARGET_FIELDS:
                continue

            eid = ent["entity_id"]
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

    # Supplement anatomical nodes from enriched file (catches any not in normalised)
    for anat in anatomical:
        eid = anat["entity_id"]
        if eid not in nodes:
            nodes[eid] = {
                "id":       eid,
                "label":    anat["entity_form"],
                "type":     "Anatomical Entity",
                "mentions": anat["mentions"],
            }

    log.info("Nodes collected: %d  (Anatomical=%d  Symptom=%d  Movement=%d)",
             len(nodes),
             sum(1 for n in nodes.values() if n["type"] == "Anatomical Entity"),
             sum(1 for n in nodes.values() if n["type"] == "Symptom"),
             sum(1 for n in nodes.values() if n["type"] == "Movement"))

    return nodes, abs_index

# ─────────────────────────────────────────────────────────────
# Step 2 – build normalised-form → entity_id lookup
# ─────────────────────────────────────────────────────────────

def build_form_lookup(nodes: dict) -> dict[str, list[str]]:
    """
    {normalised_entity_form → [entity_id, ...]}
    Multiple ids can share the same surface form (e.g. typo variants).
    """
    lookup: dict[str, list[str]] = defaultdict(list)
    for eid, node in nodes.items():
        lookup[_norm(node["label"])].append(eid)
    log.info("Form-lookup entries: %d", len(lookup))
    return lookup

# ─────────────────────────────────────────────────────────────
# Step 3a – anatomical link edges
# ─────────────────────────────────────────────────────────────

def build_anatomical_edges(anatomical: list, nodes: dict, abs_index: dict) -> list[dict]:
    """
    One edge per (anat_eid, related_eid) pair.
    We infer which abstracts support this link by scanning mentions of both
    endpoints and collecting abstract_ids they share.
    """
    edges = []
    edge_id = 0

    # Build a quick eid → set(abstract_ids) map
    eid_to_abstracts: dict[str, set] = defaultdict(set)
    for eid, node in nodes.items():
        for m in node["mentions"]:
            eid_to_abstracts[eid].add(m["abstract_id"])

    for anat in anatomical:
        anat_eid = anat["entity_id"]
        if anat_eid not in nodes:
            continue

        for rel_eid_list, direction in [
            (anat["related_symptom_entity_ids"],  "Symptom"),
            (anat["related_movement_entity_ids"], "Movement"),
        ]:
            for rel_eid in rel_eid_list:
                if rel_eid not in nodes:
                    continue
                if nodes[rel_eid]["type"] not in {"Symptom", "Movement"}:
                    continue

                # Shared abstract ids
                shared_abs = (eid_to_abstracts[anat_eid]
                              & eid_to_abstracts[rel_eid])

                abstracts_info = []
                population_seen: set = set()
                population_entities: list = []

                for ab_id in sorted(shared_abs):
                    ab_data = abs_index.get(ab_id, {})
                    abstracts_info.append({
                        "abstract_id":   ab_id,
                        "abstract_text": ab_data.get("text", ""),
                        "evidence":      "",          # anatomy links have no explicit evidence sentence
                    })
                    for pop in ab_data.get("population_entities", []):
                        k = pop["entity_id"]
                        if k not in population_seen:
                            population_seen.add(k)
                            population_entities.append(pop)

                edges.append({
                    "id":                 f"e_{edge_id:04d}",
                    "source":             anat_eid,
                    "target":             rel_eid,
                    "relation":           "anatomical_link",
                    "edge_class":         "anatomy",
                    "abstracts":          abstracts_info,
                    "population_entities": population_entities,
                })
                edge_id += 1

    log.info("Anatomical edges: %d", len(edges))
    return edges, edge_id

# ─────────────────────────────────────────────────────────────
# Step 3b – relation edges (contributes_to / associated_with)
# ─────────────────────────────────────────────────────────────

ALLOWED_RELATIONS = {"contributes_to", "associated_with"}

def build_relation_edges(
    relations: dict,
    nodes: dict,
    form_lookup: dict,
    abs_index: dict,
    edge_id_start: int,
) -> list[dict]:
    """
    For each (abstract, relation-triple) where relation ∈ ALLOWED_RELATIONS:
      • Match subject  → entity_id  (must be Symptom or Movement)
      • Match object   → entity_id  (must be Symptom or Movement)
      • Attach evidence + population from that abstract.
    Duplicate (source, target, relation) pairs across abstracts are **merged**
    into a single edge with multiple abstract entries.
    """
    # Temporary accumulator: (src_eid, tgt_eid, relation) → edge dict
    edge_acc: dict[tuple, dict] = {}
    edge_id = edge_id_start

    sym_mov_types = {"Symptom", "Movement"}

    for abstract_id, triple_list in relations.items():
        ab_data = abs_index.get(abstract_id, {})

        for triple in triple_list:
            if triple["relation"] not in ALLOWED_RELATIONS:
                continue

            subj_norm = _norm(triple["subject"])
            obj_norm  = _norm(triple["object"])

            src_ids = form_lookup.get(subj_norm, [])
            tgt_ids = form_lookup.get(obj_norm,  [])

            if not src_ids or not tgt_ids:
                log.debug("No match  abs=%s  subj='%s' obj='%s'",
                          abstract_id, triple["subject"], triple["object"])
                continue

            for src_eid in src_ids:
                for tgt_eid in tgt_ids:
                    if src_eid == tgt_eid:
                        continue
                    if nodes[src_eid]["type"] not in sym_mov_types:
                        continue
                    if nodes[tgt_eid]["type"] not in sym_mov_types:
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

                    # Add abstract entry
                    existing_abs_ids = {a["abstract_id"] for a in edge["abstracts"]}
                    if abstract_id not in existing_abs_ids:
                        edge["abstracts"].append({
                            "abstract_id":   abstract_id,
                            "abstract_text": ab_data.get("text", ""),
                            "evidence":      triple.get("evidence", ""),
                        })

                    # Merge population entities
                    existing_pop = {p["entity_id"] for p in edge["population_entities"]}
                    for pop in ab_data.get("population_entities", []):
                        if pop["entity_id"] not in existing_pop:
                            edge["population_entities"].append(pop)
                            existing_pop.add(pop["entity_id"])

    relation_edges = list(edge_acc.values())
    log.info("Relation edges: %d  (contributes_to + associated_with)", len(relation_edges))
    return relation_edges

# ─────────────────────────────────────────────────────────────
# Step 3c – METHOD 1: entity specification edges (symptom/movement → anatomy)
# ─────────────────────────────────────────────────────────────

def build_entity_specification_edges(
    entity_specs: list,
    nodes: dict,
    form_lookup: dict,
    abs_index: dict,
    edge_id_start: int,
) -> list[dict]:
    """
    METHOD 1: Link symptoms/movements to anatomies using entity_specification.json.

    For each entity_spec entry (symptom/movement with candidate anatomies and
    predicted_locations), create an edge from the symptom/movement to its
    best-ranked anatomical location.
    """
    edge_acc: dict[tuple, dict] = {}
    edge_id = edge_id_start

    sym_mov_types = {"Symptom", "Movement"}

    for spec in entity_specs:
        abstract_id = spec["abstract_id"]
        entity_form = spec["entity_form"]
        field = spec["field"]
        predicted_locs = spec.get("predicted_locations", [])

        if not predicted_locs or field not in sym_mov_types:
            continue

        ab_data = abs_index.get(abstract_id, {})

        # Find source entity (symptom/movement)
        src_norm = _norm(entity_form)
        src_ids = form_lookup.get(src_norm, [])

        if not src_ids:
            continue

        # Get target entity (predicted anatomy location)
        target_form = predicted_locs[0]  # Take the top-1 prediction
        tgt_norm = _norm(target_form)
        tgt_ids = form_lookup.get(tgt_norm, [])

        if not tgt_ids:
            continue

        for src_eid in src_ids:
            for tgt_eid in tgt_ids:
                if src_eid == tgt_eid:
                    continue
                if nodes[src_eid]["type"] not in sym_mov_types:
                    continue
                if nodes[tgt_eid]["type"] != "Anatomical Entity":
                    continue

                # Create edge key (source, target, relation_type)
                key = (src_eid, tgt_eid, "located_in")

                if key not in edge_acc:
                    edge_acc[key] = {
                        "id":                  f"e_{edge_id:04d}",
                        "source":              src_eid,
                        "target":              tgt_eid,
                        "relation":            "located_in",
                        "edge_class":          "specification",
                        "abstracts":           [],
                        "population_entities": [],
                    }
                    edge_id += 1

                edge = edge_acc[key]

                # Add abstract entry (avoid duplicates)
                existing_abs_ids = {a["abstract_id"] for a in edge["abstracts"]}
                if abstract_id not in existing_abs_ids:
                    edge["abstracts"].append({
                        "abstract_id":   abstract_id,
                        "abstract_text": ab_data.get("text", ""),
                        "evidence":      f"{entity_form} → {target_form} (entity specification ranking)",
                    })

                # Merge population entities
                existing_pop = {p["entity_id"] for p in edge["population_entities"]}
                for pop in ab_data.get("population_entities", []):
                    if pop["entity_id"] not in existing_pop:
                        edge["population_entities"].append(pop)
                        existing_pop.add(pop["entity_id"])

    spec_edges = list(edge_acc.values())
    log.info("Entity specification edges: %d", len(spec_edges))
    return spec_edges

# ─────────────────────────────────────────────────────────────
# Step 3d – METHOD 2: located_in relation edges (from relations_claude.json)
# ─────────────────────────────────────────────────────────────

def build_located_in_edges(
    relations: dict,
    nodes: dict,
    form_lookup: dict,
    abs_index: dict,
    edge_id_start: int,
) -> list[dict]:
    """
    METHOD 2: Link symptoms/movements to anatomies using explicit "located_in"
    relations from relations_claude.json.

    For each relation triple with relation_type == "located_in":
      • Subject should be Symptom or Movement
      • Object should be Anatomical Entity
      • Create edge: symptom/movement → anatomy
    """
    edge_acc: dict[tuple, dict] = {}
    edge_id = edge_id_start

    sym_mov_types = {"Symptom", "Movement"}

    for abstract_id, triple_list in relations.items():
        ab_data = abs_index.get(abstract_id, {})

        for triple in triple_list:
            if triple["relation"] != "located_in":
                continue

            subj_norm = _norm(triple["subject"])
            obj_norm  = _norm(triple["object"])

            src_ids = form_lookup.get(subj_norm, [])
            tgt_ids = form_lookup.get(obj_norm,  [])

            if not src_ids or not tgt_ids:
                log.debug("No match (located_in)  abs=%s  subj='%s' obj='%s'",
                          abstract_id, triple["subject"], triple["object"])
                continue

            for src_eid in src_ids:
                for tgt_eid in tgt_ids:
                    if src_eid == tgt_eid:
                        continue
                    if nodes[src_eid]["type"] not in sym_mov_types:
                        continue
                    if nodes[tgt_eid]["type"] != "Anatomical Entity":
                        continue

                    key = (src_eid, tgt_eid, "located_in")

                    if key not in edge_acc:
                        edge_acc[key] = {
                            "id":                  f"e_{edge_id:04d}",
                            "source":              src_eid,
                            "target":              tgt_eid,
                            "relation":            "located_in",
                            "edge_class":          "relation",
                            "abstracts":           [],
                            "population_entities": [],
                        }
                        edge_id += 1

                    edge = edge_acc[key]

                    # Add abstract entry
                    existing_abs_ids = {a["abstract_id"] for a in edge["abstracts"]}
                    if abstract_id not in existing_abs_ids:
                        edge["abstracts"].append({
                            "abstract_id":   abstract_id,
                            "abstract_text": ab_data.get("text", ""),
                            "evidence":      triple.get("evidence", ""),
                        })

                    # Merge population entities
                    existing_pop = {p["entity_id"] for p in edge["population_entities"]}
                    for pop in ab_data.get("population_entities", []):
                        if pop["entity_id"] not in existing_pop:
                            edge["population_entities"].append(pop)
                            existing_pop.add(pop["entity_id"])

    located_in_edges = list(edge_acc.values())
    log.info("Located_in relation edges: %d", len(located_in_edges))
    return located_in_edges

def main():
    log.info("Loading source files …")
    normalized      = load(str(NORMALIZED_ENTITIES_PATH))
    anatomical      = load(str(ANATOMICAL_ENTITIES_PATH))
    relations       = load(str(RELATIONS_PATH))
    entity_specs    = load(str(ENTITY_SPECIFICATION_PATH)) if Path(ENTITY_SPECIFICATION_PATH).exists() else []

    nodes, abs_index = collect_nodes(normalized, anatomical)
    form_lookup      = build_form_lookup(nodes)

    # === METHOD 1: Entity Enrichment (enriched anatomical data) ===
    log.info("\n--- Building METHOD 1: Entity Enrichment ---")
    anat_edges_m1, next_id_m1 = build_anatomical_edges(anatomical, nodes, abs_index)
    # Build other relation edges for METHOD 1 with correct starting ID
    other_rel_edges_m1 = build_relation_edges(relations, nodes, form_lookup, abs_index, next_id_m1)

    all_edges_m1 = anat_edges_m1 + other_rel_edges_m1

    graph_m1 = {
        "metadata": {
            "method": "METHOD 1: Entity Enrichment",
            "description": (
                "Knowledge graph where Anatomical Entity ↔ Symptom/Movement links are "
                "derived from enriched anatomical data (anatomical_entities_enriched.json). "
                "Anatomical links come from pre-enriched semantic relationships."
            ),
            "node_types":  ["Anatomical Entity", "Symptom", "Movement"],
            "edge_types":  ["anatomical_link", "contributes_to", "associated_with"],
            "node_count":  len(nodes),
            "edge_count":  len(all_edges_m1),
            "anatomical_link_edges": len(anat_edges_m1),
            "other_relation_edges": len(other_rel_edges_m1),
        },
        "nodes": nodes,
        "edges": all_edges_m1,
    }

    GRAPH_METHOD1_PATH.parent.mkdir(parents=True, exist_ok=True)
    save(graph_m1, str(GRAPH_METHOD1_PATH))
    log.info("✓ Saved METHOD 1: %s", GRAPH_METHOD1_PATH)
    log.info("  Summary: %d nodes, %d edges (%d anatomical_links + %d other relations)",
             len(nodes), len(all_edges_m1), len(anat_edges_m1), len(other_rel_edges_m1))

    # === METHOD 2: Located_in Relations (explicit relation triples) ===
    log.info("\n--- Building METHOD 2: Located_in Relations ---")
    located_in_edges_m2 = build_located_in_edges(relations, nodes, form_lookup, abs_index, 0)
    # Build other relation edges for METHOD 2 with correct starting ID
    next_id_m2 = len(located_in_edges_m2)
    other_rel_edges_m2 = build_relation_edges(relations, nodes, form_lookup, abs_index, next_id_m2)

    all_edges_m2 = located_in_edges_m2 + other_rel_edges_m2

    graph_m2 = {
        "metadata": {
            "method": "METHOD 2: Located_in Relations",
            "description": (
                "Knowledge graph where Symptom/Movement ↔ Anatomical Entity links are "
                "derived from explicit 'located_in' relations in relations_claude.json. "
                "Anatomical links come from extracted relation triples."
            ),
            "node_types":  ["Anatomical Entity", "Symptom", "Movement"],
            "edge_types":  ["anatomical_link", "contributes_to", "associated_with"],
            "node_count":  len(nodes),
            "edge_count":  len(all_edges_m2),
            "anatomical_link_edges": len(located_in_edges_m2),
            "other_relation_edges": len(other_rel_edges_m2),
        },
        "nodes": nodes,
        "edges": all_edges_m2,
    }

    GRAPH_METHOD2_PATH.parent.mkdir(parents=True, exist_ok=True)
    save(graph_m2, str(GRAPH_METHOD2_PATH))
    log.info("✓ Saved METHOD 2: %s", GRAPH_METHOD2_PATH)
    log.info("  Summary: %d nodes, %d edges (%d anatomical_links + %d other relations)",
             len(nodes), len(all_edges_m2), len(located_in_edges_m2), len(other_rel_edges_m2))


if __name__ == "__main__":
    main()
