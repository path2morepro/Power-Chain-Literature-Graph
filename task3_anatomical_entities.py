"""
Task 3 – Build anatomical_entities_enriched.json.

For every unique Anatomical-Entity in normalized_entities.json, record:
  • all mentions across abstracts
  • entity_ids of Symptom entities whose predicted_locations include this anatomy
  • entity_ids of Terms-of-Body-Movements entities whose predicted_locations
    include this anatomy

The source for the "predicted_locations" links is entity_specification.json.
entity_specification.json has no entity_ids, so we look them up from
normalized_entities.json via the (abstract_id, mention.position) key.

Output file: anatomical_entities_enriched.json
Schema per record:
{
    "entity_id":    str,
    "entity_form":  str,
    "field":        "Anatomical Entity",
    "mentions": [
        {
            "abstract_id":    str,
            "original_text":  str,
            "position":       int
        },
        ...
    ],
    "related_symptom_entity_ids":   [str, ...],   # deduplicated, sorted
    "related_movement_entity_ids":  [str, ...]    # deduplicated, sorted
}
"""

import json
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_json(path: str) -> dict | list:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def save_json(data, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=4, ensure_ascii=False)
    log.info("Saved → %s  (%d records)", path, len(data))


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------

def collect_anatomical_entities(normalized: dict) -> dict:
    """
    Return {entity_id: record} for every Anatomical-Entity in
    normalized_entities.json.  Accumulate all mentions.
    """
    anatomy_map: dict = {}

    for abstract_entry in normalized["abstracts"]:
        abstract_id = abstract_entry["abstract"]["abstract_id"]
        for ent in abstract_entry["entities"]:
            if ent["field"] != "Anatomical Entity":
                continue
            eid = ent["entity_id"]
            if eid not in anatomy_map:
                anatomy_map[eid] = {
                    "entity_id":   eid,
                    "entity_form": ent["entity_form"],
                    "field":       "Anatomical Entity",
                    "mentions":    [],
                    # Use sets during construction, convert to lists later
                    "_symptom_ids":  set(),
                    "_movement_ids": set(),
                }
            anatomy_map[eid]["mentions"].append({
                "abstract_id":   abstract_id,
                "original_text": ent["mention"]["original_text"],
                "position":      ent["mention"]["position"],
            })

    log.info("Found %d unique anatomical entities.", len(anatomy_map))
    return anatomy_map


def build_position_lookup(normalized: dict) -> dict:
    """(abstract_id, position) → entity record."""
    lookup: dict = {}
    for abstract_entry in normalized["abstracts"]:
        abstract_id = abstract_entry["abstract"]["abstract_id"]
        for ent in abstract_entry["entities"]:
            key = (abstract_id, ent["mention"]["position"])
            # Keep a list to handle rare position collisions; first wins for our use-case
            lookup.setdefault(key, ent)
    return lookup


def build_form_to_anatomy_ids(anatomy_map: dict) -> dict:
    """
    Lowercase entity_form → list of entity_ids.

    Multiple anatomical entities can share the same textual form
    (e.g. typo variants "medial" vs "medical" are separate entity_ids).
    """
    form_map: dict = defaultdict(list)
    for eid, data in anatomy_map.items():
        form_map[data["entity_form"].lower()].append(eid)
    return form_map


# ---------------------------------------------------------------------------
# Link symptoms / movements to anatomies via predicted_locations
# ---------------------------------------------------------------------------

def link_related_entities(
    spec_entries: list,
    anatomy_map: dict,
    form_to_anatomy_ids: dict,
    pos_lookup: dict,
) -> None:
    """
    Iterate over entity_specification.json entries (Symptom + Movement) and
    for each predicted_location, add the entry's entity_id to the matching
    anatomical entity's related set.
    """
    for item in spec_entries:
        abstract_id = item["abstract_id"]
        position    = item["mention"]["position"]
        field       = item["field"]
        key         = (abstract_id, position)

        matched = pos_lookup.get(key)
        if matched is None:
            log.debug("No normalized match for spec entry abs=%s pos=%d form='%s'",
                      abstract_id, position, item["entity_form"])
            continue

        sym_mov_eid = matched["entity_id"]

        for pred_loc in item.get("predicted_locations", []):
            anatomy_ids = form_to_anatomy_ids.get(pred_loc.lower(), [])
            for anat_eid in anatomy_ids:
                if field == "Symptom":
                    anatomy_map[anat_eid]["_symptom_ids"].add(sym_mov_eid)
                elif field == "Terms of Body Movements":
                    anatomy_map[anat_eid]["_movement_ids"].add(sym_mov_eid)


# ---------------------------------------------------------------------------
# Serialise
# ---------------------------------------------------------------------------

def finalise(anatomy_map: dict) -> list:
    """Convert sets → sorted lists and remove internal keys."""
    output = []
    for eid, data in sorted(anatomy_map.items()):
        output.append({
            "entity_id":                   data["entity_id"],
            "entity_form":                 data["entity_form"],
            "field":                       data["field"],
            "mentions":                    data["mentions"],
            "related_symptom_entity_ids":  sorted(data["_symptom_ids"]),
            "related_movement_entity_ids": sorted(data["_movement_ids"]),
        })
    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    normalized   = load_json("normalized_entities.json")
    spec_entries = load_json("entity_specification.json")

    anatomy_map         = collect_anatomical_entities(normalized)
    pos_lookup          = build_position_lookup(normalized)
    form_to_anatomy_ids = build_form_to_anatomy_ids(anatomy_map)

    link_related_entities(spec_entries, anatomy_map, form_to_anatomy_ids, pos_lookup)

    # Summary stats
    linked = sum(
        1 for d in anatomy_map.values()
        if d["_symptom_ids"] or d["_movement_ids"]
    )
    log.info(
        "%d / %d anatomical entities have at least one linked symptom or movement.",
        linked, len(anatomy_map),
    )

    output = finalise(anatomy_map)
    save_json(output, "anatomical_entities_enriched.json")


if __name__ == "__main__":
    main()
