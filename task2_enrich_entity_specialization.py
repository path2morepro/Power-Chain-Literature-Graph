"""
Task 2 – Produce entities_specialization.json: a richer version of
entity_specification.json that adds entity_id and the full mention block
(matching the style used in normalized_entities.json).

entity_specification.json  → Symptom + Terms-of-Body-Movements entries,
                              with abstract_id but **no** entity_id.
normalized_entities.json   → entity_id for every entity, keyed on
                              (abstract_id, mention.position).

Matching key: (abstract_id, mention.position)  – unique per document.

Output file: entities_specialization.json
Schema per record:
{
    "abstract_id":          str,
    "entity_id":            str | null,   # null if no match was found
    "entity_form":          str,
    "field":                str,
    "mention": {
        "original_text":    str,
        "position":         int
    },
    "candidate_anatomies":  [str, ...],
    "predicted_locations":  [str, ...]
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
# Build lookup  (abstract_id, position) → entity dict
# ---------------------------------------------------------------------------

def build_position_lookup(normalized: dict) -> dict:
    """
    Returns a dict keyed by (abstract_id, position) whose value is the
    entity record from normalized_entities.json.

    When two entities share the same position in the same abstract (edge case),
    we keep all of them in a list and resolve ties later by entity_form.
    """
    lookup: dict = {}
    for abstract_entry in normalized["abstracts"]:
        abstract_id = abstract_entry["abstract"]["abstract_id"]
        for ent in abstract_entry["entities"]:
            key = (abstract_id, ent["mention"]["position"])
            if key not in lookup:
                lookup[key] = []
            lookup[key].append(ent)
    return lookup


def resolve_match(candidates: list, entity_form: str) -> dict | None:
    """
    From a list of entities at the same (abstract_id, position), return the
    one whose entity_form most closely matches the spec entry.  Falls back to
    the first candidate when no string match is found.
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    # Prefer exact form match
    for c in candidates:
        if c["entity_form"].lower() == entity_form.lower():
            return c
    # Prefer same field prefix (Symptom / Terms of Body Movements)
    return candidates[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    normalized = load_json("normalized_entities.json")
    spec_entries = load_json("entity_specification.json")

    lookup = build_position_lookup(normalized)

    enriched = []
    stats = defaultdict(int)

    for item in spec_entries:
        abstract_id  = item["abstract_id"]
        position     = item["mention"]["position"]
        entity_form  = item["entity_form"]
        key          = (abstract_id, position)

        candidates   = lookup.get(key, [])
        matched      = resolve_match(candidates, entity_form)

        if matched is None:
            stats["unmatched"] += 1
            log.debug("No match: abs=%s pos=%d form='%s'",
                      abstract_id, position, entity_form)
        else:
            stats["matched"] += 1

        enriched_record = {
            "abstract_id":         abstract_id,
            "entity_id":           matched["entity_id"] if matched else None,
            "entity_form":         entity_form,
            "field":               item["field"],
            "mention":             item["mention"],          # original_text + position
            "candidate_anatomies": item.get("candidate_anatomies", []),
            "predicted_locations": item.get("predicted_locations", []),
        }
        enriched.append(enriched_record)

    log.info("Matched: %d / %d  (unmatched: %d)",
             stats["matched"], len(spec_entries), stats["unmatched"])
    save_json(enriched, "entities_specialization.json")


if __name__ == "__main__":
    main()
