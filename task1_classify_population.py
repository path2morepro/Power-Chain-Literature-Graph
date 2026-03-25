"""
Task 1 – Classify population entities in normalized_entities.json
using the GLiNER2 model and population_specification.json as the labelling standard.

Output file: population_classified.json
Schema per record:
{
    "abstract_id": str,
    "entity_id":   str,
    "entity_form": str,
    "classification": {
        "sex":         str | null,
        "age":         str | null,
        "level":       str | null,
        "condition":   str | null,
        "study_group": str | null
    },
    "classification_list": [str, ...],   # non-null values from the dict above
    "mention": {
        "original_text": str,
        "position":      int
    }
}
"""

import json
import logging
from pathlib import Path

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
# Classification helpers
# ---------------------------------------------------------------------------

def _context_window(abstract_text: str, position: int, window: int = 20) -> str:
    """Return a ±window-word slice around *position* in the abstract."""
    words = abstract_text.split()
    start = max(0, position - window)
    end   = min(len(words), position + window)
    return " ".join(words[start:end])


def classify_entity(
    extractor,
    entity_form: str,
    abstract_text: str,
    position: int,
    spec: dict,
) -> dict:
    """
    Use GLiNER2 to assign one label per spec field for the given population entity.

    Strategy
    --------
    1. Run GLiNER2 on *entity_form* alone – works well for descriptive phrases
       such as "nonathletic population" or "young adults".
    2. If every field stays unresolved, fall back to a ±20-word context window
       from the abstract so that implicit cues (e.g. "patients" → "patient") are
       captured.
    3. If still unresolved after the fallback, the field value remains None.

    For each field we pick the *first* label whose entity list is non-empty
    (labels are tried in the order they appear in the spec file).
    """

    # Flatten the spec into a {label: description} dict and track field membership.
    entity_types: dict[str, str] = {}
    label_to_field: dict[str, str] = {}
    for field, labels in spec.items():
        for label, desc in labels.items():
            entity_types[label] = desc
            label_to_field[label] = field

    def _run(text: str) -> dict[str, str | None]:
        """Single GLiNER2 call; returns {field: selected_label | None}."""
        result    = extractor.extract_entities(text, entity_types)
        extracted = result.get("entities", {})

        assignment: dict[str, str | None] = {field: None for field in spec}
        for label, hits in extracted.items():
            if hits:                                    # at least one span found
                field = label_to_field[label]
                if assignment[field] is None:           # keep first match per field
                    assignment[field] = label
        return assignment

    # --- primary pass: entity form -----------------------------------------
    assignment = _run(entity_form)

    # --- fallback: context window ------------------------------------------
    if all(v is None for v in assignment.values()):
        context    = _context_window(abstract_text, position)
        assignment = _run(context)

    return assignment


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    normalized = load_json("normalized_entities.json")
    spec       = load_json("population_specification.json")

    # Lazy-import so the script is importable even without gliner2 installed.
    from gliner2 import GLiNER2
    log.info("Loading GLiNER2 model …")
    extractor = GLiNER2.from_pretrained("fastino/gliner2-large-v1")

    results = []
    total_pop = 0

    for abstract_entry in normalized["abstracts"]:
        abstract_id   = abstract_entry["abstract"]["abstract_id"]
        abstract_text = abstract_entry["abstract"]["text"]

        pop_entities = [
            e for e in abstract_entry["entities"]
            if e["field"] == "Population"
        ]
        total_pop += len(pop_entities)

        for ent in pop_entities:
            log.debug("  classifying '%s' (abs=%s, eid=%s)",
                      ent["entity_form"], abstract_id, ent["entity_id"])

            classification = classify_entity(
                extractor,
                entity_form   = ent["entity_form"],
                abstract_text = abstract_text,
                position      = ent["mention"]["position"],
                spec          = spec,
            )

            classification_list = [v for v in classification.values() if v is not None]

            results.append({
                "abstract_id":       abstract_id,
                "entity_id":         ent["entity_id"],
                "entity_form":       ent["entity_form"],
                "classification":    classification,
                "classification_list": classification_list,
                "mention":           ent["mention"],
            })

    log.info("Processed %d population entities across %d abstracts.",
             total_pop, len(normalized["abstracts"]))
    save_json(results, "population_classified.json")


if __name__ == "__main__":
    main()
