"""
relation_extraction.py
======================
Open-schema relation extraction from normalized_entities.json using an LLM.

Unlike the original relations_GPT5.json (fixed 6-relation schema), this script
extracts relation phrases VERBATIM — no forced mapping to predefined types.
The model is only constrained to use entity_forms from the input entity list
as subject and object.

Input
-----
  data/processed/01_ner_normalized/normalized_entities.json

Output
------
  data/processed/03_relations_extracted/relations_open.json

Output format (keyed by abstract_id, same envelope as relations_GPT5.json):
  {
    "abs_001": [
      {
        "subject":  "flat foot",           ← must be an entity_form from that abstract
        "relation": "is a risk factor for", ← verbatim phrase, no schema constraint
        "object":   "anterior knee pain",  ← must be an entity_form from that abstract
        "evidence": "flat foot is a risk factor for anterior knee pain"
      },
      ...
    ],
    ...
  }

Usage
-----
  python -m src.pipeline.relation_extraction
  python -m src.pipeline.relation_extraction --input path/to/normalized_entities.json
  python -m src.pipeline.relation_extraction --output path/to/relations_open.json
  python -m src.pipeline.relation_extraction --dry-run     # print prompts, don't call API
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import anthropic

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # src/pipeline → project root
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
NER_NORMALIZED_DIR = DATA_PROCESSED_DIR / "01_ner_normalized"
RELATIONS_EXTRACTED_DIR = DATA_PROCESSED_DIR / "03_relations_extracted"

INPUT_PATH = NER_NORMALIZED_DIR / "normalized_entities.json"
OUTPUT_PATH = RELATIONS_EXTRACTED_DIR / "relations_open.json"

# ─────────────────────────────────────────────────────────────────────────────
# PROMPT
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a biomedical relation extraction specialist.
Your task is to identify semantic relationships between named entities in clinical abstracts.

RULES:
- Subject and object MUST be taken verbatim from the provided entity list.
- Extract the relation as a short natural-language phrase (e.g. "is a risk factor for",
  "is located in", "was measured by", "correlates with", "increases the risk of").
- Do NOT force relations into a fixed schema — use whatever phrase best describes
  the relationship as stated in the text.
- Only extract relations that are explicitly stated or strongly implied in the text.
- Do not hallucinate relations not present in the text.
- If no relation exists between any pair, return an empty array.

OUTPUT FORMAT (strict JSON — no markdown, no preamble):
[
  {"subject": "entity A", "relation": "phrase", "object": "entity B", "evidence": "snippet"}
]"""

def build_user_prompt(abstract_text: str, entity_forms: list[str]) -> str:
    entities_str = "\n".join(f"  - {e}" for e in entity_forms)
    return f"""Abstract:
{abstract_text}

Entities:
{entities_str}

Extract all relations between these entities."""


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    log.info("Saved → %s", path)


def extract_abstract_records(normalized: dict) -> list[dict]:
    """Return list of {abstract_id, text, entity_forms} dicts."""
    records = []
    for entry in normalized["abstracts"]:
        abstract_id = entry["abstract"]["abstract_id"]
        text        = entry["abstract"].get("text", "").strip()
        if not text:
            continue
        # Deduplicate entity_forms, preserve order
        seen   = set()
        forms  = []
        for ent in entry.get("entities", []):
            f = ent.get("entity_form", "").strip()
            if f and f not in seen:
                seen.add(f)
                forms.append(f)
        records.append({
            "abstract_id":   abstract_id,
            "text":          text,
            "entity_forms":  forms,
        })
    return records


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def parse_triples(raw: str, valid_forms: set[str]) -> list[dict]:
    """
    Parse the model's JSON response and filter out triples whose subject
    or object are not in the abstract's entity list.
    """
    raw = raw.strip()
    # Strip accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        triples = json.loads(raw)
    except json.JSONDecodeError as e:
        log.warning("JSON parse error: %s — raw: %s", e, raw[:200])
        return []

    if not isinstance(triples, list):
        log.warning("Expected list, got %s", type(triples))
        return []

    valid = []
    for t in triples:
        subj = str(t.get("subject", "")).strip()
        rel  = str(t.get("relation", "")).strip()
        obj  = str(t.get("object",  "")).strip()
        ev   = str(t.get("evidence","")).strip()

        if not subj or not rel or not obj:
            continue
        # Enforce that subject and object are known entity forms
        # Use case-insensitive match to be lenient
        valid_lower = {f.lower() for f in valid_forms}
        if subj.lower() not in valid_lower:
            log.debug("Skipping triple — subject not in entity list: '%s'", subj)
            continue
        if obj.lower() not in valid_lower:
            log.debug("Skipping triple — object not in entity list: '%s'", obj)
            continue
        if subj.lower() == obj.lower():
            continue

        valid.append({
            "subject":  subj,
            "relation": rel,
            "object":   obj,
            "evidence": ev,
        })

    return valid


def extract_relations_for_abstract(
    client: anthropic.Anthropic,
    abstract_id: str,
    text: str,
    entity_forms: list[str],
    dry_run: bool = False,
) -> list[dict]:
    """Call the LLM for one abstract and return parsed triples."""
    if not entity_forms:
        log.info("[%s] skipping — no entities", abstract_id)
        return []

    prompt = build_user_prompt(text, entity_forms)

    if dry_run:
        print(f"\n{'─'*60}")
        print(f"ABSTRACT: {abstract_id}")
        print(f"ENTITIES: {entity_forms}")
        print(f"PROMPT:\n{prompt}")
        return []

    try:
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text
        log.info("[%s] received %d chars", abstract_id, len(raw))

        triples = parse_triples(raw, set(entity_forms))
        log.info("[%s] extracted %d valid triples", abstract_id, len(triples))
        return triples

    except Exception as e:
        log.error("[%s] API error: %s", abstract_id, e)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Open-schema relation extraction.")
    parser.add_argument("--input",   default=str(INPUT_PATH),  help="Path to normalized_entities.json")
    parser.add_argument("--output",  default=str(OUTPUT_PATH), help="Path to write relations_open.json")
    parser.add_argument("--dry-run", action="store_true",      help="Print prompts without calling API")
    parser.add_argument("--delay",   type=float, default=0.5,  help="Seconds between API calls")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    log.info("Loading %s ...", input_path)
    normalized = load_json(input_path)
    records    = extract_abstract_records(normalized)
    log.info("Found %d abstracts to process", len(records))

    client = None if args.dry_run else anthropic.Anthropic()

    results: dict[str, list[dict]] = {}

    for i, rec in enumerate(records, 1):
        abs_id = rec["abstract_id"]
        log.info("[%d/%d] Processing %s (%d entities)",
                 i, len(records), abs_id, len(rec["entity_forms"]))

        triples = extract_relations_for_abstract(
            client,
            abs_id,
            rec["text"],
            rec["entity_forms"],
            dry_run=args.dry_run,
        )
        results[abs_id] = triples

        if not args.dry_run and i < len(records):
            time.sleep(args.delay)

    if not args.dry_run:
        save_json(results, output_path)
        total = sum(len(v) for v in results.values())
        log.info("Done. %d total triples across %d abstracts → %s",
                 total, len(results), output_path)


if __name__ == "__main__":
    main()
