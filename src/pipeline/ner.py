"""Consolidated NER pipeline for the flat-feet literature workspace.

This module centralizes the NER-related logic that was previously spread across
`entity_extraction.ipynb` and `task1_classify_population.py`.

The pipeline is intentionally exposed through the three user-requested stage
functions:

1. `entityRecoganization`
2. `entityNormalization`
3. `entityRecoganizationFineGrained`

The original notebook and task scripts are left untouched. This file is the
organized, reusable module version of that logic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# Keep logging simple so the module is usable both from the CLI and imports.
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# Resolve all important project paths relative to this file so the module works
# regardless of the current shell location.
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # Go from src/pipeline to project root
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"

DEFAULT_ARTICLES_CANDIDATES = (
    DATA_RAW_DIR / "Flatfeet_clean.csv",
    BASE_DIR / "Flatfeet_clean.csv",
)

# NER pipeline stages
RAW_ENTITIES_PATH = DATA_PROCESSED_DIR / "evaluation" / "baselines" / "entities_pretrainedmodel.json"
NORMALIZED_ENTITIES_PATH = DATA_PROCESSED_DIR / "01_ner_normalized" / "normalized_entities.json"
ROUND_TWO_ENTITIES_PATH = DATA_PROCESSED_DIR / "evaluation" / "baselines" / "entities_2nd_pretrainedmodel.json"
ENTITY_SPECIFICATION_PATH = DATA_PROCESSED_DIR / "evaluation" / "baselines" / "entity_specification.json"
POPULATION_CLASSIFIED_PATH = DATA_PROCESSED_DIR / "02_entity_enrichment" / "population_classified.json"
POPULATION_SPECIFICATION_PATH = DATA_PROCESSED_DIR / "02_entity_enrichment" / "population_specification.json"


# These schemas are copied from the notebook so the extracted JSON structure
# remains compatible with the existing downstream scripts.
ROUND_ONE_SCHEMA = {
    "Anatomical Entity": "Any structure within a biological organism.",
    "Symptom": "Medical symptoms, conditions, and clinically relevant complaints.",
    "Terms of Body Movements": "Movements, biomechanical measures, postures, alignments, and motion phrases like 'ankle dorsiflexion' or 'hip internal rotation'.",
    "Population": "The population of interest, including age, sex, sample size, clinical status, and study groups.",
    "Measurement": "Specific measurement tools, experimental methods, statistical values, angles, or scores used to draw conclusions. Examples: Clarke's Angle, Meary's angle, p < 0.05, forest plots."
}

ROUND_TWO_SCHEMAS = {
    "Symptom": {
        "anatomical structure": "Where is the symptom located?",
        "symptom": "What is the symptom, diagnosis, or condition?",
    },
    "Terms of Body Movements": {
        "anatomical structure": "Which anatomical structure is involved in the movement?",
        "movement": (
            "What is the movement, posture, alignment, or biomechanical measure?"
        ),
    },
    "Population": {
        "age": "Age, age range, or developmental stage.",
        "level": "Athelete level, activity level, fitness level, or non-athlete.",
        "clinical status": "Clinical condition, case/control status, or disease status.",
        "sex": "Sex or gender.",
        "study group": "Named cohort, subgroup, or study arm.",
    },
}


# The normalization maps below are copied from the notebook so entity IDs and
# canonical forms stay aligned with the existing project outputs.
ABBREVIATION_MAP = {
    "AP": "anterior-posterior",
    "BF": "biceps femoris",
    "BFFG": "bilateral flexible flat feet group",
    "KF": "knee flexion",
    "LBP": "low back pain",
    "MG": "medial gastrocnemius",
    "MLBP": "mechanical low back pain",
    "MTSS": "medial tibial stress syndrome",
    "NFG": "normal foot group",
    "OA": "osteoarthritis",
    "OFFG": "one-sided flexible flat feet group",
    "PF": "plantar fasciitis",
    "PFP": "patellofemoral pain",
    "PFPS": "patellofemoral pain syndrome",
    "RF": "rectus femoris",
    "TA": "tibialis anterior",
}

IRREGULAR_SINGULARS = {
    "feet": "foot",
    "teeth": "tooth",
    "men": "man",
    "women": "woman",
    "children": "child",
    "people": "person",
    "indices": "index",
}

SINGULAR_OVERRIDES = {
    "alignments": "alignment",
    "analyses": "analysis",
    "angles": "angle",
    "areas": "area",
    "changes": "change",
    "controls": "control",
    "deformations": "deformation",
    "disorders": "disorder",
    "extremities": "extremity",
    "factors": "factor",
    "flexions": "flexion",
    "grades": "grade",
    "imbalances": "imbalance",
    "individuals": "individual",
    "joints": "joint",
    "knees": "knee",
    "limbs": "limb",
    "measures": "measure",
    "motions": "motion",
    "muscles": "muscle",
    "parameters": "parameter",
    "patients": "patient",
    "pathologies": "pathology",
    "planes": "plane",
    "postures": "posture",
    "pressures": "pressure",
    "referees": "referee",
    "runners": "runner",
    "scores": "score",
    "subjects": "subject",
    "symptoms": "symptom",
    "torsions": "torsion",
    "volunteers": "volunteer",
}

TOKEN_SPLIT_PATTERN = re.compile(r"([\s/(),]+)")


# ---------------------------------------------------------------------------
# Generic I/O helpers
# ---------------------------------------------------------------------------

def ensure_intermediate_dir(directory: Path = DATA_PROCESSED_DIR) -> Path:
    """Create the processed data directory on demand."""
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def resolve_articles_csv_path(path: str | Path | None = None) -> Path:
    """Resolve the source CSV path, preferring the `Data/` location."""
    if path is not None:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = BASE_DIR / candidate
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Articles CSV not found: {candidate}")

    for candidate in DEFAULT_ARTICLES_CANDIDATES:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find Flatfeet_clean.csv in either "
        f"{DEFAULT_ARTICLES_CANDIDATES[0]} or {DEFAULT_ARTICLES_CANDIDATES[1]}"
    )


def load_json(path: str | Path) -> Any:
    """Load a JSON file using UTF-8 encoding."""
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def save_json(data: Any, path: str | Path) -> None:
    """Write JSON with stable formatting and create parent directories if needed."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=4, ensure_ascii=False)
    log.info("Saved → %s", target)


def load_abstracts_from_csv(path: str | Path | None = None) -> dict[int, str]:
    """Load abstracts using the notebook's pandas workflow.

    The loader mirrors `entity_extraction.ipynb` by reading the CSV with pandas,
    selecting the expected article columns, creating sequential article ids
    starting at 1, and returning a notebook-compatible `{id: abstract}` mapping.
    """
    csv_path = resolve_articles_csv_path(path)
    columns = [
        "Title",
        "Author",
        "Publication Year",
        "Abstract Note",
        "Journal Abbreviation",
    ]

    articles = pd.read_csv(csv_path)
    articles = articles[columns].copy()
    articles["id"] = list(range(1, len(articles) + 1))

    abstracts = dict(zip(articles["id"], articles["Abstract Note"]))

    log.info("Loaded %d abstracts from %s", len(abstracts), csv_path)
    return abstracts


def load_gliner_model():
    """Lazy-load GLiNER2 so importing this module does not require the package."""
    try:
        from gliner2 import GLiNER2
    except ImportError as exc:
        raise ImportError(
            "gliner2 is required to run the NER extraction stages. "
            "Install it in the current environment before running NER.py."
        ) from exc

    log.info("Loading GLiNER2 model …")
    return GLiNER2.from_pretrained("fastino/gliner2-large-v1")


# ---------------------------------------------------------------------------
# Stage 1 helpers: round-one entity extraction
# ---------------------------------------------------------------------------

def extract_round_one_entities(
    extractor: Any,
    text: str,
    schema: dict[str, str] | None = None,
) -> dict[str, list[str]]:
    """Run the first-pass extractor on a single abstract."""
    active_schema = schema or ROUND_ONE_SCHEMA
    return extractor.extract_entities(text, active_schema).get("entities", {})


def entityRecoganization(
    articles_path: str | Path | None = None,
    output_path: str | Path = RAW_ENTITIES_PATH,
    extractor: Any | None = None,
    schema: dict[str, str] | None = None,
) -> dict[str, dict[str, list[str]]]:
    """Run round-one entity extraction and write `entities_pretrainedmodel.json`.
    """
    ensure_intermediate_dir()
    abstracts = load_abstracts_from_csv(articles_path)
    active_extractor = extractor or load_gliner_model()

    all_results: dict[str, dict[str, list[str]]] = {}
    for abstract_id, text in abstracts.items():
        if text is None or not str(text).strip():
            log.info("[%s] skipped empty abstract", abstract_id)
            all_results[str(abstract_id)] = {}
            continue

        all_results[str(abstract_id)] = extract_round_one_entities(
            active_extractor,
            str(text),
            schema=schema,
        )

    save_json(all_results, output_path)
    return all_results


# ---------------------------------------------------------------------------
# Stage 2 helpers: entity normalization
# ---------------------------------------------------------------------------

def _tokenize_with_spans(text: str) -> list[re.Match[str]]:
    """Tokenize text into non-whitespace spans so token positions can be tracked."""
    return list(re.finditer(r"\S+", text))


def _char_to_start_token(
    token_matches: list[re.Match[str]],
    start_char: int,
) -> int | None:
    """Map a character offset back to the token index that starts there."""
    for token_index, token in enumerate(token_matches):
        if token.start() <= start_char < token.end():
            return token_index
    return None


def _collect_match_positions(
    text: str,
    token_matches: list[re.Match[str]],
    entity_text: str,
    ignore_case: bool = False,
) -> list[int]:
    """Collect all token-start positions for an entity mention inside an abstract."""
    flags = re.IGNORECASE if ignore_case else 0
    positions: list[int] = []

    for match in re.finditer(re.escape(entity_text), text, flags=flags):
        start_token = _char_to_start_token(token_matches, match.start())
        if start_token is not None:
            positions.append(start_token)

    return positions


def _is_abbreviation(text: str) -> bool:
    """Detect all-uppercase abbreviations while ignoring punctuation."""
    letters = [char for char in text if char.isalpha()]
    return bool(letters) and all(char.isupper() for char in letters)


def _singularize_word(word: str) -> str:
    """Apply the notebook's singularization rules to a single token."""
    if not word or _is_abbreviation(word):
        return word

    lower_word = word.lower()
    if lower_word in IRREGULAR_SINGULARS:
        return IRREGULAR_SINGULARS[lower_word]
    if lower_word in SINGULAR_OVERRIDES:
        return SINGULAR_OVERRIDES[lower_word]
    if len(lower_word) <= 3:
        return lower_word
    if lower_word.endswith("ies") and len(lower_word) > 4:
        return lower_word[:-3] + "y"
    if lower_word.endswith("sses") or lower_word.endswith("xes") or lower_word.endswith("zes"):
        return lower_word[:-2]
    if lower_word.endswith("s") and not lower_word.endswith("ss"):
        return lower_word[:-1]
    return lower_word


def _expand_abbreviation_token(token: str) -> str:
    """Expand uppercase abbreviation tokens before canonical normalization."""
    if not token or TOKEN_SPLIT_PATTERN.fullmatch(token):
        return token
    if _is_abbreviation(token):
        return ABBREVIATION_MAP.get(token, token.lower())
    return token.lower()


def _normalize_token(token: str) -> str:
    """Normalize a token while preserving separator tokens."""
    if not token or TOKEN_SPLIT_PATTERN.fullmatch(token):
        return token
    return _singularize_word(token)


def _normalize_hyphen_spacing(text: str) -> str:
    """Replace semantic hyphens with spaces but preserve numeric ranges."""
    return re.sub(r"(?<!\d)-(?!\d)|(?<!\d)-(?=\d)|(?<=\d)-(?!\d)", " ", text)


def _normalize_canonical_form(text: str, field: str) -> str | None:
    """Project a raw entity mention into the canonical form used by the project."""
    if not text or not text.strip():
        return None

    normalized = _normalize_hyphen_spacing(text.strip())
    normalized = "".join(
        _expand_abbreviation_token(part) for part in TOKEN_SPLIT_PATTERN.split(normalized)
    )
    normalized = "".join(
        _normalize_token(part) for part in TOKEN_SPLIT_PATTERN.split(normalized)
    )
    normalized = re.sub(r"\s+", " ", normalized).strip(" -/(),")

    # The anatomical rules below come directly from the notebook and remove
    # low-value generic or side-specific forms that would fragment the graph.
    if field == "Anatomical Entity":
        if normalized == "joint":
            return None
        if normalized.endswith(" joint"):
            normalized = normalized[:-6].strip()
            if not normalized:
                return None
        if normalized.startswith("left "):
            normalized = normalized[5:].strip()
        elif normalized.startswith("right "):
            normalized = normalized[6:].strip()
        if not normalized:
            return None

    return normalized or None


def build_canonical_entities(
    abstracts_dict: dict[int, str],
    entity_dict: dict[str, dict[str, list[str]]] | dict[int, dict[str, list[str]]],
) -> list[dict[str, Any]]:
    """Build raw canonical records grouped by field and original surface form."""
    canonical_records: list[dict[str, Any]] = []
    canonical_index: dict[tuple[str, str], int] = {}

    for abstract_id in sorted(abstracts_dict):
        text = "" if abstracts_dict[abstract_id] is None else str(abstracts_dict[abstract_id])
        abstract_entities = entity_dict.get(str(abstract_id), entity_dict.get(abstract_id, {}))
        token_matches = _tokenize_with_spans(text)

        # Cache exact and fallback matches once per abstract to avoid repeated scans.
        exact_match_cache: dict[str, list[int]] = {}
        ignorecase_match_cache: dict[str, list[int]] = {}

        for field, entities in abstract_entities.items():
            seen_entity_texts: set[str] = set()
            for entity_text in entities:
                if not isinstance(entity_text, str) or not entity_text.strip():
                    continue

                entity_text = entity_text.strip()
                if entity_text in seen_entity_texts:
                    continue
                seen_entity_texts.add(entity_text)

                if entity_text not in exact_match_cache:
                    exact_match_cache[entity_text] = _collect_match_positions(
                        text,
                        token_matches,
                        entity_text,
                    )
                match_positions = exact_match_cache[entity_text]

                if match_positions:
                    positions = match_positions
                else:
                    fallback_key = entity_text.casefold()
                    if fallback_key not in ignorecase_match_cache:
                        ignorecase_match_cache[fallback_key] = _collect_match_positions(
                            text,
                            token_matches,
                            entity_text,
                            ignore_case=True,
                        )
                    positions = ignorecase_match_cache[fallback_key]

                # Preserve the notebook's behavior: keep unmatched mentions so
                # they still remain visible in downstream artifacts.
                if not positions:
                    positions = [None]

                canonical_key = (field, entity_text)
                if canonical_key not in canonical_index:
                    canonical_index[canonical_key] = len(canonical_records)
                    canonical_records.append(
                        {
                            "canonical_id": f"ent_{len(canonical_records) + 1:03d}",
                            "canonical_form": entity_text,
                            "field": field,
                            "variants": [entity_text],
                            "occurrences": [],
                        }
                    )

                for position in positions:
                    canonical_records[canonical_index[canonical_key]]["occurrences"].append(
                        {
                            "abstract_id": f"abs_{int(abstract_id):03d}",
                            "original_text": entity_text,
                            "position": position,
                        }
                    )

    return canonical_records


def normalize_canonical_entities(
    canonical_entities: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge raw canonical records into normalized entity forms."""
    merged_records: dict[tuple[str, str], dict[str, Any]] = {}

    for record in canonical_entities:
        normalized_form = _normalize_canonical_form(record["canonical_form"], record["field"])
        if normalized_form is None:
            continue

        merge_key = (record["field"], normalized_form)
        if merge_key not in merged_records:
            merged_records[merge_key] = {
                "canonical_form": normalized_form,
                "field": record["field"],
                "variants": [],
                "occurrences": [],
            }

        merged_record = merged_records[merge_key]
        for variant in record.get("variants", []):
            if variant not in merged_record["variants"]:
                merged_record["variants"].append(variant)

        merged_record["occurrences"].extend(record.get("occurrences", []))

    normalized_records: list[dict[str, Any]] = []
    for index, merged_record in enumerate(
        sorted(
            merged_records.values(),
            key=lambda item: (item["field"], item["canonical_form"]),
        ),
        start=1,
    ):
        normalized_records.append(
            {
                "canonical_id": f"ent_{index:03d}",
                "canonical_form": merged_record["canonical_form"],
                "field": merged_record["field"],
                "variants": merged_record["variants"],
                "occurrences": merged_record["occurrences"],
            }
        )

    return normalized_records


def build_abstract_aggregated_entities(
    canonical_entities: list[dict[str, Any]],
    abstracts_dict: dict[int, str],
) -> list[dict[str, Any]]:
    """Project normalized entities into the abstract-centric JSON structure."""
    abstract_records = {
        f"abs_{int(abstract_id):03d}": {
            "abstract": {
                "abstract_id": f"abs_{int(abstract_id):03d}",
                "text": "" if abstract_text is None else str(abstract_text),
            },
            "entities": [],
            "output_format": {
                "triples": [],
            },
        }
        for abstract_id, abstract_text in abstracts_dict.items()
    }

    for entity in canonical_entities:
        for occurrence in entity.get("occurrences", []):
            abstract_id = occurrence["abstract_id"]
            abstract_records[abstract_id]["entities"].append(
                {
                    "entity_id": entity["canonical_id"],
                    "entity_form": entity["canonical_form"],
                    "field": entity["field"],
                    "mention": {
                        "original_text": occurrence["original_text"],
                        "position": occurrence["position"],
                    },
                }
            )

    # Sorting keeps the JSON deterministic and easier to compare during debugging.
    for record in abstract_records.values():
        record["entities"].sort(
            key=lambda item: (
                float("inf") if item["mention"]["position"] is None else item["mention"]["position"],
                item["entity_id"],
            )
        )

    return list(abstract_records.values())


def build_normalized_abstract_entities(
    canonical_entities_raw: list[dict[str, Any]],
    abstracts_dict: dict[int, str],
) -> list[dict[str, Any]]:
    """Convenience wrapper that normalizes and reprojects canonical records."""
    normalized_entities = normalize_canonical_entities(canonical_entities_raw)
    return build_abstract_aggregated_entities(normalized_entities, abstracts_dict)


def entityNormalization(
    articles_path: str | Path | None = None,
    extracted_entities: dict[str, dict[str, list[str]]] | None = None,
    extracted_entities_path: str | Path = RAW_ENTITIES_PATH,
    output_path: str | Path = NORMALIZED_ENTITIES_PATH,
) -> dict[str, list[dict[str, Any]]]:
    """Normalize round-one entities and write `normalized_entities.json`."""
    ensure_intermediate_dir()
    abstracts = load_abstracts_from_csv(articles_path)
    raw_entities = extracted_entities or load_json(extracted_entities_path)

    canonical_entities_raw = build_canonical_entities(abstracts, raw_entities)
    abstract_aggregated_entities = build_normalized_abstract_entities(
        canonical_entities_raw,
        abstracts,
    )

    payload = {"abstracts": abstract_aggregated_entities}
    save_json(payload, output_path)
    return payload


# ---------------------------------------------------------------------------
# Stage 3 helpers: fine-grained NER
# ---------------------------------------------------------------------------

def build_round_two_lookup(
    abstract_aggregated_entities: list[dict[str, Any]],
    extractor: Any,
    round_two_schemas: dict[str, dict[str, str]] | None = None,
) -> dict[str, dict[str, dict[str, list[str]]]]:
    """Run round-two attribute extraction on unique normalized entity forms."""
    schemas = round_two_schemas or ROUND_TWO_SCHEMAS
    round_two_lookup: dict[str, dict[str, dict[str, list[str]]]] = {
        field: {} for field in schemas
    }

    for record in abstract_aggregated_entities:
        for entity in record["entities"]:
            field = entity["field"]
            entity_form = entity["entity_form"]
            if field not in schemas:
                continue
            if entity_form not in round_two_lookup[field]:
                round_two_lookup[field][entity_form] = extractor.extract_entities(
                    entity_form,
                    schemas[field],
                ).get("entities", {})

    # Drop empty top-level groups so the JSON matches the notebook output shape.
    return {field: lookup for field, lookup in round_two_lookup.items() if lookup}


def normalize_eval_phrase(text: str | None, field: str) -> str | None:
    """Notebook-compatible normalization helper retained for reuse."""
    if text is None or not str(text).strip():
        return None
    return _normalize_canonical_form(str(text), field)


def normalize_anatomical_location(text: str | None) -> str | None:
    """Normalize candidate anatomy strings into the anatomical canonical space."""
    if text is None or not str(text).strip():
        return None
    return _normalize_canonical_form(str(text), "Anatomical Entity")


def hash_feature(feature: str, dim: int) -> int:
    """Hash a sparse text feature into a fixed feature space."""
    digest = hashlib.md5(feature.encode("utf-8")).hexdigest()
    return int(digest, 16) % dim


def build_text_embedding(text: str, dim: int = 512) -> np.ndarray:
    """Build a hashed text embedding using tokens and character trigrams."""
    normalized_text = re.sub(r"\s+", " ", text.lower()).strip()
    vector = np.zeros(dim, dtype=float)

    if not normalized_text:
        return vector

    for token in normalized_text.split():
        vector[hash_feature(f"tok:{token}", dim)] += 1.0

    padded = f"  {normalized_text} "
    for index in range(len(padded) - 2):
        trigram = padded[index:index + 3]
        vector[hash_feature(f"tri:{trigram}", dim)] += 1.0

    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector

    return vector / norm


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Compute cosine similarity for two dense normalized vectors."""
    if left.size == 0 or right.size == 0:
        return 0.0
    return float(np.dot(left, right))


def rank_anatomical_candidates(
    query_text: str,
    candidate_forms: list[str],
    top_k: int = 1,
    dim: int = 512,
) -> list[str]:
    """Rank candidate anatomy forms using the notebook's hashed-text baseline."""
    if not query_text or not candidate_forms:
        return []

    query_embedding = build_text_embedding(query_text, dim=dim)
    scored_candidates: list[tuple[float, str]] = []
    for candidate in candidate_forms:
        candidate_embedding = build_text_embedding(candidate, dim=dim)
        score = cosine_similarity(query_embedding, candidate_embedding)
        scored_candidates.append((score, candidate))

    scored_candidates.sort(key=lambda item: (-item[0], item[1]))
    return [candidate for _, candidate in scored_candidates[:top_k]]


def build_abstract_anatomy_lookup(
    abstract_entity_records: list[dict[str, Any]],
) -> dict[str, list[str]]:
    """Build `{abstract_id: [normalized anatomy forms...]}` for fine-grained linking."""
    abstract_anatomy_lookup: dict[str, list[str]] = {}

    for record in abstract_entity_records:
        abstract_id = record["abstract"]["abstract_id"]
        anatomy_forms: list[str] = []

        for entity in record["entities"]:
            if entity["field"] != "Anatomical Entity":
                continue

            anatomy_form = normalize_anatomical_location(entity["entity_form"])
            if anatomy_form and anatomy_form not in anatomy_forms:
                anatomy_forms.append(anatomy_form)

        abstract_anatomy_lookup[abstract_id] = anatomy_forms

    return abstract_anatomy_lookup


def build_entity_specification_cases(
    abstract_entity_records: list[dict[str, Any]],
    top_k: int = 1,
    embedding_dim: int = 512,
) -> list[dict[str, Any]]:
    """Build the anatomy-linking benchmark cases stored in `entity_specification.json`."""
    abstract_anatomy_lookup = build_abstract_anatomy_lookup(abstract_entity_records)
    method2_cases: list[dict[str, Any]] = []

    for record in abstract_entity_records:
        abstract_id = record["abstract"]["abstract_id"]
        candidate_forms = abstract_anatomy_lookup.get(abstract_id, [])

        for entity in record["entities"]:
            if entity["field"] not in {"Symptom", "Terms of Body Movements"}:
                continue

            predictions = rank_anatomical_candidates(
                entity["entity_form"],
                candidate_forms,
                top_k=top_k,
                dim=embedding_dim,
            )

            method2_cases.append(
                {
                    "abstract_id": abstract_id,
                    "entity_form": entity["entity_form"],
                    "field": entity["field"],
                    "mention": entity["mention"],
                    "candidate_anatomies": candidate_forms,
                    "predicted_locations": predictions,
                }
            )

    return method2_cases


def entityRecoganizationFineGrained(
    normalized_data: dict[str, list[dict[str, Any]]] | None = None,
    normalized_path: str | Path = NORMALIZED_ENTITIES_PATH,
    round_two_output_path: str | Path = ROUND_TWO_ENTITIES_PATH,
    entity_specification_output_path: str | Path = ENTITY_SPECIFICATION_PATH,
    extractor: Any | None = None,
    round_two_schemas: dict[str, dict[str, str]] | None = None,
    top_k: int = 1,
    embedding_dim: int = 512,
) -> dict[str, Any]:
    """Run the fine-grained NER stage and write both downstream JSON artifacts."""
    ensure_intermediate_dir()
    normalized_payload = normalized_data or load_json(normalized_path)
    abstract_aggregated_entities = normalized_payload["abstracts"]
    active_extractor = extractor or load_gliner_model()

    round_two_lookup = build_round_two_lookup(
        abstract_aggregated_entities,
        active_extractor,
        round_two_schemas=round_two_schemas,
    )

    round_two_payload = {
        "round2_lookup": round_two_lookup,
        "abstracts": abstract_aggregated_entities,
    }
    save_json(round_two_payload, round_two_output_path)

    entity_specification = build_entity_specification_cases(
        abstract_aggregated_entities,
        top_k=top_k,
        embedding_dim=embedding_dim,
    )
    save_json(entity_specification, entity_specification_output_path)

    return {
        "round2_lookup": round_two_lookup,
        "abstracts": abstract_aggregated_entities,
        "entity_specification": entity_specification,
    }


# ---------------------------------------------------------------------------
# Population classification helpers copied from task1_classify_population.py
# ---------------------------------------------------------------------------

def _context_window(abstract_text: str, position: int | None, window: int = 20) -> str:
    """Return a ±window-word slice around a population mention."""
    words = abstract_text.split()
    if position is None:
        return " ".join(words[: window * 2])

    start = max(0, position - window)
    end = min(len(words), position + window)
    return " ".join(words[start:end])


def classify_entity(
    extractor: Any,
    entity_form: str,
    abstract_text: str,
    position: int | None,
    spec: dict[str, dict[str, str]],
) -> dict[str, str | None]:
    """Classify a population mention using the same two-pass strategy as task 1."""
    entity_types: dict[str, str] = {}
    label_to_field: dict[str, str] = {}

    # Flatten the hierarchical spec so GLiNER2 can score every label in one pass.
    for field, labels in spec.items():
        for label, description in labels.items():
            entity_types[label] = description
            label_to_field[label] = field

    def _run(text: str) -> dict[str, str | None]:
        """Run one GLiNER2 call and keep the first non-empty label per field."""
        result = extractor.extract_entities(text, entity_types)
        extracted = result.get("entities", {})

        assignment: dict[str, str | None] = {field: None for field in spec}
        for label, hits in extracted.items():
            if not hits:
                continue
            field = label_to_field[label]
            if assignment[field] is None:
                assignment[field] = label
        return assignment

    assignment = _run(entity_form)
    if all(value is None for value in assignment.values()):
        assignment = _run(_context_window(abstract_text, position))

    return assignment


def classify_population_entities(
    normalized_data: dict[str, list[dict[str, Any]]] | None = None,
    normalized_path: str | Path = NORMALIZED_ENTITIES_PATH,
    population_spec_path: str | Path = POPULATION_SPECIFICATION_PATH,
    output_path: str | Path = POPULATION_CLASSIFIED_PATH,
    extractor: Any | None = None,
) -> list[dict[str, Any]]:
    """Classify normalized population entities and write `population_classified.json`."""
    ensure_intermediate_dir()
    normalized_payload = normalized_data or load_json(normalized_path)
    population_spec = load_json(population_spec_path)
    active_extractor = extractor or load_gliner_model()

    results: list[dict[str, Any]] = []
    total_pop = 0

    for abstract_entry in normalized_payload["abstracts"]:
        abstract_id = abstract_entry["abstract"]["abstract_id"]
        abstract_text = abstract_entry["abstract"]["text"]

        population_entities = [
            entity for entity in abstract_entry["entities"] if entity["field"] == "Population"
        ]
        total_pop += len(population_entities)

        for entity in population_entities:
            classification = classify_entity(
                active_extractor,
                entity_form=entity["entity_form"],
                abstract_text=abstract_text,
                position=entity["mention"]["position"],
                spec=population_spec,
            )

            results.append(
                {
                    "abstract_id": abstract_id,
                    "entity_id": entity["entity_id"],
                    "entity_form": entity["entity_form"],
                    "classification": classification,
                    "classification_list": [
                        value for value in classification.values() if value is not None
                    ],
                    "mention": entity["mention"],
                }
            )

    log.info(
        "Processed %d population entities across %d abstracts.",
        total_pop,
        len(normalized_payload["abstracts"]),
    )
    save_json(results, output_path)
    return results


# ---------------------------------------------------------------------------
# High-level orchestration
# ---------------------------------------------------------------------------

def run_pipeline(
    articles_path: str | Path | None = None,
    classify_population: bool = False,
) -> dict[str, Any]:
    """Run the consolidated NER workflow end-to-end."""
    extractor = load_gliner_model()
    round_one = entityRecoganization(
        articles_path=articles_path,
        extractor=extractor,
    )
    normalized = entityNormalization(
        articles_path=articles_path,
        extracted_entities=round_one,
    )
    fine_grained = entityRecoganizationFineGrained(
        normalized_data=normalized,
        extractor=extractor,
    )

    result = {
        "entities_pretrainedmodel": round_one,
        "normalized_entities": normalized,
        "fine_grained": fine_grained,
    }

    if classify_population:
        result["population_classified"] = classify_population_entities(
            normalized_data=normalized,
            extractor=extractor,
        )

    return result


def parse_args() -> argparse.Namespace:
    """Parse CLI flags for the module entrypoint."""
    parser = argparse.ArgumentParser(description="Run the consolidated NER pipeline.")
    parser.add_argument(
        "--articles",
        default=None,
        help="Optional path to Flatfeet_clean.csv. Defaults to Data/Flatfeet_clean.csv.",
    )
    parser.add_argument(
        "--classify-population",
        action="store_true",
        help="Also run the population-classification stage from task1_classify_population.py.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for local execution."""
    args = parse_args()
    run_pipeline(
        articles_path=args.articles,
        classify_population=args.classify_population,
    )


if __name__ == "__main__":
    main()
