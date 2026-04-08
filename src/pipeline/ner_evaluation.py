"""NER evaluation utilities for round-one and round-two experiments.

This module reorganizes the notebook-based evaluation code into a reusable,
documented Python module.

"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from src.pipeline import ner


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


BASE_DIR = Path(__file__).resolve().parent.parent.parent  # src/pipeline → project root
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
                     

# Baselines and predictions
PRED_ROUND_ONE_PATH = DATA_PROCESSED_DIR / "evaluation" / "baselines" / "entities_pretrainedmodel.json"
GOLD_ROUND_ONE_PATH = DATA_PROCESSED_DIR / "evaluation" / "extraction_gold_standard" / "entities_manual_gold.json"
ROUND_TWO_METHOD1_PATH = DATA_PROCESSED_DIR / "evaluation" / "baselines" / "entities_2nd_pretrainedmodel.json"
ROUND_TWO_METHOD2_PATH = DATA_PROCESSED_DIR / "evaluation" / "baselines" / "entity_specification.json"
GOLD_STANDARD_PATH = DATA_RAW_DIR / "golden_standard.csv"

# Round 1 evaluation outputs
ROUND_ONE_PRED_WITH_MENTIONS_PATH = DATA_PROCESSED_DIR / "evaluation" / "ner_round1" / "entities_pretrainedmodel_with_mentions.json"
ROUND_ONE_GOLD_WITH_MENTIONS_PATH = DATA_PROCESSED_DIR / "evaluation" / "ner_round1" / "entities_gold_with_mentions.json"
ROUND_ONE_METRICS_PATH = DATA_PROCESSED_DIR / "evaluation" / "ner_round1" / "metrics.csv"

# Round 1 visualization data (JSON/CSV for rendering)
OUTPUTS_VIS_DIR = BASE_DIR / "outputs" / "visualizations" / "ner_round1"
ROUND_ONE_COUNTS_PATH = OUTPUTS_VIS_DIR / "count_comparison.csv"
ROUND_ONE_COMPARISON_JSON_PATH = OUTPUTS_VIS_DIR / "comparison.json"

# Round 1 normalized entities (deduplicated forms for visualization)
ROUND_ONE_NORMALIZED_PRED_PATH = OUTPUTS_VIS_DIR / "normalized_pred.json"
ROUND_ONE_NORMALIZED_GOLD_PATH = OUTPUTS_VIS_DIR / "normalized_gold.json"

# Round 2 evaluation outputs
ROUND_TWO_BERT_RESULTS_PATH = DATA_PROCESSED_DIR / "evaluation" / "ner_round2" / "bert_results.json"
ROUND_TWO_EVALUATION_JSON_PATH = DATA_PROCESSED_DIR / "evaluation" / "ner_round2" / "evaluation.json"
ROUND_TWO_EVALUATION_TABLE_PATH = DATA_PROCESSED_DIR / "evaluation" / "ner_round2" / "evaluation.csv"

DEFAULT_BERT_MODEL_SPECS = {
    "biobert": "dmis-lab/biobert-base-cased-v1.1",
    "pubmedbert": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "scibert": "allenai/scibert_scivocab_uncased",
}


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def load_json(path: str | Path) -> Any:
    """Load JSON with UTF-8 decoding."""
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def save_json(data: Any, path: str | Path) -> None:
    """Save JSON and create parent directories when needed."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=4, ensure_ascii=False)
    log.info("Saved → %s", target)


def abstract_key_to_int(raw_key: str | int) -> int:
    """Normalize abstract keys like `1` and `Abstract 1` into a shared integer id."""
    if isinstance(raw_key, int):
        return raw_key

    text = str(raw_key).strip()
    match = re.search(r"(\d+)$", text)
    if not match:
        raise ValueError(f"Could not parse abstract id from key: {raw_key}")
    return int(match.group(1))


def abstract_id_label(index: int) -> str:
    """Convert an integer abstract index into the canonical `abs_###` id."""
    return f"abs_{index:03d}"


def normalize_surface_text(text: str) -> str:
    """Normalize text lightly for evaluation matching without changing semantics."""
    return re.sub(r"\s+", " ", str(text).strip().lower()).replace("’", "'")


def f1_score(precision: float, recall: float) -> float:
    """Compute the harmonic mean when both inputs are non-zero."""
    if precision == 0 or recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Round-one mention enrichment
# ---------------------------------------------------------------------------

def normalize_round_one_pred_entities(raw_data: dict[str, Any]) -> dict[int, dict[str, list[str]]]:
    """Normalize the pretrained extractor JSON into `{abstract_index: {field: entities}}`."""
    normalized: dict[int, dict[str, list[str]]] = {}
    for raw_key, fields in raw_data.items():
        normalized[abstract_key_to_int(raw_key)] = fields
    return normalized


def normalize_round_one_gold_entities(raw_data: dict[str, Any]) -> dict[int, dict[str, list[str]]]:
    """Normalize the GPT-5 JSON into `{abstract_index: {field: entities}}`."""
    normalized: dict[int, dict[str, list[str]]] = {}
    for raw_key, payload in raw_data.items():
        normalized[abstract_key_to_int(raw_key)] = payload.get("entities", {})
    return normalized


def collect_mention_occurrences(text: str, entity_text: str) -> list[dict[str, Any]]:
    """Locate all occurrences of one entity string inside an abstract.

    The matching strategy intentionally mirrors the raw-entity position logic in
    `ner.py`: exact match first, then an ignore-case fallback, and finally a
    placeholder unmatched mention when the text cannot be located.
    """
    token_matches = ner._tokenize_with_spans(text)

    match_spans = list(re.finditer(re.escape(entity_text), text))
    if not match_spans:
        match_spans = list(re.finditer(re.escape(entity_text), text, flags=re.IGNORECASE))

    occurrences: list[dict[str, Any]] = []
    if not match_spans:
        occurrences.append(
            {
                "original_text": entity_text,
                "position": None,
                "token_length": len(re.findall(r"\S+", entity_text)),
                "char_start": None,
                "char_end": None,
            }
        )
        return occurrences

    for match in match_spans:
        start_token = ner._char_to_start_token(token_matches, match.start())
        occurrences.append(
            {
                "original_text": entity_text,
                "position": start_token,
                "token_length": len(re.findall(r"\S+", entity_text)),
                "char_start": match.start(),
                "char_end": match.end(),
            }
        )

    return occurrences


def build_round_one_mention_payload(
    abstracts: dict[int, str],
    entity_lookup: dict[int, dict[str, list[str]]],
    source: str,
) -> dict[str, list[dict[str, Any]]]:
    """Convert raw round-one outputs into mention-aware abstract records."""
    abstract_records: list[dict[str, Any]] = []

    for abstract_index in sorted(abstracts):
        abstract_text = "" if abstracts[abstract_index] is None else str(abstracts[abstract_index])
        raw_fields = entity_lookup.get(abstract_index, {})

        entities: list[dict[str, Any]] = []
        for field, entity_list in raw_fields.items():
            seen_texts: set[str] = set()
            for entity_text in entity_list:
                if not isinstance(entity_text, str) or not entity_text.strip():
                    continue

                entity_text = entity_text.strip()
                if entity_text in seen_texts:
                    continue
                seen_texts.add(entity_text)

                for occurrence in collect_mention_occurrences(abstract_text, entity_text):
                    entities.append(
                        {
                            "source": source,
                            "field": field,
                            "entity_form": entity_text,
                            "mention": {
                                "original_text": occurrence["original_text"],
                                "position": occurrence["position"],
                                "token_length": occurrence["token_length"],
                            },
                            "char_start": occurrence["char_start"],
                            "char_end": occurrence["char_end"],
                        }
                    )

        entities.sort(
            key=lambda item: (
                float("inf") if item["mention"]["position"] is None else item["mention"]["position"],
                item["field"],
                normalize_surface_text(item["entity_form"]),
            )
        )

        abstract_records.append(
            {
                "abstract": {
                    "abstract_id": abstract_id_label(abstract_index),
                    "text": abstract_text,
                },
                "entities": entities,
            }
        )

    return {"abstracts": abstract_records}


def flatten_round_one_mentions(payload: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Flatten the mention-aware payload for tabular comparison."""
    rows: list[dict[str, Any]] = []
    for abstract_entry in payload["abstracts"]:
        abstract_id = abstract_entry["abstract"]["abstract_id"]
        for entity in abstract_entry["entities"]:
            rows.append(
                {
                    "abstract_id": abstract_id,
                    "field": entity["field"],
                    "entity_form": entity["entity_form"],
                    "normalized_form": normalize_surface_text(entity["entity_form"]),
                    "position": entity["mention"]["position"],
                    "token_length": entity["mention"]["token_length"],
                    "source": entity["source"],
                }
            )
    return rows


def mention_key(row: dict[str, Any]) -> tuple[str, str, str]:
    """Build the form-level comparison key for entity-based metrics.

    Standard NER evaluation: an entity form is considered extracted once per
    abstract/field combination, regardless of how many times the text appears
    or where it occurs in the abstract. Multiple identical extractions count as 1.
    """
    return (
        row["abstract_id"],
        row["field"],
        row["normalized_form"],
    )


def mention_key_sort_value(key: tuple[str, str, str]) -> tuple[str, str, str]:
    """Provide a stable sort key for form-level entity keys."""
    return key


def build_round_one_normalized_entities(
    mention_payload: dict[str, list[dict[str, Any]]],
    source: str,
) -> dict[str, list[dict[str, Any]]]:
    """Extract unique entity forms from mention payload (deduplicated per abstract/field).

    Takes mention-level data and deduplicates by (field, normalized_form) per abstract,
    removing position information. Useful for form-level visualization and comparison.
    """
    abstract_records: list[dict[str, Any]] = []

    for abstract_entry in mention_payload["abstracts"]:
        abstract_id = abstract_entry["abstract"]["abstract_id"]
        abstract_text = abstract_entry["abstract"]["text"]

        # Deduplicate entities by (field, normalized_form) within this abstract
        seen_forms: set[tuple[str, str]] = set()
        normalized_entities: list[dict[str, Any]] = []

        for entity in abstract_entry["entities"]:
            normalized_form = normalize_surface_text(entity["entity_form"])
            form_key = (entity["field"], normalized_form)
            if form_key not in seen_forms:
                seen_forms.add(form_key)
                normalized_entities.append(
                    {
                        "field": entity["field"],
                        "entity_form": entity["entity_form"],
                        "normalized_form": normalized_form,
                        "source": source,
                    }
                )

        # Sort by field then by entity form
        normalized_entities.sort(key=lambda e: (e["field"], e["entity_form"]))

        abstract_records.append(
            {
                "abstract": {
                    "abstract_id": abstract_id,
                    "text": abstract_text,
                },
                "entities": normalized_entities,
            }
        )

    return {"abstracts": abstract_records}



def build_round_one_metrics(
    pred_payload: dict[str, list[dict[str, Any]]],
    gold_payload: dict[str, list[dict[str, Any]]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Compute form-level entity metrics and count comparisons for round one.

    Metrics are computed at the entity-form level (abstract_id, field, normalized_form).
    An entity is counted once per abstract/field, regardless of how many times it appears.
    This is the standard NER evaluation approach in the biomedical NLP literature.
    """
    pred_rows = flatten_round_one_mentions(pred_payload)
    gold_rows = flatten_round_one_mentions(gold_payload)

    pred_df = pd.DataFrame(pred_rows)
    gold_df = pd.DataFrame(gold_rows)

    pred_keys = {mention_key(row) for row in pred_rows}
    gold_keys = {mention_key(row) for row in gold_rows}
    matched_keys = pred_keys & gold_keys

    metric_rows: list[dict[str, Any]] = []
    all_fields = sorted(set(pred_df.get("field", pd.Series(dtype=str))) | set(gold_df.get("field", pd.Series(dtype=str))))
    for field in all_fields:
        field_pred_keys = {key for key in pred_keys if key[1] == field}
        field_gold_keys = {key for key in gold_keys if key[1] == field}
        field_matches = field_pred_keys & field_gold_keys

        pred_count = len(field_pred_keys)
        gold_count = len(field_gold_keys)
        match_count = len(field_matches)
        precision = match_count / pred_count if pred_count else 0.0
        recall = match_count / gold_count if gold_count else 0.0

        metric_rows.append(
            {
                "field": field,
                "pred_mentions": pred_count,
                "gold_mentions": gold_count,
                "matched_mentions": match_count,
                "precision": precision,
                "recall": recall,
                "f1": f1_score(precision, recall),
            }
        )

    overall_precision = len(matched_keys) / len(pred_keys) if pred_keys else 0.0
    overall_recall = len(matched_keys) / len(gold_keys) if gold_keys else 0.0
    metric_rows.append(
        {
            "field": "OVERALL",
            "pred_mentions": len(pred_keys),
            "gold_mentions": len(gold_keys),
            "matched_mentions": len(matched_keys),
            "precision": overall_precision,
            "recall": overall_recall,
            "f1": f1_score(overall_precision, overall_recall),
        }
    )

    # Overall metrics excluding "Measurement" field
    pred_keys_excl_meas = {key for key in pred_keys if key[1] != "Measurement"}
    gold_keys_excl_meas = {key for key in gold_keys if key[1] != "Measurement"}
    matched_keys_excl_meas = pred_keys_excl_meas & gold_keys_excl_meas

    overall_precision_excl = len(matched_keys_excl_meas) / len(pred_keys_excl_meas) if pred_keys_excl_meas else 0.0
    overall_recall_excl = len(matched_keys_excl_meas) / len(gold_keys_excl_meas) if gold_keys_excl_meas else 0.0
    metric_rows.append(
        {
            "field": "OVERALL_EXCL_MEASUREMENT",
            "pred_mentions": len(pred_keys_excl_meas),
            "gold_mentions": len(gold_keys_excl_meas),
            "matched_mentions": len(matched_keys_excl_meas),
            "precision": overall_precision_excl,
            "recall": overall_recall_excl,
            "f1": f1_score(overall_precision_excl, overall_recall_excl),
        }
    )

    count_rows: list[dict[str, Any]] = []
    abstract_ids = sorted(set(pred_df.get("abstract_id", pd.Series(dtype=str))) | set(gold_df.get("abstract_id", pd.Series(dtype=str))))
    for abstract_id in abstract_ids:
        pred_subset = pred_df[pred_df["abstract_id"] == abstract_id] if not pred_df.empty else pd.DataFrame()
        gold_subset = gold_df[gold_df["abstract_id"] == abstract_id] if not gold_df.empty else pd.DataFrame()
        fields = sorted(set(pred_subset.get("field", pd.Series(dtype=str))) | set(gold_subset.get("field", pd.Series(dtype=str))))

        for field in fields:
            pred_field = pred_subset[pred_subset["field"] == field] if not pred_subset.empty else pd.DataFrame()
            gold_field = gold_subset[gold_subset["field"] == field] if not gold_subset.empty else pd.DataFrame()
            count_rows.append(
                {
                    "abstract_id": abstract_id,
                    "field": field,
                    "pred_unique_entity_count": pred_field["normalized_form"].nunique() if not pred_field.empty else 0,
                    "gold_unique_entity_count": gold_field["normalized_form"].nunique() if not gold_field.empty else 0,
                    "pred_mention_count": len(pred_field),
                    "gold_mention_count": len(gold_field),
                    "unique_entity_delta": (
                        (pred_field["normalized_form"].nunique() if not pred_field.empty else 0)
                        - (gold_field["normalized_form"].nunique() if not gold_field.empty else 0)
                    ),
                    "mention_delta": len(pred_field) - len(gold_field),
                }
            )

    metrics_df = pd.DataFrame(metric_rows)
    counts_df = pd.DataFrame(count_rows)

    comparison_payload = {
        "metrics": metric_rows,
        "matched_keys": [list(key) for key in sorted(matched_keys, key=mention_key_sort_value)],
        "pred_only_keys": [list(key) for key in sorted(pred_keys - gold_keys, key=mention_key_sort_value)],
        "gold_only_keys": [list(key) for key in sorted(gold_keys - pred_keys, key=mention_key_sort_value)],
    }
    return metrics_df, counts_df, comparison_payload


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Round-one orchestration
# ---------------------------------------------------------------------------

def run_round_one_evaluation() -> dict[str, Any]:
    """Run the mention-aware first-round comparison and save all artifacts."""
    abstracts = ner.load_abstracts_from_csv()
    pred_raw = load_json(PRED_ROUND_ONE_PATH)
    gold_raw = load_json(GOLD_ROUND_ONE_PATH)

    pred_entities = normalize_round_one_pred_entities(pred_raw)
    gold_entities = normalize_round_one_gold_entities(gold_raw)

    pred_payload = build_round_one_mention_payload(abstracts, pred_entities, source="PRED")
    gold_payload = build_round_one_mention_payload(abstracts, gold_entities, source="GOLD")

    save_json(pred_payload, ROUND_ONE_PRED_WITH_MENTIONS_PATH)
    save_json(gold_payload, ROUND_ONE_GOLD_WITH_MENTIONS_PATH)

    # Build and save normalized entities (deduplicated forms for visualization)
    normalized_pred = build_round_one_normalized_entities(pred_payload, source="PRED")
    normalized_gold = build_round_one_normalized_entities(gold_payload, source="GOLD")
    save_json(normalized_pred, ROUND_ONE_NORMALIZED_PRED_PATH)
    save_json(normalized_gold, ROUND_ONE_NORMALIZED_GOLD_PATH)

    metrics_df, counts_df, comparison_payload = build_round_one_metrics(pred_payload, gold_payload)
    metrics_df.to_csv(ROUND_ONE_METRICS_PATH, index=False)
    counts_df.to_csv(ROUND_ONE_COUNTS_PATH, index=False)
    save_json(comparison_payload, ROUND_ONE_COMPARISON_JSON_PATH)

    return {
        "pred_with_mentions": pred_payload,
        "gold_with_mentions": gold_payload,
        "normalized_pred": normalized_pred,
        "normalized_gold": normalized_gold,
        "metrics": metrics_df.to_dict(orient="records"),
        "count_comparison": counts_df.to_dict(orient="records"),
        "comparison": comparison_payload,
    }


# ---------------------------------------------------------------------------
# Round-two BERT method utilities
# ---------------------------------------------------------------------------

def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Average token embeddings while respecting the attention mask."""
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def encode_texts_with_transformer(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: str,
    batch_size: int = 16,
) -> dict[str, np.ndarray]:
    """Encode unique texts into normalized sentence vectors."""
    vectors: dict[str, np.ndarray] = {}
    ordered_texts = list(dict.fromkeys(texts))
    if not ordered_texts:
        return vectors

    model.eval()
    with torch.no_grad():
        for start in range(0, len(ordered_texts), batch_size):
            batch = ordered_texts[start:start + batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded)
            pooled = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            for text, vector in zip(batch, pooled.cpu().numpy()):
                vectors[text] = vector

    return vectors


def rank_anatomical_candidates_with_vectors(
    query_text: str,
    candidate_forms: list[str],
    vector_lookup: dict[str, np.ndarray],
    top_k: int = 1,
) -> list[str]:
    """Rank anatomy candidates with dense transformer embeddings."""
    if not query_text or not candidate_forms or query_text not in vector_lookup:
        return []

    query_vector = vector_lookup[query_text]
    scored: list[tuple[float, str]] = []
    for candidate in candidate_forms:
        if candidate not in vector_lookup:
            continue
        score = float(np.dot(query_vector, vector_lookup[candidate]))
        scored.append((score, candidate))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [candidate for _, candidate in scored[:top_k]]


def normalize_method1_predictions(attributes: dict[str, list[str]]) -> list[str]:
    """Normalize method-1 anatomy predictions into the canonical anatomy space."""
    predictions: list[str] = []
    for item in attributes.get("anatomical structure", []):
        normalized_item = ner.normalize_anatomical_location(item)
        if normalized_item and normalized_item not in predictions:
            predictions.append(normalized_item)
    return predictions


def location_match(predicted_location: str | None, gold_location: str | None) -> bool:
    """Allow exact matches and simple token-subset matches for anatomy labels."""
    if not predicted_location or not gold_location:
        return False
    if predicted_location == gold_location:
        return True

    predicted_tokens = set(predicted_location.split())
    gold_tokens = set(gold_location.split())
    return gold_tokens.issubset(predicted_tokens) or predicted_tokens.issubset(gold_tokens)


def summarize_method(evaluation_rows: list[dict[str, Any]], prediction_key: str, correct_key: str) -> dict[str, float]:
    """Summarize coverage, accuracy, and recall for one evaluation output column."""
    total = len(evaluation_rows)
    predicted = sum(1 for row in evaluation_rows if row[prediction_key])
    correct = sum(1 for row in evaluation_rows if row[correct_key])
    return {
        "total": total,
        "predicted": predicted,
        "coverage": (predicted / total * 100) if total else 0.0,
        "correct": correct,
        "accuracy": (correct / predicted * 100) if predicted else 0.0,
        "recall": (correct / total * 100) if total else 0.0,
    }


def load_method2_cases(
    method1_data: dict[str, Any],
    entity_specification_path: str | Path = ROUND_TWO_METHOD2_PATH,
    top_k: int = 1,
    embedding_dim: int = 512,
) -> list[dict[str, Any]]:
    """Load existing method-2 cases when available, otherwise rebuild them."""
    path = Path(entity_specification_path)
    if path.exists():
        return load_json(path)
    return ner.build_entity_specification_cases(
        method1_data["abstracts"],
        top_k=top_k,
        embedding_dim=embedding_dim,
    )


def run_bert_method2_variants(
    method2_cases: list[dict[str, Any]],
    abstract_anatomy_lookup: dict[str, list[str]],
    bert_model_specs: dict[str, str] | None = None,
    top_k: int = 1,
    bert_batch_size: int = 16,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Run the BERT-based ranking variants from the notebook."""
    model_specs = bert_model_specs or DEFAULT_BERT_MODEL_SPECS
    device = "cuda" if torch.cuda.is_available() else "cpu"

    bert_method2_results: dict[str, Any] = {}
    bert_method2_cases: dict[str, list[dict[str, Any]]] = {}

    all_anatomy_forms = sorted(
        {candidate for candidates in abstract_anatomy_lookup.values() for candidate in candidates}
    )
    query_forms = sorted({case["entity_form"] for case in method2_cases})
    texts_to_encode = all_anatomy_forms + query_forms

    for model_name, model_id in model_specs.items():
        log.info("Loading %s: %s on %s", model_name, model_id, device)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id, use_safetensors=True).to(device)

        vector_lookup = encode_texts_with_transformer(
            texts_to_encode,
            tokenizer,
            model,
            device=device,
            batch_size=bert_batch_size,
        )

        cases: list[dict[str, Any]] = []
        for case in method2_cases:
            predictions = rank_anatomical_candidates_with_vectors(
                case["entity_form"],
                case["candidate_anatomies"],
                vector_lookup,
                top_k=top_k,
            )
            cases.append(
                {
                    "abstract_id": case["abstract_id"],
                    "entity_form": case["entity_form"],
                    "field": case["field"],
                    "mention": case["mention"],
                    "candidate_anatomies": case["candidate_anatomies"],
                    "predicted_locations": predictions,
                }
            )

        bert_method2_cases[model_name] = cases
        bert_method2_results[model_name] = {
            "model_id": model_id,
            "device": device,
            "cases": cases,
        }

    return bert_method2_results, bert_method2_cases


def evaluate_round_two_predictions(
    method1_data: dict[str, Any],
    method2_cases: list[dict[str, Any]],
    bert_method2_cases: dict[str, list[dict[str, Any]]],
    golden_standard_path: str | Path = GOLD_STANDARD_PATH,
    top_k: int = 1,
    embedding_dim: int = 512,
) -> tuple[list[dict[str, Any]], dict[str, Any], pd.DataFrame]:
    """Evaluate method 1, the hashed baseline, and all BERT variants."""
    gold_rows = pd.read_csv(golden_standard_path)
    round_two_lookup = method1_data["round2_lookup"]
    abstract_entity_records = method1_data["abstracts"]
    abstract_anatomy_lookup = ner.build_abstract_anatomy_lookup(abstract_entity_records)

    bert_case_lookup: dict[str, dict[tuple[str, str, str, int | None], list[str]]] = {}
    for model_name, cases in bert_method2_cases.items():
        model_lookup: dict[tuple[str, str, str, int | None], list[str]] = {}
        for case in cases:
            key = (
                case["abstract_id"],
                case["field"],
                case["entity_form"],
                case["mention"]["position"],
            )
            model_lookup[key] = case["predicted_locations"]
        bert_case_lookup[model_name] = model_lookup

    field_by_type = {
        "symptom": "Symptom",
        "movement": "Terms of Body Movements",
    }

    evaluation_rows: list[dict[str, Any]] = []
    for _, row in gold_rows.iterrows():
        gold_type = str(row["Type"]).strip().lower()
        if gold_type not in field_by_type:
            continue

        abstract_id = abstract_id_label(int(row["Abstract"]))
        field = field_by_type[gold_type]
        query_form = ner.normalize_eval_phrase(row["Full Phrase"], field)
        gold_location = ner.normalize_anatomical_location(row["Anatomical Location"])

        method1_attributes = round_two_lookup.get(field, {}).get(query_form, {})
        method1_predictions = normalize_method1_predictions(method1_attributes)
        method1_correct = any(location_match(prediction, gold_location) for prediction in method1_predictions)

        traditional_predictions = ner.rank_anatomical_candidates(
            query_form,
            abstract_anatomy_lookup.get(abstract_id, []),
            top_k=top_k,
            dim=embedding_dim,
        )
        traditional_correct = any(
            location_match(prediction, gold_location) for prediction in traditional_predictions
        )

        bert_predictions: dict[str, list[str]] = {}
        bert_correct: dict[str, bool] = {}
        for model_name, model_lookup in bert_case_lookup.items():
            matching_predictions: list[str] = []
            for key, predictions in model_lookup.items():
                case_abstract_id, case_field, case_query_form, _ = key
                if (
                    case_abstract_id == abstract_id
                    and case_field == field
                    and case_query_form == query_form
                ):
                    matching_predictions = predictions
                    break

            bert_predictions[model_name] = matching_predictions
            bert_correct[model_name] = any(
                location_match(prediction, gold_location) for prediction in matching_predictions
            )

        evaluation_rows.append(
            {
                "abstract_id": abstract_id,
                "type": gold_type,
                "full_phrase": row["Full Phrase"],
                "gold_location": gold_location,
                "method1_predictions": method1_predictions,
                "method1_correct": method1_correct,
                "traditional_method2_predictions": traditional_predictions,
                "traditional_method2_correct": traditional_correct,
                "bert_predictions": bert_predictions,
                "bert_correct": bert_correct,
            }
        )

    evaluation_summary = {
        "method1_round2_extraction": summarize_method(
            evaluation_rows,
            "method1_predictions",
            "method1_correct",
        ),
        "traditional_method2_similarity_benchmark": summarize_method(
            evaluation_rows,
            "traditional_method2_predictions",
            "traditional_method2_correct",
        ),
    }

    for model_name in bert_method2_cases:
        predicted = sum(1 for row in evaluation_rows if row["bert_predictions"].get(model_name))
        correct = sum(1 for row in evaluation_rows if row["bert_correct"].get(model_name))
        total = len(evaluation_rows)
        evaluation_summary[f"{model_name}_method2"] = {
            "total": total,
            "predicted": predicted,
            "coverage": (predicted / total * 100) if total else 0.0,
            "correct": correct,
            "accuracy": (correct / predicted * 100) if predicted else 0.0,
            "recall": (correct / total * 100) if total else 0.0,
        }

    evaluation_table = pd.DataFrame(
        [
            {
                "method": method_name,
                "total": summary["total"],
                "predicted": summary["predicted"],
                "correct": summary["correct"],
                "coverage": summary["coverage"],
                "accuracy": summary["accuracy"],
                "recall": summary["recall"],
            }
            for method_name, summary in evaluation_summary.items()
        ]
    )

    return evaluation_rows, evaluation_summary, evaluation_table


# ---------------------------------------------------------------------------
# Round-two orchestration
# ---------------------------------------------------------------------------

def run_round_two_evaluation(
    top_k: int = 1,
    embedding_dim: int = 512,
    bert_batch_size: int = 16,
) -> dict[str, Any]:
    """Run the second-round BERT comparison and save its outputs."""
    method1_data = load_json(ROUND_TWO_METHOD1_PATH)
    abstract_entity_records = method1_data["abstracts"]
    abstract_anatomy_lookup = ner.build_abstract_anatomy_lookup(abstract_entity_records)
    method2_cases = load_method2_cases(
        method1_data,
        entity_specification_path=ROUND_TWO_METHOD2_PATH,
        top_k=top_k,
        embedding_dim=embedding_dim,
    )

    bert_results, bert_cases = run_bert_method2_variants(
        method2_cases,
        abstract_anatomy_lookup,
        bert_model_specs=DEFAULT_BERT_MODEL_SPECS,
        top_k=top_k,
        bert_batch_size=bert_batch_size,
    )

    evaluation_rows, evaluation_summary, evaluation_table = evaluate_round_two_predictions(
        method1_data,
        method2_cases,
        bert_cases,
        golden_standard_path=GOLD_STANDARD_PATH,
        top_k=top_k,
        embedding_dim=embedding_dim,
    )

    save_json(bert_results, ROUND_TWO_BERT_RESULTS_PATH)
    save_json(
        {
            "evaluation_rows": evaluation_rows,
            "evaluation_summary": evaluation_summary,
        },
        ROUND_TWO_EVALUATION_JSON_PATH,
    )
    evaluation_table.to_csv(ROUND_TWO_EVALUATION_TABLE_PATH, index=False)

    return {
        "bert_results": bert_results,
        "evaluation_rows": evaluation_rows,
        "evaluation_summary": evaluation_summary,
        "evaluation_table": evaluation_table.to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """CLI arguments for selectively running one or both evaluation blocks."""
    parser = argparse.ArgumentParser(description="Run the reorganized NER evaluation workflows.")
    parser.add_argument(
        "--mode",
        choices=["round1", "round2", "all"],
        default="all",
        help="Choose which evaluation block to run.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Number of candidate anatomies kept for method-2 ranking.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=512,
        help="Embedding dimension for the hashed baseline in round two.",
    )
    parser.add_argument(
        "--bert-batch-size",
        type=int,
        default=16,
        help="Batch size used when encoding texts with transformer models.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()

    if args.mode in {"round1", "all"}:
        run_round_one_evaluation()
    if args.mode in {"round2", "all"}:
        run_round_two_evaluation(
            top_k=args.top_k,
            embedding_dim=args.embedding_dim,
            bert_batch_size=args.bert_batch_size,
        )


if __name__ == "__main__":
    main()
