"""
relation_extraction_evaluation.py
==================================
Evaluates extracted relations against manually annotated gold standard.

Compares relations extracted via LLM (relations_Claude.json) against the
manually curated gold standard (relations_manual_gold.json) to measure:
  - Precision, Recall, F1 for triple matching (subject, relation, object)
  - Exact matches vs. partial matches
  - Per-relation-type performance
  - Per-abstract statistics

Input
-----
  data/processed/03_relations_extracted/relations_Claude.json     (predicted)
  data/processed/evaluation/extraction_gold_standard/relations_manual_gold.json  (gold standard)

Output
------
  data/processed/evaluation/relation_extraction/
    ├── metrics.json           # Aggregate metrics
    ├── evaluation.csv         # Per-abstract breakdown
    ├── per_relation_metrics.json  # Metrics by relation type
    └── evaluation_report.txt  # Human-readable summary
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # src/pipeline → project root
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Input paths
PREDICTED_RELATIONS_PATH = DATA_PROCESSED_DIR / "03_relations_extracted" / "relations_Claude.json"
GOLD_STANDARD_PATH = DATA_PROCESSED_DIR / "evaluation" / "extraction_gold_standard" / "relations_manual_gold.json"

# Output paths
EVAL_OUTPUT_DIR = DATA_PROCESSED_DIR / "evaluation" / "relation_extraction"
METRICS_PATH = EVAL_OUTPUT_DIR / "metrics.json"
EVALUATION_CSV_PATH = EVAL_OUTPUT_DIR / "evaluation.csv"
PER_RELATION_METRICS_PATH = EVAL_OUTPUT_DIR / "per_relation_metrics.json"
REPORT_PATH = EVAL_OUTPUT_DIR / "evaluation_report.txt"


# ─────────────────────────────────────────────────────────────────────────────
# Generic I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_json(path: str | Path) -> Any:
    """Load JSON file using UTF-8 encoding."""
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str | Path) -> None:
    """Save JSON file with proper formatting."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    log.info(f"Saved → {target}")


def save_csv(rows: list[dict], path: str | Path) -> None:
    """Save list of dicts to CSV."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        log.warning(f"No rows to save to {target}")
        return
    with target.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"Saved → {target}")


# ─────────────────────────────────────────────────────────────────────────────
# Normalization helpers
# ─────────────────────────────────────────────────────────────────────────────

def normalize_triple(subject: str, relation: str, obj: str) -> tuple:
    """Normalize a triple for comparison (lowercase, strip whitespace)."""
    return (
        subject.lower().strip(),
        relation.lower().strip(),
        obj.lower().strip(),
    )


def normalize_entities_in_triple(subject: str, relation: str, obj: str) -> tuple:
    """Normalize triple without strict entity matching (for partial evaluation)."""
    return normalize_triple(subject, relation, obj)


# ─────────────────────────────────────────────────────────────────────────────
# Triple comparison
# ─────────────────────────────────────────────────────────────────────────────

def triples_match_exact(pred: dict, gold: dict) -> bool:
    """Check if predicted and gold triples match exactly (all three components)."""
    pred_triple = normalize_triple(pred["subject"], pred["relation"], pred["object"])
    gold_triple = normalize_triple(gold["subject"], gold["relation"], gold["object"])
    return pred_triple == gold_triple


def entities_match(pred: dict, gold: dict) -> bool:
    """Check if subject and object match (relation may differ)."""
    pred_entities = (pred["subject"].lower().strip(), pred["object"].lower().strip())
    gold_entities = (gold["subject"].lower().strip(), gold["object"].lower().strip())
    return pred_entities == gold_entities


def relation_matches_between_same_entities(pred: dict, gold: dict) -> bool:
    """Check if entities match AND relation types match."""
    return entities_match(pred, gold) and \
           pred["relation"].lower().strip() == gold["relation"].lower().strip()


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(tp: int, fp: int, fn: int) -> dict:
    """Compute precision, recall, F1 from TP, FP, FN counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def evaluate_abstract(
    predicted_relations: list[dict],
    gold_relations: list[dict],
) -> dict:
    """
    Evaluate a single abstract's relations.

    Returns:
      {
        "total_gold": int,
        "total_predicted": int,
        "exact_matches": int,
        "entity_matches": int,
        "metrics_exact": {...},
        "metrics_entity": {...},
        "per_relation_type": {...}
      }
    """
    gold_set = {normalize_triple(r["subject"], r["relation"], r["object"]) for r in gold_relations}
    predicted_set = {normalize_triple(r["subject"], r["relation"], r["object"]) for r in predicted_relations}

    # Exact match: triple (subject, relation, object) matches
    exact_matches = gold_set & predicted_set
    fp_exact = predicted_set - gold_set
    fn_exact = gold_set - predicted_set

    metrics_exact = compute_metrics(len(exact_matches), len(fp_exact), len(fn_exact))

    # Entity match: (subject, object) matches
    gold_entities = {(r["subject"].lower().strip(), r["object"].lower().strip()) for r in gold_relations}
    predicted_entities = {(r["subject"].lower().strip(), r["object"].lower().strip()) for r in predicted_relations}

    entity_matches = gold_entities & predicted_entities
    fp_entity = predicted_entities - gold_entities
    fn_entity = gold_entities - predicted_entities

    metrics_entity = compute_metrics(len(entity_matches), len(fp_entity), len(fn_entity))

    # Per-relation-type breakdown
    per_relation_metrics: dict[str, dict] = {}
    for relation_type in set(r["relation"].lower().strip() for r in gold_relations + predicted_relations):
        gold_rel = [r for r in gold_relations if r["relation"].lower().strip() == relation_type]
        pred_rel = [r for r in predicted_relations if r["relation"].lower().strip() == relation_type]

        gold_rel_set = {normalize_triple(r["subject"], r["relation"], r["object"]) for r in gold_rel}
        pred_rel_set = {normalize_triple(r["subject"], r["relation"], r["object"]) for r in pred_rel}

        matches = gold_rel_set & pred_rel_set
        fp = pred_rel_set - gold_rel_set
        fn = gold_rel_set - pred_rel_set

        per_relation_metrics[relation_type] = compute_metrics(len(matches), len(fp), len(fn))

    return {
        "total_gold": len(gold_relations),
        "total_predicted": len(predicted_relations),
        "exact_matches": len(exact_matches),
        "entity_matches": len(entity_matches),
        "metrics_exact": metrics_exact,
        "metrics_entity": metrics_entity,
        "per_relation_type": per_relation_metrics,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    predicted_path: str | Path = PREDICTED_RELATIONS_PATH,
    gold_standard_path: str | Path = GOLD_STANDARD_PATH,
    output_dir: str | Path = EVAL_OUTPUT_DIR,
) -> dict:
    """
    Run full relation extraction evaluation.

    Returns:
      {
        "summary": {...},
        "per_abstract": {...}
      }
    """
    predicted_path = Path(predicted_path)
    gold_standard_path = Path(gold_standard_path)
    output_dir = Path(output_dir)

    if not gold_standard_path.exists():
        log.error(f"Gold standard not found: {gold_standard_path}")
        return {}

    if not predicted_path.exists():
        log.warning(f"Predicted relations not found: {predicted_path}")
        predicted = {}
    else:
        predicted = load_json(predicted_path)

    log.info(f"Loading gold standard from {gold_standard_path}")
    gold_standard = load_json(gold_standard_path)

    # Evaluate each abstract
    per_abstract_results = {}
    abstract_csv_rows = []

    all_gold_count = 0
    all_predicted_count = 0
    all_exact_matches = 0
    all_exact_tp = 0
    all_exact_fp = 0
    all_exact_fn = 0

    all_rel_type_metrics: dict[str, dict] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for abstract_id in sorted(gold_standard.keys()):
        gold_rels = gold_standard.get(abstract_id, [])
        pred_rels = predicted.get(abstract_id, [])

        result = evaluate_abstract(pred_rels, gold_rels)
        per_abstract_results[abstract_id] = result

        all_gold_count += result["total_gold"]
        all_predicted_count += result["total_predicted"]
        all_exact_matches += result["exact_matches"]
        all_exact_tp += result["metrics_exact"]["tp"]
        all_exact_fp += result["metrics_exact"]["fp"]
        all_exact_fn += result["metrics_exact"]["fn"]

        # Accumulate per-relation-type metrics
        for rel_type, metrics in result["per_relation_type"].items():
            all_rel_type_metrics[rel_type]["tp"] += metrics["tp"]
            all_rel_type_metrics[rel_type]["fp"] += metrics["fp"]
            all_rel_type_metrics[rel_type]["fn"] += metrics["fn"]

        abstract_csv_rows.append({
            "abstract_id": abstract_id,
            "gold_count": result["total_gold"],
            "predicted_count": result["total_predicted"],
            "exact_matches": result["exact_matches"],
            "entity_matches": result["entity_matches"],
            "exact_precision": result["metrics_exact"]["precision"],
            "exact_recall": result["metrics_exact"]["recall"],
            "exact_f1": result["metrics_exact"]["f1"],
            "entity_precision": result["metrics_entity"]["precision"],
            "entity_recall": result["metrics_entity"]["recall"],
            "entity_f1": result["metrics_entity"]["f1"],
        })

    # Aggregate metrics
    summary = compute_metrics(all_exact_tp, all_exact_fp, all_exact_fn)
    summary.update({
        "total_gold_relations": all_gold_count,
        "total_predicted_relations": all_predicted_count,
        "abstracts_evaluated": len(per_abstract_results),
    })

    # Per-relation-type aggregate
    per_relation_metrics_agg = {}
    for rel_type, counts in all_rel_type_metrics.items():
        per_relation_metrics_agg[rel_type] = compute_metrics(counts["tp"], counts["fp"], counts["fn"])

    result_data = {
        "summary": summary,
        "per_abstract": per_abstract_results,
    }

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(summary, METRICS_PATH)
    save_csv(abstract_csv_rows, EVALUATION_CSV_PATH)
    save_json(per_relation_metrics_agg, PER_RELATION_METRICS_PATH)

    # Generate report
    report = generate_report(summary, per_relation_metrics_agg, abstract_csv_rows)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    log.info(f"Saved → {REPORT_PATH}")

    return result_data


def generate_report(
    summary: dict,
    per_relation_metrics: dict,
    abstract_rows: list[dict],
) -> str:
    """Generate human-readable evaluation report."""
    report = []
    report.append("=" * 80)
    report.append("RELATION EXTRACTION EVALUATION REPORT")
    report.append("=" * 80)
    report.append("")

    report.append("OVERALL METRICS (Exact Triple Match)")
    report.append("-" * 80)
    report.append(f"Total Gold Relations:           {summary['total_gold_relations']}")
    report.append(f"Total Predicted Relations:      {summary['total_predicted_relations']}")
    report.append(f"True Positives:                 {summary['tp']}")
    report.append(f"False Positives:                {summary['fp']}")
    report.append(f"False Negatives:                {summary['fn']}")
    report.append(f"Precision:                      {summary['precision']:.4f}")
    report.append(f"Recall:                         {summary['recall']:.4f}")
    report.append(f"F1-Score:                       {summary['f1']:.4f}")
    report.append(f"Abstracts Evaluated:            {summary['abstracts_evaluated']}")
    report.append("")

    report.append("PER-RELATION-TYPE METRICS")
    report.append("-" * 80)
    for rel_type in sorted(per_relation_metrics.keys()):
        metrics = per_relation_metrics[rel_type]
        report.append(f"\n{rel_type}:")
        report.append(f"  Precision: {metrics['precision']:.4f}")
        report.append(f"  Recall:    {metrics['recall']:.4f}")
        report.append(f"  F1-Score:  {metrics['f1']:.4f}")
        report.append(f"  (TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']})")
    report.append("")

    report.append("TOP 10 ABSTRACTS BY F1-SCORE")
    report.append("-" * 80)
    sorted_abstracts = sorted(abstract_rows, key=lambda x: x["exact_f1"], reverse=True)
    for row in sorted_abstracts[:10]:
        report.append(
            f"{row['abstract_id']:8s}  "
            f"Gold={row['gold_count']:2d}  Pred={row['predicted_count']:2d}  "
            f"Match={row['exact_matches']:2d}  F1={row['exact_f1']:.4f}"
        )
    report.append("")

    report.append("BOTTOM 10 ABSTRACTS BY F1-SCORE")
    report.append("-" * 80)
    for row in sorted_abstracts[-10:]:
        report.append(
            f"{row['abstract_id']:8s}  "
            f"Gold={row['gold_count']:2d}  Pred={row['predicted_count']:2d}  "
            f"Match={row['exact_matches']:2d}  F1={row['exact_f1']:.4f}"
        )
    report.append("")

    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate relation extraction against gold standard"
    )
    parser.add_argument(
        "--predicted",
        type=Path,
        default=PREDICTED_RELATIONS_PATH,
        help=f"Path to predicted relations JSON (default: {PREDICTED_RELATIONS_PATH})",
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=GOLD_STANDARD_PATH,
        help=f"Path to gold standard relations JSON (default: {GOLD_STANDARD_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EVAL_OUTPUT_DIR,
        help=f"Output directory for evaluation results (default: {EVAL_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    run_evaluation(
        predicted_path=args.predicted,
        gold_standard_path=args.gold,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
