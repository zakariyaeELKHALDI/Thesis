"""Prepare the versioned retrieval-depth expansion without a new API call."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
import argparse
import json
import os

import numpy as np

from geotech_rag.retrieval_evaluation import (
    RETRIEVAL_PREPARATION_SUMMARY_SCHEMA_VERSION,
    RetrievalEvaluationError,
    RetrievalEvaluationIntegrityError,
    RetrievalEvaluationSchemaError,
    _read_json_lines,
    _read_json_object,
    _resolve_output_paths,
    _validate_candidate_pool,
    _validate_completed_judgements,
    build_blinded_judgement_records,
    build_ranked_candidate_records,
    calculate_file_sha256,
    canonical_json_bytes,
    load_retrieval_evaluation_config,
    load_retrieval_evaluation_inputs,
    search_faiss_pool,
    serialise_json_lines,
)


LINEAGE_SCHEMA_VERSION = "1.0"

LINEAGE_TOP_LEVEL_KEYS = {
    "decision_status",
    "judgement_reuse",
    "lineage_schema_version",
    "query_embedding_reuse",
    "source_v1",
    "target_v2",
}

PREFIX_MATCH_FIELDS = [
    "candidate_schema_version",
    "chunk_character_count",
    "chunk_id",
    "chunk_text_sha256",
    "chunk_token_count",
    "index_position",
    "parent_record_id",
    "pdf_page_index",
    "pdf_page_number",
    "printed_page_number",
    "query_text_sha256",
    "question_id",
    "question_position",
    "rank",
    "similarity_score",
    "source_id",
]


@dataclass(frozen=True)
class RetrievalExpansionPreparationResult:
    """Paths, fingerprints and reuse counts from v2 preparation."""

    query_embedding_matrix_path: Path
    ranked_candidate_pool_path: Path
    blinded_judgement_file_path: Path
    summary_path: Path
    query_embedding_matrix_sha256: str
    ranked_candidate_pool_sha256: str
    blinded_judgement_file_sha256: str
    summary_sha256: str
    candidate_count: int
    reused_judgement_count: int
    new_judgement_count: int


def _require_mapping(value: object, context: str) -> dict[str, Any]:
    """Require one JSON object."""

    if not isinstance(value, dict):
        raise RetrievalEvaluationSchemaError(f"{context} must be an object.")
    return value


def _require_exact_keys(
    value: dict[str, Any],
    expected: set[str],
    context: str,
) -> None:
    """Reject missing or additional contract fields."""

    if set(value) != expected:
        raise RetrievalEvaluationSchemaError(
            f"{context} has unexpected fields: {sorted(set(value) ^ expected)}"
        )


def _resolve_project_path(root: Path, value: object, context: str) -> Path:
    """Resolve one safe project-relative lineage path."""

    if not isinstance(value, str) or not value:
        raise RetrievalEvaluationSchemaError(f"{context} must be a non-empty path.")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise RetrievalEvaluationSchemaError(
            f"{context} must be a safe project-relative path."
        )
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise RetrievalEvaluationSchemaError(
            f"{context} resolves outside the project root."
        )
    return resolved


def _require_file_hash(path: Path, expected_hash: object, context: str) -> None:
    """Require one file and its frozen SHA-256 fingerprint."""

    if not path.is_file():
        raise RetrievalEvaluationIntegrityError(f"{context} is missing: {path}")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise RetrievalEvaluationSchemaError(f"{context} has an invalid SHA-256.")
    if calculate_file_sha256(path) != expected_hash:
        raise RetrievalEvaluationIntegrityError(f"{context} fingerprint differs.")


def load_retrieval_evaluation_v2_lineage(path: str | Path) -> dict[str, Any]:
    """Load the frozen v1-to-v2 reuse contract."""

    lineage_path = Path(path)
    try:
        raw = json.loads(lineage_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RetrievalEvaluationSchemaError(
            "Cannot read the v2 lineage contract as valid JSON."
        ) from error

    lineage = _require_mapping(raw, "V2 lineage contract")
    _require_exact_keys(lineage, LINEAGE_TOP_LEVEL_KEYS, "V2 lineage contract")
    if lineage["lineage_schema_version"] != LINEAGE_SCHEMA_VERSION:
        raise RetrievalEvaluationSchemaError("Unsupported v2 lineage schema version.")
    if lineage["decision_status"] != "frozen_before_v2_expanded_search":
        raise RetrievalEvaluationSchemaError("Unexpected v2 lineage decision status.")

    for key in (
        "judgement_reuse",
        "query_embedding_reuse",
        "source_v1",
        "target_v2",
    ):
        _require_mapping(lineage[key], key)
    return lineage


def validate_lineage_against_config(
    lineage: dict[str, Any],
    config: dict[str, Any],
    config_path: Path,
    root: Path,
) -> None:
    """Validate that the lineage and v2 configuration describe one run."""

    target = lineage["target_v2"]
    reuse = lineage["judgement_reuse"]
    matrix = lineage["query_embedding_reuse"]
    question_count = config["inputs"]["evaluation_questions"]["question_count"]
    config_relative_path = config_path.resolve().relative_to(root).as_posix()

    expected_target = {
        "candidate_depths": config["search"]["candidate_depths"],
        "configuration_id": config["configuration_id"],
        "configuration_relative_path": config_relative_path,
        "configuration_sha256": calculate_file_sha256(config_path),
        "diagnostic_tail_end_rank": config["search"]["judgement_pool_depth"],
        "diagnostic_tail_start_rank": config["search"]["maximum_candidate_depth"] + 1,
        "judgement_pool_depth": config["search"]["judgement_pool_depth"],
        "maximum_candidate_depth": config["search"]["maximum_candidate_depth"],
        "protocol_amendment_commit": "98f3e92",
    }
    if target != expected_target:
        raise RetrievalEvaluationIntegrityError(
            "The v2 lineage target differs from the v2 configuration."
        )

    existing_start = reuse.get("existing_rank_start")
    existing_end = reuse.get("existing_rank_end")
    new_start = reuse.get("new_rank_start")
    new_end = reuse.get("new_rank_end")
    if (existing_start, existing_end, new_start, new_end) != (1, 20, 21, 30):
        raise RetrievalEvaluationIntegrityError("Unexpected v2 judgement rank ranges.")
    if reuse.get("prefix_match_fields") != PREFIX_MATCH_FIELDS:
        raise RetrievalEvaluationIntegrityError("The frozen prefix fields changed.")
    if reuse.get("transfer_method") != "validated_question_id_chunk_id_pair":
        raise RetrievalEvaluationIntegrityError("Unexpected judgement transfer method.")
    if reuse.get("transferred_values") != ["relevance_grade", "judgement_note"]:
        raise RetrievalEvaluationIntegrityError("Unexpected transferred judgement fields.")

    expected_reused = question_count * (existing_end - existing_start + 1)
    expected_new = question_count * (new_end - new_start + 1)
    expected_total = question_count * config["search"]["judgement_pool_depth"]
    if reuse.get("reused_judgement_count") != expected_reused:
        raise RetrievalEvaluationIntegrityError("Unexpected reused judgement count.")
    if reuse.get("new_judgement_count") != expected_new:
        raise RetrievalEvaluationIntegrityError("Unexpected new judgement count.")
    if reuse.get("total_v2_candidate_count") != expected_total:
        raise RetrievalEvaluationIntegrityError("Unexpected total v2 candidate count.")

    if matrix.get("api_request_required") is not False:
        raise RetrievalEvaluationIntegrityError("V2 must not request new query embeddings.")
    if matrix.get("expected_matrix_shape") != [
        question_count,
        config["query_embedding"]["dimensions"],
    ]:
        raise RetrievalEvaluationIntegrityError("Unexpected reused matrix shape contract.")
    if matrix.get("expected_dtype") != config["query_embedding"]["dtype"]:
        raise RetrievalEvaluationIntegrityError("Unexpected reused matrix dtype contract.")


def load_reused_query_matrix(
    path: Path,
    *,
    expected_sha256: str,
    expected_shape: list[int],
    expected_dtype: str,
    norm_tolerance: float,
) -> np.ndarray:
    """Load and validate the exact frozen v1 query matrix without normalising it again."""

    _require_file_hash(path, expected_sha256, "Source query-embedding matrix")
    try:
        matrix = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise RetrievalEvaluationIntegrityError(
            "Cannot load the frozen source query matrix."
        ) from error
    if list(matrix.shape) != expected_shape:
        raise RetrievalEvaluationIntegrityError("Source query matrix shape differs.")
    if str(matrix.dtype) != expected_dtype:
        raise RetrievalEvaluationIntegrityError("Source query matrix dtype differs.")
    if not matrix.flags.c_contiguous:
        raise RetrievalEvaluationIntegrityError("Source query matrix is not C-contiguous.")
    if not np.isfinite(matrix).all():
        raise RetrievalEvaluationIntegrityError("Source query matrix is not finite.")
    norms = np.linalg.norm(matrix, axis=1)
    if np.max(np.abs(norms - 1.0)) > norm_tolerance:
        raise RetrievalEvaluationIntegrityError("Source query matrix is not L2-normalised.")
    return matrix


def migrate_reused_judgements(
    source_candidates: list[dict[str, Any]],
    source_judgements: list[dict[str, Any]],
    target_candidates: list[dict[str, Any]],
    target_judgements: list[dict[str, Any]],
    *,
    prefix_match_fields: list[str],
    reused_rank_end: int,
    expected_reused_count: int,
    expected_new_count: int,
) -> list[dict[str, Any]]:
    """Transfer verified grades to matching v2 pairs and leave only new ranks blank."""

    # The transfer join is the frozen question-chunk pair, never list position.
    source_candidates_by_pair = {
        (record["question_id"], record["chunk_id"]): record
        for record in source_candidates
    }
    if len(source_candidates_by_pair) != len(source_candidates):
        raise RetrievalEvaluationIntegrityError(
            "Source question-chunk pairs are not unique."
        )
    source_judgements_by_id = {
        record["judgement_id"]: record
        for record in source_judgements
    }
    if len(source_judgements_by_id) != len(source_judgements):
        raise RetrievalEvaluationIntegrityError("Source judgement IDs are not unique.")
    target_candidates_by_id = {
        record["judgement_id"]: record
        for record in target_candidates
    }
    if len(target_candidates_by_id) != len(target_candidates):
        raise RetrievalEvaluationIntegrityError("Target candidate IDs are not unique.")
    if len(target_judgements) != len(target_candidates):
        raise RetrievalEvaluationIntegrityError(
            "Target judgement and candidate counts differ."
        )

    migrated: list[dict[str, Any]] = []
    reused_count = 0
    new_count = 0
    for target_judgement in target_judgements:
        target_candidate = target_candidates_by_id.get(
            target_judgement["judgement_id"]
        )
        if target_candidate is None:
            raise RetrievalEvaluationIntegrityError(
                "A target judgement has no candidate."
            )
        output = dict(target_judgement)
        rank = target_candidate["rank"]
        if rank <= reused_rank_end:
            pair = (
                target_candidate["question_id"],
                target_candidate["chunk_id"],
            )
            source_candidate = source_candidates_by_pair.get(pair)
            if source_candidate is None:
                raise RetrievalEvaluationIntegrityError(
                    "A reused v2 question-chunk pair has no v1 candidate."
                )
            for field in prefix_match_fields:
                if source_candidate.get(field) != target_candidate.get(field):
                    raise RetrievalEvaluationIntegrityError(
                        f"V1 and v2 candidate prefixes differ on {field}."
                    )
            source_judgement = source_judgements_by_id.get(
                source_candidate["judgement_id"]
            )
            if source_judgement is None:
                raise RetrievalEvaluationIntegrityError(
                    "A reused v1 candidate has no completed judgement."
                )
            if (
                source_judgement["query_text"] != target_judgement["query_text"]
                or source_judgement["chunk_text"] != target_judgement["chunk_text"]
            ):
                raise RetrievalEvaluationIntegrityError(
                    "V1 and v2 judgement text differs for a reused pair."
                )
            grade = source_judgement["relevance_grade"]
            note = source_judgement["judgement_note"]
            if isinstance(grade, bool) or grade not in {0, 1, 2}:
                raise RetrievalEvaluationSchemaError(
                    "A reused v1 judgement is incomplete or invalid."
                )
            if grade in {1, 2} and (
                not isinstance(note, str) or not note.strip()
            ):
                raise RetrievalEvaluationSchemaError(
                    "A reused positive judgement has no evidence note."
                )
            if grade == 0 and note is not None:
                raise RetrievalEvaluationSchemaError(
                    "A reused grade-0 judgement must have a null note."
                )
            output["relevance_grade"] = grade
            output["judgement_note"] = note
            reused_count += 1
        else:
            if (
                target_judgement["relevance_grade"] is not None
                or target_judgement["judgement_note"] is not None
            ):
                raise RetrievalEvaluationIntegrityError(
                    "A new v2 judgement is unexpectedly completed."
                )
            new_count += 1
        migrated.append(output)

    if reused_count != expected_reused_count:
        raise RetrievalEvaluationIntegrityError("Reused judgement count differs.")
    if new_count != expected_new_count:
        raise RetrievalEvaluationIntegrityError("New judgement count differs.")
    if [record["judgement_id"] for record in migrated] != sorted(
        record["judgement_id"] for record in migrated
    ):
        raise RetrievalEvaluationIntegrityError(
            "Migrated judgements are not in blinded order."
        )
    return migrated


def _load_source_v1_evidence(
    lineage: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    """Validate and load every frozen v1 artifact needed by v2."""

    source = lineage["source_v1"]
    matrix_contract = lineage["query_embedding_reuse"]
    path_fields = {
        "config": ("configuration_relative_path", "configuration_sha256"),
        "matrix": (None, "source_sha256"),
        "pool": ("ranked_candidate_pool_relative_path", "ranked_candidate_pool_sha256"),
        "judgements": ("completed_judgement_relative_path", "completed_judgement_sha256"),
        "summary": ("preparation_summary_relative_path", "preparation_summary_sha256"),
        "metrics": ("initial_metrics_relative_path", "initial_metrics_sha256"),
        "table": ("initial_metrics_table_relative_path", "initial_metrics_table_sha256"),
    }
    paths: dict[str, Path] = {}
    for name, (path_key, hash_key) in path_fields.items():
        if name == "matrix":
            relative_path = matrix_contract.get("source_relative_path")
            expected_hash = matrix_contract.get(hash_key)
        else:
            relative_path = source.get(path_key)
            expected_hash = source.get(hash_key)
        path = _resolve_project_path(root, relative_path, f"source_v1.{name}")
        _require_file_hash(path, expected_hash, f"Frozen v1 {name}")
        paths[name] = path

    source_config = load_retrieval_evaluation_config(paths["config"])
    if source_config["configuration_id"] != source.get("configuration_id"):
        raise RetrievalEvaluationIntegrityError("The v1 configuration ID differs.")
    source_inputs = load_retrieval_evaluation_inputs(paths["config"], root)
    source_candidates = _read_json_lines(paths["pool"], "Frozen v1 candidate pool")
    _validate_candidate_pool(source_candidates, source_inputs)
    source_judgements = _read_json_lines(
        paths["judgements"],
        "Frozen v1 completed judgements",
    )
    _validate_completed_judgements(
        source_judgements,
        source_candidates,
        source_inputs,
    )
    if len(source_judgements) != source.get("completed_judgement_count"):
        raise RetrievalEvaluationIntegrityError("The v1 completed count differs.")

    source_summary = _read_json_object(paths["summary"], "Frozen v1 summary")
    if source_summary.get("relevance_judgement_template_sha256") != source.get(
        "relevance_judgement_template_sha256"
    ):
        raise RetrievalEvaluationIntegrityError(
            "The v1 empty judgement-template fingerprint differs."
        )
    source_metrics = _read_json_object(paths["metrics"], "Frozen v1 metrics")
    if source_metrics.get("candidate_range_adequacy_gate", {}).get("triggered") is not True:
        raise RetrievalEvaluationIntegrityError("The v1 adequacy gate did not trigger.")
    if source_metrics.get("selected_depth_k") is not None:
        raise RetrievalEvaluationIntegrityError("The v1 metrics selected a depth.")
    if source_metrics.get("relevance_judgements_sha256") != source.get(
        "completed_judgement_sha256"
    ):
        raise RetrievalEvaluationIntegrityError(
            "The v1 metrics reference different completed judgements."
        )
    if source_metrics.get("ranked_candidate_pool_sha256") != source.get(
        "ranked_candidate_pool_sha256"
    ):
        raise RetrievalEvaluationIntegrityError(
            "The v1 metrics reference a different candidate pool."
        )
    return {
        "candidates": source_candidates,
        "config": source_config,
        "inputs": source_inputs,
        "judgements": source_judgements,
        "matrix_path": paths["matrix"],
        "metrics": source_metrics,
        "paths": paths,
        "summary": source_summary,
    }


def prepare_retrieval_evaluation_v2(
    config_path: str | Path,
    lineage_path: str | Path,
    project_root: str | Path,
) -> RetrievalExpansionPreparationResult:
    """Prepare v2 by reusing frozen v1 evidence and searching locally to rank 30."""

    root = Path(project_root).resolve()
    resolved_config_path = Path(config_path)
    if not resolved_config_path.is_absolute():
        resolved_config_path = root / resolved_config_path
    resolved_lineage_path = Path(lineage_path)
    if not resolved_lineage_path.is_absolute():
        resolved_lineage_path = root / resolved_lineage_path

    config = load_retrieval_evaluation_config(resolved_config_path)
    lineage = load_retrieval_evaluation_v2_lineage(resolved_lineage_path)
    validate_lineage_against_config(
        lineage,
        config,
        resolved_config_path,
        root,
    )
    paths = _resolve_output_paths(config, root)
    preparation_paths = [
        paths["matrix"],
        paths["pool"],
        paths["judgements"],
        paths["summary"],
    ]
    existing = [path for path in preparation_paths if path.exists()]
    if existing:
        raise RetrievalEvaluationError(
            "V2 preparation output already exists; refusing to replace it."
        )
    if paths["metrics"].exists() or paths["table"].exists():
        raise RetrievalEvaluationError(
            "V2 metric output exists before preparation."
        )
    if len(set(preparation_paths)) != len(preparation_paths):
        raise RetrievalEvaluationIntegrityError("V2 output paths are not unique.")

    source = _load_source_v1_evidence(lineage, root)
    source_paths = set(source["paths"].values())
    if any(path in source_paths for path in preparation_paths):
        raise RetrievalEvaluationIntegrityError(
            "A v2 output path overlaps a frozen v1 artifact."
        )

    inputs = load_retrieval_evaluation_inputs(resolved_config_path, root)
    if inputs.config["inputs"] != source["config"]["inputs"]:
        raise RetrievalEvaluationIntegrityError("V1 and v2 retrieval inputs differ.")
    if config["inputs"]["embedding_index"]["faiss_index_sha256"] != lineage[
        "source_v1"
    ]["faiss_index_sha256"]:
        raise RetrievalEvaluationIntegrityError("The v2 FAISS index differs from v1.")
    if config["inputs"]["evaluation_questions"]["questions_sha256"] != lineage[
        "source_v1"
    ]["question_sha256"]:
        raise RetrievalEvaluationIntegrityError("The v2 questions differ from v1.")

    matrix_contract = lineage["query_embedding_reuse"]
    query_contract = config["query_embedding"]
    query_matrix = load_reused_query_matrix(
        source["matrix_path"],
        expected_sha256=matrix_contract["source_sha256"],
        expected_shape=matrix_contract["expected_matrix_shape"],
        expected_dtype=matrix_contract["expected_dtype"],
        norm_tolerance=query_contract["post_normalisation_norm_tolerance"],
    )
    scores, positions = search_faiss_pool(
        inputs.faiss_index,
        query_matrix,
        config["search"]["judgement_pool_depth"],
    )
    candidates = build_ranked_candidate_records(
        config["configuration_id"],
        inputs.question_records,
        inputs.chunk_records,
        inputs.chunk_token_counts,
        inputs.mapping_records,
        scores,
        positions,
    )
    blank_judgements = build_blinded_judgement_records(
        config["configuration_id"],
        inputs.question_records,
        inputs.chunk_records,
        candidates,
    )
    reuse = lineage["judgement_reuse"]
    judgements = migrate_reused_judgements(
        source["candidates"],
        source["judgements"],
        candidates,
        blank_judgements,
        prefix_match_fields=reuse["prefix_match_fields"],
        reused_rank_end=reuse["existing_rank_end"],
        expected_reused_count=reuse["reused_judgement_count"],
        expected_new_count=reuse["new_judgement_count"],
    )

    for path in preparation_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=root) as temporary_directory:
        temporary_root = Path(temporary_directory)
        temporary_matrix = temporary_root / "query-embeddings.npy"
        temporary_pool = temporary_root / "ranked-candidate-pool.jsonl"
        temporary_judgements = temporary_root / "relevance-judgements.jsonl"
        temporary_summary = temporary_root / "retrieval-depth-evaluation-summary.json"

        temporary_matrix.write_bytes(source["matrix_path"].read_bytes())
        temporary_pool.write_bytes(serialise_json_lines(candidates))
        temporary_judgements.write_bytes(serialise_json_lines(judgements))
        norms = np.linalg.norm(query_matrix, axis=1)
        summary = {
            "api_request_made": False,
            "candidate_count": len(candidates),
            "completed_reused_judgement_count": reuse["reused_judgement_count"],
            "configuration_id": config["configuration_id"],
            "configuration_relative_path": resolved_config_path.resolve().relative_to(root).as_posix(),
            "configuration_sha256": calculate_file_sha256(resolved_config_path),
            "faiss_index_sha256": config["inputs"]["embedding_index"]["faiss_index_sha256"],
            "ground_truth_accessed": False,
            "judgement_pool_depth": config["search"]["judgement_pool_depth"],
            "lineage_relative_path": resolved_lineage_path.resolve().relative_to(root).as_posix(),
            "lineage_sha256": calculate_file_sha256(resolved_lineage_path),
            "new_unjudged_count": reuse["new_judgement_count"],
            "prefix_match_passed": True,
            "query_embedding_reused": True,
            "question_count": len(inputs.question_records),
            "question_embedding_matrix_relative_path": paths["matrix"].relative_to(root).as_posix(),
            "question_embedding_matrix_sha256": calculate_file_sha256(temporary_matrix),
            "question_embedding_matrix_shape": list(query_matrix.shape),
            "question_embedding_norm_maximum": float(norms.max()),
            "question_embedding_norm_minimum": float(norms.min()),
            "question_embedding_token_count": source["summary"]["question_embedding_token_count"],
            "questions_sha256": config["inputs"]["evaluation_questions"]["questions_sha256"],
            "ranked_candidate_pool_relative_path": paths["pool"].relative_to(root).as_posix(),
            "ranked_candidate_pool_sha256": calculate_file_sha256(temporary_pool),
            "relevance_judgement_template_relative_path": paths["judgements"].relative_to(root).as_posix(),
            "relevance_judgement_template_sha256": calculate_file_sha256(temporary_judgements),
            "retrieval_preparation_summary_schema_version": RETRIEVAL_PREPARATION_SUMMARY_SCHEMA_VERSION,
            "selected_depth_k": None,
            "selection_status": "unresolved_pending_additional_relevance_judgements",
            "source_completed_judgements_sha256": lineage["source_v1"]["completed_judgement_sha256"],
            "source_configuration_id": lineage["source_v1"]["configuration_id"],
            "source_ranked_candidate_pool_sha256": lineage["source_v1"]["ranked_candidate_pool_sha256"],
        }
        temporary_summary.write_bytes(canonical_json_bytes(summary))

        if not np.array_equal(
            np.load(temporary_matrix, allow_pickle=False),
            query_matrix,
        ):
            raise RetrievalEvaluationIntegrityError(
                "Reused query matrix changed during staging."
            )
        if _read_json_lines(temporary_pool, "Staged v2 pool") != candidates:
            raise RetrievalEvaluationIntegrityError("V2 pool changed during staging.")
        if _read_json_lines(temporary_judgements, "Staged v2 judgements") != judgements:
            raise RetrievalEvaluationIntegrityError(
                "V2 judgements changed during staging."
            )
        if _read_json_object(temporary_summary, "Staged v2 summary") != summary:
            raise RetrievalEvaluationIntegrityError("V2 summary changed during staging.")

        matrix_sha256 = calculate_file_sha256(temporary_matrix)
        pool_sha256 = calculate_file_sha256(temporary_pool)
        judgement_sha256 = calculate_file_sha256(temporary_judgements)
        summary_sha256 = calculate_file_sha256(temporary_summary)
        os.replace(temporary_matrix, paths["matrix"])
        os.replace(temporary_pool, paths["pool"])
        os.replace(temporary_judgements, paths["judgements"])
        os.replace(temporary_summary, paths["summary"])

    return RetrievalExpansionPreparationResult(
        query_embedding_matrix_path=paths["matrix"],
        ranked_candidate_pool_path=paths["pool"],
        blinded_judgement_file_path=paths["judgements"],
        summary_path=paths["summary"],
        query_embedding_matrix_sha256=matrix_sha256,
        ranked_candidate_pool_sha256=pool_sha256,
        blinded_judgement_file_sha256=judgement_sha256,
        summary_sha256=summary_sha256,
        candidate_count=len(candidates),
        reused_judgement_count=reuse["reused_judgement_count"],
        new_judgement_count=reuse["new_judgement_count"],
    )


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the local v2 expansion command-line interface."""

    parser = argparse.ArgumentParser(
        description="Prepare the versioned retrieval-depth expansion."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/retrieval-evaluation-config-v2.json"),
    )
    parser.add_argument(
        "--lineage",
        type=Path,
        default=Path("configs/retrieval-evaluation-v2-lineage.json"),
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    return parser


def main() -> None:
    """Prepare v2 without an API call and report only safe metadata."""

    arguments = _build_argument_parser().parse_args()
    result = prepare_retrieval_evaluation_v2(
        arguments.config,
        arguments.lineage,
        arguments.project_root,
    )
    print("Retrieval-depth v2 preparation: PASSED")
    print("  Candidate pairs:", result.candidate_count)
    print("  Reused judgements:", result.reused_judgement_count)
    print("  New judgements:", result.new_judgement_count)
    print("  Query matrix SHA-256:", result.query_embedding_matrix_sha256)
    print("  Candidate pool SHA-256:", result.ranked_candidate_pool_sha256)
    print("  Judgement file SHA-256:", result.blinded_judgement_file_sha256)
    print("  Summary SHA-256:", result.summary_sha256)
    print("  API request made: False")
    print("  Ground truth accessed: False")


if __name__ == "__main__":
    main()
