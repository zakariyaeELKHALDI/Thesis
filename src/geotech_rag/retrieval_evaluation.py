"""Prepare and score a leakage-controlled retrieval-depth evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Protocol
import argparse
import csv
import io
import json
import math
import os

import faiss
import numpy as np
import tiktoken

from geotech_rag.embedding_index import (
    build_openai_embedding_client,
    load_embedding_index_config,
    load_validated_chunk_records,
    normalise_embedding_matrix,
    validate_raw_embedding_matrix,
)
from geotech_rag.evaluation_questions import load_evaluation_questions


RETRIEVAL_EVALUATION_CONFIG_SCHEMA_VERSION = "1.0"
RANKED_CANDIDATE_SCHEMA_VERSION = "1.0"
RELEVANCE_JUDGEMENT_SCHEMA_VERSION = "1.0"
RETRIEVAL_PREPARATION_SUMMARY_SCHEMA_VERSION = "1.0"
RETRIEVAL_METRICS_SCHEMA_VERSION = "1.0"

TOP_LEVEL_KEYS = {
    "candidate_range_adequacy_gate",
    "configuration_id",
    "decision_status",
    "inputs",
    "leakage_controls",
    "metrics",
    "outputs",
    "protocol_boundary",
    "query_embedding",
    "relevance_judgements",
    "retrieval_evaluation_config_schema_version",
    "search",
    "selection_rule",
}

QUESTION_INPUT_KEYS = {
    "configuration_relative_path",
    "configuration_sha256",
    "question_count",
    "question_projection_sha256",
    "questions_relative_path",
    "questions_sha256",
}

INDEX_INPUT_KEYS = {
    "chunk_count",
    "configuration_relative_path",
    "configuration_sha256",
    "embedding_matrix_relative_path",
    "embedding_matrix_sha256",
    "faiss_index_relative_path",
    "faiss_index_sha256",
    "index_mapping_relative_path",
    "index_mapping_sha256",
    "input_chunks_relative_path",
    "input_chunks_sha256",
    "summary_relative_path",
    "summary_sha256",
}

QUERY_EMBEDDING_KEYS = {
    "batch_count",
    "client_class",
    "client_package",
    "client_package_version",
    "dimensions",
    "dimensions_argument",
    "dtype",
    "expected_query_count",
    "finite_values_required",
    "memory_layout",
    "normalisation",
    "normalise_query_vectors",
    "ordered_batch_method",
    "post_normalisation_norm_tolerance",
    "requested_model",
    "token_encoding",
}

SEARCH_KEYS = {
    "backend_package",
    "backend_package_version",
    "candidate_depths",
    "exact_search",
    "index_class",
    "judgement_pool_depth",
    "mapping_position_field",
    "maximum_candidate_depth",
    "metric",
    "rank_origin",
    "reference_depth",
    "reference_depth_basis",
    "reference_package",
    "reference_package_version",
    "reference_signature",
    "score_order",
    "search_once_at_pool_depth_and_score_nested_prefixes",
    "similarity_interpretation",
}

RELEVANCE_JUDGEMENT_CONFIG_KEYS = {
    "assessor_count",
    "blinded_order_method",
    "ground_truth_answers_prohibited",
    "hidden_during_judgement",
    "judgement_context",
    "judgement_unit",
    "model_generated_answers_prohibited",
    "positive_grade_note_required",
    "relevance_grades",
}

METRIC_CONFIG_KEYS = {
    "conventional_recall_reported",
    "direct_evidence_grade",
    "hit_rate_denominator",
    "ndcg_gain_function",
    "primary_metric",
    "recall_omission_reason",
    "reported_metrics",
    "useful_evidence_minimum_grade",
}

ADEQUACY_GATE_KEYS = {
    "action_if_triggered",
    "pool_depth_is_not_a_candidate_depth",
    "trigger",
}

SELECTION_RULE_KEYS = {
    "applicable_only_if_candidate_range_adequacy_gate_passes",
    "mrr_ndcg_precision_are_reported_not_selection_tiebreakers",
    "ordered_steps",
    "selected_depth_k",
    "selection_status",
}

LEAKAGE_CONTROL_KEYS = {
    "answer_generation_allowed_during_retrieval_evaluation",
    "evaluation_questions_must_not_enter_corpus_index",
    "ground_truth_available_during_relevance_judgement",
    "ground_truth_available_during_retrieval",
    "only_question_and_retrieved_corpus_chunk_may_be_judged",
    "protocol_frozen_before_query_embedding_and_search",
    "question_embeddings_used_only_as_runtime_queries",
}

OUTPUT_KEYS = {
    "blinded_judgement_file_relative_path",
    "copyrighted_question_and_chunk_text_tracked_by_git",
    "metrics_relative_path",
    "metrics_table_relative_path",
    "notebook_relative_path",
    "query_embedding_matrix_relative_path",
    "ranked_candidate_pool_relative_path",
    "summary_relative_path",
    "tracked_metrics_must_not_contain_question_or_chunk_text",
}

MAPPING_KEYS = {
    "chunk_id",
    "index_mapping_schema_version",
    "index_position",
    "parent_record_id",
    "pdf_page_index",
    "pdf_page_number",
    "printed_page_number",
    "source_id",
}

CANDIDATE_KEYS = {
    "candidate_schema_version",
    "chunk_character_count",
    "chunk_id",
    "chunk_text_sha256",
    "chunk_token_count",
    "configuration_id",
    "index_position",
    "judgement_id",
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
}

JUDGEMENT_KEYS = {
    "chunk_text",
    "configuration_id",
    "judgement_id",
    "judgement_note",
    "judgement_schema_version",
    "query_text",
    "relevance_grade",
}


class RetrievalEvaluationError(RuntimeError):
    """Base error for retrieval-evaluation failures."""


class RetrievalEvaluationConfigError(RetrievalEvaluationError):
    """Raised when the retrieval-evaluation configuration is invalid."""


class RetrievalEvaluationIntegrityError(RetrievalEvaluationError):
    """Raised when a frozen input or generated artifact is inconsistent."""


class RetrievalEvaluationSchemaError(RetrievalEvaluationError):
    """Raised when a record violates a frozen schema."""


class RetrievalEvaluationLeakageError(RetrievalEvaluationError):
    """Raised when a retrieval-evaluation leakage boundary is violated."""


class EmbeddingClient(Protocol):
    """Minimal ordered document-embedding client used by this stage."""

    def embed_documents(self, texts: list[str]) -> Any:
        """Embed ordered texts and return one vector for each input."""


@dataclass(frozen=True)
class RetrievalEvaluationInputs:
    """Validated questions, chunks, mapping and production FAISS index."""

    config: dict[str, Any]
    embedding_config: dict[str, Any]
    question_records: tuple[dict[str, Any], ...]
    chunk_records: tuple[dict[str, Any], ...]
    chunk_token_counts: tuple[int, ...]
    mapping_records: tuple[dict[str, Any], ...]
    faiss_index: faiss.Index


@dataclass(frozen=True)
class RetrievalPreparationResult:
    """Paths, fingerprints and counts from preparation."""

    query_embedding_matrix_path: Path
    ranked_candidate_pool_path: Path
    blinded_judgement_file_path: Path
    summary_path: Path
    query_embedding_matrix_sha256: str
    ranked_candidate_pool_sha256: str
    blinded_judgement_file_sha256: str
    summary_sha256: str
    question_count: int
    pool_depth: int
    candidate_count: int


@dataclass(frozen=True)
class RetrievalMetricsResult:
    """Paths, fingerprints and selection status from scoring."""

    metrics_path: Path
    metrics_table_path: Path
    metrics_sha256: str
    metrics_table_sha256: str
    selected_depth_k: int | None
    candidate_range_adequacy_gate_triggered: bool
    completed_judgement_count: int


def calculate_file_sha256(path: Path) -> str:
    """Calculate a file fingerprint without changing the file."""

    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    """Return deterministic indented JSON bytes."""

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")


def serialise_json_lines(records: list[dict[str, Any]]) -> bytes:
    """Return deterministic compact JSONL bytes."""

    return "".join(
        json.dumps(
            record,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for record in records
    ).encode("utf-8")


def _require_mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RetrievalEvaluationConfigError(f"{context} must be an object.")
    return value


def _require_exact_keys(
    value: dict[str, Any],
    expected: set[str],
    context: str,
    *,
    error_class: type[RetrievalEvaluationError] = RetrievalEvaluationConfigError,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise error_class(
            f"{context} has invalid keys; missing={missing}, "
            f"unexpected={unexpected}."
        )


def _require_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise RetrievalEvaluationConfigError(
            f"{context} must be a non-empty string."
        )
    return value


def _require_integer(value: object, context: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RetrievalEvaluationConfigError(
            f"{context} must be an integer greater than or equal to {minimum}."
        )
    return value


def _require_boolean(value: object, context: str) -> bool:
    if not isinstance(value, bool):
        raise RetrievalEvaluationConfigError(f"{context} must be boolean.")
    return value


def _validate_sha256(value: object, context: str) -> str:
    text = _require_string(value, context)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise RetrievalEvaluationConfigError(
            f"{context} must be a lowercase SHA-256 value."
        )
    return text


def _validate_relative_path(value: object, context: str) -> Path:
    text = _require_string(value, context)
    path = Path(text)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != text:
        raise RetrievalEvaluationConfigError(
            f"{context} must be a normalised relative POSIX path."
        )
    return path


def _resolve_within_directory(
    root: Path,
    relative_path: str,
    allowed_relative_directory: str,
    context: str,
) -> Path:
    path = (root / relative_path).resolve()
    directory = (root / allowed_relative_directory).resolve()
    try:
        path.relative_to(directory)
    except ValueError as error:
        raise RetrievalEvaluationLeakageError(
            f"{context} must remain within {allowed_relative_directory}."
        ) from error
    return path


def _validate_input_section(inputs: dict[str, Any]) -> None:
    _require_exact_keys(
        inputs,
        {"embedding_index", "evaluation_questions"},
        "inputs",
    )
    questions = _require_mapping(
        inputs["evaluation_questions"],
        "inputs.evaluation_questions",
    )
    index = _require_mapping(inputs["embedding_index"], "inputs.embedding_index")
    _require_exact_keys(questions, QUESTION_INPUT_KEYS, "inputs.evaluation_questions")
    _require_exact_keys(index, INDEX_INPUT_KEYS, "inputs.embedding_index")

    for field_name in (
        "configuration_relative_path",
        "questions_relative_path",
    ):
        _validate_relative_path(
            questions[field_name],
            f"inputs.evaluation_questions.{field_name}",
        )
    for field_name in (
        "configuration_sha256",
        "questions_sha256",
        "question_projection_sha256",
    ):
        _validate_sha256(
            questions[field_name],
            f"inputs.evaluation_questions.{field_name}",
        )
    _require_integer(
        questions["question_count"],
        "inputs.evaluation_questions.question_count",
        1,
    )

    for field_name in (
        "configuration_relative_path",
        "summary_relative_path",
        "embedding_matrix_relative_path",
        "faiss_index_relative_path",
        "index_mapping_relative_path",
        "input_chunks_relative_path",
    ):
        _validate_relative_path(index[field_name], f"inputs.embedding_index.{field_name}")
    for field_name in (
        "configuration_sha256",
        "summary_sha256",
        "embedding_matrix_sha256",
        "faiss_index_sha256",
        "index_mapping_sha256",
        "input_chunks_sha256",
    ):
        _validate_sha256(index[field_name], f"inputs.embedding_index.{field_name}")
    _require_integer(index["chunk_count"], "inputs.embedding_index.chunk_count", 1)


def _validate_query_embedding(value: dict[str, Any]) -> None:
    _require_exact_keys(value, QUERY_EMBEDDING_KEYS, "query_embedding")
    for field_name in (
        "client_class",
        "client_package",
        "client_package_version",
        "requested_model",
        "token_encoding",
        "ordered_batch_method",
        "dtype",
        "memory_layout",
        "normalisation",
    ):
        _require_string(value[field_name], f"query_embedding.{field_name}")
    if value["dimensions_argument"] is not None:
        _require_integer(
            value["dimensions_argument"],
            "query_embedding.dimensions_argument",
            1,
        )
    _require_integer(value["dimensions"], "query_embedding.dimensions", 1)
    _require_integer(
        value["expected_query_count"],
        "query_embedding.expected_query_count",
        1,
    )
    if value["batch_count"] != 1:
        raise RetrievalEvaluationConfigError("query_embedding.batch_count must be 1.")
    if value["ordered_batch_method"] != "embed_documents":
        raise RetrievalEvaluationConfigError(
            "query_embedding.ordered_batch_method must be embed_documents."
        )
    if value["dtype"] != "float32" or value["memory_layout"] != "C_contiguous":
        raise RetrievalEvaluationConfigError(
            "Query vectors must use contiguous float32 representation."
        )
    for field_name in ("finite_values_required", "normalise_query_vectors"):
        if _require_boolean(value[field_name], f"query_embedding.{field_name}") is not True:
            raise RetrievalEvaluationConfigError(
                f"query_embedding.{field_name} must remain true."
            )
    tolerance = value["post_normalisation_norm_tolerance"]
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
        raise RetrievalEvaluationConfigError(
            "query_embedding.post_normalisation_norm_tolerance must be numeric."
        )
    if not math.isfinite(float(tolerance)) or float(tolerance) <= 0.0:
        raise RetrievalEvaluationConfigError(
            "Query norm tolerance must be positive and finite."
        )


def _validate_search(value: dict[str, Any]) -> None:
    _require_exact_keys(value, SEARCH_KEYS, "search")
    for field_name in (
        "backend_package",
        "backend_package_version",
        "index_class",
        "metric",
        "similarity_interpretation",
        "reference_depth_basis",
        "reference_package",
        "reference_package_version",
        "reference_signature",
        "mapping_position_field",
        "score_order",
    ):
        _require_string(value[field_name], f"search.{field_name}")

    raw_depths = value["candidate_depths"]
    if not isinstance(raw_depths, list) or not raw_depths:
        raise RetrievalEvaluationConfigError("search.candidate_depths must be non-empty.")
    depths = [
        _require_integer(depth, "search.candidate_depths item", 1)
        for depth in raw_depths
    ]
    if depths != sorted(set(depths)):
        raise RetrievalEvaluationConfigError(
            "search.candidate_depths must be unique and strictly increasing."
        )
    if value["maximum_candidate_depth"] != max(depths):
        raise RetrievalEvaluationConfigError(
            "search.maximum_candidate_depth does not match candidate_depths."
        )
    pool_depth = _require_integer(
        value["judgement_pool_depth"],
        "search.judgement_pool_depth",
        1,
    )
    if pool_depth <= max(depths):
        raise RetrievalEvaluationConfigError(
            "The judgement pool must exceed the maximum candidate depth."
        )
    if value["reference_depth"] not in depths:
        raise RetrievalEvaluationConfigError(
            "search.reference_depth must be one of the candidate depths."
        )
    if value["rank_origin"] != 1:
        raise RetrievalEvaluationConfigError("search.rank_origin must be 1.")
    if value["mapping_position_field"] != "index_position":
        raise RetrievalEvaluationConfigError(
            "search.mapping_position_field must be index_position."
        )
    if value["score_order"] != "descending":
        raise RetrievalEvaluationConfigError("search.score_order must be descending.")
    for field_name in (
        "exact_search",
        "search_once_at_pool_depth_and_score_nested_prefixes",
    ):
        if _require_boolean(value[field_name], f"search.{field_name}") is not True:
            raise RetrievalEvaluationConfigError(f"search.{field_name} must remain true.")


def _validate_relevance_judgements(value: dict[str, Any]) -> None:
    _require_exact_keys(
        value,
        RELEVANCE_JUDGEMENT_CONFIG_KEYS,
        "relevance_judgements",
    )
    if value["assessor_count"] != 1:
        raise RetrievalEvaluationConfigError(
            "relevance_judgements.assessor_count must be 1."
        )
    for field_name in ("judgement_unit", "blinded_order_method"):
        _require_string(value[field_name], f"relevance_judgements.{field_name}")
    grades = value["relevance_grades"]
    if not isinstance(grades, list) or len(grades) != 3:
        raise RetrievalEvaluationConfigError(
            "relevance_judgements.relevance_grades must contain three grades."
        )
    if [grade.get("grade") for grade in grades if isinstance(grade, dict)] != [0, 1, 2]:
        raise RetrievalEvaluationConfigError("Relevance grades must be exactly 0, 1 and 2.")
    for position, grade in enumerate(grades):
        mapping = _require_mapping(grade, f"relevance grade {position}")
        _require_exact_keys(mapping, {"definition", "grade", "label"}, f"relevance grade {position}")
        _require_string(mapping["label"], f"relevance grade {position}.label")
        _require_string(mapping["definition"], f"relevance grade {position}.definition")
    if value["judgement_context"] != ["question_text", "retrieved_chunk_text"]:
        raise RetrievalEvaluationConfigError("Unexpected relevance-judgement context.")
    if value["hidden_during_judgement"] != [
        "retrieval_rank",
        "similarity_score",
        "candidate_depth_membership",
    ]:
        raise RetrievalEvaluationConfigError("Unexpected relevance-judgement blinding.")
    for field_name in (
        "positive_grade_note_required",
        "ground_truth_answers_prohibited",
        "model_generated_answers_prohibited",
    ):
        if _require_boolean(value[field_name], f"relevance_judgements.{field_name}") is not True:
            raise RetrievalEvaluationConfigError(
                f"relevance_judgements.{field_name} must remain true."
            )


def _validate_metrics(value: dict[str, Any], question_count: int) -> None:
    _require_exact_keys(value, METRIC_CONFIG_KEYS, "metrics")
    _require_string(value["primary_metric"], "metrics.primary_metric")
    _require_string(value["ndcg_gain_function"], "metrics.ndcg_gain_function")
    _require_string(value["recall_omission_reason"], "metrics.recall_omission_reason")
    reported = value["reported_metrics"]
    if not isinstance(reported, list) or not reported or not all(
        isinstance(metric, str) and metric for metric in reported
    ):
        raise RetrievalEvaluationConfigError("metrics.reported_metrics is invalid.")
    if len(set(reported)) != len(reported):
        raise RetrievalEvaluationConfigError("metrics.reported_metrics contains duplicates.")
    if value["hit_rate_denominator"] != question_count:
        raise RetrievalEvaluationConfigError(
            "metrics.hit_rate_denominator must match the question count."
        )
    if value["direct_evidence_grade"] != 2:
        raise RetrievalEvaluationConfigError("metrics.direct_evidence_grade must be 2.")
    if value["useful_evidence_minimum_grade"] != 1:
        raise RetrievalEvaluationConfigError(
            "metrics.useful_evidence_minimum_grade must be 1."
        )
    if value["conventional_recall_reported"] is not False:
        raise RetrievalEvaluationConfigError(
            "Conventional recall must remain disabled for pooled judgements."
        )


def _validate_controls_and_outputs(config: dict[str, Any]) -> None:
    gate = _require_mapping(
        config["candidate_range_adequacy_gate"],
        "candidate_range_adequacy_gate",
    )
    selection = _require_mapping(config["selection_rule"], "selection_rule")
    leakage = _require_mapping(config["leakage_controls"], "leakage_controls")
    outputs = _require_mapping(config["outputs"], "outputs")
    _require_exact_keys(gate, ADEQUACY_GATE_KEYS, "candidate_range_adequacy_gate")
    _require_exact_keys(selection, SELECTION_RULE_KEYS, "selection_rule")
    _require_exact_keys(leakage, LEAKAGE_CONTROL_KEYS, "leakage_controls")
    _require_exact_keys(outputs, OUTPUT_KEYS, "outputs")

    _require_string(gate["trigger"], "candidate_range_adequacy_gate.trigger")
    _require_string(
        gate["action_if_triggered"],
        "candidate_range_adequacy_gate.action_if_triggered",
    )
    if gate["pool_depth_is_not_a_candidate_depth"] is not True:
        raise RetrievalEvaluationConfigError(
            "The pool depth must not become a candidate depth."
        )

    if selection["selected_depth_k"] is not None:
        raise RetrievalEvaluationConfigError(
            "selection_rule.selected_depth_k must remain unresolved."
        )
    if selection["selection_status"] != "unresolved_pending_relevance_judgements":
        raise RetrievalEvaluationConfigError("Unexpected selection status.")
    if selection["ordered_steps"] != [
        "retain_depths_with_maximum_direct_evidence_hit_count",
        "retain_depths_with_maximum_useful_evidence_hit_count",
        "select_smallest_remaining_depth",
    ]:
        raise RetrievalEvaluationConfigError("The frozen selection rule changed.")
    for field_name in (
        "applicable_only_if_candidate_range_adequacy_gate_passes",
        "mrr_ndcg_precision_are_reported_not_selection_tiebreakers",
    ):
        if selection[field_name] is not True:
            raise RetrievalEvaluationConfigError(f"selection_rule.{field_name} must be true.")

    expected_leakage = {
        "answer_generation_allowed_during_retrieval_evaluation": False,
        "evaluation_questions_must_not_enter_corpus_index": True,
        "ground_truth_available_during_relevance_judgement": False,
        "ground_truth_available_during_retrieval": False,
        "only_question_and_retrieved_corpus_chunk_may_be_judged": True,
        "protocol_frozen_before_query_embedding_and_search": True,
        "question_embeddings_used_only_as_runtime_queries": True,
    }
    if leakage != expected_leakage:
        raise RetrievalEvaluationLeakageError("The frozen leakage controls changed.")

    for field_name in OUTPUT_KEYS - {
        "copyrighted_question_and_chunk_text_tracked_by_git",
        "tracked_metrics_must_not_contain_question_or_chunk_text",
    }:
        _validate_relative_path(outputs[field_name], f"outputs.{field_name}")
    if outputs["copyrighted_question_and_chunk_text_tracked_by_git"] is not False:
        raise RetrievalEvaluationLeakageError(
            "Copyrighted question and chunk text cannot be marked for Git tracking."
        )
    if outputs["tracked_metrics_must_not_contain_question_or_chunk_text"] is not True:
        raise RetrievalEvaluationLeakageError(
            "Tracked metrics must remain free of question and chunk text."
        )


def load_retrieval_evaluation_config(path: str | Path) -> dict[str, Any]:
    """Load and strictly validate the frozen retrieval-evaluation contract."""

    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RetrievalEvaluationConfigError(
            "Cannot read the retrieval-evaluation configuration as valid JSON."
        ) from error
    config = _require_mapping(raw, "Retrieval-evaluation configuration")
    _require_exact_keys(config, TOP_LEVEL_KEYS, "Retrieval-evaluation configuration")
    if config["retrieval_evaluation_config_schema_version"] != (
        RETRIEVAL_EVALUATION_CONFIG_SCHEMA_VERSION
    ):
        raise RetrievalEvaluationConfigError(
            "Unsupported retrieval-evaluation configuration schema version."
        )
    _require_string(config["configuration_id"], "configuration_id")
    if config["decision_status"] != "protocol_frozen_before_query_embedding_and_search":
        raise RetrievalEvaluationConfigError("Unexpected decision status.")

    boundary = _require_mapping(config["protocol_boundary"], "protocol_boundary")
    expected_boundary = {
        "faiss_search_run": False,
        "ground_truth_accessed": False,
        "question_embeddings_generated": False,
        "relevance_judgements_created": False,
    }
    if boundary != expected_boundary:
        raise RetrievalEvaluationLeakageError(
            "The pre-retrieval protocol boundary must remain entirely false."
        )

    inputs = _require_mapping(config["inputs"], "inputs")
    _validate_input_section(inputs)
    questions = inputs["evaluation_questions"]
    query_embedding = _require_mapping(config["query_embedding"], "query_embedding")
    search = _require_mapping(config["search"], "search")
    judgements = _require_mapping(config["relevance_judgements"], "relevance_judgements")
    metrics = _require_mapping(config["metrics"], "metrics")
    _validate_query_embedding(query_embedding)
    _validate_search(search)
    _validate_relevance_judgements(judgements)
    _validate_metrics(metrics, questions["question_count"])
    _validate_controls_and_outputs(config)
    if query_embedding["expected_query_count"] != questions["question_count"]:
        raise RetrievalEvaluationConfigError(
            "Expected query count differs from the frozen question count."
        )
    if search["judgement_pool_depth"] > inputs["embedding_index"]["chunk_count"]:
        raise RetrievalEvaluationConfigError(
            "Judgement pool depth exceeds the corpus index size."
        )
    return config


def _read_json_object(path: Path, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RetrievalEvaluationIntegrityError(
            f"Cannot read {context} as valid JSON."
        ) from error
    if not isinstance(value, dict):
        raise RetrievalEvaluationSchemaError(f"{context} must be a JSON object.")
    return value


def _read_json_lines(path: Path, context: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise RetrievalEvaluationIntegrityError(f"Cannot read {context}.") from error
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line:
            raise RetrievalEvaluationSchemaError(
                f"{context} contains an empty line at {line_number}."
            )
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise RetrievalEvaluationSchemaError(
                f"{context} line {line_number} is not valid JSON."
            ) from error
        if not isinstance(record, dict):
            raise RetrievalEvaluationSchemaError(
                f"{context} line {line_number} must be an object."
            )
        records.append(record)
    return records


def _require_file_fingerprint(path: Path, expected_sha256: str, context: str) -> None:
    if not path.is_file():
        raise RetrievalEvaluationIntegrityError(f"{context} is missing: {path}.")
    if calculate_file_sha256(path) != expected_sha256:
        raise RetrievalEvaluationIntegrityError(f"{context} fingerprint differs.")


def _validate_mapping_records(
    mapping_records: list[dict[str, Any]],
    chunk_records: list[dict[str, Any]],
) -> None:
    if len(mapping_records) != len(chunk_records):
        raise RetrievalEvaluationIntegrityError(
            "Index mapping count differs from the chunk count."
        )
    provenance_fields = (
        "chunk_id",
        "parent_record_id",
        "pdf_page_index",
        "pdf_page_number",
        "printed_page_number",
        "source_id",
    )
    for position, (mapping, chunk) in enumerate(
        zip(mapping_records, chunk_records, strict=True)
    ):
        _require_exact_keys(
            mapping,
            MAPPING_KEYS,
            f"Index mapping record {position}",
            error_class=RetrievalEvaluationSchemaError,
        )
        if mapping["index_position"] != position:
            raise RetrievalEvaluationIntegrityError(
                f"Index mapping position {position} is not contiguous."
            )
        for field_name in provenance_fields:
            if mapping[field_name] != chunk[field_name]:
                raise RetrievalEvaluationIntegrityError(
                    f"Index mapping record {position} differs on {field_name}."
                )


def load_retrieval_evaluation_inputs(
    config_path: str | Path,
    project_root: str | Path,
) -> RetrievalEvaluationInputs:
    """Load every frozen input without embedding a query or searching FAISS."""

    root = Path(project_root).resolve()
    resolved_config_path = Path(config_path)
    if not resolved_config_path.is_absolute():
        resolved_config_path = root / resolved_config_path
    resolved_config_path = resolved_config_path.resolve()
    try:
        resolved_config_path.relative_to(root)
    except ValueError as error:
        raise RetrievalEvaluationConfigError(
            "The retrieval-evaluation config must remain within the project."
        ) from error
    config = load_retrieval_evaluation_config(resolved_config_path)

    question_input = config["inputs"]["evaluation_questions"]
    index_input = config["inputs"]["embedding_index"]
    question_config_path = _resolve_within_directory(
        root,
        question_input["configuration_relative_path"],
        "configs",
        "Evaluation-question configuration",
    )
    embedding_config_path = _resolve_within_directory(
        root,
        index_input["configuration_relative_path"],
        "configs",
        "Embedding-index configuration",
    )
    summary_path = _resolve_within_directory(
        root,
        index_input["summary_relative_path"],
        "data/processed/audit",
        "Embedding-index summary",
    )
    question_path = _resolve_within_directory(
        root,
        question_input["questions_relative_path"],
        "data/raw/evaluation/questions",
        "Evaluation-question file",
    )
    matrix_path = _resolve_within_directory(
        root,
        index_input["embedding_matrix_relative_path"],
        "data/processed/embeddings",
        "Corpus embedding matrix",
    )
    faiss_path = _resolve_within_directory(
        root,
        index_input["faiss_index_relative_path"],
        "data/processed/index",
        "FAISS index",
    )
    mapping_path = _resolve_within_directory(
        root,
        index_input["index_mapping_relative_path"],
        "data/processed/index",
        "Index mapping",
    )
    chunks_path = _resolve_within_directory(
        root,
        index_input["input_chunks_relative_path"],
        "data/processed/corpus",
        "Corpus chunks",
    )

    _require_file_fingerprint(
        question_config_path,
        question_input["configuration_sha256"],
        "Evaluation-question configuration",
    )
    _require_file_fingerprint(
        question_path,
        question_input["questions_sha256"],
        "Evaluation-question file",
    )
    _require_file_fingerprint(
        embedding_config_path,
        index_input["configuration_sha256"],
        "Embedding-index configuration",
    )
    _require_file_fingerprint(
        summary_path,
        index_input["summary_sha256"],
        "Embedding-index summary",
    )
    _require_file_fingerprint(
        matrix_path,
        index_input["embedding_matrix_sha256"],
        "Corpus embedding matrix",
    )
    _require_file_fingerprint(
        faiss_path,
        index_input["faiss_index_sha256"],
        "FAISS index",
    )
    _require_file_fingerprint(
        mapping_path,
        index_input["index_mapping_sha256"],
        "Index mapping",
    )
    _require_file_fingerprint(
        chunks_path,
        index_input["input_chunks_sha256"],
        "Corpus chunks",
    )

    question_dataset = load_evaluation_questions(
        question_config_path,
        project_root=root,
    )
    if question_dataset.question_path != question_path.resolve():
        raise RetrievalEvaluationIntegrityError(
            "Question loader resolved a different question file."
        )
    if question_dataset.questions_sha256 != question_input["questions_sha256"]:
        raise RetrievalEvaluationIntegrityError("Question fingerprint is inconsistent.")
    if question_dataset.question_projection_sha256 != (
        question_input["question_projection_sha256"]
    ):
        raise RetrievalEvaluationIntegrityError(
            "Question projection fingerprint is inconsistent."
        )
    if len(question_dataset.records) != question_input["question_count"]:
        raise RetrievalEvaluationIntegrityError("Question count is inconsistent.")

    embedding_config = load_embedding_index_config(embedding_config_path)
    chunk_records, token_counts = load_validated_chunk_records(
        embedding_config,
        root,
    )
    if len(chunk_records) != index_input["chunk_count"]:
        raise RetrievalEvaluationIntegrityError("Chunk count is inconsistent.")
    summary = _read_json_object(summary_path, "Embedding-index summary")
    if summary.get("faiss_index_sha256") != index_input["faiss_index_sha256"]:
        raise RetrievalEvaluationIntegrityError("Summary FAISS fingerprint is inconsistent.")
    if summary.get("index_mapping_sha256") != index_input["index_mapping_sha256"]:
        raise RetrievalEvaluationIntegrityError("Summary mapping fingerprint is inconsistent.")
    if summary.get("embedding_matrix_sha256") != index_input["embedding_matrix_sha256"]:
        raise RetrievalEvaluationIntegrityError("Summary matrix fingerprint is inconsistent.")

    mapping_records = _read_json_lines(mapping_path, "Index mapping")
    _validate_mapping_records(mapping_records, chunk_records)
    try:
        index = faiss.read_index(str(faiss_path))
    except RuntimeError as error:
        raise RetrievalEvaluationIntegrityError("Cannot load the production FAISS index.") from error
    search = config["search"]
    query = config["query_embedding"]
    if type(index).__name__ != search["index_class"]:
        raise RetrievalEvaluationIntegrityError("FAISS index class differs from the contract.")
    if index.d != query["dimensions"]:
        raise RetrievalEvaluationIntegrityError("FAISS dimensions differ from query dimensions.")
    if index.ntotal != index_input["chunk_count"]:
        raise RetrievalEvaluationIntegrityError("FAISS vector count differs from the contract.")

    embedding = embedding_config["embedding"]
    vectors = embedding_config["vector_representation"]
    faiss_config = embedding_config["faiss"]
    comparisons = {
        "client_package": embedding["client_package"],
        "client_package_version": embedding["client_package_version"],
        "client_class": embedding["client_class"],
        "requested_model": embedding["requested_model"],
        "dimensions": embedding["dimensions"],
        "dimensions_argument": embedding["dimensions_argument"],
        "token_encoding": embedding["token_encoding"],
        "dtype": vectors["dtype"],
        "memory_layout": vectors["memory_layout"],
        "normalisation": vectors["normalisation"],
        "normalise_query_vectors": vectors["normalise_query_vectors"],
        "post_normalisation_norm_tolerance": vectors[
            "post_normalisation_norm_tolerance"
        ],
    }
    for field_name, expected in comparisons.items():
        if query[field_name] != expected:
            raise RetrievalEvaluationIntegrityError(
                f"Query embedding contract differs on {field_name}."
            )
    search_comparisons = {
        "backend_package": faiss_config["package"],
        "backend_package_version": faiss_config["package_version"],
        "index_class": faiss_config["index_class"],
        "metric": faiss_config["metric"],
        "similarity_interpretation": faiss_config["similarity_interpretation"],
        "exact_search": not faiss_config["approximate_search"],
    }
    for field_name, expected in search_comparisons.items():
        if search[field_name] != expected:
            raise RetrievalEvaluationIntegrityError(
                f"Search contract differs on {field_name}."
            )

    return RetrievalEvaluationInputs(
        config=config,
        embedding_config=embedding_config,
        question_records=question_dataset.records,
        chunk_records=tuple(chunk_records),
        chunk_token_counts=tuple(token_counts),
        mapping_records=tuple(mapping_records),
        faiss_index=index,
    )


def judgement_id(configuration_id: str, question_id: str, chunk_id: str) -> str:
    """Build the deterministic identifier and blinded ordering key for one pair."""

    return sha256(
        "\x00".join((configuration_id, question_id, chunk_id)).encode("utf-8")
    ).hexdigest()


def search_faiss_pool(
    index: faiss.Index,
    normalised_queries: np.ndarray,
    pool_depth: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one exact pool-depth search and validate positions and score order."""

    if normalised_queries.ndim != 2 or normalised_queries.dtype != np.float32:
        raise RetrievalEvaluationIntegrityError(
            "FAISS query matrix must be a two-dimensional float32 matrix."
        )
    if not normalised_queries.flags.c_contiguous:
        raise RetrievalEvaluationIntegrityError("FAISS query matrix must be C-contiguous.")
    if pool_depth <= 0 or pool_depth > index.ntotal:
        raise RetrievalEvaluationIntegrityError("Invalid FAISS judgement pool depth.")
    scores, positions = index.search(normalised_queries, pool_depth)
    if scores.shape != positions.shape or scores.shape != (
        len(normalised_queries),
        pool_depth,
    ):
        raise RetrievalEvaluationIntegrityError("FAISS returned an unexpected result shape.")
    if not np.isfinite(scores).all():
        raise RetrievalEvaluationIntegrityError("FAISS returned a non-finite score.")
    if np.any(positions < 0) or np.any(positions >= index.ntotal):
        raise RetrievalEvaluationIntegrityError("FAISS returned an invalid index position.")
    if np.any(scores[:, :-1] < scores[:, 1:]):
        raise RetrievalEvaluationIntegrityError("FAISS scores are not in descending order.")
    for row in positions:
        if len(set(int(position) for position in row)) != pool_depth:
            raise RetrievalEvaluationIntegrityError(
                "FAISS returned duplicate positions for one question."
            )
    return scores, positions


def build_ranked_candidate_records(
    configuration_id: str,
    question_records: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    chunk_records: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    chunk_token_counts: tuple[int, ...] | list[int],
    mapping_records: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    scores: np.ndarray,
    positions: np.ndarray,
) -> list[dict[str, Any]]:
    """Join ranked positions to stable chunk provenance without copying text."""

    if scores.shape != positions.shape or scores.ndim != 2:
        raise RetrievalEvaluationIntegrityError("Ranked score and position matrices differ.")
    if scores.shape[0] != len(question_records):
        raise RetrievalEvaluationIntegrityError("Ranked rows differ from question count.")
    if not (
        len(chunk_records) == len(chunk_token_counts) == len(mapping_records)
    ):
        raise RetrievalEvaluationIntegrityError("Chunk, token and mapping counts differ.")

    records: list[dict[str, Any]] = []
    observed_pairs: set[tuple[str, str]] = set()
    for question_position, question in enumerate(question_records):
        if question["position"] != question_position:
            raise RetrievalEvaluationIntegrityError("Question positions are not contiguous.")
        for rank_offset in range(scores.shape[1]):
            rank = rank_offset + 1
            index_position = int(positions[question_position, rank_offset])
            if index_position < 0 or index_position >= len(chunk_records):
                raise RetrievalEvaluationIntegrityError("Candidate index position is invalid.")
            chunk = chunk_records[index_position]
            mapping = mapping_records[index_position]
            if mapping["index_position"] != index_position:
                raise RetrievalEvaluationIntegrityError("Candidate mapping order is invalid.")
            if mapping["chunk_id"] != chunk["chunk_id"]:
                raise RetrievalEvaluationIntegrityError("Candidate chunk mapping differs.")
            pair = (question["question_id"], chunk["chunk_id"])
            if pair in observed_pairs:
                raise RetrievalEvaluationIntegrityError("Duplicate question-chunk pair.")
            observed_pairs.add(pair)
            record = {
                "candidate_schema_version": RANKED_CANDIDATE_SCHEMA_VERSION,
                "chunk_character_count": chunk["character_count"],
                "chunk_id": chunk["chunk_id"],
                "chunk_text_sha256": sha256(chunk["text"].encode("utf-8")).hexdigest(),
                "chunk_token_count": chunk_token_counts[index_position],
                "configuration_id": configuration_id,
                "index_position": index_position,
                "judgement_id": judgement_id(
                    configuration_id,
                    question["question_id"],
                    chunk["chunk_id"],
                ),
                "parent_record_id": mapping["parent_record_id"],
                "pdf_page_index": mapping["pdf_page_index"],
                "pdf_page_number": mapping["pdf_page_number"],
                "printed_page_number": mapping["printed_page_number"],
                "query_text_sha256": question["query_text_sha256"],
                "question_id": question["question_id"],
                "question_position": question_position,
                "rank": rank,
                "similarity_score": float(scores[question_position, rank_offset]),
                "source_id": mapping["source_id"],
            }
            _require_exact_keys(
                record,
                CANDIDATE_KEYS,
                "Ranked candidate record",
                error_class=RetrievalEvaluationSchemaError,
            )
            records.append(record)
    return records


def build_blinded_judgement_records(
    configuration_id: str,
    question_records: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    chunk_records: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Create deterministic rank-blinded manual-judgement records."""

    questions_by_id = {record["question_id"]: record for record in question_records}
    chunks_by_id = {record["chunk_id"]: record for record in chunk_records}
    if len(questions_by_id) != len(question_records):
        raise RetrievalEvaluationIntegrityError("Question identifiers are not unique.")
    if len(chunks_by_id) != len(chunk_records):
        raise RetrievalEvaluationIntegrityError("Chunk identifiers are not unique.")

    blinded: list[dict[str, Any]] = []
    for candidate in candidate_records:
        _require_exact_keys(
            candidate,
            CANDIDATE_KEYS,
            "Ranked candidate record",
            error_class=RetrievalEvaluationSchemaError,
        )
        question = questions_by_id.get(candidate["question_id"])
        chunk = chunks_by_id.get(candidate["chunk_id"])
        if question is None or chunk is None:
            raise RetrievalEvaluationIntegrityError(
                "Candidate cannot be joined to question and chunk text."
            )
        expected_id = judgement_id(
            configuration_id,
            question["question_id"],
            chunk["chunk_id"],
        )
        if candidate["judgement_id"] != expected_id:
            raise RetrievalEvaluationIntegrityError("Candidate judgement ID differs.")
        record = {
            "chunk_text": chunk["text"],
            "configuration_id": configuration_id,
            "judgement_id": expected_id,
            "judgement_note": None,
            "judgement_schema_version": RELEVANCE_JUDGEMENT_SCHEMA_VERSION,
            "query_text": question["query_text"],
            "relevance_grade": None,
        }
        _require_exact_keys(
            record,
            JUDGEMENT_KEYS,
            "Blinded judgement record",
            error_class=RetrievalEvaluationSchemaError,
        )
        blinded.append(record)
    blinded.sort(key=lambda record: record["judgement_id"])
    if len({record["judgement_id"] for record in blinded}) != len(blinded):
        raise RetrievalEvaluationIntegrityError("Judgement identifiers are not unique.")
    return blinded


def _resolve_output_paths(config: dict[str, Any], root: Path) -> dict[str, Path]:
    outputs = config["outputs"]
    return {
        "matrix": _resolve_within_directory(
            root,
            outputs["query_embedding_matrix_relative_path"],
            "data/processed/evaluation/retrieval",
            "Query embedding output",
        ),
        "pool": _resolve_within_directory(
            root,
            outputs["ranked_candidate_pool_relative_path"],
            "data/processed/evaluation/retrieval",
            "Ranked pool output",
        ),
        "judgements": _resolve_within_directory(
            root,
            outputs["blinded_judgement_file_relative_path"],
            "data/raw/evaluation/retrieval-judgements",
            "Blinded judgement output",
        ),
        "summary": _resolve_within_directory(
            root,
            outputs["summary_relative_path"],
            "data/processed/audit",
            "Retrieval summary output",
        ),
        "metrics": _resolve_within_directory(
            root,
            outputs["metrics_relative_path"],
            "results/metrics",
            "Retrieval metrics output",
        ),
        "table": _resolve_within_directory(
            root,
            outputs["metrics_table_relative_path"],
            "results/tables",
            "Retrieval metrics table output",
        ),
    }


def _contains_completed_judgements(path: Path) -> bool:
    if not path.is_file():
        return False
    for record in _read_json_lines(path, "Existing relevance judgements"):
        if record.get("relevance_grade") is not None:
            return True
    return False


def prepare_retrieval_evaluation(
    config_path: str | Path,
    project_root: str | Path,
    *,
    embedding_client: EmbeddingClient | None = None,
    overwrite: bool = False,
) -> RetrievalPreparationResult:
    """Embed questions once, search at pool depth and write blinded evidence."""

    root = Path(project_root).resolve()
    resolved_config_path = Path(config_path)
    if not resolved_config_path.is_absolute():
        resolved_config_path = root / resolved_config_path
    config = load_retrieval_evaluation_config(resolved_config_path)
    paths = _resolve_output_paths(config, root)
    preparation_paths = [paths["matrix"], paths["pool"], paths["judgements"], paths["summary"]]
    existing = [path for path in preparation_paths if path.exists()]
    if existing and not overwrite:
        raise RetrievalEvaluationError(
            "Retrieval-evaluation preparation output already exists; "
            "select explicit overwrite to replace it."
        )
    if overwrite and _contains_completed_judgements(paths["judgements"]):
        raise RetrievalEvaluationError(
            "Refusing to overwrite a relevance-judgement file containing completed grades."
        )
    if overwrite and (paths["metrics"].exists() or paths["table"].exists()):
        raise RetrievalEvaluationError(
            "Refusing to replace preparation artifacts while scored metrics exist."
        )

    inputs = load_retrieval_evaluation_inputs(resolved_config_path, root)
    client = embedding_client
    if client is None:
        client = build_openai_embedding_client(inputs.embedding_config, root)
    query_texts = [record["query_text"] for record in inputs.question_records]
    try:
        raw_vectors = client.embed_documents(query_texts)
    except Exception as error:
        raise RetrievalEvaluationError("Question embedding request failed.") from error
    query_contract = config["query_embedding"]
    raw_matrix = validate_raw_embedding_matrix(
        raw_vectors,
        query_contract["expected_query_count"],
        query_contract["dimensions"],
    )
    normalised_matrix = normalise_embedding_matrix(
        raw_matrix,
        query_contract["post_normalisation_norm_tolerance"],
    )
    scores, positions = search_faiss_pool(
        inputs.faiss_index,
        normalised_matrix,
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
    judgements = build_blinded_judgement_records(
        config["configuration_id"],
        inputs.question_records,
        inputs.chunk_records,
        candidates,
    )

    for path in preparation_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=root) as temporary_directory:
        temporary_root = Path(temporary_directory)
        temporary_matrix = temporary_root / "query-embeddings.npy"
        temporary_pool = temporary_root / "ranked-candidate-pool.jsonl"
        temporary_judgements = temporary_root / "relevance-judgements.jsonl"
        temporary_summary = temporary_root / "retrieval-depth-evaluation-summary.json"
        np.save(temporary_matrix, normalised_matrix, allow_pickle=False)
        temporary_pool.write_bytes(serialise_json_lines(candidates))
        temporary_judgements.write_bytes(serialise_json_lines(judgements))

        encoding = tiktoken.get_encoding(query_contract["token_encoding"])
        query_token_counts = [len(encoding.encode(text)) for text in query_texts]
        norms = np.linalg.norm(normalised_matrix, axis=1)
        summary = {
            "candidate_count": len(candidates),
            "configuration_id": config["configuration_id"],
            "configuration_relative_path": resolved_config_path.resolve().relative_to(root).as_posix(),
            "configuration_sha256": calculate_file_sha256(resolved_config_path),
            "faiss_index_sha256": config["inputs"]["embedding_index"]["faiss_index_sha256"],
            "ground_truth_accessed": False,
            "judgement_pool_depth": config["search"]["judgement_pool_depth"],
            "question_count": len(inputs.question_records),
            "question_embedding_matrix_relative_path": paths["matrix"].relative_to(root).as_posix(),
            "question_embedding_matrix_sha256": calculate_file_sha256(temporary_matrix),
            "question_embedding_matrix_shape": list(normalised_matrix.shape),
            "question_embedding_norm_maximum": float(norms.max()),
            "question_embedding_norm_minimum": float(norms.min()),
            "question_embedding_token_count": sum(query_token_counts),
            "questions_sha256": config["inputs"]["evaluation_questions"]["questions_sha256"],
            "ranked_candidate_pool_relative_path": paths["pool"].relative_to(root).as_posix(),
            "ranked_candidate_pool_sha256": calculate_file_sha256(temporary_pool),
            "relevance_judgement_template_relative_path": paths["judgements"].relative_to(root).as_posix(),
            "relevance_judgement_template_sha256": calculate_file_sha256(temporary_judgements),
            "retrieval_preparation_summary_schema_version": RETRIEVAL_PREPARATION_SUMMARY_SCHEMA_VERSION,
            "selected_depth_k": None,
            "selection_status": "unresolved_pending_relevance_judgements",
        }
        temporary_summary.write_bytes(canonical_json_bytes(summary))

        reloaded_matrix = np.load(temporary_matrix, allow_pickle=False)
        if not np.array_equal(reloaded_matrix, normalised_matrix):
            raise RetrievalEvaluationIntegrityError("Query matrix changed after round trip.")
        if _read_json_lines(temporary_pool, "Staged candidate pool") != candidates:
            raise RetrievalEvaluationIntegrityError("Candidate pool changed after round trip.")
        if _read_json_lines(temporary_judgements, "Staged judgement file") != judgements:
            raise RetrievalEvaluationIntegrityError("Judgement file changed after round trip.")
        if _read_json_object(temporary_summary, "Staged retrieval summary") != summary:
            raise RetrievalEvaluationIntegrityError("Retrieval summary changed after round trip.")

        matrix_sha256 = calculate_file_sha256(temporary_matrix)
        pool_sha256 = calculate_file_sha256(temporary_pool)
        judgement_sha256 = calculate_file_sha256(temporary_judgements)
        summary_sha256 = calculate_file_sha256(temporary_summary)
        os.replace(temporary_matrix, paths["matrix"])
        os.replace(temporary_pool, paths["pool"])
        os.replace(temporary_judgements, paths["judgements"])
        os.replace(temporary_summary, paths["summary"])

    return RetrievalPreparationResult(
        query_embedding_matrix_path=paths["matrix"],
        ranked_candidate_pool_path=paths["pool"],
        blinded_judgement_file_path=paths["judgements"],
        summary_path=paths["summary"],
        query_embedding_matrix_sha256=matrix_sha256,
        ranked_candidate_pool_sha256=pool_sha256,
        blinded_judgement_file_sha256=judgement_sha256,
        summary_sha256=summary_sha256,
        question_count=len(inputs.question_records),
        pool_depth=config["search"]["judgement_pool_depth"],
        candidate_count=len(candidates),
    )


def _validate_candidate_pool(
    records: list[dict[str, Any]],
    inputs: RetrievalEvaluationInputs,
) -> None:
    config = inputs.config
    expected_count = len(inputs.question_records) * config["search"]["judgement_pool_depth"]
    if len(records) != expected_count:
        raise RetrievalEvaluationIntegrityError("Candidate pool count is inconsistent.")
    chunks = {record["chunk_id"]: record for record in inputs.chunk_records}
    expected_position = 0
    seen_ids: set[str] = set()
    for question_position, question in enumerate(inputs.question_records):
        for rank in range(1, config["search"]["judgement_pool_depth"] + 1):
            record = records[expected_position]
            expected_position += 1
            _require_exact_keys(
                record,
                CANDIDATE_KEYS,
                f"Candidate record {expected_position}",
                error_class=RetrievalEvaluationSchemaError,
            )
            if record["question_position"] != question_position or record["rank"] != rank:
                raise RetrievalEvaluationIntegrityError("Candidate pool order is inconsistent.")
            if record["question_id"] != question["question_id"]:
                raise RetrievalEvaluationIntegrityError("Candidate question ID is inconsistent.")
            if record["query_text_sha256"] != question["query_text_sha256"]:
                raise RetrievalEvaluationIntegrityError("Candidate query fingerprint differs.")
            chunk = chunks.get(record["chunk_id"])
            if chunk is None:
                raise RetrievalEvaluationIntegrityError("Candidate references an unknown chunk.")
            if record["index_position"] < 0 or record["index_position"] >= len(inputs.mapping_records):
                raise RetrievalEvaluationIntegrityError("Candidate index position is invalid.")
            mapping = inputs.mapping_records[record["index_position"]]
            if mapping["chunk_id"] != record["chunk_id"]:
                raise RetrievalEvaluationIntegrityError("Candidate mapping differs.")
            if record["chunk_text_sha256"] != sha256(chunk["text"].encode("utf-8")).hexdigest():
                raise RetrievalEvaluationIntegrityError("Candidate chunk fingerprint differs.")
            if record["chunk_token_count"] != inputs.chunk_token_counts[
                record["index_position"]
            ]:
                raise RetrievalEvaluationIntegrityError(
                    "Candidate chunk token count differs."
                )
            if record["chunk_character_count"] != chunk["character_count"]:
                raise RetrievalEvaluationIntegrityError(
                    "Candidate chunk character count differs."
                )
            for field_name in (
                "parent_record_id",
                "pdf_page_index",
                "pdf_page_number",
                "printed_page_number",
                "source_id",
            ):
                if record[field_name] != mapping[field_name]:
                    raise RetrievalEvaluationIntegrityError(
                        f"Candidate provenance differs on {field_name}."
                    )
            expected_id = judgement_id(
                config["configuration_id"],
                record["question_id"],
                record["chunk_id"],
            )
            if record["judgement_id"] != expected_id or expected_id in seen_ids:
                raise RetrievalEvaluationIntegrityError("Candidate judgement ID is invalid.")
            seen_ids.add(expected_id)
        question_records = records[
            question_position * config["search"]["judgement_pool_depth"] :
            (question_position + 1) * config["search"]["judgement_pool_depth"]
        ]
        scores = [record["similarity_score"] for record in question_records]
        if not all(isinstance(score, (int, float)) and math.isfinite(float(score)) for score in scores):
            raise RetrievalEvaluationSchemaError("Candidate score is not finite.")
        if any(first < second for first, second in zip(scores, scores[1:])):
            raise RetrievalEvaluationIntegrityError("Candidate scores are not descending.")


def _validate_completed_judgements(
    records: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    inputs: RetrievalEvaluationInputs,
) -> dict[str, int]:
    if len(records) != len(candidates):
        raise RetrievalEvaluationIntegrityError("Judgement count differs from candidate count.")
    record_ids = []
    for position, record in enumerate(records):
        if not isinstance(record, dict):
            raise RetrievalEvaluationSchemaError(
                f"Judgement record {position} must be an object."
            )
        record_id = record.get("judgement_id")
        if not isinstance(record_id, str) or not record_id:
            raise RetrievalEvaluationSchemaError(
                f"Judgement record {position} has no valid judgement ID."
            )
        record_ids.append(record_id)
    if record_ids != sorted(record_ids):
        raise RetrievalEvaluationIntegrityError("Judgement records are not in blinded order.")
    questions = {record["question_id"]: record for record in inputs.question_records}
    chunks = {record["chunk_id"]: record for record in inputs.chunk_records}
    expected_by_id = {record["judgement_id"]: record for record in candidates}
    grades: dict[str, int] = {}
    for position, record in enumerate(records):
        _require_exact_keys(
            record,
            JUDGEMENT_KEYS,
            f"Judgement record {position}",
            error_class=RetrievalEvaluationSchemaError,
        )
        if record["judgement_schema_version"] != RELEVANCE_JUDGEMENT_SCHEMA_VERSION:
            raise RetrievalEvaluationSchemaError("Unexpected judgement schema version.")
        candidate = expected_by_id.get(record["judgement_id"])
        if candidate is None or record["judgement_id"] in grades:
            raise RetrievalEvaluationIntegrityError("Judgement set is incomplete or duplicated.")
        if record["configuration_id"] != inputs.config["configuration_id"]:
            raise RetrievalEvaluationIntegrityError("Judgement configuration ID differs.")
        question = questions.get(candidate["question_id"])
        chunk = chunks.get(candidate["chunk_id"])
        if question is None or chunk is None:
            raise RetrievalEvaluationIntegrityError("Judgement references unknown text.")
        if record["query_text"] != question["query_text"]:
            raise RetrievalEvaluationIntegrityError("Judgement question text differs.")
        if record["chunk_text"] != chunk["text"]:
            raise RetrievalEvaluationIntegrityError("Judgement chunk text differs.")
        grade = record["relevance_grade"]
        if isinstance(grade, bool) or grade not in (0, 1, 2):
            raise RetrievalEvaluationSchemaError(
                "Every relevance grade must be completed with 0, 1 or 2."
            )
        note = record["judgement_note"]
        if grade > 0 and (not isinstance(note, str) or not note.strip()):
            raise RetrievalEvaluationSchemaError(
                "Grades 1 and 2 require a non-empty judgement note."
            )
        if grade == 0 and note is not None and not isinstance(note, str):
            raise RetrievalEvaluationSchemaError(
                "A grade-0 judgement note must be a string or null."
            )
        grades[record["judgement_id"]] = grade
    if set(grades) != set(expected_by_id):
        raise RetrievalEvaluationIntegrityError("Judgement identifiers differ from the pool.")
    return grades


def wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float]:
    """Return a two-sided Wilson score interval for one proportion."""

    if total <= 0 or successes < 0 or successes > total:
        raise RetrievalEvaluationError("Invalid Wilson interval counts.")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [max(0.0, centre - margin), min(1.0, centre + margin)]


def _discounted_cumulative_gain(grades: list[int]) -> float:
    return sum(
        (2**grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(grades, start=1)
    )


def calculate_retrieval_metrics(
    config: dict[str, Any],
    candidate_records: list[dict[str, Any]],
    grades_by_judgement_id: dict[str, int],
) -> dict[str, Any]:
    """Calculate frozen nested-depth metrics and apply the adequacy gate."""

    candidate_depths = config["search"]["candidate_depths"]
    pool_depth = config["search"]["judgement_pool_depth"]
    question_count = config["inputs"]["evaluation_questions"]["question_count"]
    grouped: dict[str, list[dict[str, Any]]] = {}
    question_order: list[str] = []
    for record in candidate_records:
        question_id = record["question_id"]
        if question_id not in grouped:
            grouped[question_id] = []
            question_order.append(question_id)
        enriched = dict(record)
        enriched["relevance_grade"] = grades_by_judgement_id[record["judgement_id"]]
        grouped[question_id].append(enriched)
    if len(grouped) != question_count:
        raise RetrievalEvaluationIntegrityError("Metric question count is inconsistent.")
    for question_id, records in grouped.items():
        if len(records) != pool_depth or [record["rank"] for record in records] != list(
            range(1, pool_depth + 1)
        ):
            raise RetrievalEvaluationIntegrityError(
                f"Candidate ranks are incomplete for {question_id}."
            )

    maximum_depth = max(candidate_depths)
    gate_questions: list[str] = []
    per_question: list[dict[str, Any]] = []
    for question_id in question_order:
        records = grouped[question_id]
        direct_ranks = [
            record["rank"] for record in records if record["relevance_grade"] == 2
        ]
        useful_ranks = [
            record["rank"] for record in records if record["relevance_grade"] >= 1
        ]
        if not any(rank <= maximum_depth for rank in direct_ranks) and any(
            maximum_depth < rank <= pool_depth for rank in direct_ranks
        ):
            gate_questions.append(question_id)
        per_question.append(
            {
                "direct_evidence_in_candidate_range": any(
                    rank <= maximum_depth for rank in direct_ranks
                ),
                "direct_evidence_in_diagnostic_tail": any(
                    maximum_depth < rank <= pool_depth for rank in direct_ranks
                ),
                "first_direct_evidence_rank": min(direct_ranks) if direct_ranks else None,
                "first_useful_evidence_rank": min(useful_ranks) if useful_ranks else None,
                "question_id": question_id,
            }
        )

    rows: list[dict[str, Any]] = []
    for depth in candidate_depths:
        direct_hits = 0
        useful_hits = 0
        reciprocal_ranks: list[float] = []
        ndcg_values: list[float] = []
        useful_precisions: list[float] = []
        context_token_totals: list[int] = []
        context_character_totals: list[int] = []
        for question_id in question_order:
            pool = grouped[question_id]
            prefix = pool[:depth]
            prefix_grades = [record["relevance_grade"] for record in prefix]
            direct_ranks = [
                record["rank"] for record in prefix if record["relevance_grade"] == 2
            ]
            direct_hit = bool(direct_ranks)
            useful_count = sum(grade >= 1 for grade in prefix_grades)
            direct_hits += int(direct_hit)
            useful_hits += int(useful_count > 0)
            reciprocal_ranks.append(1.0 / min(direct_ranks) if direct_hit else 0.0)
            actual_dcg = _discounted_cumulative_gain(prefix_grades)
            ideal_grades = sorted(
                (record["relevance_grade"] for record in pool),
                reverse=True,
            )[:depth]
            ideal_dcg = _discounted_cumulative_gain(ideal_grades)
            ndcg_values.append(actual_dcg / ideal_dcg if ideal_dcg else 0.0)
            useful_precisions.append(useful_count / depth)
            context_token_totals.append(
                sum(record["chunk_token_count"] for record in prefix)
            )
            context_character_totals.append(
                sum(record["chunk_character_count"] for record in prefix)
            )
        rows.append(
            {
                "depth_k": depth,
                "direct_evidence_hit_count": direct_hits,
                "direct_evidence_hit_rate": direct_hits / question_count,
                "direct_evidence_hit_rate_wilson_95_interval": wilson_interval(
                    direct_hits,
                    question_count,
                ),
                "direct_evidence_mrr": sum(reciprocal_ranks) / question_count,
                "graded_ndcg": sum(ndcg_values) / question_count,
                "mean_retrieved_context_characters": sum(context_character_totals)
                / question_count,
                "mean_retrieved_context_tokens": sum(context_token_totals)
                / question_count,
                "useful_evidence_hit_count": useful_hits,
                "useful_evidence_hit_rate": useful_hits / question_count,
                "useful_evidence_hit_rate_wilson_95_interval": wilson_interval(
                    useful_hits,
                    question_count,
                ),
                "useful_evidence_precision": sum(useful_precisions) / question_count,
            }
        )

    gate_triggered = bool(gate_questions)
    selected_depth: int | None = None
    if not gate_triggered:
        maximum_direct_hits = max(row["direct_evidence_hit_count"] for row in rows)
        eligible = [
            row for row in rows if row["direct_evidence_hit_count"] == maximum_direct_hits
        ]
        maximum_useful_hits = max(row["useful_evidence_hit_count"] for row in eligible)
        eligible = [
            row for row in eligible if row["useful_evidence_hit_count"] == maximum_useful_hits
        ]
        selected_depth = min(row["depth_k"] for row in eligible)

    return {
        "candidate_depth_metrics": rows,
        "candidate_range_adequacy_gate": {
            "diagnostic_tail_end_rank": pool_depth,
            "diagnostic_tail_start_rank": maximum_depth + 1,
            "question_ids_triggering_gate": gate_questions,
            "triggered": gate_triggered,
        },
        "per_question_diagnostics": per_question,
        "selected_depth_k": selected_depth,
        "selection_status": (
            "candidate_range_expansion_required"
            if gate_triggered
            else "selected_by_frozen_rule"
        ),
    }


def _metrics_table_bytes(rows: list[dict[str, Any]]) -> bytes:
    fieldnames = [
        "depth_k",
        "direct_evidence_hit_count",
        "direct_evidence_hit_rate",
        "direct_evidence_hit_rate_wilson_95_lower",
        "direct_evidence_hit_rate_wilson_95_upper",
        "useful_evidence_hit_count",
        "useful_evidence_hit_rate",
        "useful_evidence_hit_rate_wilson_95_lower",
        "useful_evidence_hit_rate_wilson_95_upper",
        "direct_evidence_mrr",
        "graded_ndcg",
        "useful_evidence_precision",
        "mean_retrieved_context_tokens",
        "mean_retrieved_context_characters",
    ]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        direct_interval = row["direct_evidence_hit_rate_wilson_95_interval"]
        useful_interval = row["useful_evidence_hit_rate_wilson_95_interval"]
        writer.writerow(
            {
                **{
                    key: value
                    for key, value in row.items()
                    if key not in {
                        "direct_evidence_hit_rate_wilson_95_interval",
                        "useful_evidence_hit_rate_wilson_95_interval",
                    }
                },
                "direct_evidence_hit_rate_wilson_95_lower": direct_interval[0],
                "direct_evidence_hit_rate_wilson_95_upper": direct_interval[1],
                "useful_evidence_hit_rate_wilson_95_lower": useful_interval[0],
                "useful_evidence_hit_rate_wilson_95_upper": useful_interval[1],
            }
        )
    return output.getvalue().encode("utf-8")


def score_retrieval_judgements(
    config_path: str | Path,
    project_root: str | Path,
    *,
    overwrite: bool = False,
) -> RetrievalMetricsResult:
    """Validate completed blinded judgements and write text-free metrics."""

    root = Path(project_root).resolve()
    resolved_config_path = Path(config_path)
    if not resolved_config_path.is_absolute():
        resolved_config_path = root / resolved_config_path
    config = load_retrieval_evaluation_config(resolved_config_path)
    paths = _resolve_output_paths(config, root)
    if (paths["metrics"].exists() or paths["table"].exists()) and not overwrite:
        raise RetrievalEvaluationError(
            "Retrieval metric output already exists; select explicit overwrite to replace it."
        )
    inputs = load_retrieval_evaluation_inputs(resolved_config_path, root)
    if not paths["summary"].is_file():
        raise RetrievalEvaluationIntegrityError(
            f"Retrieval preparation summary is missing: {paths['summary']}."
        )
    summary = _read_json_object(paths["summary"], "Retrieval preparation summary")
    if summary.get("configuration_sha256") != calculate_file_sha256(resolved_config_path):
        raise RetrievalEvaluationIntegrityError(
            "Retrieval preparation summary references a different configuration."
        )
    if summary.get("retrieval_preparation_summary_schema_version") != (
        RETRIEVAL_PREPARATION_SUMMARY_SCHEMA_VERSION
    ):
        raise RetrievalEvaluationSchemaError(
            "Unexpected retrieval preparation summary schema version."
        )
    if summary.get("question_count") != len(inputs.question_records):
        raise RetrievalEvaluationIntegrityError(
            "Retrieval preparation question count differs."
        )
    if summary.get("candidate_count") != (
        len(inputs.question_records) * config["search"]["judgement_pool_depth"]
    ):
        raise RetrievalEvaluationIntegrityError(
            "Retrieval preparation candidate count differs."
        )
    if summary.get("questions_sha256") != config["inputs"][
        "evaluation_questions"
    ]["questions_sha256"]:
        raise RetrievalEvaluationIntegrityError(
            "Retrieval preparation question fingerprint differs."
        )
    if not paths["matrix"].is_file():
        raise RetrievalEvaluationIntegrityError(
            f"Question embedding matrix is missing: {paths['matrix']}."
        )
    if summary.get("question_embedding_matrix_sha256") != calculate_file_sha256(
        paths["matrix"]
    ):
        raise RetrievalEvaluationIntegrityError(
            "Question embedding matrix fingerprint differs."
        )
    if not paths["pool"].is_file() or not paths["judgements"].is_file():
        raise RetrievalEvaluationIntegrityError(
            "Candidate pool or completed judgement file is missing."
        )
    if summary.get("ranked_candidate_pool_sha256") != calculate_file_sha256(paths["pool"]):
        raise RetrievalEvaluationIntegrityError("Candidate pool fingerprint differs.")
    candidates = _read_json_lines(paths["pool"], "Ranked candidate pool")
    _validate_candidate_pool(candidates, inputs)
    judgements = _read_json_lines(paths["judgements"], "Relevance judgements")
    grades = _validate_completed_judgements(judgements, candidates, inputs)
    calculated = calculate_retrieval_metrics(config, candidates, grades)
    judgement_sha256 = calculate_file_sha256(paths["judgements"])
    metrics = {
        **calculated,
        "candidate_count": len(candidates),
        "completed_judgement_count": len(judgements),
        "configuration_id": config["configuration_id"],
        "configuration_relative_path": resolved_config_path.resolve().relative_to(root).as_posix(),
        "configuration_sha256": calculate_file_sha256(resolved_config_path),
        "ground_truth_accessed": False,
        "question_count": len(inputs.question_records),
        "questions_sha256": config["inputs"]["evaluation_questions"]["questions_sha256"],
        "ranked_candidate_pool_sha256": calculate_file_sha256(paths["pool"]),
        "relevance_judgements_sha256": judgement_sha256,
        "retrieval_metrics_schema_version": RETRIEVAL_METRICS_SCHEMA_VERSION,
    }
    metrics_bytes = canonical_json_bytes(metrics)
    table_bytes = _metrics_table_bytes(calculated["candidate_depth_metrics"])
    paths["metrics"].parent.mkdir(parents=True, exist_ok=True)
    paths["table"].parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=root) as temporary_directory:
        temporary_root = Path(temporary_directory)
        temporary_metrics = temporary_root / "retrieval-depth-evaluation.json"
        temporary_table = temporary_root / "retrieval-depth-metrics.csv"
        temporary_metrics.write_bytes(metrics_bytes)
        temporary_table.write_bytes(table_bytes)
        if _read_json_object(temporary_metrics, "Staged metrics") != metrics:
            raise RetrievalEvaluationIntegrityError("Retrieval metrics changed after round trip.")
        metrics_sha256 = calculate_file_sha256(temporary_metrics)
        table_sha256 = calculate_file_sha256(temporary_table)
        os.replace(temporary_metrics, paths["metrics"])
        os.replace(temporary_table, paths["table"])
    return RetrievalMetricsResult(
        metrics_path=paths["metrics"],
        metrics_table_path=paths["table"],
        metrics_sha256=metrics_sha256,
        metrics_table_sha256=table_sha256,
        selected_depth_k=calculated["selected_depth_k"],
        candidate_range_adequacy_gate_triggered=calculated[
            "candidate_range_adequacy_gate"
        ]["triggered"],
        completed_judgement_count=len(judgements),
    )


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare blinded retrieval-depth evidence or score completed "
            "relevance judgements."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/retrieval-evaluation-config.json"),
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--score-judgements",
        action="store_true",
        help="Score completed local relevance judgements without an API call.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace the output for the selected operation.",
    )
    return parser


def main() -> None:
    arguments = _build_argument_parser().parse_args()
    if arguments.score_judgements:
        result = score_retrieval_judgements(
            arguments.config,
            arguments.project_root,
            overwrite=arguments.overwrite,
        )
        print("Retrieval-depth scoring: PASSED")
        print("  Completed judgements:", result.completed_judgement_count)
        print(
            "  Candidate-range adequacy gate triggered:",
            result.candidate_range_adequacy_gate_triggered,
        )
        print("  Selected retrieval depth k:", result.selected_depth_k)
        print("  Metrics:", result.metrics_path)
        print("  Metrics SHA-256:", result.metrics_sha256)
        print("  Metrics table:", result.metrics_table_path)
        print("  Metrics table SHA-256:", result.metrics_table_sha256)
        print("  Ground truth accessed: False")
        return

    result = prepare_retrieval_evaluation(
        arguments.config,
        arguments.project_root,
        overwrite=arguments.overwrite,
    )
    print("Retrieval-evaluation preparation: PASSED")
    print("  Questions:", result.question_count)
    print("  Judgement pool depth:", result.pool_depth)
    print("  Candidate pairs:", result.candidate_count)
    print("  Query embedding matrix:", result.query_embedding_matrix_path)
    print("  Query matrix SHA-256:", result.query_embedding_matrix_sha256)
    print("  Ranked candidate pool:", result.ranked_candidate_pool_path)
    print("  Candidate pool SHA-256:", result.ranked_candidate_pool_sha256)
    print("  Blinded judgement file:", result.blinded_judgement_file_path)
    print("  Judgement template SHA-256:", result.blinded_judgement_file_sha256)
    print("  Summary:", result.summary_path)
    print("  Summary SHA-256:", result.summary_sha256)
    print("  Ground truth accessed: False")
    print("  Selected retrieval depth k: unresolved")


if __name__ == "__main__":
    main()
