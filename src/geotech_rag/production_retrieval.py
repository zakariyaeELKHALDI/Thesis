"""Load and search the frozen production retrieval configuration."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

import faiss
import numpy as np
import tiktoken

from geotech_rag.corpus_manifest import calculate_file_sha256
from geotech_rag.embedding_index import build_openai_embedding_client


PRODUCTION_RETRIEVAL_CONFIG_SCHEMA_VERSION = "1.0"
RETRIEVAL_RESULT_SCHEMA_VERSION = "1.0"

TOP_LEVEL_KEYS = {
    "configuration_id",
    "context_assembly_boundary",
    "decision_status",
    "experimental_controls",
    "inputs",
    "integrity_controls",
    "leakage_controls",
    "production_retrieval_config_schema_version",
    "query_embedding",
    "request_policy",
    "returned_record_contract",
    "search",
    "selection_evidence",
}
CONTEXT_KEYS = {
    "all_retrieved_records_available_to_context_assembly",
    "context_format_not_selected_by_this_configuration",
    "retrieved_record_count",
    "silent_filtering_prohibited",
    "silent_rank_truncation_prohibited",
    "silent_reranking_prohibited",
    "transformation_requires_separate_versioned_configuration",
}
EXPERIMENTAL_CONTROL_KEYS = {
    "benchmark_contexts_must_be_identical_across_generator_models",
    "benchmark_query_embeddings_reused_across_generator_models",
    "corpus_index_fixed_across_generator_models",
    "only_generator_model_may_change_in_main_model_comparison",
    "retrieval_depth_fixed_across_generator_models",
    "retrieval_settings_fixed_across_generator_models",
}
INPUT_KEYS = {
    "chunk_embeddings",
    "chunk_records",
    "chunking_configuration",
    "embedding_index_configuration",
    "embedding_index_summary",
    "faiss_index",
    "index_mapping",
}
CHUNK_EMBEDDING_KEYS = {
    "dtype",
    "relative_path",
    "sha256",
    "shape",
}
CHUNK_RECORD_KEYS = {
    "record_count",
    "relative_path",
    "schema_version",
    "sha256",
}
CONFIG_REFERENCE_KEYS = {"relative_path", "sha256"}
EMBEDDING_CONFIG_REFERENCE_KEYS = {
    "configuration_id",
    "relative_path",
    "sha256",
}
SUMMARY_REFERENCE_KEYS = {
    "relative_path",
    "schema_version",
    "sha256",
}
FAISS_INPUT_KEYS = {
    "dimensions",
    "index_class",
    "relative_path",
    "sha256",
    "total_vectors",
}
MAPPING_INPUT_KEYS = {
    "record_count",
    "relative_path",
    "schema_version",
    "sha256",
}
INTEGRITY_CONTROL_KEYS = {
    "all_input_fingerprints_required_before_search",
    "chunk_and_mapping_identifiers_must_match",
    "chunk_count_must_equal_index_total",
    "embedding_dimensions_must_equal_index_dimensions",
    "faiss_index_class_must_match",
    "finite_query_vector_values_required",
    "index_mapping_position_must_equal_record_offset",
    "query_norm_tolerance",
    "query_vectors_must_be_c_contiguous_float32",
    "zero_length_query_vectors_prohibited",
}
LEAKAGE_CONTROL_KEYS = {
    "answer_keys_available_during_retrieval",
    "ground_truth_answers_available_during_retrieval",
    "relevance_grades_used_by_production_search",
    "relevance_notes_used_by_production_search",
    "retrieval_search_uses_only_query_embedding_and_frozen_index",
}
QUERY_EMBEDDING_KEYS = {
    "benchmark_mode",
    "common_vector_contract",
    "interactive_mode",
}
BENCHMARK_MODE_KEYS = {
    "api_request_required",
    "enabled",
    "expected_query_count",
    "perform_faiss_search_from_frozen_matrix",
    "query_embedding_matrix_relative_path",
    "query_embedding_matrix_sha256",
    "query_embedding_matrix_shape",
    "question_configuration_relative_path",
    "question_configuration_sha256",
    "question_order_source",
    "question_projection_sha256",
    "questions_relative_path",
    "questions_sha256",
    "reuse_ranked_candidate_pool_as_runtime_input",
}
COMMON_VECTOR_KEYS = {
    "dimensions",
    "dtype",
    "finite_values_required",
    "memory_layout",
    "normalisation",
    "normalise_query_vectors",
    "post_normalisation_norm_tolerance",
}
INTERACTIVE_MODE_KEYS = {
    "api_key_environment_variable",
    "api_key_file_relative_path",
    "api_request_required",
    "client_class",
    "client_package",
    "client_package_version",
    "dimensions_argument",
    "enabled",
    "inputs_per_request",
    "request_method",
    "requested_model",
    "response_embedding_fingerprint_recorded_per_run",
    "response_model_alias_not_assumed_stable",
    "token_encoding",
}
REQUEST_POLICY_KEYS = {
    "check_embedding_context_length",
    "honour_server_retry_after",
    "max_retries",
    "maximum_inputs_per_request",
    "maximum_tokens_per_input",
    "retry_max_seconds",
    "retry_min_seconds",
    "show_progress_bar",
    "skip_empty",
    "tiktoken_enabled",
    "tiktoken_model_name",
    "timeout_seconds",
}
RETURNED_RECORD_KEYS = {
    "chunk_record_join_key",
    "chunk_text_loaded_from_frozen_chunk_records",
    "index_mapping_fields",
    "rank_field",
    "rank_origin",
    "record_schema_version",
    "similarity_score_field",
    "source_provenance_required",
}
SEARCH_KEYS = {
    "backend_package",
    "backend_package_version",
    "exact_search",
    "index_class",
    "mapping_position_field",
    "metric",
    "preserve_faiss_returned_order",
    "rank_origin",
    "retrieval_depth_k",
    "score_order",
    "similarity_interpretation",
}
SELECTION_EVIDENCE_KEYS = {
    "candidate_range_adequacy_gate_triggered",
    "completed_judgement_count",
    "completion_audit_relative_path",
    "completion_audit_sha256",
    "decision_document_commit",
    "decision_document_relative_path",
    "decision_document_sha256",
    "direct_evidence_hit_count_at_selected_depth",
    "evaluation_result_commit",
    "further_depth_expansion_required",
    "metrics_relative_path",
    "metrics_sha256",
    "metrics_table_relative_path",
    "metrics_table_sha256",
    "question_count",
    "scoring_audit_relative_path",
    "scoring_audit_sha256",
    "selected_depth_k",
    "selection_status",
    "useful_evidence_hit_count_at_selected_depth",
    "v3_configuration_relative_path",
    "v3_configuration_sha256",
}
MAPPING_FIELDS = {
    "chunk_id",
    "index_mapping_schema_version",
    "index_position",
    "parent_record_id",
    "pdf_page_index",
    "pdf_page_number",
    "printed_page_number",
    "source_id",
}


class ProductionRetrievalError(RuntimeError):
    """Base exception for production retrieval failures."""


class ProductionRetrievalConfigError(ProductionRetrievalError):
    """Raised when the frozen production configuration is invalid."""


class ProductionRetrievalIntegrityError(ProductionRetrievalError):
    """Raised when a configured artifact fails validation."""


class QueryEmbeddingClient(Protocol):
    """Minimum interface required for one interactive query request."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed one ordered list of texts."""


@dataclass(frozen=True)
class RetrievalResult:
    """One query result without retaining the private query text."""

    mode: str
    query_identifier: str
    query_embedding_sha256: str
    query_token_count: int | None
    api_request_made: bool
    records: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible copy of the retrieval result."""

        return {
            "api_request_made": self.api_request_made,
            "mode": self.mode,
            "query_embedding_sha256": self.query_embedding_sha256,
            "query_identifier": self.query_identifier,
            "query_token_count": self.query_token_count,
            "records": [dict(record) for record in self.records],
        }


def _require_mapping(value: Any, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProductionRetrievalConfigError(
            f"{description} must be a JSON object."
        )
    return value


def _require_exact_keys(
    value: dict[str, Any],
    expected: set[str],
    description: str,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ProductionRetrievalConfigError(
            f"{description} has missing fields {missing} and "
            f"unexpected fields {unexpected}."
        )


def _require_bool(value: Any, description: str) -> bool:
    if type(value) is not bool:
        raise ProductionRetrievalConfigError(
            f"{description} must be a boolean."
        )
    return value


def _require_int(
    value: Any,
    description: str,
    *,
    minimum: int = 0,
) -> int:
    if type(value) is not int or value < minimum:
        raise ProductionRetrievalConfigError(
            f"{description} must be an integer >= {minimum}."
        )
    return value


def _require_number(
    value: Any,
    description: str,
    *,
    minimum: float = 0.0,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProductionRetrievalConfigError(
            f"{description} must be numeric."
        )
    numeric = float(value)
    if not np.isfinite(numeric) or numeric < minimum:
        raise ProductionRetrievalConfigError(
            f"{description} must be finite and >= {minimum}."
        )
    return numeric


def _require_string(value: Any, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProductionRetrievalConfigError(
            f"{description} must be a non-empty string."
        )
    return value


def _validate_sha256(value: Any, description: str) -> str:
    text = _require_string(value, description)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ProductionRetrievalConfigError(
            f"{description} must be a lowercase SHA-256 fingerprint."
        )
    return text


def _validate_relative_path(value: Any, description: str) -> str:
    text = _require_string(value, description)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or "\\" in text
        or path.as_posix() != text
    ):
        raise ProductionRetrievalConfigError(
            f"{description} must be a normalised repository-relative path."
        )
    return text


def _validate_reference(
    value: Any,
    expected_keys: set[str],
    description: str,
) -> dict[str, Any]:
    record = _require_mapping(value, description)
    _require_exact_keys(record, expected_keys, description)
    _validate_relative_path(record["relative_path"], f"{description} path")
    _validate_sha256(record["sha256"], f"{description} SHA-256")
    return record


def _require_all_true(value: dict[str, Any], description: str) -> None:
    for key, item in value.items():
        if _require_bool(item, f"{description}.{key}") is not True:
            raise ProductionRetrievalConfigError(
                f"{description}.{key} must remain true."
            )


def _load_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProductionRetrievalIntegrityError(
            f"Cannot load {description}: {path}."
        ) from error
    if not isinstance(value, dict):
        raise ProductionRetrievalIntegrityError(
            f"{description} must contain a JSON object."
        )
    return value


def _read_json_lines(path: Path, description: str) -> list[dict[str, Any]]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ProductionRetrievalIntegrityError(
            f"Cannot read {description}: {path}."
        ) from error
    if not payload.endswith(b"\n"):
        raise ProductionRetrievalIntegrityError(
            f"{description} must end with a newline."
        )
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        try:
            record = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ProductionRetrievalIntegrityError(
                f"Invalid {description} record at line {line_number}."
            ) from error
        if not isinstance(record, dict):
            raise ProductionRetrievalIntegrityError(
                f"{description} line {line_number} is not an object."
            )
        records.append(record)
    return records


def _installed_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError as error:
        raise ProductionRetrievalIntegrityError(
            f"Required package is not installed: {package_name}."
        ) from error


def load_production_retrieval_config(path: str | Path) -> dict[str, Any]:
    """Load and strictly validate the frozen production contract."""

    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProductionRetrievalConfigError(
            f"Cannot load production retrieval configuration: {config_path}."
        ) from error
    config = _require_mapping(raw, "Production retrieval configuration")
    _require_exact_keys(config, TOP_LEVEL_KEYS, "Production retrieval configuration")

    if config["production_retrieval_config_schema_version"] != (
        PRODUCTION_RETRIEVAL_CONFIG_SCHEMA_VERSION
    ):
        raise ProductionRetrievalConfigError(
            "Unsupported production retrieval configuration schema."
        )
    _require_string(config["configuration_id"], "Configuration ID")
    if config["decision_status"] != (
        "frozen_before_production_retriever_implementation"
    ):
        raise ProductionRetrievalConfigError("Unexpected decision status.")

    context = _require_mapping(
        config["context_assembly_boundary"], "Context assembly boundary"
    )
    _require_exact_keys(context, CONTEXT_KEYS, "Context assembly boundary")
    _require_all_true(
        {key: value for key, value in context.items() if key != "retrieved_record_count"},
        "context_assembly_boundary",
    )
    context_count = _require_int(
        context["retrieved_record_count"], "Retrieved record count", minimum=1
    )

    controls = _require_mapping(
        config["experimental_controls"], "Experimental controls"
    )
    _require_exact_keys(controls, EXPERIMENTAL_CONTROL_KEYS, "Experimental controls")
    _require_all_true(controls, "experimental_controls")

    inputs = _require_mapping(config["inputs"], "Inputs")
    _require_exact_keys(inputs, INPUT_KEYS, "Inputs")
    chunk_embeddings = _validate_reference(
        inputs["chunk_embeddings"], CHUNK_EMBEDDING_KEYS, "Chunk embeddings"
    )
    if chunk_embeddings["dtype"] != "float32":
        raise ProductionRetrievalConfigError("Chunk embeddings must be float32.")
    shape = chunk_embeddings["shape"]
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(type(item) is not int or item <= 0 for item in shape)
    ):
        raise ProductionRetrievalConfigError("Invalid chunk-embedding shape.")

    chunk_records = _validate_reference(
        inputs["chunk_records"], CHUNK_RECORD_KEYS, "Chunk records"
    )
    chunk_count = _require_int(
        chunk_records["record_count"], "Chunk-record count", minimum=1
    )
    _require_string(chunk_records["schema_version"], "Chunk schema version")
    _validate_reference(
        inputs["chunking_configuration"],
        CONFIG_REFERENCE_KEYS,
        "Chunking configuration",
    )
    embedding_reference = _validate_reference(
        inputs["embedding_index_configuration"],
        EMBEDDING_CONFIG_REFERENCE_KEYS,
        "Embedding-index configuration",
    )
    _require_string(
        embedding_reference["configuration_id"],
        "Embedding-index configuration ID",
    )
    summary_reference = _validate_reference(
        inputs["embedding_index_summary"],
        SUMMARY_REFERENCE_KEYS,
        "Embedding-index summary",
    )
    _require_string(summary_reference["schema_version"], "Summary schema version")
    faiss_input = _validate_reference(
        inputs["faiss_index"], FAISS_INPUT_KEYS, "FAISS index"
    )
    dimensions = _require_int(
        faiss_input["dimensions"], "FAISS dimensions", minimum=1
    )
    total_vectors = _require_int(
        faiss_input["total_vectors"], "FAISS vector count", minimum=1
    )
    if faiss_input["index_class"] != "IndexFlatIP":
        raise ProductionRetrievalConfigError("Only IndexFlatIP is supported.")
    mapping_input = _validate_reference(
        inputs["index_mapping"], MAPPING_INPUT_KEYS, "Index mapping"
    )
    mapping_count = _require_int(
        mapping_input["record_count"], "Mapping record count", minimum=1
    )
    _require_string(mapping_input["schema_version"], "Mapping schema version")
    if not (
        shape == [chunk_count, dimensions]
        and chunk_count == mapping_count == total_vectors
    ):
        raise ProductionRetrievalConfigError(
            "Configured chunk, embedding, mapping and index counts differ."
        )

    integrity = _require_mapping(config["integrity_controls"], "Integrity controls")
    _require_exact_keys(integrity, INTEGRITY_CONTROL_KEYS, "Integrity controls")
    _require_all_true(
        {key: value for key, value in integrity.items() if key != "query_norm_tolerance"},
        "integrity_controls",
    )
    tolerance = _require_number(
        integrity["query_norm_tolerance"], "Query norm tolerance", minimum=0.0
    )
    if tolerance <= 0.0:
        raise ProductionRetrievalConfigError("Query norm tolerance must be positive.")

    leakage = _require_mapping(config["leakage_controls"], "Leakage controls")
    _require_exact_keys(leakage, LEAKAGE_CONTROL_KEYS, "Leakage controls")
    if any(
        _require_bool(leakage[key], f"leakage_controls.{key}")
        for key in (
            "answer_keys_available_during_retrieval",
            "ground_truth_answers_available_during_retrieval",
            "relevance_grades_used_by_production_search",
            "relevance_notes_used_by_production_search",
        )
    ):
        raise ProductionRetrievalConfigError("A leakage prohibition was disabled.")
    if leakage["retrieval_search_uses_only_query_embedding_and_frozen_index"] is not True:
        raise ProductionRetrievalConfigError("Unexpected retrieval search inputs.")

    query = _require_mapping(config["query_embedding"], "Query embedding")
    _require_exact_keys(query, QUERY_EMBEDDING_KEYS, "Query embedding")
    benchmark = _require_mapping(query["benchmark_mode"], "Benchmark mode")
    _require_exact_keys(benchmark, BENCHMARK_MODE_KEYS, "Benchmark mode")
    if not (
        benchmark["enabled"] is True
        and benchmark["api_request_required"] is False
        and benchmark["perform_faiss_search_from_frozen_matrix"] is True
        and benchmark["reuse_ranked_candidate_pool_as_runtime_input"] is False
    ):
        raise ProductionRetrievalConfigError("Unexpected benchmark query policy.")
    query_count = _require_int(
        benchmark["expected_query_count"], "Benchmark query count", minimum=1
    )
    for key in (
        "query_embedding_matrix_relative_path",
        "question_configuration_relative_path",
        "questions_relative_path",
    ):
        _validate_relative_path(benchmark[key], f"benchmark_mode.{key}")
    for key in (
        "query_embedding_matrix_sha256",
        "question_configuration_sha256",
        "question_projection_sha256",
        "questions_sha256",
    ):
        _validate_sha256(benchmark[key], f"benchmark_mode.{key}")
    matrix_shape = benchmark["query_embedding_matrix_shape"]
    if matrix_shape != [query_count, dimensions]:
        raise ProductionRetrievalConfigError("Invalid benchmark matrix shape.")
    _require_string(benchmark["question_order_source"], "Question order source")

    vector = _require_mapping(query["common_vector_contract"], "Vector contract")
    _require_exact_keys(vector, COMMON_VECTOR_KEYS, "Vector contract")
    if not (
        vector["dimensions"] == dimensions
        and vector["dtype"] == "float32"
        and vector["memory_layout"] == "C_contiguous"
        and vector["normalisation"] == "explicit_l2"
        and vector["finite_values_required"] is True
        and vector["normalise_query_vectors"] is True
        and float(vector["post_normalisation_norm_tolerance"]) == tolerance
    ):
        raise ProductionRetrievalConfigError("Unexpected query-vector contract.")

    interactive = _require_mapping(query["interactive_mode"], "Interactive mode")
    _require_exact_keys(interactive, INTERACTIVE_MODE_KEYS, "Interactive mode")
    if not (
        interactive["enabled"] is True
        and interactive["api_request_required"] is True
        and interactive["inputs_per_request"] == 1
        and interactive["request_method"] == "embed_documents"
        and interactive["response_embedding_fingerprint_recorded_per_run"] is True
        and interactive["response_model_alias_not_assumed_stable"] is True
    ):
        raise ProductionRetrievalConfigError("Unexpected interactive query policy.")
    for key in (
        "api_key_environment_variable",
        "api_key_file_relative_path",
        "client_class",
        "client_package",
        "client_package_version",
        "requested_model",
        "token_encoding",
    ):
        _require_string(interactive[key], f"interactive_mode.{key}")
    _validate_relative_path(
        interactive["api_key_file_relative_path"], "Interactive API-key path"
    )
    if interactive["dimensions_argument"] is not None:
        raise ProductionRetrievalConfigError("Dimensions argument must remain null.")

    policy = _require_mapping(config["request_policy"], "Request policy")
    _require_exact_keys(policy, REQUEST_POLICY_KEYS, "Request policy")
    for key in (
        "check_embedding_context_length",
        "honour_server_retry_after",
        "tiktoken_enabled",
    ):
        if _require_bool(policy[key], f"request_policy.{key}") is not True:
            raise ProductionRetrievalConfigError(f"request_policy.{key} must be true.")
    for key in ("show_progress_bar", "skip_empty"):
        if _require_bool(policy[key], f"request_policy.{key}") is not False:
            raise ProductionRetrievalConfigError(f"request_policy.{key} must be false.")
    for key in (
        "max_retries",
        "maximum_inputs_per_request",
        "maximum_tokens_per_input",
        "retry_max_seconds",
        "retry_min_seconds",
    ):
        _require_int(policy[key], f"request_policy.{key}", minimum=0)
    _require_number(policy["timeout_seconds"], "Request timeout", minimum=0.0)
    _require_string(policy["tiktoken_model_name"], "Tiktoken model name")
    if policy["maximum_inputs_per_request"] != 1:
        raise ProductionRetrievalConfigError("Interactive requests must contain one input.")

    returned = _require_mapping(config["returned_record_contract"], "Returned record contract")
    _require_exact_keys(returned, RETURNED_RECORD_KEYS, "Returned record contract")
    if set(returned["index_mapping_fields"]) != MAPPING_FIELDS:
        raise ProductionRetrievalConfigError("Unexpected returned mapping fields.")
    if not (
        returned["chunk_record_join_key"] == "chunk_id"
        and returned["chunk_text_loaded_from_frozen_chunk_records"] is True
        and returned["rank_field"] == "retrieval_rank"
        and returned["rank_origin"] == 1
        and returned["record_schema_version"] == RETRIEVAL_RESULT_SCHEMA_VERSION
        and returned["similarity_score_field"] == "similarity_score"
        and returned["source_provenance_required"] is True
    ):
        raise ProductionRetrievalConfigError("Unexpected returned-record contract.")

    search = _require_mapping(config["search"], "Search")
    _require_exact_keys(search, SEARCH_KEYS, "Search")
    depth = _require_int(search["retrieval_depth_k"], "Retrieval depth", minimum=1)
    if not (
        search["backend_package"] == "faiss-cpu"
        and search["exact_search"] is True
        and search["index_class"] == "IndexFlatIP"
        and search["mapping_position_field"] == "index_position"
        and search["metric"] == "inner_product"
        and search["preserve_faiss_returned_order"] is True
        and search["rank_origin"] == 1
        and search["score_order"] == "descending"
        and search["similarity_interpretation"]
        == "cosine_similarity_for_l2_normalised_vectors"
    ):
        raise ProductionRetrievalConfigError("Unexpected search contract.")
    _require_string(search["backend_package_version"], "FAISS package version")
    if depth != context_count or depth > total_vectors:
        raise ProductionRetrievalConfigError("Retrieval-depth fields are inconsistent.")

    selection = _require_mapping(config["selection_evidence"], "Selection evidence")
    _require_exact_keys(selection, SELECTION_EVIDENCE_KEYS, "Selection evidence")
    if not (
        selection["selected_depth_k"] == depth
        and selection["selection_status"] == "selected_by_frozen_rule"
        and selection["candidate_range_adequacy_gate_triggered"] is False
        and selection["further_depth_expansion_required"] is False
        and selection["completed_judgement_count"] > 0
        and selection["question_count"] == query_count
        and selection["direct_evidence_hit_count_at_selected_depth"] <= query_count
        and selection["useful_evidence_hit_count_at_selected_depth"] <= query_count
    ):
        raise ProductionRetrievalConfigError("Unexpected selection evidence.")
    for key in (
        "completion_audit_relative_path",
        "decision_document_relative_path",
        "metrics_relative_path",
        "metrics_table_relative_path",
        "scoring_audit_relative_path",
        "v3_configuration_relative_path",
    ):
        _validate_relative_path(selection[key], f"selection_evidence.{key}")
    for key in (
        "completion_audit_sha256",
        "decision_document_sha256",
        "metrics_sha256",
        "metrics_table_sha256",
        "scoring_audit_sha256",
        "v3_configuration_sha256",
    ):
        _validate_sha256(selection[key], f"selection_evidence.{key}")
    for key in ("decision_document_commit", "evaluation_result_commit"):
        commit = _require_string(selection[key], f"selection_evidence.{key}")
        if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
            raise ProductionRetrievalConfigError(f"Invalid Git commit in {key}.")

    return config


def _resolve_artifact(
    root: Path,
    relative_path: str,
    description: str,
) -> Path:
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ProductionRetrievalIntegrityError(
            f"{description} escapes the project root."
        ) from error
    if not path.is_file():
        raise ProductionRetrievalIntegrityError(
            f"Missing {description}: {path}."
        )
    return path


def _verify_artifact(
    root: Path,
    relative_path: str,
    expected_sha256: str,
    description: str,
) -> Path:
    path = _resolve_artifact(root, relative_path, description)
    if calculate_file_sha256(path) != expected_sha256:
        raise ProductionRetrievalIntegrityError(
            f"{description} fingerprint differs from the frozen contract."
        )
    return path


def _validate_package_versions(config: dict[str, Any]) -> None:
    search = config["search"]
    interactive = config["query_embedding"]["interactive_mode"]
    for package_name, expected in (
        (search["backend_package"], search["backend_package_version"]),
        (interactive["client_package"], interactive["client_package_version"]),
    ):
        actual = _installed_version(package_name)
        if actual != expected:
            raise ProductionRetrievalIntegrityError(
                f"Package {package_name} has version {actual}; expected {expected}."
            )


class ProductionRetriever:
    """Validated exact retriever for benchmark and interactive query modes."""

    def __init__(
        self,
        *,
        config: dict[str, Any],
        project_root: Path,
        chunks: list[dict[str, Any]],
        mapping: list[dict[str, Any]],
        index: Any,
        benchmark_question_ids: list[str],
        benchmark_query_matrix: np.ndarray,
        embedding_index_config: dict[str, Any],
    ) -> None:
        self.config = config
        self.project_root = project_root
        self._chunks = tuple(chunks)
        self._mapping = tuple(mapping)
        self._index = index
        self._benchmark_question_ids = tuple(benchmark_question_ids)
        self._benchmark_positions = {
            question_id: position
            for position, question_id in enumerate(benchmark_question_ids)
        }
        self._benchmark_query_matrix = benchmark_query_matrix
        self._embedding_index_config = embedding_index_config

    @property
    def retrieval_depth_k(self) -> int:
        """Return the frozen production retrieval depth."""

        return int(self.config["search"]["retrieval_depth_k"])

    @property
    def benchmark_question_ids(self) -> tuple[str, ...]:
        """Return the frozen benchmark order without question text."""

        return self._benchmark_question_ids

    @property
    def chunk_count(self) -> int:
        """Return the validated indexed chunk count."""

        return len(self._chunks)

    def _prepare_query_vector(
        self,
        vector: Any,
        *,
        already_normalised: bool,
    ) -> np.ndarray:
        dimensions = self.config["query_embedding"]["common_vector_contract"][
            "dimensions"
        ]
        try:
            matrix = np.asarray(vector, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise ProductionRetrievalIntegrityError(
                "Query embedding cannot be converted to float32."
            ) from error
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.shape != (1, dimensions):
            raise ProductionRetrievalIntegrityError(
                f"Query embedding must have shape (1, {dimensions})."
            )
        matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        if not np.isfinite(matrix).all():
            raise ProductionRetrievalIntegrityError(
                "Query embedding contains non-finite values."
            )
        norm = float(np.linalg.norm(matrix[0]))
        if norm == 0.0:
            raise ProductionRetrievalIntegrityError(
                "Query embedding has zero length."
            )
        tolerance = self.config["integrity_controls"]["query_norm_tolerance"]
        if already_normalised:
            if not np.isclose(norm, 1.0, atol=tolerance, rtol=0.0):
                raise ProductionRetrievalIntegrityError(
                    "Frozen benchmark query embedding is not L2-normalised."
                )
        else:
            faiss.normalize_L2(matrix)
            normalised_norm = float(np.linalg.norm(matrix[0]))
            if not np.isclose(
                normalised_norm,
                1.0,
                atol=tolerance,
                rtol=0.0,
            ):
                raise ProductionRetrievalIntegrityError(
                    "Interactive query normalisation failed."
                )
        return matrix

    def _search(
        self,
        matrix: np.ndarray,
        *,
        mode: str,
        query_identifier: str,
        query_token_count: int | None,
        api_request_made: bool,
    ) -> RetrievalResult:
        depth = self.retrieval_depth_k
        scores, positions = self._index.search(matrix, depth)
        if scores.shape != (1, depth) or positions.shape != (1, depth):
            raise ProductionRetrievalIntegrityError(
                "FAISS returned an unexpected result shape."
            )
        if not np.isfinite(scores).all():
            raise ProductionRetrievalIntegrityError(
                "FAISS returned a non-finite similarity score."
            )
        if np.any(positions < 0) or np.any(positions >= len(self._chunks)):
            raise ProductionRetrievalIntegrityError(
                "FAISS returned an invalid index position."
            )
        if np.any(scores[0][:-1] < scores[0][1:]):
            raise ProductionRetrievalIntegrityError(
                "FAISS scores are not in descending order."
            )

        records: list[dict[str, Any]] = []
        for rank, (score, position_value) in enumerate(
            zip(scores[0], positions[0], strict=True),
            start=1,
        ):
            position = int(position_value)
            mapping = self._mapping[position]
            chunk = self._chunks[position]
            record = {
                "chunk_id": mapping["chunk_id"],
                "index_mapping_schema_version": mapping[
                    "index_mapping_schema_version"
                ],
                "index_position": position,
                "parent_record_id": mapping["parent_record_id"],
                "pdf_page_index": mapping["pdf_page_index"],
                "pdf_page_number": mapping["pdf_page_number"],
                "printed_page_number": mapping["printed_page_number"],
                "retrieval_result_schema_version": (
                    RETRIEVAL_RESULT_SCHEMA_VERSION
                ),
                "retrieval_rank": rank,
                "similarity_score": float(score),
                "source_id": mapping["source_id"],
                "text": chunk["text"],
            }
            records.append(record)

        return RetrievalResult(
            mode=mode,
            query_identifier=query_identifier,
            query_embedding_sha256=sha256(
                matrix.tobytes(order="C")
            ).hexdigest(),
            query_token_count=query_token_count,
            api_request_made=api_request_made,
            records=tuple(records),
        )

    def retrieve_benchmark(self, question_id: str) -> RetrievalResult:
        """Search one benchmark row using its frozen query embedding."""

        if question_id not in self._benchmark_positions:
            raise ProductionRetrievalError(
                f"Unknown benchmark question identifier: {question_id}."
            )
        position = self._benchmark_positions[question_id]
        matrix = self._prepare_query_vector(
            self._benchmark_query_matrix[position],
            already_normalised=True,
        )
        return self._search(
            matrix,
            mode="benchmark",
            query_identifier=question_id,
            query_token_count=None,
            api_request_made=False,
        )

    def retrieve_all_benchmark(self) -> tuple[RetrievalResult, ...]:
        """Search all benchmark rows in their frozen question order."""

        return tuple(
            self.retrieve_benchmark(question_id)
            for question_id in self._benchmark_question_ids
        )

    def retrieve_interactive(
        self,
        query_text: str,
        *,
        embedding_client: QueryEmbeddingClient | None = None,
        query_identifier: str | None = None,
    ) -> RetrievalResult:
        """Embed and search one new question without retaining its text."""

        if not isinstance(query_text, str) or not query_text.strip():
            raise ProductionRetrievalError(
                "Interactive query text must be a non-empty string."
            )
        interactive = self.config["query_embedding"]["interactive_mode"]
        encoding = tiktoken.get_encoding(interactive["token_encoding"])
        token_count = len(
            encoding.encode(query_text, disallowed_special=())
        )
        context_limit = self._embedding_index_config["embedding"][
            "embedding_context_length"
        ]
        if token_count > context_limit:
            raise ProductionRetrievalError(
                f"Interactive query has {token_count} tokens; "
                f"maximum is {context_limit}."
            )
        client = embedding_client
        if client is None:
            client = build_openai_embedding_client(
                self._embedding_index_config,
                self.project_root,
            )
        try:
            vectors = client.embed_documents([query_text])
        except Exception as error:
            raise ProductionRetrievalError(
                "Interactive query embedding request failed."
            ) from error
        if not isinstance(vectors, Sequence) or len(vectors) != 1:
            raise ProductionRetrievalIntegrityError(
                "Interactive embedding response must contain one vector."
            )
        matrix = self._prepare_query_vector(
            vectors[0],
            already_normalised=False,
        )
        text_hash = sha256(query_text.encode("utf-8")).hexdigest()
        identifier = query_identifier or f"interactive-{text_hash}"
        if not isinstance(identifier, str) or not identifier:
            raise ProductionRetrievalError(
                "Query identifier must be a non-empty string."
            )
        return self._search(
            matrix,
            mode="interactive",
            query_identifier=identifier,
            query_token_count=token_count,
            api_request_made=True,
        )


def load_production_retriever(
    config_path: str | Path,
    project_root: str | Path,
) -> ProductionRetriever:
    """Validate every frozen input and load the exact production index."""

    config = load_production_retrieval_config(config_path)
    root = Path(project_root).resolve()
    if not root.is_dir():
        raise ProductionRetrievalIntegrityError(
            f"Project root does not exist: {root}."
        )
    _validate_package_versions(config)

    inputs = config["inputs"]
    verified_inputs: dict[str, Path] = {}
    for key, description in (
        ("chunk_embeddings", "chunk embedding matrix"),
        ("chunk_records", "chunk records"),
        ("chunking_configuration", "chunking configuration"),
        ("embedding_index_configuration", "embedding-index configuration"),
        ("embedding_index_summary", "embedding-index summary"),
        ("faiss_index", "FAISS index"),
        ("index_mapping", "index mapping"),
    ):
        record = inputs[key]
        verified_inputs[key] = _verify_artifact(
            root,
            record["relative_path"],
            record["sha256"],
            description,
        )

    selection = config["selection_evidence"]
    selection_paths: dict[str, Path] = {}
    for key, description in (
        ("completion_audit", "v3 completion audit"),
        ("decision_document", "retrieval-depth decision"),
        ("metrics", "v3 retrieval metrics"),
        ("metrics_table", "v3 retrieval metrics table"),
        ("scoring_audit", "v3 scoring audit"),
        ("v3_configuration", "v3 retrieval configuration"),
    ):
        selection_paths[key] = _verify_artifact(
            root,
            selection[f"{key}_relative_path"],
            selection[f"{key}_sha256"],
            description,
        )

    benchmark = config["query_embedding"]["benchmark_mode"]
    question_config_path = _verify_artifact(
        root,
        benchmark["question_configuration_relative_path"],
        benchmark["question_configuration_sha256"],
        "evaluation-question configuration",
    )
    questions_path = _verify_artifact(
        root,
        benchmark["questions_relative_path"],
        benchmark["questions_sha256"],
        "benchmark questions",
    )
    query_matrix_path = _verify_artifact(
        root,
        benchmark["query_embedding_matrix_relative_path"],
        benchmark["query_embedding_matrix_sha256"],
        "benchmark query matrix",
    )

    embedding_config = _load_json_object(
        verified_inputs["embedding_index_configuration"],
        "embedding-index configuration",
    )
    if embedding_config.get("retrieval", {}).get("k", "missing") is not None:
        raise ProductionRetrievalIntegrityError(
            "Historical embedding configuration no longer has unresolved k."
        )
    interactive = config["query_embedding"]["interactive_mode"]
    embedding = embedding_config.get("embedding", {})
    for field, expected in (
        ("client_class", interactive["client_class"]),
        ("client_package", interactive["client_package"]),
        ("client_package_version", interactive["client_package_version"]),
        ("dimensions", config["inputs"]["faiss_index"]["dimensions"]),
        ("dimensions_argument", interactive["dimensions_argument"]),
        ("requested_model", interactive["requested_model"]),
        ("token_encoding", interactive["token_encoding"]),
    ):
        if embedding.get(field) != expected:
            raise ProductionRetrievalIntegrityError(
                f"Embedding-index field {field} differs from production."
            )

    summary = _load_json_object(
        verified_inputs["embedding_index_summary"],
        "embedding-index summary",
    )
    summary_expectations = {
        "chunk_count": inputs["chunk_records"]["record_count"],
        "embedding_dimensions": inputs["faiss_index"]["dimensions"],
        "embedding_matrix_sha256": inputs["chunk_embeddings"]["sha256"],
        "faiss_index_class": inputs["faiss_index"]["index_class"],
        "faiss_index_sha256": inputs["faiss_index"]["sha256"],
        "index_mapping_record_count": inputs["index_mapping"]["record_count"],
        "index_mapping_sha256": inputs["index_mapping"]["sha256"],
        "input_chunks_sha256": inputs["chunk_records"]["sha256"],
        "normalisation": "explicit_l2",
        "requested_embedding_model": interactive["requested_model"],
    }
    for field, expected in summary_expectations.items():
        if summary.get(field) != expected:
            raise ProductionRetrievalIntegrityError(
                f"Embedding-index summary field {field} is inconsistent."
            )

    metrics = _load_json_object(selection_paths["metrics"], "v3 retrieval metrics")
    completion = _load_json_object(
        selection_paths["completion_audit"], "v3 completion audit"
    )
    scoring = _load_json_object(selection_paths["scoring_audit"], "v3 scoring audit")
    v3_config = _load_json_object(
        selection_paths["v3_configuration"], "v3 retrieval configuration"
    )
    if not (
        metrics.get("selected_depth_k") == config["search"]["retrieval_depth_k"]
        and metrics.get("selection_status") == "selected_by_frozen_rule"
        and metrics.get("candidate_range_adequacy_gate", {}).get("triggered") is False
        and completion.get("completed_judgement_count")
        == selection["completed_judgement_count"]
        and completion.get("remaining_judgement_count") == 0
        and scoring.get("independent_selection_rule_matches") is True
        and scoring.get("selected_depth_k") == config["search"]["retrieval_depth_k"]
        and scoring.get("further_depth_expansion_required") is False
        and v3_config.get("selection_rule", {}).get("selected_depth_k") is None
    ):
        raise ProductionRetrievalIntegrityError(
            "Selection evidence differs from the production contract."
        )

    chunks = _read_json_lines(verified_inputs["chunk_records"], "chunk records")
    mapping = _read_json_lines(verified_inputs["index_mapping"], "index mapping")
    expected_count = inputs["chunk_records"]["record_count"]
    if len(chunks) != expected_count or len(mapping) != expected_count:
        raise ProductionRetrievalIntegrityError(
            "Chunk or mapping count differs from the production contract."
        )
    chunk_ids: set[str] = set()
    for position, (chunk, mapping_record) in enumerate(
        zip(chunks, mapping, strict=True)
    ):
        if set(mapping_record) != MAPPING_FIELDS:
            raise ProductionRetrievalIntegrityError(
                f"Mapping record {position} has unexpected fields."
            )
        if mapping_record["index_position"] != position:
            raise ProductionRetrievalIntegrityError(
                f"Mapping record {position} has the wrong position."
            )
        for field in (
            "chunk_id",
            "parent_record_id",
            "pdf_page_index",
            "pdf_page_number",
            "printed_page_number",
            "source_id",
        ):
            if mapping_record[field] != chunk.get(field):
                raise ProductionRetrievalIntegrityError(
                    f"Chunk and mapping {field} differ at position {position}."
                )
        if not isinstance(chunk.get("text"), str) or not chunk["text"]:
            raise ProductionRetrievalIntegrityError(
                f"Chunk {position} has invalid text."
            )
        chunk_id = mapping_record["chunk_id"]
        if chunk_id in chunk_ids:
            raise ProductionRetrievalIntegrityError(
                f"Duplicate chunk identifier at position {position}."
            )
        chunk_ids.add(chunk_id)

    try:
        index = faiss.read_index(str(verified_inputs["faiss_index"]))
    except RuntimeError as error:
        raise ProductionRetrievalIntegrityError(
            "Cannot load the frozen FAISS index."
        ) from error
    if not (
        type(index).__name__ == inputs["faiss_index"]["index_class"]
        and index.d == inputs["faiss_index"]["dimensions"]
        and index.ntotal == expected_count
    ):
        raise ProductionRetrievalIntegrityError(
            "Loaded FAISS index differs from the production contract."
        )

    question_config = _load_json_object(
        question_config_path, "evaluation-question configuration"
    )
    if question_config.get("retrieval_evaluation", {}).get(
        "selected_depth_k", "missing"
    ) is not None:
        raise ProductionRetrievalIntegrityError(
            "Historical question configuration no longer has unresolved k."
        )
    question_ids = question_config.get("benchmark", {}).get("question_ids")
    if not isinstance(question_ids, list) or any(
        not isinstance(question_id, str) or not question_id
        for question_id in question_ids
    ):
        raise ProductionRetrievalIntegrityError(
            "Evaluation-question configuration has invalid identifiers."
        )
    question_records = _read_json_lines(questions_path, "benchmark questions")
    if len(question_records) != benchmark["expected_query_count"]:
        raise ProductionRetrievalIntegrityError(
            "Benchmark question count differs from the production contract."
        )
    observed_ids = []
    for position, question in enumerate(question_records):
        if question.get("position") != position:
            raise ProductionRetrievalIntegrityError(
                f"Benchmark question {position} has the wrong position."
            )
        if question.get("ground_truth_available_during_generation") is not False:
            raise ProductionRetrievalIntegrityError(
                f"Benchmark question {position} allows ground truth."
            )
        observed_ids.append(question.get("question_id"))
    if observed_ids != question_ids:
        raise ProductionRetrievalIntegrityError(
            "Benchmark question order differs from its configuration."
        )

    try:
        query_matrix = np.load(query_matrix_path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ProductionRetrievalIntegrityError(
            "Cannot load the frozen benchmark query matrix."
        ) from error
    expected_shape = tuple(benchmark["query_embedding_matrix_shape"])
    if not (
        query_matrix.shape == expected_shape
        and query_matrix.dtype == np.float32
        and query_matrix.flags.c_contiguous
        and np.isfinite(query_matrix).all()
    ):
        raise ProductionRetrievalIntegrityError(
            "Benchmark query matrix has the wrong representation."
        )
    norms = np.linalg.norm(query_matrix, axis=1)
    tolerance = config["integrity_controls"]["query_norm_tolerance"]
    if not np.allclose(norms, 1.0, atol=tolerance, rtol=0.0):
        raise ProductionRetrievalIntegrityError(
            "Benchmark query matrix is not L2-normalised."
        )

    return ProductionRetriever(
        config=config,
        project_root=root,
        chunks=chunks,
        mapping=mapping,
        index=index,
        benchmark_question_ids=question_ids,
        benchmark_query_matrix=query_matrix,
        embedding_index_config=embedding_config,
    )


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the frozen production retrieval contract."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Production retrieval configuration path.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Repository root. Defaults to the current directory.",
    )
    return parser


def main() -> None:
    """Validate resources without searching or making an API request."""

    arguments = _build_argument_parser().parse_args()
    retriever = load_production_retriever(
        arguments.config,
        arguments.project_root,
    )
    print("Production retrieval preflight: PASSED")
    print("  Configuration ID:", retriever.config["configuration_id"])
    print("  Indexed chunks:", retriever.chunk_count)
    print("  Retrieval depth k:", retriever.retrieval_depth_k)
    print("  Benchmark questions:", len(retriever.benchmark_question_ids))
    print("  Benchmark search performed: False")
    print("  Interactive embedding API request made: False")
    print("  Question or chunk text displayed: False")
    print("  Ground truth accessed: False")


if __name__ == "__main__":
    main()
