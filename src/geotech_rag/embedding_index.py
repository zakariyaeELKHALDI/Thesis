"""Generate validated embeddings and build a persistent FAISS index."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from math import ceil, isfinite
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Protocol

import faiss
import numpy as np
import tiktoken
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from geotech_rag.corpus_export import (
    serialise_json_lines,
    serialise_json_object,
)
from geotech_rag.corpus_manifest import calculate_file_sha256


EMBEDDING_INDEX_CONFIG_SCHEMA_VERSION = "1.0"
INDEX_MAPPING_SCHEMA_VERSION = "1.0"
EMBEDDING_INDEX_SUMMARY_SCHEMA_VERSION = "1.0"
EMBEDDING_BATCH_SCHEMA_VERSION = "1.0"

CHUNKS_RELATIVE_PATH = Path(
    "data/processed/corpus/chunks.jsonl"
)
CHUNKING_SUMMARY_RELATIVE_PATH = Path(
    "data/processed/audit/chunking-summary.json"
)
EMBEDDING_MATRIX_RELATIVE_PATH = Path(
    "data/processed/embeddings/chunk-embeddings.npy"
)
FAISS_INDEX_RELATIVE_PATH = Path(
    "data/processed/index/chunks.faiss"
)
INDEX_MAPPING_RELATIVE_PATH = Path(
    "data/processed/index/chunk-mapping.jsonl"
)
EMBEDDING_INDEX_SUMMARY_RELATIVE_PATH = Path(
    "data/processed/audit/embedding-index-summary.json"
)
EMBEDDING_STAGING_RELATIVE_DIRECTORY = Path(
    "data/processed/embeddings/staging"
)

TOP_LEVEL_CONFIG_KEYS = {
    "api_constraints",
    "configuration_id",
    "decision_status",
    "embedding",
    "embedding_index_config_schema_version",
    "expected_source_specific_audit",
    "faiss",
    "input",
    "output",
    "request_policy",
    "retrieval",
    "vector_representation",
}
INPUT_CONFIG_KEYS = {
    "chunk_schema_version",
    "chunking_summary_relative_path",
    "chunking_summary_sha256",
    "record_count",
    "relative_path",
    "sha256",
    "size_bytes",
}
EMBEDDING_CONFIG_KEYS = {
    "api_key_environment_variable",
    "api_key_file_relative_path",
    "client_class",
    "client_package",
    "client_package_version",
    "dimensions",
    "dimensions_argument",
    "embedding_context_length",
    "observed_response_model_is_regeneration_requirement",
    "observed_smoke_test_response_model",
    "requested_model",
    "token_encoding",
}
REQUEST_POLICY_KEYS = {
    "account_rate_limit_is_runtime_specific",
    "batch_size",
    "check_embedding_context_length",
    "honour_server_retry_after",
    "max_retries",
    "resumable_batch_staging",
    "retry_max_seconds",
    "retry_min_seconds",
    "show_progress_bar",
    "skip_empty",
    "tiktoken_enabled",
    "tiktoken_model_name",
    "timeout_seconds",
}
API_CONSTRAINT_KEYS = {
    "account_rate_limit_not_frozen",
    "evidence_checked_date",
    "maximum_inputs_per_request",
    "maximum_tokens_per_input",
    "maximum_total_tokens_per_request",
}
VECTOR_REPRESENTATION_KEYS = {
    "dtype",
    "finite_values_required",
    "memory_layout",
    "normalisation",
    "normalise_corpus_vectors",
    "normalise_query_vectors",
    "numpy_package_version",
    "post_normalisation_norm_tolerance",
}
FAISS_CONFIG_KEYS = {
    "approximate_search",
    "index_class",
    "metric",
    "package",
    "package_version",
    "similarity_interpretation",
    "training_required",
}
RETRIEVAL_CONFIG_KEYS = {
    "decision_status",
    "k",
}
OUTPUT_CONFIG_KEYS = {
    "embedding_matrix_relative_path",
    "faiss_index_relative_path",
    "index_mapping_relative_path",
    "index_mapping_schema_version",
    "overwrite_policy",
    "staging_relative_directory",
    "summary_relative_path",
    "summary_schema_version",
    "summary_written_last",
}
EXPECTED_AUDIT_KEYS = {
    "duplicate_text_occurrence_count",
    "embedding_matrix_shape",
    "faiss_index_dimension",
    "faiss_index_total",
    "index_mapping_record_count",
    "input_chunk_count",
    "input_total_character_count",
    "input_total_token_count",
    "inputs_over_api_token_limit",
    "maximum_chunk_token_count",
    "maximum_inputs_in_one_batch",
    "maximum_tokens_in_one_batch",
    "median_chunk_token_count",
    "minimum_chunk_token_count",
    "p95_chunk_token_count",
    "request_batch_count",
}
CHUNK_KEYS = {
    "character_count",
    "chunk_bbox",
    "chunk_id",
    "chunk_index",
    "chunk_schema_version",
    "coordinate_origin",
    "coordinate_unit",
    "line_count",
    "page_height",
    "page_state",
    "page_width",
    "parent_character_end",
    "parent_character_start",
    "parent_line_end_index",
    "parent_line_start_index",
    "parent_record_id",
    "parent_record_schema_version",
    "pdf_page_index",
    "pdf_page_number",
    "printed_page_number",
    "region_bbox",
    "region_index",
    "source_id",
    "source_sha256",
    "text",
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
BATCH_METADATA_KEYS = {
    "batch_index",
    "batch_schema_version",
    "configuration_sha256",
    "dimensions",
    "dtype",
    "end_position",
    "input_chunk_ids_sha256",
    "input_chunks_sha256",
    "input_count",
    "input_text_sha256",
    "input_token_count",
    "matrix_sha256",
    "requested_model",
    "start_position",
}


class EmbeddingIndexError(RuntimeError):
    """Raised when embedding or indexing violates the frozen contract."""


class EmbeddingClient(Protocol):
    """Minimal client interface used by production and deterministic tests."""

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Return one embedding vector for every input text in order."""


@dataclass(frozen=True)
class EmbeddingIndexResult:
    """Paths, fingerprints and counts from one completed production run."""

    embedding_matrix_path: Path
    faiss_index_path: Path
    index_mapping_path: Path
    summary_path: Path
    embedding_matrix_sha256: str
    faiss_index_sha256: str
    index_mapping_sha256: str
    summary_sha256: str
    chunk_count: int
    dimensions: int
    batch_count: int


def _require_exact_keys(
    value: Any,
    expected_keys: set[str],
    description: str,
) -> dict[str, Any]:
    """Require an object containing exactly the expected keys."""

    if not isinstance(value, dict):
        raise EmbeddingIndexError(
            f"{description} must be an object."
        )

    actual_keys = set(value)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise EmbeddingIndexError(
            f"{description} has invalid fields; "
            f"missing={missing}, unexpected={unexpected}."
        )

    return value


def _require_integer(
    value: Any,
    description: str,
    minimum: int | None = None,
) -> int:
    """Return an integer while rejecting booleans."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise EmbeddingIndexError(
            f"{description} must be an integer."
        )
    if minimum is not None and value < minimum:
        raise EmbeddingIndexError(
            f"{description} must be at least {minimum}."
        )
    return value


def _require_finite_number(
    value: Any,
    description: str,
    minimum_exclusive: float | None = None,
) -> float:
    """Return a finite number while rejecting booleans."""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
    ):
        raise EmbeddingIndexError(
            f"{description} must be a finite number."
        )

    result = float(value)
    if minimum_exclusive is not None and result <= minimum_exclusive:
        raise EmbeddingIndexError(
            f"{description} must be greater than {minimum_exclusive}."
        )
    return result


def _require_boolean(
    value: Any,
    expected: bool,
    description: str,
) -> None:
    """Require one explicit Boolean contract value."""

    if value is not expected:
        raise EmbeddingIndexError(
            f"{description} must equal {expected}."
        )


def _require_non_empty_string(
    value: Any,
    description: str,
) -> str:
    """Return a non-empty string."""

    if not isinstance(value, str) or not value:
        raise EmbeddingIndexError(
            f"{description} must be a non-empty string."
        )
    return value


def _validate_sha256(
    value: Any,
    description: str,
) -> str:
    """Validate and return a lowercase SHA-256 value."""

    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(
            character not in "0123456789abcdef"
            for character in value
        )
    ):
        raise EmbeddingIndexError(
            f"{description} is not a valid SHA-256."
        )
    return value


def _validate_relative_path(
    value: Any,
    description: str,
) -> Path:
    """Validate a normalised repository-relative POSIX path."""

    path_text = _require_non_empty_string(value, description)
    path = Path(path_text)
    if (
        path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != path_text
    ):
        raise EmbeddingIndexError(
            f"{description} must be a normalised "
            "repository-relative POSIX path."
        )
    return path


def _require_within_directory(
    path: Path,
    directory: Path,
    description: str,
) -> None:
    """Require a resolved path to remain inside one directory."""

    try:
        path.resolve().relative_to(directory.resolve())
    except ValueError as error:
        raise EmbeddingIndexError(
            f"{description} lies outside {directory}."
        ) from error


def _installed_package_version(package_name: str) -> str:
    """Return the installed version of one required package."""

    try:
        return version(package_name)
    except PackageNotFoundError as error:
        raise EmbeddingIndexError(
            f"Required package is not installed: {package_name}."
        ) from error


def _validate_expected_path(
    value: Any,
    expected: Path,
    description: str,
) -> None:
    """Require one exact authorised repository-relative path."""

    actual = _validate_relative_path(value, description)
    if actual != expected:
        raise EmbeddingIndexError(
            f"Unexpected {description.lower()}."
        )


def _validate_input_config(raw_config: Any) -> dict[str, Any]:
    """Validate the exact chunk-input identity and paths."""

    config = _require_exact_keys(
        raw_config,
        INPUT_CONFIG_KEYS,
        "Embedding input configuration",
    )
    _validate_expected_path(
        config["relative_path"],
        CHUNKS_RELATIVE_PATH,
        "Embedding input path",
    )
    _validate_expected_path(
        config["chunking_summary_relative_path"],
        CHUNKING_SUMMARY_RELATIVE_PATH,
        "Chunking-summary input path",
    )
    _validate_sha256(
        config["sha256"],
        "Expected chunks fingerprint",
    )
    _validate_sha256(
        config["chunking_summary_sha256"],
        "Expected chunking-summary fingerprint",
    )
    _require_integer(
        config["size_bytes"],
        "Expected chunks byte size",
        minimum=1,
    )
    _require_integer(
        config["record_count"],
        "Expected chunk count",
        minimum=1,
    )
    if config["chunk_schema_version"] != "1.0":
        raise EmbeddingIndexError(
            "Unsupported chunk schema version."
        )
    return config


def _validate_embedding_config(
    raw_config: Any,
) -> dict[str, Any]:
    """Validate the embedding client and remote-model contract."""

    config = _require_exact_keys(
        raw_config,
        EMBEDDING_CONFIG_KEYS,
        "Embedding configuration",
    )
    if config["client_class"] != "OpenAIEmbeddings":
        raise EmbeddingIndexError(
            "Only OpenAIEmbeddings is supported."
        )
    if config["client_package"] != "langchain-openai":
        raise EmbeddingIndexError(
            "Unexpected embedding-client package."
        )

    package_version = _require_non_empty_string(
        config["client_package_version"],
        "Embedding-client package version",
    )
    if _installed_package_version(config["client_package"]) != (
        package_version
    ):
        raise EmbeddingIndexError(
            "Installed embedding-client version differs from configuration."
        )

    if config["requested_model"] != "text-embedding-ada-002":
        raise EmbeddingIndexError(
            "Unexpected reconstructed-baseline embedding model."
        )
    _require_non_empty_string(
        config["observed_smoke_test_response_model"],
        "Observed smoke-test response model",
    )
    _require_boolean(
        config["observed_response_model_is_regeneration_requirement"],
        False,
        "Observed response-model regeneration requirement",
    )

    dimensions = _require_integer(
        config["dimensions"],
        "Embedding dimensions",
        minimum=1,
    )
    if dimensions != 1536:
        raise EmbeddingIndexError(
            "text-embedding-ada-002 must use 1536 dimensions."
        )
    if config["dimensions_argument"] is not None:
        raise EmbeddingIndexError(
            "The dimensions argument must remain omitted for ada-002."
        )
    if config["token_encoding"] != "cl100k_base":
        raise EmbeddingIndexError(
            "Unexpected embedding token encoding."
        )
    if config["embedding_context_length"] != 8191:
        raise EmbeddingIndexError(
            "Unexpected LangChain embedding context length."
        )
    if config["api_key_environment_variable"] != "OPENAI_API_KEY":
        raise EmbeddingIndexError(
            "Unexpected API-key environment variable."
        )
    if config["api_key_file_relative_path"] != ".env":
        raise EmbeddingIndexError(
            "Unexpected API-key file path."
        )
    return config


def _validate_request_policy(
    raw_config: Any,
) -> dict[str, Any]:
    """Validate batching, retry and context-check behaviour."""

    config = _require_exact_keys(
        raw_config,
        REQUEST_POLICY_KEYS,
        "Embedding request policy",
    )
    _require_integer(
        config["batch_size"],
        "Embedding batch size",
        minimum=1,
    )
    if config["batch_size"] != 256:
        raise EmbeddingIndexError(
            "The reconstructed baseline batch size must equal 256."
        )
    _require_integer(
        config["max_retries"],
        "Maximum retries",
        minimum=0,
    )
    retry_minimum = _require_integer(
        config["retry_min_seconds"],
        "Minimum retry seconds",
        minimum=0,
    )
    retry_maximum = _require_integer(
        config["retry_max_seconds"],
        "Maximum retry seconds",
        minimum=0,
    )
    if retry_maximum < retry_minimum:
        raise EmbeddingIndexError(
            "Maximum retry seconds cannot be smaller than the minimum."
        )
    _require_finite_number(
        config["timeout_seconds"],
        "Request timeout",
        minimum_exclusive=0.0,
    )

    expected_booleans = {
        "account_rate_limit_is_runtime_specific": True,
        "check_embedding_context_length": True,
        "honour_server_retry_after": True,
        "resumable_batch_staging": True,
        "show_progress_bar": False,
        "skip_empty": False,
        "tiktoken_enabled": True,
    }
    for field_name, expected_value in expected_booleans.items():
        _require_boolean(
            config[field_name],
            expected_value,
            f"Request-policy {field_name}",
        )

    if config["tiktoken_model_name"] != "text-embedding-ada-002":
        raise EmbeddingIndexError(
            "Unexpected tokenizer model name."
        )
    return config


def _validate_api_constraints(
    raw_config: Any,
) -> dict[str, Any]:
    """Validate the endpoint constraints recorded by the audit."""

    config = _require_exact_keys(
        raw_config,
        API_CONSTRAINT_KEYS,
        "Embedding API constraints",
    )
    _require_non_empty_string(
        config["evidence_checked_date"],
        "API-constraint evidence date",
    )
    for field_name in (
        "maximum_inputs_per_request",
        "maximum_tokens_per_input",
        "maximum_total_tokens_per_request",
    ):
        _require_integer(
            config[field_name],
            f"API constraint {field_name}",
            minimum=1,
        )
    _require_boolean(
        config["account_rate_limit_not_frozen"],
        True,
        "Account rate-limit policy",
    )
    return config


def _validate_vector_config(
    raw_config: Any,
) -> dict[str, Any]:
    """Validate the NumPy representation and normalisation rules."""

    config = _require_exact_keys(
        raw_config,
        VECTOR_REPRESENTATION_KEYS,
        "Vector representation",
    )
    numpy_version = _require_non_empty_string(
        config["numpy_package_version"],
        "NumPy package version",
    )
    if _installed_package_version("numpy") != numpy_version:
        raise EmbeddingIndexError(
            "Installed NumPy version differs from configuration."
        )
    if config["dtype"] != "float32":
        raise EmbeddingIndexError(
            "Only float32 embedding vectors are supported."
        )
    if config["memory_layout"] != "C_contiguous":
        raise EmbeddingIndexError(
            "Only C-contiguous embedding matrices are supported."
        )
    if config["normalisation"] != "explicit_l2":
        raise EmbeddingIndexError(
            "Only explicit L2 normalisation is supported."
        )
    for field_name in (
        "finite_values_required",
        "normalise_corpus_vectors",
        "normalise_query_vectors",
    ):
        _require_boolean(
            config[field_name],
            True,
            f"Vector setting {field_name}",
        )
    _require_finite_number(
        config["post_normalisation_norm_tolerance"],
        "Vector-norm tolerance",
        minimum_exclusive=0.0,
    )
    return config


def _validate_faiss_config(
    raw_config: Any,
) -> dict[str, Any]:
    """Validate the exact FAISS index and similarity contract."""

    config = _require_exact_keys(
        raw_config,
        FAISS_CONFIG_KEYS,
        "FAISS configuration",
    )
    if config["package"] != "faiss-cpu":
        raise EmbeddingIndexError(
            "Unexpected FAISS package."
        )
    package_version = _require_non_empty_string(
        config["package_version"],
        "FAISS package version",
    )
    if _installed_package_version(config["package"]) != package_version:
        raise EmbeddingIndexError(
            "Installed FAISS version differs from configuration."
        )
    expected_values = {
        "approximate_search": False,
        "index_class": "IndexFlatIP",
        "metric": "inner_product",
        "similarity_interpretation": (
            "cosine_similarity_for_l2_normalised_vectors"
        ),
        "training_required": False,
    }
    for field_name, expected_value in expected_values.items():
        if config[field_name] != expected_value:
            raise EmbeddingIndexError(
                f"FAISS {field_name} must equal {expected_value!r}."
            )
    return config


def _validate_retrieval_config(
    raw_config: Any,
) -> dict[str, Any]:
    """Ensure retrieval depth remains outside index construction."""

    config = _require_exact_keys(
        raw_config,
        RETRIEVAL_CONFIG_KEYS,
        "Retrieval configuration",
    )
    if config["k"] is not None:
        raise EmbeddingIndexError(
            "Retrieval depth k must remain unresolved during indexing."
        )
    if config["decision_status"] != (
        "unresolved_pending_retrieval_evaluation"
    ):
        raise EmbeddingIndexError(
            "Unexpected retrieval-depth decision status."
        )
    return config


def _validate_output_config(
    raw_config: Any,
) -> dict[str, Any]:
    """Validate every output path and publication rule."""

    config = _require_exact_keys(
        raw_config,
        OUTPUT_CONFIG_KEYS,
        "Embedding-index output configuration",
    )
    expected_paths = {
        "embedding_matrix_relative_path": EMBEDDING_MATRIX_RELATIVE_PATH,
        "faiss_index_relative_path": FAISS_INDEX_RELATIVE_PATH,
        "index_mapping_relative_path": INDEX_MAPPING_RELATIVE_PATH,
        "summary_relative_path": EMBEDDING_INDEX_SUMMARY_RELATIVE_PATH,
        "staging_relative_directory": EMBEDDING_STAGING_RELATIVE_DIRECTORY,
    }
    for field_name, expected_path in expected_paths.items():
        _validate_expected_path(
            config[field_name],
            expected_path,
            field_name,
        )
    if config["index_mapping_schema_version"] != (
        INDEX_MAPPING_SCHEMA_VERSION
    ):
        raise EmbeddingIndexError(
            "Unsupported index-mapping schema version."
        )
    if config["summary_schema_version"] != (
        EMBEDDING_INDEX_SUMMARY_SCHEMA_VERSION
    ):
        raise EmbeddingIndexError(
            "Unsupported embedding-index summary schema version."
        )
    if config["overwrite_policy"] != "require_explicit_flag":
        raise EmbeddingIndexError(
            "Unexpected embedding-index overwrite policy."
        )
    _require_boolean(
        config["summary_written_last"],
        True,
        "Summary-written-last policy",
    )
    return config


def _validate_expected_audit(
    raw_config: Any,
    input_config: dict[str, Any],
    embedding_config: dict[str, Any],
    request_policy: dict[str, Any],
) -> dict[str, Any]:
    """Validate the internally consistent source-specific expectations."""

    audit = _require_exact_keys(
        raw_config,
        EXPECTED_AUDIT_KEYS,
        "Expected embedding-index audit",
    )
    integer_fields = EXPECTED_AUDIT_KEYS - {
        "embedding_matrix_shape",
        "median_chunk_token_count",
    }
    for field_name in integer_fields:
        _require_integer(
            audit[field_name],
            f"Expected {field_name}",
            minimum=0,
        )
    _require_finite_number(
        audit["median_chunk_token_count"],
        "Expected median chunk token count",
    )

    matrix_shape = audit["embedding_matrix_shape"]
    if not isinstance(matrix_shape, list) or len(matrix_shape) != 2:
        raise EmbeddingIndexError(
            "Expected embedding matrix shape must contain two integers."
        )
    for position, value in enumerate(matrix_shape):
        _require_integer(
            value,
            f"Expected embedding matrix shape position {position}",
            minimum=1,
        )

    record_count = input_config["record_count"]
    dimensions = embedding_config["dimensions"]
    batch_size = request_policy["batch_size"]
    if matrix_shape != [record_count, dimensions]:
        raise EmbeddingIndexError(
            "Expected embedding matrix shape is inconsistent."
        )
    if audit["input_chunk_count"] != record_count:
        raise EmbeddingIndexError(
            "Expected input chunk count is inconsistent."
        )
    if audit["faiss_index_dimension"] != dimensions:
        raise EmbeddingIndexError(
            "Expected FAISS dimension is inconsistent."
        )
    if audit["faiss_index_total"] != record_count:
        raise EmbeddingIndexError(
            "Expected FAISS total is inconsistent."
        )
    if audit["index_mapping_record_count"] != record_count:
        raise EmbeddingIndexError(
            "Expected mapping count is inconsistent."
        )
    if audit["request_batch_count"] != ceil(
        record_count / batch_size
    ):
        raise EmbeddingIndexError(
            "Expected request batch count is inconsistent."
        )
    if audit["maximum_inputs_in_one_batch"] > batch_size:
        raise EmbeddingIndexError(
            "Expected maximum batch input count exceeds batch size."
        )
    return audit


def load_embedding_index_config(
    config_path: str | Path,
) -> dict[str, Any]:
    """Load and validate the frozen embedding-index configuration."""

    path = Path(config_path)
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise EmbeddingIndexError(
            f"Cannot read embedding-index configuration: {path}."
        ) from error
    except json.JSONDecodeError as error:
        raise EmbeddingIndexError(
            f"Embedding-index configuration is invalid JSON: {path}."
        ) from error

    config = _require_exact_keys(
        config,
        TOP_LEVEL_CONFIG_KEYS,
        "Embedding-index configuration",
    )
    if config["embedding_index_config_schema_version"] != (
        EMBEDDING_INDEX_CONFIG_SCHEMA_VERSION
    ):
        raise EmbeddingIndexError(
            "Unsupported embedding-index config schema version."
        )
    _require_non_empty_string(
        config["configuration_id"],
        "Embedding-index configuration identifier",
    )
    if config["decision_status"] != (
        "accepted_pending_production_implementation"
    ):
        raise EmbeddingIndexError(
            "Unexpected embedding-index decision status."
        )

    input_config = _validate_input_config(config["input"])
    embedding_config = _validate_embedding_config(config["embedding"])
    request_policy = _validate_request_policy(config["request_policy"])
    api_constraints = _validate_api_constraints(config["api_constraints"])
    _validate_vector_config(config["vector_representation"])
    _validate_faiss_config(config["faiss"])
    _validate_retrieval_config(config["retrieval"])
    _validate_output_config(config["output"])
    _validate_expected_audit(
        config["expected_source_specific_audit"],
        input_config,
        embedding_config,
        request_policy,
    )

    if request_policy["batch_size"] > api_constraints[
        "maximum_inputs_per_request"
    ]:
        raise EmbeddingIndexError(
            "Embedding batch size exceeds the recorded API input limit."
        )
    expected_audit = config["expected_source_specific_audit"]
    if expected_audit["maximum_chunk_token_count"] > (
        api_constraints["maximum_tokens_per_input"]
    ):
        raise EmbeddingIndexError(
            "Expected chunk tokens exceed the recorded per-input API limit."
        )
    if expected_audit["maximum_tokens_in_one_batch"] > (
        api_constraints["maximum_total_tokens_per_request"]
    ):
        raise EmbeddingIndexError(
            "Expected batch tokens exceed the recorded API request limit."
        )
    return config


def _read_utf8_jsonl(
    path: Path,
    description: str,
) -> list[dict[str, Any]]:
    """Read a non-empty newline-terminated UTF-8 JSONL file."""

    if not path.is_file():
        raise EmbeddingIndexError(
            f"{description} is missing: {path}."
        )
    try:
        file_bytes = path.read_bytes()
        decoded_text = file_bytes.decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise EmbeddingIndexError(
            f"Cannot read {description.lower()} as UTF-8: {path}."
        ) from error
    if not file_bytes.endswith(b"\n"):
        raise EmbeddingIndexError(
            f"{description} must end with a newline."
        )
    try:
        records = [
            json.loads(line)
            for line in decoded_text.splitlines()
        ]
    except json.JSONDecodeError as error:
        raise EmbeddingIndexError(
            f"{description} contains invalid JSON."
        ) from error
    if not records:
        raise EmbeddingIndexError(
            f"{description} contains no records."
        )
    return records


def _validate_chunk_records(
    records: list[dict[str, Any]],
    expected_schema_version: str,
) -> None:
    """Validate chunk identity, ordering and required retrieval provenance."""

    chunk_ids: set[str] = set()
    previous_parent_id: str | None = None
    previous_chunk_index: int | None = None
    completed_parent_ids: set[str] = set()
    common_source: tuple[str, str] | None = None

    for position, raw_record in enumerate(records):
        record = _require_exact_keys(
            raw_record,
            CHUNK_KEYS,
            f"Chunk record {position}",
        )
        if record["chunk_schema_version"] != expected_schema_version:
            raise EmbeddingIndexError(
                "Chunk record has an unsupported schema version."
            )

        chunk_id = _require_non_empty_string(
            record["chunk_id"],
            "Chunk identifier",
        )
        parent_record_id = _require_non_empty_string(
            record["parent_record_id"],
            "Parent record identifier",
        )
        chunk_index = _require_integer(
            record["chunk_index"],
            "Chunk index",
            minimum=0,
        )
        text = _require_non_empty_string(
            record["text"],
            "Chunk text",
        )
        if chunk_id in chunk_ids:
            raise EmbeddingIndexError(
                f"Duplicate chunk identifier: {chunk_id}."
            )
        chunk_ids.add(chunk_id)

        expected_chunk_id = (
            f"{parent_record_id}:chunk-{chunk_index:04d}"
        )
        if chunk_id != expected_chunk_id:
            raise EmbeddingIndexError(
                "Chunk identifier is inconsistent with its parent and index."
            )
        if record["character_count"] != len(text):
            raise EmbeddingIndexError(
                "Chunk character count differs from its text."
            )

        source_id = _require_non_empty_string(
            record["source_id"],
            "Chunk source identifier",
        )
        source_sha256 = _validate_sha256(
            record["source_sha256"],
            "Chunk source fingerprint",
        )
        source = (source_id, source_sha256)
        if common_source is None:
            common_source = source
        elif source != common_source:
            raise EmbeddingIndexError(
                "Chunk records contain mixed source identities."
            )

        pdf_page_index = _require_integer(
            record["pdf_page_index"],
            "Chunk PDF page index",
            minimum=0,
        )
        pdf_page_number = _require_integer(
            record["pdf_page_number"],
            "Chunk PDF page number",
            minimum=1,
        )
        if pdf_page_number != pdf_page_index + 1:
            raise EmbeddingIndexError(
                "Chunk PDF page identifiers are inconsistent."
            )

        if parent_record_id != previous_parent_id:
            if parent_record_id in completed_parent_ids:
                raise EmbeddingIndexError(
                    "Chunk parent groups are not contiguous."
                )
            if previous_parent_id is not None:
                completed_parent_ids.add(previous_parent_id)
            if chunk_index != 0:
                raise EmbeddingIndexError(
                    "The first chunk of a parent must have index zero."
                )
            previous_parent_id = parent_record_id
        elif previous_chunk_index is None or chunk_index != (
            previous_chunk_index + 1
        ):
            raise EmbeddingIndexError(
                "Chunk indices are not consecutive within their parent."
            )
        previous_chunk_index = chunk_index


def _nearest_rank_percentile(
    values: Sequence[int],
    percentile: float,
) -> int:
    """Return the nearest-rank percentile for non-empty integers."""

    if not values:
        raise EmbeddingIndexError(
            "Cannot calculate a percentile for an empty sequence."
        )
    rank = max(1, ceil(percentile * len(values)))
    return sorted(values)[rank - 1]


def calculate_embedding_input_audit(
    chunk_records: list[dict[str, Any]],
    token_counts: list[int],
    batch_size: int,
    maximum_tokens_per_input: int,
    dimensions: int,
) -> dict[str, Any]:
    """Calculate every source-specific pre-embedding acceptance value."""

    if len(chunk_records) != len(token_counts):
        raise EmbeddingIndexError(
            "Chunk and token-count lengths differ."
        )
    if not chunk_records:
        raise EmbeddingIndexError(
            "At least one chunk is required for the input audit."
        )

    batch_input_counts = []
    batch_token_totals = []
    for start in range(0, len(chunk_records), batch_size):
        end = min(start + batch_size, len(chunk_records))
        batch_input_counts.append(end - start)
        batch_token_totals.append(sum(token_counts[start:end]))

    texts = [record["text"] for record in chunk_records]
    return {
        "duplicate_text_occurrence_count": (
            len(texts) - len(set(texts))
        ),
        "embedding_matrix_shape": [
            len(chunk_records),
            dimensions,
        ],
        "faiss_index_dimension": dimensions,
        "faiss_index_total": len(chunk_records),
        "index_mapping_record_count": len(chunk_records),
        "input_chunk_count": len(chunk_records),
        "input_total_character_count": sum(map(len, texts)),
        "input_total_token_count": sum(token_counts),
        "inputs_over_api_token_limit": sum(
            count > maximum_tokens_per_input
            for count in token_counts
        ),
        "maximum_chunk_token_count": max(token_counts),
        "maximum_inputs_in_one_batch": max(batch_input_counts),
        "maximum_tokens_in_one_batch": max(batch_token_totals),
        "median_chunk_token_count": statistics.median(
            token_counts
        ),
        "minimum_chunk_token_count": min(token_counts),
        "p95_chunk_token_count": _nearest_rank_percentile(
            token_counts,
            0.95,
        ),
        "request_batch_count": len(batch_input_counts),
    }


def load_validated_chunk_records(
    config: dict[str, Any],
    project_root: str | Path,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Load the exact chunk export and reproduce its frozen token audit."""

    root = Path(project_root).resolve()
    input_config = config["input"]
    chunks_path = root / input_config["relative_path"]
    chunking_summary_path = (
        root / input_config["chunking_summary_relative_path"]
    )
    _require_within_directory(
        chunks_path,
        root / "data/processed/corpus",
        "Embedding input",
    )
    _require_within_directory(
        chunking_summary_path,
        root / "data/processed/audit",
        "Chunking-summary input",
    )

    if not chunks_path.is_file():
        raise EmbeddingIndexError(
            f"Chunk export is missing: {chunks_path}."
        )
    if chunks_path.stat().st_size != input_config["size_bytes"]:
        raise EmbeddingIndexError(
            "Chunk export byte size differs from configuration."
        )
    if calculate_file_sha256(chunks_path) != input_config["sha256"]:
        raise EmbeddingIndexError(
            "Chunk export fingerprint differs from configuration."
        )
    if not chunking_summary_path.is_file():
        raise EmbeddingIndexError(
            f"Chunking summary is missing: {chunking_summary_path}."
        )
    if calculate_file_sha256(chunking_summary_path) != (
        input_config["chunking_summary_sha256"]
    ):
        raise EmbeddingIndexError(
            "Chunking-summary fingerprint differs from configuration."
        )

    try:
        chunking_summary = json.loads(
            chunking_summary_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise EmbeddingIndexError(
            "Cannot read the chunking summary as valid JSON."
        ) from error
    if chunking_summary.get("chunks_sha256") != input_config["sha256"]:
        raise EmbeddingIndexError(
            "Chunking summary references a different chunk export."
        )
    if chunking_summary.get("chunk_count") != input_config["record_count"]:
        raise EmbeddingIndexError(
            "Chunking summary contains an unexpected chunk count."
        )

    records = _read_utf8_jsonl(
        chunks_path,
        "Chunk export",
    )
    if len(records) != input_config["record_count"]:
        raise EmbeddingIndexError(
            "Chunk record count differs from configuration."
        )
    _validate_chunk_records(
        records,
        input_config["chunk_schema_version"],
    )

    encoding = tiktoken.get_encoding(
        config["embedding"]["token_encoding"]
    )
    token_counts = [
        len(encoding.encode(record["text"]))
        for record in records
    ]
    actual_audit = calculate_embedding_input_audit(
        records,
        token_counts,
        config["request_policy"]["batch_size"],
        config["api_constraints"]["maximum_tokens_per_input"],
        config["embedding"]["dimensions"],
    )
    expected_audit = config["expected_source_specific_audit"]
    if actual_audit != expected_audit:
        differing_fields = sorted(
            field_name
            for field_name in EXPECTED_AUDIT_KEYS
            if actual_audit[field_name] != expected_audit[field_name]
        )
        raise EmbeddingIndexError(
            "Embedding input audit differs from configuration; "
            f"fields={differing_fields}."
        )
    return records, token_counts


def build_index_mapping_records(
    chunk_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Map every matrix row and FAISS position to chunk provenance."""

    mapping_records = []
    for index_position, chunk_record in enumerate(chunk_records):
        mapping_record = {
            "chunk_id": chunk_record["chunk_id"],
            "index_mapping_schema_version": INDEX_MAPPING_SCHEMA_VERSION,
            "index_position": index_position,
            "parent_record_id": chunk_record["parent_record_id"],
            "pdf_page_index": chunk_record["pdf_page_index"],
            "pdf_page_number": chunk_record["pdf_page_number"],
            "printed_page_number": chunk_record["printed_page_number"],
            "source_id": chunk_record["source_id"],
        }
        _require_exact_keys(
            mapping_record,
            MAPPING_KEYS,
            f"Index mapping record {index_position}",
        )
        mapping_records.append(mapping_record)
    return mapping_records


def validate_raw_embedding_matrix(
    raw_vectors: Any,
    expected_rows: int,
    expected_dimensions: int,
) -> np.ndarray:
    """Convert one client response into a finite contiguous float32 matrix."""

    try:
        matrix = np.asarray(raw_vectors, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as error:
        raise EmbeddingIndexError(
            "Embedding response cannot be converted to float32."
        ) from error
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)
    expected_shape = (expected_rows, expected_dimensions)
    if matrix.shape != expected_shape:
        raise EmbeddingIndexError(
            f"Unexpected embedding matrix shape: {matrix.shape}; "
            f"expected {expected_shape}."
        )
    if not np.isfinite(matrix).all():
        raise EmbeddingIndexError(
            "Embedding matrix contains a non-finite value."
        )
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms == 0.0):
        raise EmbeddingIndexError(
            "Embedding matrix contains a zero-length vector."
        )
    return matrix


def normalise_embedding_matrix(
    raw_matrix: np.ndarray,
    norm_tolerance: float,
) -> np.ndarray:
    """Return an explicitly L2-normalised contiguous float32 matrix."""

    matrix = np.ascontiguousarray(
        raw_matrix.copy(),
        dtype=np.float32,
    )
    if not np.isfinite(matrix).all():
        raise EmbeddingIndexError(
            "Cannot normalise a matrix containing non-finite values."
        )
    if np.any(np.linalg.norm(matrix, axis=1) == 0.0):
        raise EmbeddingIndexError(
            "Cannot normalise a zero-length embedding vector."
        )
    faiss.normalize_L2(matrix)
    norms = np.linalg.norm(matrix, axis=1)
    if not np.allclose(
        norms,
        np.ones_like(norms),
        rtol=0.0,
        atol=norm_tolerance,
    ):
        raise EmbeddingIndexError(
            "Explicit L2 normalisation failed its norm tolerance."
        )
    return matrix


def _projection_sha256(values: Any) -> str:
    """Fingerprint a deterministic compact JSON projection."""

    serialised = json.dumps(
        values,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(serialised).hexdigest()


def _batch_paths(
    staging_directory: Path,
    batch_index: int,
) -> tuple[Path, Path]:
    """Return stable vector and metadata paths for one batch."""

    stem = f"batch-{batch_index:05d}"
    return (
        staging_directory / f"{stem}.npy",
        staging_directory / f"{stem}.json",
    )


def _expected_batch_metadata(
    config: dict[str, Any],
    configuration_sha256: str,
    chunk_records: list[dict[str, Any]],
    token_counts: list[int],
    batch_index: int,
    start_position: int,
    end_position: int,
) -> dict[str, Any]:
    """Build the immutable metadata fields for one staged request."""

    batch_records = chunk_records[start_position:end_position]
    return {
        "batch_index": batch_index,
        "batch_schema_version": EMBEDDING_BATCH_SCHEMA_VERSION,
        "configuration_sha256": configuration_sha256,
        "dimensions": config["embedding"]["dimensions"],
        "dtype": config["vector_representation"]["dtype"],
        "end_position": end_position,
        "input_chunk_ids_sha256": _projection_sha256(
            [record["chunk_id"] for record in batch_records]
        ),
        "input_chunks_sha256": config["input"]["sha256"],
        "input_count": len(batch_records),
        "input_text_sha256": _projection_sha256(
            [record["text"] for record in batch_records]
        ),
        "input_token_count": sum(
            token_counts[start_position:end_position]
        ),
        "requested_model": config["embedding"]["requested_model"],
        "start_position": start_position,
    }


def _write_staged_batch(
    vector_path: Path,
    metadata_path: Path,
    matrix: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    """Atomically publish one validated raw embedding batch and metadata."""

    vector_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        dir=vector_path.parent,
    ) as temporary_directory:
        temporary_root = Path(temporary_directory)
        temporary_vector_path = temporary_root / vector_path.name
        temporary_metadata_path = temporary_root / metadata_path.name
        with temporary_vector_path.open("wb") as handle:
            np.save(handle, matrix, allow_pickle=False)
        stored_matrix = np.load(
            temporary_vector_path,
            allow_pickle=False,
        )
        if not np.array_equal(stored_matrix, matrix):
            raise EmbeddingIndexError(
                "Staged embedding batch failed its exact NumPy round trip."
            )

        completed_metadata = {
            **metadata,
            "matrix_sha256": calculate_file_sha256(
                temporary_vector_path
            ),
        }
        metadata_bytes = serialise_json_object(completed_metadata)
        if json.loads(metadata_bytes.decode("utf-8")) != (
            completed_metadata
        ):
            raise EmbeddingIndexError(
                "Staged batch metadata failed its exact JSON round trip."
            )
        temporary_metadata_path.write_bytes(metadata_bytes)

        # Publish metadata last so it marks a complete batch pair.
        os.replace(temporary_vector_path, vector_path)
        os.replace(temporary_metadata_path, metadata_path)


def _load_staged_batch(
    vector_path: Path,
    metadata_path: Path,
    expected_metadata: dict[str, Any],
) -> np.ndarray:
    """Validate and load one complete resumable batch pair."""

    if vector_path.exists() != metadata_path.exists():
        raise EmbeddingIndexError(
            "A staged embedding batch is incomplete."
        )
    if not vector_path.is_file() or not metadata_path.is_file():
        raise EmbeddingIndexError(
            "A staged embedding batch is not a regular file pair."
        )
    try:
        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise EmbeddingIndexError(
            "Staged embedding metadata is invalid."
        ) from error
    metadata = _require_exact_keys(
        metadata,
        BATCH_METADATA_KEYS,
        "Staged embedding metadata",
    )
    for field_name, expected_value in expected_metadata.items():
        if metadata[field_name] != expected_value:
            raise EmbeddingIndexError(
                "Staged embedding metadata differs from the current "
                f"run at field {field_name}."
            )
    _validate_sha256(
        metadata["matrix_sha256"],
        "Staged embedding matrix fingerprint",
    )
    if calculate_file_sha256(vector_path) != metadata["matrix_sha256"]:
        raise EmbeddingIndexError(
            "Staged embedding matrix fingerprint differs from metadata."
        )
    try:
        matrix = np.load(vector_path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise EmbeddingIndexError(
            "Cannot load the staged embedding matrix."
        ) from error
    return validate_raw_embedding_matrix(
        matrix,
        expected_metadata["input_count"],
        expected_metadata["dimensions"],
    )


def embed_chunks_with_staging(
    config: dict[str, Any],
    configuration_sha256: str,
    chunk_records: list[dict[str, Any]],
    token_counts: list[int],
    staging_directory: str | Path,
    embedding_client: EmbeddingClient,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Embed ordered chunks in resumable, fingerprinted request batches."""

    directory = Path(staging_directory)
    directory.mkdir(parents=True, exist_ok=True)
    batch_size = config["request_policy"]["batch_size"]
    dimensions = config["embedding"]["dimensions"]
    batch_matrices = []
    completed_metadata_records = []

    for batch_index, start_position in enumerate(
        range(0, len(chunk_records), batch_size)
    ):
        end_position = min(
            start_position + batch_size,
            len(chunk_records),
        )
        vector_path, metadata_path = _batch_paths(
            directory,
            batch_index,
        )
        expected_metadata = _expected_batch_metadata(
            config,
            configuration_sha256,
            chunk_records,
            token_counts,
            batch_index,
            start_position,
            end_position,
        )

        if vector_path.exists() or metadata_path.exists():
            matrix = _load_staged_batch(
                vector_path,
                metadata_path,
                expected_metadata,
            )
            action = "resumed"
        else:
            texts = [
                record["text"]
                for record in chunk_records[
                    start_position:end_position
                ]
            ]
            try:
                raw_vectors = embedding_client.embed_documents(texts)
            except Exception as error:
                raise EmbeddingIndexError(
                    f"Embedding request failed for batch {batch_index}."
                ) from error
            matrix = validate_raw_embedding_matrix(
                raw_vectors,
                len(texts),
                dimensions,
            )
            _write_staged_batch(
                vector_path,
                metadata_path,
                matrix,
                expected_metadata,
            )
            action = "embedded"

        completed_metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
        batch_matrices.append(matrix)
        completed_metadata_records.append(completed_metadata)
        if progress_callback is not None:
            progress_callback(
                f"Batch {batch_index + 1}/"
                f"{config['expected_source_specific_audit']['request_batch_count']} "
                f"{action}: positions {start_position}:{end_position}"
            )

    combined_matrix = np.ascontiguousarray(
        np.vstack(batch_matrices),
        dtype=np.float32,
    )
    combined_matrix = validate_raw_embedding_matrix(
        combined_matrix,
        len(chunk_records),
        dimensions,
    )
    return combined_matrix, completed_metadata_records


def build_faiss_index(
    normalised_matrix: np.ndarray,
) -> faiss.IndexFlatIP:
    """Build an exact inner-product index over normalised vectors."""

    if normalised_matrix.ndim != 2:
        raise EmbeddingIndexError(
            "FAISS input matrix must be two-dimensional."
        )
    if normalised_matrix.dtype != np.float32:
        raise EmbeddingIndexError(
            "FAISS input matrix must use float32."
        )
    if not normalised_matrix.flags.c_contiguous:
        raise EmbeddingIndexError(
            "FAISS input matrix must be C-contiguous."
        )
    index = faiss.IndexFlatIP(normalised_matrix.shape[1])
    index.add(normalised_matrix)
    if index.d != normalised_matrix.shape[1]:
        raise EmbeddingIndexError(
            "FAISS index dimension differs from the matrix."
        )
    if index.ntotal != normalised_matrix.shape[0]:
        raise EmbeddingIndexError(
            "FAISS index vector count differs from the matrix."
        )
    return index


def _validate_faiss_round_trip(
    index: faiss.IndexFlatIP,
    index_path: Path,
    normalised_matrix: np.ndarray,
) -> faiss.Index:
    """Reload an index and require identical representative searches."""

    try:
        reloaded_index = faiss.read_index(str(index_path))
    except RuntimeError as error:
        raise EmbeddingIndexError(
            "Cannot reload the staged FAISS index."
        ) from error
    if type(reloaded_index).__name__ != "IndexFlatIP":
        raise EmbeddingIndexError(
            "Reloaded FAISS index has an unexpected type."
        )
    if reloaded_index.d != index.d:
        raise EmbeddingIndexError(
            "Reloaded FAISS index has an unexpected dimension."
        )
    if reloaded_index.ntotal != index.ntotal:
        raise EmbeddingIndexError(
            "Reloaded FAISS index has an unexpected vector count."
        )

    representative_positions = sorted(
        {
            0,
            len(normalised_matrix) // 2,
            len(normalised_matrix) - 1,
        }
    )
    queries = np.ascontiguousarray(
        normalised_matrix[representative_positions],
        dtype=np.float32,
    )
    search_depth = min(5, len(normalised_matrix))
    original_scores, original_positions = index.search(
        queries,
        search_depth,
    )
    reloaded_scores, reloaded_positions = reloaded_index.search(
        queries,
        search_depth,
    )
    if not np.array_equal(original_positions, reloaded_positions):
        raise EmbeddingIndexError(
            "FAISS positions changed after index reload."
        )
    if not np.array_equal(original_scores, reloaded_scores):
        raise EmbeddingIndexError(
            "FAISS scores changed after index reload."
        )
    return reloaded_index


def build_openai_embedding_client(
    config: dict[str, Any],
    project_root: str | Path,
) -> OpenAIEmbeddings:
    """Build the exact production LangChain embedding integration."""

    root = Path(project_root).resolve()
    env_path = root / config["embedding"][
        "api_key_file_relative_path"
    ]
    _require_within_directory(
        env_path,
        root,
        "API-key file",
    )
    load_dotenv(env_path, override=False)
    environment_name = config["embedding"][
        "api_key_environment_variable"
    ]
    if not os.getenv(environment_name):
        raise EmbeddingIndexError(
            f"{environment_name} is not available to the process."
        )

    request_policy = config["request_policy"]
    return OpenAIEmbeddings(
        model=config["embedding"]["requested_model"],
        dimensions=config["embedding"]["dimensions_argument"],
        chunk_size=request_policy["batch_size"],
        max_retries=request_policy["max_retries"],
        timeout=request_policy["timeout_seconds"],
        show_progress_bar=request_policy["show_progress_bar"],
        skip_empty=request_policy["skip_empty"],
        check_embedding_ctx_length=request_policy[
            "check_embedding_context_length"
        ],
        tiktoken_enabled=request_policy["tiktoken_enabled"],
        tiktoken_model_name=request_policy["tiktoken_model_name"],
        retry_min_seconds=request_policy["retry_min_seconds"],
        retry_max_seconds=request_policy["retry_max_seconds"],
    )


def _resolve_project_path(
    project_root: Path,
    relative_path: Path,
    allowed_directory: Path,
    description: str,
) -> Path:
    """Resolve a path within its authorised project directory."""

    path = project_root / relative_path
    directory = project_root / allowed_directory
    _require_within_directory(path, directory, description)
    return path


def _validate_mapping_round_trip(
    serialised: bytes,
    records: list[dict[str, Any]],
) -> None:
    """Require exact index-mapping recovery before publication."""

    if not serialised.endswith(b"\n"):
        raise EmbeddingIndexError(
            "Index mapping has no final newline."
        )
    recovered = [
        json.loads(line)
        for line in serialised.decode("utf-8").splitlines()
    ]
    if recovered != records:
        raise EmbeddingIndexError(
            "Index mapping failed its exact JSONL round trip."
        )


def _validate_summary_round_trip(
    serialised: bytes,
    record: dict[str, Any],
) -> None:
    """Require exact summary recovery before publication."""

    if not serialised.endswith(b"\n"):
        raise EmbeddingIndexError(
            "Embedding-index summary has no final newline."
        )
    if json.loads(serialised.decode("utf-8")) != record:
        raise EmbeddingIndexError(
            "Embedding-index summary failed its exact JSON round trip."
        )


def _build_embedding_index_summary(
    config: dict[str, Any],
    config_path: Path,
    project_root: Path,
    chunk_records: list[dict[str, Any]],
    token_counts: list[int],
    batch_metadata_records: list[dict[str, Any]],
    normalised_matrix: np.ndarray,
    output_paths: dict[str, Path],
    staged_paths: dict[str, Path],
    output_fingerprints: dict[str, str],
) -> dict[str, Any]:
    """Build the deterministic final embedding-index audit record."""

    relative_config_path = config_path.relative_to(project_root).as_posix()
    return {
        "batch_count": len(batch_metadata_records),
        "batch_metadata_projection_sha256": _projection_sha256(
            batch_metadata_records
        ),
        "batch_size": config["request_policy"]["batch_size"],
        "chunk_count": len(chunk_records),
        "configuration_id": config["configuration_id"],
        "configuration_relative_path": relative_config_path,
        "configuration_sha256": calculate_file_sha256(config_path),
        "embedding_client_class": config["embedding"]["client_class"],
        "embedding_client_package": config["embedding"]["client_package"],
        "embedding_client_package_version": config["embedding"][
            "client_package_version"
        ],
        "embedding_dimensions": int(normalised_matrix.shape[1]),
        "embedding_dtype": str(normalised_matrix.dtype),
        "embedding_matrix_relative_path": output_paths[
            "embedding_matrix"
        ].relative_to(project_root).as_posix(),
        "embedding_matrix_sha256": output_fingerprints[
            "embedding_matrix"
        ],
        "embedding_matrix_size_bytes": staged_paths[
            "embedding_matrix"
        ].stat().st_size,
        "embedding_norm_maximum": float(
            np.linalg.norm(normalised_matrix, axis=1).max()
        ),
        "embedding_norm_minimum": float(
            np.linalg.norm(normalised_matrix, axis=1).min()
        ),
        "embedding_token_count": sum(token_counts),
        "faiss_index_class": config["faiss"]["index_class"],
        "faiss_index_relative_path": output_paths[
            "faiss_index"
        ].relative_to(project_root).as_posix(),
        "faiss_index_sha256": output_fingerprints["faiss_index"],
        "faiss_index_size_bytes": staged_paths[
            "faiss_index"
        ].stat().st_size,
        "faiss_package_version": config["faiss"]["package_version"],
        "input_chunk_schema_version": config["input"][
            "chunk_schema_version"
        ],
        "input_chunks_relative_path": config["input"]["relative_path"],
        "input_chunks_sha256": config["input"]["sha256"],
        "input_chunking_summary_sha256": config["input"][
            "chunking_summary_sha256"
        ],
        "index_mapping_record_count": len(chunk_records),
        "index_mapping_relative_path": output_paths[
            "index_mapping"
        ].relative_to(project_root).as_posix(),
        "index_mapping_schema_version": INDEX_MAPPING_SCHEMA_VERSION,
        "index_mapping_sha256": output_fingerprints[
            "index_mapping"
        ],
        "index_mapping_size_bytes": staged_paths[
            "index_mapping"
        ].stat().st_size,
        "normalisation": config["vector_representation"][
            "normalisation"
        ],
        "requested_embedding_model": config["embedding"][
            "requested_model"
        ],
        "similarity_interpretation": config["faiss"][
            "similarity_interpretation"
        ],
        "summary_schema_version": EMBEDDING_INDEX_SUMMARY_SCHEMA_VERSION,
        "token_encoding": config["embedding"]["token_encoding"],
        "tiktoken_package_version": _installed_package_version(
            "tiktoken"
        ),
    }


def export_embedding_index(
    config_path: str | Path,
    project_root: str | Path | None = None,
    overwrite: bool = False,
    embedding_client: EmbeddingClient | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> EmbeddingIndexResult:
    """Validate, embed, index and publish the configured production outputs."""

    root = (
        Path.cwd()
        if project_root is None
        else Path(project_root)
    ).resolve()
    resolved_config_path = Path(config_path)
    if not resolved_config_path.is_absolute():
        resolved_config_path = root / resolved_config_path
    _require_within_directory(
        resolved_config_path,
        root / "configs",
        "Embedding-index configuration",
    )
    config = load_embedding_index_config(resolved_config_path)

    output_config = config["output"]
    output_paths = {
        "embedding_matrix": _resolve_project_path(
            root,
            Path(output_config["embedding_matrix_relative_path"]),
            Path("data/processed/embeddings"),
            "Embedding matrix output",
        ),
        "faiss_index": _resolve_project_path(
            root,
            Path(output_config["faiss_index_relative_path"]),
            Path("data/processed/index"),
            "FAISS index output",
        ),
        "index_mapping": _resolve_project_path(
            root,
            Path(output_config["index_mapping_relative_path"]),
            Path("data/processed/index"),
            "Index mapping output",
        ),
        "summary": _resolve_project_path(
            root,
            Path(output_config["summary_relative_path"]),
            Path("data/processed/audit"),
            "Embedding-index summary output",
        ),
    }
    staging_directory = _resolve_project_path(
        root,
        Path(output_config["staging_relative_directory"]),
        Path("data/processed/embeddings"),
        "Embedding staging directory",
    )

    existing_outputs = [
        path
        for path in output_paths.values()
        if path.exists()
    ]
    if existing_outputs and not overwrite:
        raise EmbeddingIndexError(
            "Embedding-index output already exists; select explicit "
            "overwrite to replace it."
        )

    chunk_records, token_counts = load_validated_chunk_records(
        config,
        root,
    )
    client = (
        build_openai_embedding_client(config, root)
        if embedding_client is None
        else embedding_client
    )
    configuration_sha256 = calculate_file_sha256(
        resolved_config_path
    )
    raw_matrix, batch_metadata_records = embed_chunks_with_staging(
        config,
        configuration_sha256,
        chunk_records,
        token_counts,
        staging_directory,
        client,
        progress_callback,
    )
    normalised_matrix = normalise_embedding_matrix(
        raw_matrix,
        config["vector_representation"][
            "post_normalisation_norm_tolerance"
        ],
    )
    expected_shape = tuple(
        config["expected_source_specific_audit"][
            "embedding_matrix_shape"
        ]
    )
    if normalised_matrix.shape != expected_shape:
        raise EmbeddingIndexError(
            "Final embedding matrix shape differs from configuration."
        )

    index = build_faiss_index(normalised_matrix)
    mapping_records = build_index_mapping_records(chunk_records)
    mapping_bytes = serialise_json_lines(mapping_records)
    _validate_mapping_round_trip(mapping_bytes, mapping_records)

    with TemporaryDirectory(dir=root) as temporary_directory:
        temporary_root = Path(temporary_directory)
        staged_paths = {
            "embedding_matrix": temporary_root / "chunk-embeddings.npy",
            "faiss_index": temporary_root / "chunks.faiss",
            "index_mapping": temporary_root / "chunk-mapping.jsonl",
            "summary": temporary_root / "embedding-index-summary.json",
        }

        with staged_paths["embedding_matrix"].open("wb") as handle:
            np.save(handle, normalised_matrix, allow_pickle=False)
        reloaded_matrix = np.load(
            staged_paths["embedding_matrix"],
            allow_pickle=False,
        )
        if not np.array_equal(reloaded_matrix, normalised_matrix):
            raise EmbeddingIndexError(
                "Final embedding matrix failed its exact NumPy round trip."
            )

        faiss.write_index(index, str(staged_paths["faiss_index"]))
        _validate_faiss_round_trip(
            index,
            staged_paths["faiss_index"],
            normalised_matrix,
        )
        staged_paths["index_mapping"].write_bytes(mapping_bytes)

        output_fingerprints = {
            key: calculate_file_sha256(staged_paths[key])
            for key in (
                "embedding_matrix",
                "faiss_index",
                "index_mapping",
            )
        }
        summary = _build_embedding_index_summary(
            config,
            resolved_config_path,
            root,
            chunk_records,
            token_counts,
            batch_metadata_records,
            normalised_matrix,
            output_paths,
            staged_paths,
            output_fingerprints,
        )
        summary_bytes = serialise_json_object(summary)
        _validate_summary_round_trip(summary_bytes, summary)
        staged_paths["summary"].write_bytes(summary_bytes)
        summary_sha256 = calculate_file_sha256(
            staged_paths["summary"]
        )

        for output_path in output_paths.values():
            output_path.parent.mkdir(parents=True, exist_ok=True)

        # Publish the summary last so its presence marks a complete final set.
        for key in (
            "embedding_matrix",
            "faiss_index",
            "index_mapping",
            "summary",
        ):
            os.replace(staged_paths[key], output_paths[key])

    return EmbeddingIndexResult(
        embedding_matrix_path=output_paths["embedding_matrix"],
        faiss_index_path=output_paths["faiss_index"],
        index_mapping_path=output_paths["index_mapping"],
        summary_path=output_paths["summary"],
        embedding_matrix_sha256=output_fingerprints[
            "embedding_matrix"
        ],
        faiss_index_sha256=output_fingerprints["faiss_index"],
        index_mapping_sha256=output_fingerprints[
            "index_mapping"
        ],
        summary_sha256=summary_sha256,
        chunk_count=len(chunk_records),
        dimensions=normalised_matrix.shape[1],
        batch_count=len(batch_metadata_records),
    )


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the production embedding-index command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate validated chunk embeddings and build the "
            "persistent exact FAISS index."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/embedding-index-config.json",
        help="Repository-relative embedding-index configuration path.",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root containing configs and processed data.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing final embedding-index outputs.",
    )
    return parser


def main() -> None:
    """Run production embedding generation and FAISS index construction."""

    arguments = _build_argument_parser().parse_args()
    result = export_embedding_index(
        config_path=arguments.config,
        project_root=arguments.project_root,
        overwrite=arguments.overwrite,
        progress_callback=print,
    )
    print("Embedding and FAISS indexing: PASSED")
    print("  Retrieval chunks:", result.chunk_count)
    print("  Embedding dimensions:", result.dimensions)
    print("  Request batches:", result.batch_count)
    print("  Embedding matrix:", result.embedding_matrix_path)
    print(
        "  Embedding matrix SHA-256:",
        result.embedding_matrix_sha256,
    )
    print("  FAISS index:", result.faiss_index_path)
    print("  FAISS index SHA-256:", result.faiss_index_sha256)
    print("  Index mapping:", result.index_mapping_path)
    print(
        "  Index mapping SHA-256:",
        result.index_mapping_sha256,
    )
    print("  Summary:", result.summary_path)
    print("  Summary SHA-256:", result.summary_sha256)


if __name__ == "__main__":
    main()
