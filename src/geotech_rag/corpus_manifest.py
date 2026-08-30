"""Load and validate the versioned retrieval-corpus manifest."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version
from pathlib import Path
from typing import Any
import json

import pymupdf


SUPPORTED_MANIFEST_SCHEMA_VERSIONS = frozenset(
    {"1.0"}
)
HASH_BLOCK_SIZE_BYTES = 1024 * 1024
SHA256_HEXADECIMAL_CHARACTERS = frozenset(
    "0123456789abcdef"
)

EXPECTED_CORPUS_ROLE = "retrieval_corpus"
EXPECTED_RAW_CORPUS_DIRECTORY = "data/raw/corpus/"
EXPECTED_INTERIM_DIRECTORY = "data/interim/"
EXPECTED_EVALUATION_DIRECTORY = (
    "data/raw/evaluation/questions/"
)
EXPECTED_GROUND_TRUTH_DIRECTORY = (
    "data/raw/ground_truth/"
)
REQUIRED_EXCLUDED_CONTENT = frozenset(
    {
        "chapter problem regions",
        "answers to selected problems",
        "index",
    }
)


class CorpusManifestError(RuntimeError):
    """Base error for corpus-manifest validation."""


class ManifestValidationError(CorpusManifestError):
    """Raised when the manifest structure or policy is invalid."""


class SourceIntegrityError(CorpusManifestError):
    """Raised when the local source does not match the manifest."""


@dataclass(frozen=True, slots=True)
class CorpusSourceValidation:
    """Verified identity of one local corpus source."""

    manifest_path: Path
    source_path: Path
    source_id: str
    sha256: str
    size_bytes: int
    pdf_page_count: int
    parser_distribution: str
    parser_version: str

    def as_dict(self) -> dict[str, str | int]:
        """Return a serialisable validation summary."""

        return {
            "manifest_path": self.manifest_path.as_posix(),
            "source_path": self.source_path.as_posix(),
            "source_id": self.source_id,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "pdf_page_count": self.pdf_page_count,
            "parser_distribution": (
                self.parser_distribution
            ),
            "parser_version": self.parser_version,
        }


def _require_mapping(
    parent: Mapping[str, Any],
    key: str,
    context: str,
) -> Mapping[str, Any]:
    """Return one required JSON object."""

    value = parent.get(key)

    if not isinstance(value, Mapping):
        raise ManifestValidationError(
            f"{context}.{key} must be a JSON object."
        )

    return value


def _require_string(
    parent: Mapping[str, Any],
    key: str,
    context: str,
) -> str:
    """Return one required non-empty string."""

    value = parent.get(key)

    if not isinstance(value, str) or not value.strip():
        raise ManifestValidationError(
            f"{context}.{key} must be a "
            "non-empty string."
        )

    return value


def _require_string_list(
    parent: Mapping[str, Any],
    key: str,
    context: str,
) -> tuple[str, ...]:
    """Return one required array of non-empty strings."""

    value = parent.get(key)

    if (
        not isinstance(value, list)
        or not value
        or any(
            not isinstance(item, str)
            or not item.strip()
            for item in value
        )
    ):
        raise ManifestValidationError(
            f"{context}.{key} must be a "
            "non-empty array of strings."
        )

    return tuple(value)


def _require_positive_integer(
    parent: Mapping[str, Any],
    key: str,
    context: str,
) -> int:
    """Return one required positive integer."""

    value = parent.get(key)

    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
    ):
        raise ManifestValidationError(
            f"{context}.{key} must be a "
            "positive integer."
        )

    return value


def _require_false(
    parent: Mapping[str, Any],
    key: str,
    context: str,
) -> None:
    """Require a safety policy to be explicitly false."""

    if parent.get(key) is not False:
        raise ManifestValidationError(
            f"{context}.{key} must be false."
        )


def load_corpus_manifest(
    manifest_path: str | Path,
) -> dict[str, Any]:
    """
    Load one JSON corpus manifest.

    File-access and JSON-decoding failures are converted into
    project-specific validation errors for consistent API handling.
    """

    resolved_manifest_path = Path(
        manifest_path
    ).resolve()

    try:
        manifest_text = (
            resolved_manifest_path.read_text(
                encoding="utf-8"
            )
        )
    except OSError as error:
        raise ManifestValidationError(
            "The corpus manifest could not be read: "
            f"{resolved_manifest_path}"
        ) from error

    try:
        manifest = json.loads(manifest_text)
    except json.JSONDecodeError as error:
        raise ManifestValidationError(
            "The corpus manifest is not valid JSON: "
            f"{resolved_manifest_path}"
        ) from error

    if not isinstance(manifest, dict):
        raise ManifestValidationError(
            "The corpus manifest root must be "
            "a JSON object."
        )

    return manifest


def _validate_manifest_structure(
    manifest: Mapping[str, Any],
) -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
]:
    """
    Validate the schema fields required before source access.

    Returns the source, boundary and extraction records used by
    the file-integrity stage.
    """

    schema_version = _require_string(
        manifest,
        "manifest_schema_version",
        "manifest",
    )

    if (
        schema_version
        not in SUPPORTED_MANIFEST_SCHEMA_VERSIONS
    ):
        raise ManifestValidationError(
            "Unsupported manifest schema version: "
            f"{schema_version}"
        )

    corpus_record = _require_mapping(
        manifest,
        "corpus",
        "manifest",
    )

    if (
        _require_string(
            corpus_record,
            "role",
            "manifest.corpus",
        )
        != EXPECTED_CORPUS_ROLE
    ):
        raise ManifestValidationError(
            "manifest.corpus.role must be "
            f"{EXPECTED_CORPUS_ROLE}."
        )

    if (
        _require_string(
            corpus_record,
            "processed_output_directory",
            "manifest.corpus",
        )
        != EXPECTED_INTERIM_DIRECTORY
    ):
        raise ManifestValidationError(
            "manifest.corpus."
            "processed_output_directory must be "
            f"{EXPECTED_INTERIM_DIRECTORY}."
        )

    declared_source_count = (
        _require_positive_integer(
            corpus_record,
            "source_document_count",
            "manifest.corpus",
        )
    )

    sources = manifest.get("sources")

    if not isinstance(sources, list):
        raise ManifestValidationError(
            "manifest.sources must be a JSON array."
        )

    if len(sources) != declared_source_count:
        raise ManifestValidationError(
            "The declared source count does not "
            "match manifest.sources."
        )

    if declared_source_count != 1:
        raise ManifestValidationError(
            "The reconstructed baseline must contain "
            "exactly one textbook source."
        )

    source_record = sources[0]

    if not isinstance(source_record, Mapping):
        raise ManifestValidationError(
            "manifest.sources[0] must be "
            "a JSON object."
        )

    _require_string(
        source_record,
        "source_id",
        "manifest.sources[0]",
    )
    _require_string(
        source_record,
        "relative_path",
        "manifest.sources[0]",
    )

    if (
        _require_string(
            source_record,
            "media_type",
            "manifest.sources[0]",
        )
        != "application/pdf"
    ):
        raise ManifestValidationError(
            "The baseline source media type must "
            "be application/pdf."
        )

    expected_sha256 = _require_string(
        source_record,
        "sha256",
        "manifest.sources[0]",
    )

    if (
        len(expected_sha256) != 64
        or any(
            character
            not in SHA256_HEXADECIMAL_CHARACTERS
            for character in expected_sha256
        )
    ):
        raise ManifestValidationError(
            "manifest.sources[0].sha256 must be "
            "a lowercase SHA-256 digest."
        )

    _require_positive_integer(
        source_record,
        "size_bytes",
        "manifest.sources[0]",
    )
    _require_positive_integer(
        source_record,
        "pdf_page_count",
        "manifest.sources[0]",
    )

    repository_policy = _require_mapping(
        source_record,
        "repository_policy",
        "manifest.sources[0]",
    )
    _require_false(
        repository_policy,
        "tracked_by_git",
        (
            "manifest.sources[0]."
            "repository_policy"
        ),
    )
    _require_false(
        repository_policy,
        "copyrighted_source_text_in_manifest",
        (
            "manifest.sources[0]."
            "repository_policy"
        ),
    )

    pipeline_boundaries = _require_mapping(
        manifest,
        "pipeline_boundaries",
        "manifest",
    )

    raw_corpus_directory = _require_string(
        pipeline_boundaries,
        "only_raw_corpus_input_directory",
        "manifest.pipeline_boundaries",
    )

    if (
        raw_corpus_directory
        != EXPECTED_RAW_CORPUS_DIRECTORY
    ):
        raise ManifestValidationError(
            "The raw corpus input directory must be "
            f"{EXPECTED_RAW_CORPUS_DIRECTORY}."
        )

    evaluation_directory = _require_string(
        pipeline_boundaries,
        "evaluation_questions_directory",
        "manifest.pipeline_boundaries",
    )

    if (
        evaluation_directory
        != EXPECTED_EVALUATION_DIRECTORY
    ):
        raise ManifestValidationError(
            "The evaluation questions directory must be "
            f"{EXPECTED_EVALUATION_DIRECTORY}."
        )

    ground_truth_directory = _require_string(
        pipeline_boundaries,
        "ground_truth_directory",
        "manifest.pipeline_boundaries",
    )

    if (
        ground_truth_directory
        != EXPECTED_GROUND_TRUTH_DIRECTORY
    ):
        raise ManifestValidationError(
            "The ground-truth directory must be "
            f"{EXPECTED_GROUND_TRUTH_DIRECTORY}."
        )

    excluded_content = frozenset(
        _require_string_list(
            pipeline_boundaries,
            "excluded_from_retrieval_corpus",
            "manifest.pipeline_boundaries",
        )
    )
    missing_exclusions = (
        REQUIRED_EXCLUDED_CONTENT
        - excluded_content
    )

    if missing_exclusions:
        raise ManifestValidationError(
            "The manifest is missing required corpus "
            "exclusions: "
            + ", ".join(
                sorted(missing_exclusions)
            )
        )

    _require_false(
        pipeline_boundaries,
        "direct_indexing_of_raw_pdf_allowed",
        "manifest.pipeline_boundaries",
    )
    _require_false(
        pipeline_boundaries,
        "ground_truth_available_during_generation",
        "manifest.pipeline_boundaries",
    )

    extraction_decision = _require_mapping(
        manifest,
        "extraction_decision",
        "manifest",
    )
    _require_string(
        extraction_decision,
        "production_library",
        "manifest.extraction_decision",
    )
    _require_string(
        extraction_decision,
        "package_version",
        "manifest.extraction_decision",
    )
    _require_false(
        extraction_decision,
        "formula_reconstruction_in_baseline",
        "manifest.extraction_decision",
    )

    return (
        source_record,
        pipeline_boundaries,
        extraction_decision,
    )


def _resolve_project_path(
    project_root: Path,
    relative_path: str,
    field_name: str,
) -> Path:
    """Resolve one relative path without allowing project escape."""

    candidate_path = Path(relative_path)

    if candidate_path.is_absolute():
        raise ManifestValidationError(
            f"{field_name} must be relative."
        )

    resolved_project_root = project_root.resolve()
    resolved_path = (
        resolved_project_root / candidate_path
    ).resolve()

    if not resolved_path.is_relative_to(
        resolved_project_root
    ):
        raise ManifestValidationError(
            f"{field_name} escapes the project root."
        )

    return resolved_path


def calculate_file_sha256(
    source_path: str | Path,
) -> str:
    """Calculate one file's SHA-256 digest in memory-safe blocks."""

    resolved_source_path = Path(source_path)
    source_hasher = sha256()

    try:
        with resolved_source_path.open("rb") as source_file:
            for file_block in iter(
                lambda: source_file.read(
                    HASH_BLOCK_SIZE_BYTES
                ),
                b"",
            ):
                source_hasher.update(file_block)
    except OSError as error:
        raise SourceIntegrityError(
            "The corpus source could not be read: "
            f"{resolved_source_path}"
        ) from error

    return source_hasher.hexdigest()


def validate_corpus_manifest(
    manifest_path: str | Path,
    project_root: str | Path | None = None,
) -> CorpusSourceValidation:
    """
    Validate manifest structure, source identity and parser version.

    When ``project_root`` is omitted, the root is inferred from a
    manifest stored directly inside the project's ``configs`` folder.
    """

    resolved_manifest_path = Path(
        manifest_path
    ).resolve()
    manifest = load_corpus_manifest(
        resolved_manifest_path
    )

    (
        source_record,
        pipeline_boundaries,
        extraction_decision,
    ) = _validate_manifest_structure(manifest)

    if project_root is None:
        resolved_project_root = (
            resolved_manifest_path.parent.parent
        )
    else:
        resolved_project_root = Path(
            project_root
        ).resolve()

    raw_corpus_directory = _resolve_project_path(
        resolved_project_root,
        _require_string(
            pipeline_boundaries,
            "only_raw_corpus_input_directory",
            "manifest.pipeline_boundaries",
        ),
        (
            "manifest.pipeline_boundaries."
            "only_raw_corpus_input_directory"
        ),
    )
    source_path = _resolve_project_path(
        resolved_project_root,
        _require_string(
            source_record,
            "relative_path",
            "manifest.sources[0]",
        ),
        "manifest.sources[0].relative_path",
    )

    if not source_path.is_relative_to(
        raw_corpus_directory
    ):
        raise ManifestValidationError(
            "The corpus source is outside the "
            "permitted raw corpus directory."
        )

    if not source_path.is_file():
        raise SourceIntegrityError(
            f"Corpus source not found: {source_path}"
        )

    actual_sha256 = calculate_file_sha256(
        source_path
    )
    expected_sha256 = _require_string(
        source_record,
        "sha256",
        "manifest.sources[0]",
    )

    if actual_sha256 != expected_sha256:
        raise SourceIntegrityError(
            "The corpus source SHA-256 does not "
            "match the manifest."
        )

    actual_size_bytes = source_path.stat().st_size
    expected_size_bytes = (
        _require_positive_integer(
            source_record,
            "size_bytes",
            "manifest.sources[0]",
        )
    )

    if actual_size_bytes != expected_size_bytes:
        raise SourceIntegrityError(
            "The corpus source byte size does not "
            "match the manifest."
        )

    try:
        with pymupdf.open(
            str(source_path)
        ) as pdf_document:
            actual_page_count = (
                pdf_document.page_count
            )
    except Exception as error:
        raise SourceIntegrityError(
            "The corpus source could not be opened "
            "as a PDF."
        ) from error

    expected_page_count = (
        _require_positive_integer(
            source_record,
            "pdf_page_count",
            "manifest.sources[0]",
        )
    )

    if actual_page_count != expected_page_count:
        raise SourceIntegrityError(
            "The corpus source page count does not "
            "match the manifest."
        )

    parser_distribution = _require_string(
        extraction_decision,
        "production_library",
        "manifest.extraction_decision",
    )
    expected_parser_version = _require_string(
        extraction_decision,
        "package_version",
        "manifest.extraction_decision",
    )

    try:
        installed_parser_version = version(
            parser_distribution
        )
    except PackageNotFoundError as error:
        raise SourceIntegrityError(
            "The production parser distribution "
            f"is not installed: {parser_distribution}"
        ) from error

    if (
        installed_parser_version
        != expected_parser_version
    ):
        raise SourceIntegrityError(
            "The installed production parser version "
            "does not match the manifest."
        )

    return CorpusSourceValidation(
        manifest_path=resolved_manifest_path,
        source_path=source_path,
        source_id=_require_string(
            source_record,
            "source_id",
            "manifest.sources[0]",
        ),
        sha256=actual_sha256,
        size_bytes=actual_size_bytes,
        pdf_page_count=actual_page_count,
        parser_distribution=parser_distribution,
        parser_version=installed_parser_version,
    )
