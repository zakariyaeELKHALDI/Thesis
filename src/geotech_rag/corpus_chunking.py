"""Build deterministic retrieval chunks from validated corpus regions."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter

from geotech_rag.corpus_export import (
    COORDINATE_ORIGIN,
    COORDINATE_UNIT,
    LINE_REGION_COORDINATE_TOLERANCE,
    RECORD_SCHEMA_VERSION,
    serialise_json_lines,
    serialise_json_object,
)
from geotech_rag.corpus_manifest import calculate_file_sha256


CHUNKING_CONFIG_SCHEMA_VERSION = "1.1"
CHUNK_SCHEMA_VERSION = "1.0"
CHUNKING_SUMMARY_SCHEMA_VERSION = "1.0"

REGIONS_RELATIVE_PATH = Path(
    "data/interim/corpus/regions.jsonl"
)
CHUNKS_RELATIVE_PATH = Path(
    "data/processed/corpus/chunks.jsonl"
)
CHUNKING_SUMMARY_RELATIVE_PATH = Path(
    "data/processed/audit/chunking-summary.json"
)

RETRIEVAL_PAGE_STATES = {
    "fully_retained",
    "partially_retained",
}
REGION_KEYS = {
    "character_count",
    "coordinate_origin",
    "coordinate_unit",
    "line_count",
    "lines",
    "page_height",
    "page_state",
    "page_width",
    "pdf_page_index",
    "pdf_page_number",
    "printed_page_number",
    "record_id",
    "record_schema_version",
    "region_bbox",
    "region_index",
    "source_id",
    "source_sha256",
    "text",
}
LINE_KEYS = {
    "bbox",
    "block_number",
    "direction",
    "line_index",
    "line_number",
    "span_count",
    "text",
    "writing_mode",
}
SPLITTER_KEYS = {
    "add_start_index",
    "chunk_overlap",
    "chunk_size",
    "class_name",
    "embedded_line_newline_sentinel",
    "is_separator_regex",
    "keep_separator",
    "length_function",
    "package",
    "package_version",
    "protect_embedded_line_newlines",
    "separator",
    "strip_whitespace",
}
AUDIT_KEYS = {
    "adjacent_chunk_pair_count",
    "chunk_count",
    "chunk_projection_sha256",
    "chunks_with_formula_controls",
    "maximum_chunk_character_count",
    "maximum_chunks_per_parent_record",
    "oversized_chunk_count",
    "single_chunk_parent_record_count",
    "total_chunk_character_count",
}


class CorpusChunkingError(RuntimeError):
    """Raised when chunking violates the frozen contract."""


@dataclass(frozen=True)
class CorpusChunkingResult:
    """Paths, fingerprints and counts from one chunking run."""

    chunks_path: Path
    chunking_summary_path: Path
    chunks_sha256: str
    chunking_summary_sha256: str
    chunk_count: int
    parent_record_count: int


def _require_exact_keys(
    value: Any,
    expected_keys: set[str],
    description: str,
) -> dict[str, Any]:
    """Require an object containing exactly the expected keys."""

    if not isinstance(value, dict):
        raise CorpusChunkingError(
            f"{description} must be an object."
        )

    actual_keys = set(value)

    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise CorpusChunkingError(
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
        raise CorpusChunkingError(
            f"{description} must be an integer."
        )

    if minimum is not None and value < minimum:
        raise CorpusChunkingError(
            f"{description} must be at least {minimum}."
        )

    return value


def _require_finite_number(
    value: Any,
    description: str,
) -> float:
    """Return a finite numeric value while rejecting booleans."""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
    ):
        raise CorpusChunkingError(
            f"{description} must be a finite number."
        )

    return float(value)


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
        raise CorpusChunkingError(
            f"{description} is not a valid SHA-256."
        )

    return value


def _validate_relative_path(
    value: Any,
    description: str,
) -> Path:
    """Validate a normalised repository-relative POSIX path."""

    if not isinstance(value, str) or not value:
        raise CorpusChunkingError(
            f"{description} must be a non-empty string."
        )

    path = Path(value)

    if (
        path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != value
    ):
        raise CorpusChunkingError(
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
        raise CorpusChunkingError(
            f"{description} lies outside {directory}."
        ) from error


def _validate_bbox(
    value: Any,
    description: str,
) -> list[float]:
    """Validate and normalise a four-coordinate bounding box."""

    if not isinstance(value, list) or len(value) != 4:
        raise CorpusChunkingError(
            f"{description} must contain four coordinates."
        )

    bbox = [
        _require_finite_number(
            coordinate,
            f"{description} coordinate",
        )
        for coordinate in value
    ]

    if bbox[2] < bbox[0] or bbox[3] < bbox[1]:
        raise CorpusChunkingError(
            f"{description} has reversed coordinates."
        )

    return bbox


def _installed_package_version(package_name: str) -> str:
    """Return the installed version of a required package."""

    try:
        return version(package_name)
    except PackageNotFoundError as error:
        raise CorpusChunkingError(
            f"Required package is not installed: {package_name}."
        ) from error


def _validate_splitter_config(
    raw_config: Any,
) -> dict[str, Any]:
    """Validate the exact audited splitter configuration."""

    config = _require_exact_keys(
        raw_config,
        SPLITTER_KEYS,
        "Splitter configuration",
    )

    if config["class_name"] != "CharacterTextSplitter":
        raise CorpusChunkingError(
            "Only CharacterTextSplitter is supported."
        )

    if config["package"] != "langchain-text-splitters":
        raise CorpusChunkingError(
            "Unexpected text-splitter package."
        )

    package_version = config["package_version"]

    if not isinstance(package_version, str):
        raise CorpusChunkingError(
            "Splitter package version must be a string."
        )

    if _installed_package_version(config["package"]) != (
        package_version
    ):
        raise CorpusChunkingError(
            "Installed text-splitter version differs from "
            "the chunking configuration."
        )

    if config["separator"] != "\n":
        raise CorpusChunkingError(
            "The production separator must be one newline."
        )

    chunk_size = _require_integer(
        config["chunk_size"],
        "Chunk size",
        minimum=1,
    )
    chunk_overlap = _require_integer(
        config["chunk_overlap"],
        "Chunk overlap",
        minimum=0,
    )

    if chunk_overlap >= chunk_size:
        raise CorpusChunkingError(
            "Chunk overlap must be smaller than chunk size."
        )

    if config["length_function"] != "len":
        raise CorpusChunkingError(
            "Only Python len is supported."
        )

    expected_values = {
        "add_start_index": True,
        "is_separator_regex": False,
        "keep_separator": False,
        "protect_embedded_line_newlines": True,
        "strip_whitespace": False,
    }

    for field_name, expected_value in expected_values.items():
        if config[field_name] is not expected_value:
            raise CorpusChunkingError(
                f"Splitter {field_name} must equal "
                f"{expected_value}."
            )

    if config["embedded_line_newline_sentinel"] != "U+E000":
        raise CorpusChunkingError(
            "The embedded-line newline sentinel must be U+E000."
        )

    return config


def load_chunking_config(
    config_path: str | Path,
) -> dict[str, Any]:
    """Load and validate the frozen chunking configuration."""

    path = Path(config_path)

    try:
        config = json.loads(
            path.read_text(encoding="utf-8")
        )
    except OSError as error:
        raise CorpusChunkingError(
            f"Cannot read chunking configuration: {path}."
        ) from error
    except json.JSONDecodeError as error:
        raise CorpusChunkingError(
            f"Chunking configuration is invalid JSON: {path}."
        ) from error

    config = _require_exact_keys(
        config,
        {
            "chunking_config_schema_version",
            "expected_source_specific_audit",
            "input",
            "output",
            "oversized_chunk_policy",
            "splitter",
        },
        "Chunking configuration",
    )

    if config["chunking_config_schema_version"] != (
        CHUNKING_CONFIG_SCHEMA_VERSION
    ):
        raise CorpusChunkingError(
            "Unsupported chunking-config schema version."
        )

    input_config = _require_exact_keys(
        config["input"],
        {
            "character_count",
            "line_count",
            "record_count",
            "record_schema_version",
            "relative_path",
            "sha256",
        },
        "Chunking input configuration",
    )
    input_path = _validate_relative_path(
        input_config["relative_path"],
        "Chunking input path",
    )

    if input_path != REGIONS_RELATIVE_PATH:
        raise CorpusChunkingError(
            "Only the validated regions.jsonl export may "
            "enter production chunking."
        )

    _validate_sha256(
        input_config["sha256"],
        "Expected input fingerprint",
    )

    if input_config["record_schema_version"] != (
        RECORD_SCHEMA_VERSION
    ):
        raise CorpusChunkingError(
            "Unsupported parent record schema version."
        )

    for field_name in (
        "record_count",
        "line_count",
        "character_count",
    ):
        _require_integer(
            input_config[field_name],
            f"Input {field_name}",
            minimum=1,
        )

    config["splitter"] = _validate_splitter_config(
        config["splitter"]
    )

    if config["oversized_chunk_policy"] != "error":
        raise CorpusChunkingError(
            "Only the fail-on-oversize policy is supported."
        )

    output_config = _require_exact_keys(
        config["output"],
        {
            "chunk_schema_version",
            "chunks_relative_path",
            "summary_relative_path",
            "summary_schema_version",
        },
        "Chunking output configuration",
    )

    if output_config["chunk_schema_version"] != (
        CHUNK_SCHEMA_VERSION
    ):
        raise CorpusChunkingError(
            "Unsupported chunk schema version."
        )

    if output_config["summary_schema_version"] != (
        CHUNKING_SUMMARY_SCHEMA_VERSION
    ):
        raise CorpusChunkingError(
            "Unsupported chunking-summary schema version."
        )

    chunks_path = _validate_relative_path(
        output_config["chunks_relative_path"],
        "Chunks output path",
    )
    summary_path = _validate_relative_path(
        output_config["summary_relative_path"],
        "Chunking-summary output path",
    )

    if chunks_path != CHUNKS_RELATIVE_PATH:
        raise CorpusChunkingError(
            "Unexpected production chunks output path."
        )

    if summary_path != CHUNKING_SUMMARY_RELATIVE_PATH:
        raise CorpusChunkingError(
            "Unexpected production summary output path."
        )

    expected_audit = _require_exact_keys(
        config["expected_source_specific_audit"],
        AUDIT_KEYS,
        "Expected source-specific audit",
    )

    for field_name in AUDIT_KEYS - {
        "chunk_projection_sha256"
    }:
        _require_integer(
            expected_audit[field_name],
            f"Expected {field_name}",
            minimum=0,
        )

    _validate_sha256(
        expected_audit["chunk_projection_sha256"],
        "Expected chunk projection fingerprint",
    )

    return config


def load_region_records(
    regions_path: str | Path,
    expected_sha256: str,
) -> list[dict[str, Any]]:
    """Load the exact configured UTF-8 JSONL region export."""

    path = Path(regions_path)

    if not path.is_file():
        raise CorpusChunkingError(
            f"Region export is missing: {path}."
        )

    _validate_sha256(
        expected_sha256,
        "Expected region-export fingerprint",
    )
    actual_sha256 = calculate_file_sha256(path)

    if actual_sha256 != expected_sha256:
        raise CorpusChunkingError(
            "Region export fingerprint differs from the "
            "chunking configuration."
        )

    try:
        file_bytes = path.read_bytes()
        decoded_text = file_bytes.decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise CorpusChunkingError(
            f"Cannot read region export as UTF-8: {path}."
        ) from error

    if not file_bytes.endswith(b"\n"):
        raise CorpusChunkingError(
            "Region export must end with a newline."
        )

    try:
        records = [
            json.loads(line)
            for line in decoded_text.splitlines()
        ]
    except json.JSONDecodeError as error:
        raise CorpusChunkingError(
            "Region export contains invalid JSON."
        ) from error

    if not records:
        raise CorpusChunkingError(
            "Region export contains no records."
        )

    return records


def _validate_region_records(
    region_records: list[dict[str, Any]],
) -> None:
    """Validate record order, identity and line reconstruction."""

    if not isinstance(region_records, list) or not region_records:
        raise CorpusChunkingError(
            "At least one region record is required."
        )

    previous_key: tuple[int, int] | None = None
    record_ids: set[str] = set()
    common_source: tuple[str, str] | None = None

    for record_position, raw_record in enumerate(
        region_records
    ):
        record = _require_exact_keys(
            raw_record,
            REGION_KEYS,
            f"Region record {record_position}",
        )

        if record["record_schema_version"] != (
            RECORD_SCHEMA_VERSION
        ):
            raise CorpusChunkingError(
                "Region record has an unsupported schema version."
            )

        record_id = record["record_id"]
        source_id = record["source_id"]
        source_sha256 = record["source_sha256"]

        if not isinstance(record_id, str) or not record_id:
            raise CorpusChunkingError(
                "Region record identifier must be non-empty."
            )

        if record_id in record_ids:
            raise CorpusChunkingError(
                f"Duplicate region record identifier: {record_id}."
            )

        record_ids.add(record_id)

        if not isinstance(source_id, str) or not source_id:
            raise CorpusChunkingError(
                "Region source identifier must be non-empty."
            )

        _validate_sha256(
            source_sha256,
            "Region source fingerprint",
        )
        source = (source_id, source_sha256)

        if common_source is None:
            common_source = source
        elif source != common_source:
            raise CorpusChunkingError(
                "Region records contain mixed source identities."
            )

        pdf_page_index = _require_integer(
            record["pdf_page_index"],
            "PDF page index",
            minimum=0,
        )
        pdf_page_number = _require_integer(
            record["pdf_page_number"],
            "PDF page number",
            minimum=1,
        )
        region_index = _require_integer(
            record["region_index"],
            "Region index",
            minimum=0,
        )

        if pdf_page_number != pdf_page_index + 1:
            raise CorpusChunkingError(
                "PDF page identifiers are inconsistent."
            )

        expected_record_id = (
            f"{source_id}:pdf-{pdf_page_number:04d}:"
            f"region-{region_index:02d}"
        )

        if record_id != expected_record_id:
            raise CorpusChunkingError(
                "Region record identifier is inconsistent "
                "with its source and page metadata."
            )

        record_key = (pdf_page_index, region_index)

        if previous_key is not None and record_key <= previous_key:
            raise CorpusChunkingError(
                "Region records are not strictly ordered."
            )

        previous_key = record_key
        printed_page_number = record["printed_page_number"]

        if printed_page_number is not None:
            _require_integer(
                printed_page_number,
                "Printed page number",
                minimum=1,
            )

        if record["page_state"] not in RETRIEVAL_PAGE_STATES:
            raise CorpusChunkingError(
                "Region record has an invalid page state."
            )

        page_width = _require_finite_number(
            record["page_width"],
            "Page width",
        )
        page_height = _require_finite_number(
            record["page_height"],
            "Page height",
        )

        if page_width <= 0.0 or page_height <= 0.0:
            raise CorpusChunkingError(
                "Page dimensions must be positive."
            )

        if record["coordinate_origin"] != COORDINATE_ORIGIN:
            raise CorpusChunkingError(
                "Unexpected coordinate origin."
            )

        if record["coordinate_unit"] != COORDINATE_UNIT:
            raise CorpusChunkingError(
                "Unexpected coordinate unit."
            )

        region_bbox = _validate_bbox(
            record["region_bbox"],
            "Region bounding box",
        )

        if (
            region_bbox[0] < 0.0
            or region_bbox[1] < 0.0
            or region_bbox[2] > page_width
            or region_bbox[3] > page_height
        ):
            raise CorpusChunkingError(
                "Region bounding box lies outside its page."
            )

        text = record["text"]
        lines = record["lines"]

        if not isinstance(text, str) or not text:
            raise CorpusChunkingError(
                "Region text must be non-empty."
            )

        if not isinstance(lines, list) or not lines:
            raise CorpusChunkingError(
                "Region lines must be a non-empty array."
            )

        line_texts = []

        for expected_line_index, raw_line in enumerate(lines):
            line = _require_exact_keys(
                raw_line,
                LINE_KEYS,
                f"Line {expected_line_index} in {record_id}",
            )
            line_index = _require_integer(
                line["line_index"],
                "Line index",
                minimum=0,
            )

            if line_index != expected_line_index:
                raise CorpusChunkingError(
                    "Line indices must be contiguous from zero."
                )

            line_text = line["text"]

            if (
                not isinstance(line_text, str)
                or not line_text
            ):
                raise CorpusChunkingError(
                    "Positioned-line text must be non-empty."
                )

            line_texts.append(line_text)
            line_bbox = _validate_bbox(
                line["bbox"],
                "Line bounding box",
            )

            if (
                line_bbox[0]
                < region_bbox[0]
                - LINE_REGION_COORDINATE_TOLERANCE
                or line_bbox[1]
                < region_bbox[1]
                - LINE_REGION_COORDINATE_TOLERANCE
                or line_bbox[2]
                > region_bbox[2]
                + LINE_REGION_COORDINATE_TOLERANCE
                or line_bbox[3]
                > region_bbox[3]
                + LINE_REGION_COORDINATE_TOLERANCE
            ):
                raise CorpusChunkingError(
                    "Line bounding box lies outside its region."
                )

            for field_name, minimum in (
                ("block_number", 0),
                ("line_number", 0),
                ("span_count", 1),
                ("writing_mode", 0),
            ):
                _require_integer(
                    line[field_name],
                    f"Line {field_name}",
                    minimum=minimum,
                )

            direction = line["direction"]

            if not isinstance(direction, list) or len(direction) != 2:
                raise CorpusChunkingError(
                    "Line direction must contain two coordinates."
                )

            for coordinate in direction:
                _require_finite_number(
                    coordinate,
                    "Line direction coordinate",
                )

        if "\n".join(line_texts) != text:
            raise CorpusChunkingError(
                "Region text differs from its positioned lines."
            )

        line_count = _require_integer(
            record["line_count"],
            "Region line count",
            minimum=1,
        )
        character_count = _require_integer(
            record["character_count"],
            "Region character count",
            minimum=1,
        )

        if line_count != len(lines):
            raise CorpusChunkingError(
                "Region line count is inconsistent."
            )

        if character_count != len(text):
            raise CorpusChunkingError(
                "Region character count is inconsistent."
            )


def _make_splitter(
    splitter_config: dict[str, Any],
) -> CharacterTextSplitter:
    """Construct the exact configured text splitter."""

    return CharacterTextSplitter(
        separator=splitter_config["separator"],
        chunk_size=splitter_config["chunk_size"],
        chunk_overlap=splitter_config["chunk_overlap"],
        length_function=len,
        keep_separator=splitter_config["keep_separator"],
        add_start_index=splitter_config["add_start_index"],
        strip_whitespace=splitter_config["strip_whitespace"],
        is_separator_regex=splitter_config[
            "is_separator_regex"
        ],
    )


def _line_boundaries(
    lines: list[dict[str, Any]],
) -> tuple[dict[int, int], dict[int, int]]:
    """Map parent character boundaries to half-open line indices."""

    start_to_line: dict[int, int] = {}
    end_to_line: dict[int, int] = {}
    character_position = 0

    for line_index, line in enumerate(lines):
        start_to_line[character_position] = line_index
        character_position += len(line["text"])
        end_to_line[character_position] = line_index + 1

        if line_index < len(lines) - 1:
            character_position += 1

    return start_to_line, end_to_line


def _chunk_bbox(
    lines: list[dict[str, Any]],
) -> list[float]:
    """Return the bounding box enclosing all complete chunk lines."""

    return [
        min(line["bbox"][0] for line in lines),
        min(line["bbox"][1] for line in lines),
        max(line["bbox"][2] for line in lines),
        max(line["bbox"][3] for line in lines),
    ]


def _contains_formula_control(text: str) -> bool:
    """Identify preserved non-layout C0 control characters."""

    return any(
        ord(character) < 32
        and character not in "\t\n\r"
        for character in text
    )


def _protect_embedded_line_newlines(
    region_record: dict[str, Any],
    sentinel: str,
) -> str:
    """Protect U+000A values stored inside positioned lines."""

    original_text = region_record["text"]

    if sentinel in original_text:
        raise CorpusChunkingError(
            "The configured temporary sentinel occurs in "
            "the parent text."
        )

    protected_text = "\n".join(
        line["text"].replace("\n", sentinel)
        for line in region_record["lines"]
    )

    if len(protected_text) != len(original_text):
        raise CorpusChunkingError(
            "Embedded-newline protection changed parent length."
        )

    if protected_text.replace(sentinel, "\n") != original_text:
        raise CorpusChunkingError(
            "Embedded-newline protection is not reversible."
        )

    return protected_text


def build_chunk_records(
    region_records: list[dict[str, Any]],
    splitter_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Split validated regions and preserve exact provenance."""

    _validate_region_records(region_records)
    splitter_config = _validate_splitter_config(splitter_config)
    splitter = _make_splitter(splitter_config)
    chunk_size = splitter_config["chunk_size"]
    chunk_overlap = splitter_config["chunk_overlap"]
    sentinel = "\ue000"
    chunk_records: list[dict[str, Any]] = []
    chunk_ids: set[str] = set()

    for region_record in region_records:
        parent_text = region_record["text"]
        lines = region_record["lines"]
        start_to_line, end_to_line = _line_boundaries(lines)
        splitter_input = _protect_embedded_line_newlines(
            region_record,
            sentinel,
        )
        documents = splitter.create_documents([splitter_input])

        if not documents:
            raise CorpusChunkingError(
                "A non-empty region produced no chunks."
            )

        covered_lines: set[int] = set()
        previous_start: int | None = None
        previous_end: int | None = None

        for chunk_index, document in enumerate(documents):
            if not isinstance(document, Document):
                raise CorpusChunkingError(
                    "Splitter returned an unexpected object."
                )

            protected_chunk_text = document.page_content
            chunk_text = protected_chunk_text.replace(
                sentinel,
                "\n",
            )
            parent_start = document.metadata.get("start_index")

            if not isinstance(chunk_text, str) or not chunk_text:
                raise CorpusChunkingError(
                    "Splitter emitted an empty chunk."
                )

            if len(chunk_text) > chunk_size:
                raise CorpusChunkingError(
                    "Splitter emitted an oversized chunk."
                )

            parent_start = _require_integer(
                parent_start,
                "Chunk parent-character start",
                minimum=0,
            )
            parent_end = parent_start + len(
                protected_chunk_text
            )

            if len(chunk_text) != len(protected_chunk_text):
                raise CorpusChunkingError(
                    "Restoring embedded newlines changed chunk length."
                )

            if parent_text[parent_start:parent_end] != chunk_text:
                raise CorpusChunkingError(
                    "Chunk text is not its declared parent substring."
                )

            if parent_start not in start_to_line:
                raise CorpusChunkingError(
                    "Chunk starts inside a positioned line."
                )

            if parent_end not in end_to_line:
                raise CorpusChunkingError(
                    "Chunk ends inside a positioned line."
                )

            line_start = start_to_line[parent_start]
            line_end = end_to_line[parent_end]

            if line_end <= line_start:
                raise CorpusChunkingError(
                    "Chunk contains no complete positioned line."
                )

            if previous_start is None:
                if parent_start != 0:
                    raise CorpusChunkingError(
                        "First chunk does not start at parent offset zero."
                    )
            else:
                assert previous_end is not None

                if parent_start <= previous_start:
                    raise CorpusChunkingError(
                        "Chunk starts are not strictly increasing."
                    )

                if parent_start > previous_end + 1:
                    raise CorpusChunkingError(
                        "Chunk sequence leaves uncovered parent text."
                    )

                if parent_start < previous_end - chunk_overlap:
                    raise CorpusChunkingError(
                        "Effective overlap exceeds the configured target."
                    )

            selected_lines = lines[line_start:line_end]
            covered_lines.update(range(line_start, line_end))
            chunk_id = (
                f"{region_record['record_id']}:"
                f"chunk-{chunk_index:04d}"
            )

            if chunk_id in chunk_ids:
                raise CorpusChunkingError(
                    f"Duplicate chunk identifier: {chunk_id}."
                )

            chunk_ids.add(chunk_id)
            chunk_records.append(
                {
                    "character_count": len(chunk_text),
                    "chunk_bbox": _chunk_bbox(selected_lines),
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index,
                    "chunk_schema_version": CHUNK_SCHEMA_VERSION,
                    "coordinate_origin": region_record[
                        "coordinate_origin"
                    ],
                    "coordinate_unit": region_record[
                        "coordinate_unit"
                    ],
                    "line_count": line_end - line_start,
                    "page_height": region_record["page_height"],
                    "page_state": region_record["page_state"],
                    "page_width": region_record["page_width"],
                    "parent_character_end": parent_end,
                    "parent_character_start": parent_start,
                    "parent_line_end_index": line_end,
                    "parent_line_start_index": line_start,
                    "parent_record_id": region_record["record_id"],
                    "parent_record_schema_version": region_record[
                        "record_schema_version"
                    ],
                    "pdf_page_index": region_record[
                        "pdf_page_index"
                    ],
                    "pdf_page_number": region_record[
                        "pdf_page_number"
                    ],
                    "printed_page_number": region_record[
                        "printed_page_number"
                    ],
                    "region_bbox": region_record["region_bbox"],
                    "region_index": region_record["region_index"],
                    "source_id": region_record["source_id"],
                    "source_sha256": region_record["source_sha256"],
                    "text": chunk_text,
                }
            )
            previous_start = parent_start
            previous_end = parent_end

        if previous_end != len(parent_text):
            raise CorpusChunkingError(
                "Final chunk does not reach the end of its parent."
            )

        if covered_lines != set(range(len(lines))):
            raise CorpusChunkingError(
                "At least one positioned line is absent from chunks."
            )

    return chunk_records


def chunk_projection_sha256(
    chunk_records: list[dict[str, Any]],
) -> str:
    """Fingerprint the ordered parent, index and text projection."""

    projection = [
        {
            "chunk_index": record["chunk_index"],
            "parent_record_id": record["parent_record_id"],
            "text": record["text"],
        }
        for record in chunk_records
    ]

    return sha256(
        serialise_json_lines(projection)
    ).hexdigest()


def calculate_chunk_audit(
    chunk_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Calculate the frozen source-specific chunk statistics."""

    if not chunk_records:
        raise CorpusChunkingError(
            "At least one chunk record is required."
        )

    counts_by_parent = Counter(
        record["parent_record_id"]
        for record in chunk_records
    )

    return {
        "adjacent_chunk_pair_count": sum(
            count - 1 for count in counts_by_parent.values()
        ),
        "chunk_count": len(chunk_records),
        "chunk_projection_sha256": chunk_projection_sha256(
            chunk_records
        ),
        "chunks_with_formula_controls": sum(
            _contains_formula_control(record["text"])
            for record in chunk_records
        ),
        "maximum_chunk_character_count": max(
            record["character_count"]
            for record in chunk_records
        ),
        "maximum_chunks_per_parent_record": max(
            counts_by_parent.values()
        ),
        "oversized_chunk_count": 0,
        "single_chunk_parent_record_count": sum(
            count == 1 for count in counts_by_parent.values()
        ),
        "total_chunk_character_count": sum(
            record["character_count"]
            for record in chunk_records
        ),
    }


def validate_expected_chunk_audit(
    chunk_records: list[dict[str, Any]],
    expected_audit: dict[str, Any],
) -> dict[str, Any]:
    """Require every frozen audit value to be reproduced."""

    expected_audit = _require_exact_keys(
        expected_audit,
        AUDIT_KEYS,
        "Expected source-specific audit",
    )
    actual_audit = calculate_chunk_audit(chunk_records)

    if actual_audit != expected_audit:
        differences = {
            key: {
                "actual": actual_audit.get(key),
                "expected": expected_audit.get(key),
            }
            for key in sorted(AUDIT_KEYS)
            if actual_audit.get(key) != expected_audit.get(key)
        }
        raise CorpusChunkingError(
            "Chunk output differs from the frozen audit: "
            f"{differences}."
        )

    return actual_audit


def build_chunking_summary(
    config: dict[str, Any],
    config_path: Path,
    project_root: Path,
    region_records: list[dict[str, Any]],
    chunk_records: list[dict[str, Any]],
    chunks_path: Path,
    chunks_sha256: str,
) -> dict[str, Any]:
    """Build the deterministic chunking-summary record."""

    audit = validate_expected_chunk_audit(
        chunk_records,
        config["expected_source_specific_audit"],
    )
    input_config = config["input"]
    output_config = config["output"]
    splitter_config = config["splitter"]

    try:
        config_relative_path = config_path.resolve().relative_to(
            project_root.resolve()
        ).as_posix()
        chunks_relative_path = chunks_path.resolve().relative_to(
            project_root.resolve()
        ).as_posix()
    except ValueError as error:
        raise CorpusChunkingError(
            "Summary path lies outside the project root."
        ) from error

    return {
        "add_start_index": splitter_config["add_start_index"],
        "adjacent_chunk_pair_count": audit[
            "adjacent_chunk_pair_count"
        ],
        "chunk_count": audit["chunk_count"],
        "chunk_overlap": splitter_config["chunk_overlap"],
        "chunk_projection_sha256": audit[
            "chunk_projection_sha256"
        ],
        "chunk_schema_version": output_config[
            "chunk_schema_version"
        ],
        "chunk_size": splitter_config["chunk_size"],
        "chunking_config_relative_path": config_relative_path,
        "chunking_config_sha256": calculate_file_sha256(
            config_path
        ),
        "chunking_summary_schema_version": output_config[
            "summary_schema_version"
        ],
        "chunks_relative_path": chunks_relative_path,
        "chunks_sha256": _validate_sha256(
            chunks_sha256,
            "Chunks output fingerprint",
        ),
        "chunks_with_formula_controls": audit[
            "chunks_with_formula_controls"
        ],
        "input_character_count": input_config[
            "character_count"
        ],
        "input_line_count": input_config["line_count"],
        "input_record_count": input_config["record_count"],
        "input_record_schema_version": input_config[
            "record_schema_version"
        ],
        "input_regions_relative_path": input_config[
            "relative_path"
        ],
        "input_regions_sha256": input_config["sha256"],
        "embedded_line_newline_sentinel": splitter_config[
            "embedded_line_newline_sentinel"
        ],
        "is_separator_regex": splitter_config[
            "is_separator_regex"
        ],
        "keep_separator": splitter_config["keep_separator"],
        "length_function": splitter_config["length_function"],
        "maximum_chunk_character_count": audit[
            "maximum_chunk_character_count"
        ],
        "maximum_chunks_per_parent_record": audit[
            "maximum_chunks_per_parent_record"
        ],
        "oversized_chunk_count": audit["oversized_chunk_count"],
        "oversized_chunk_policy": config[
            "oversized_chunk_policy"
        ],
        "protect_embedded_line_newlines": splitter_config[
            "protect_embedded_line_newlines"
        ],
        "separator": splitter_config["separator"],
        "single_chunk_parent_record_count": audit[
            "single_chunk_parent_record_count"
        ],
        "source_id": region_records[0]["source_id"],
        "source_sha256": region_records[0]["source_sha256"],
        "splitter_class_name": splitter_config["class_name"],
        "splitter_package": splitter_config["package"],
        "splitter_package_version": splitter_config[
            "package_version"
        ],
        "strip_whitespace": splitter_config[
            "strip_whitespace"
        ],
        "total_chunk_character_count": audit[
            "total_chunk_character_count"
        ],
    }


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


def _validate_jsonl_round_trip(
    serialised: bytes,
    records: list[dict[str, Any]],
) -> None:
    """Require exact JSONL recovery before publication."""

    if not serialised.endswith(b"\n"):
        raise CorpusChunkingError(
            "Chunk JSONL has no final newline."
        )

    recovered = [
        json.loads(line)
        for line in serialised.decode("utf-8").splitlines()
    ]

    if recovered != records:
        raise CorpusChunkingError(
            "Chunk JSONL failed its exact round trip."
        )


def _validate_json_object_round_trip(
    serialised: bytes,
    record: dict[str, Any],
) -> None:
    """Require exact summary recovery before publication."""

    if not serialised.endswith(b"\n"):
        raise CorpusChunkingError(
            "Chunking summary has no final newline."
        )

    if json.loads(serialised.decode("utf-8")) != record:
        raise CorpusChunkingError(
            "Chunking summary failed its exact round trip."
        )


def export_chunked_corpus(
    config_path: str | Path,
    project_root: str | Path | None = None,
    overwrite: bool = False,
) -> CorpusChunkingResult:
    """Validate, chunk and publish the deterministic outputs."""

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
        "Chunking configuration",
    )
    config = load_chunking_config(resolved_config_path)
    regions_path = _resolve_project_path(
        root,
        Path(config["input"]["relative_path"]),
        Path("data/interim/corpus"),
        "Chunking input",
    )
    chunks_path = _resolve_project_path(
        root,
        Path(config["output"]["chunks_relative_path"]),
        Path("data/processed/corpus"),
        "Chunks output",
    )
    summary_path = _resolve_project_path(
        root,
        Path(config["output"]["summary_relative_path"]),
        Path("data/processed/audit"),
        "Chunking-summary output",
    )

    if not overwrite and (
        chunks_path.exists() or summary_path.exists()
    ):
        raise CorpusChunkingError(
            "Chunking output already exists; select explicit "
            "overwrite to replace it."
        )

    region_records = load_region_records(
        regions_path,
        config["input"]["sha256"],
    )
    _validate_region_records(region_records)

    if len(region_records) != config["input"]["record_count"]:
        raise CorpusChunkingError(
            "Input record count differs from configuration."
        )

    if sum(
        record["line_count"] for record in region_records
    ) != config["input"]["line_count"]:
        raise CorpusChunkingError(
            "Input line count differs from configuration."
        )

    if sum(
        record["character_count"] for record in region_records
    ) != config["input"]["character_count"]:
        raise CorpusChunkingError(
            "Input character count differs from configuration."
        )

    chunk_records = build_chunk_records(
        region_records,
        config["splitter"],
    )
    validate_expected_chunk_audit(
        chunk_records,
        config["expected_source_specific_audit"],
    )
    chunks_bytes = serialise_json_lines(chunk_records)
    _validate_jsonl_round_trip(chunks_bytes, chunk_records)

    with TemporaryDirectory(dir=root) as temporary_directory:
        staging_directory = Path(temporary_directory)
        staged_chunks_path = staging_directory / "chunks.jsonl"
        staged_summary_path = (
            staging_directory / "chunking-summary.json"
        )
        staged_chunks_path.write_bytes(chunks_bytes)
        chunks_sha256 = calculate_file_sha256(
            staged_chunks_path
        )
        summary = build_chunking_summary(
            config,
            resolved_config_path,
            root,
            region_records,
            chunk_records,
            chunks_path,
            chunks_sha256,
        )
        summary_bytes = serialise_json_object(summary)
        _validate_json_object_round_trip(summary_bytes, summary)
        staged_summary_path.write_bytes(summary_bytes)
        chunking_summary_sha256 = calculate_file_sha256(
            staged_summary_path
        )

        chunks_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staged_chunks_path, chunks_path)
        os.replace(staged_summary_path, summary_path)

    return CorpusChunkingResult(
        chunks_path=chunks_path,
        chunking_summary_path=summary_path,
        chunks_sha256=chunks_sha256,
        chunking_summary_sha256=chunking_summary_sha256,
        chunk_count=len(chunk_records),
        parent_record_count=len(region_records),
    )


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the production chunking command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic retrieval chunks from the "
            "validated region export."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/chunking-config.json",
        help="Repository-relative chunking configuration path.",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root containing configs and data.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing chunking outputs.",
    )
    return parser


def main() -> None:
    """Run the deterministic production chunker."""

    arguments = _build_argument_parser().parse_args()
    result = export_chunked_corpus(
        config_path=arguments.config,
        project_root=arguments.project_root,
        overwrite=arguments.overwrite,
    )
    print("Corpus chunking: PASSED")
    print("  Parent records:", result.parent_record_count)
    print("  Retrieval chunks:", result.chunk_count)
    print("  Chunks path:", result.chunks_path)
    print("  Chunks SHA-256:", result.chunks_sha256)
    print("  Summary path:", result.chunking_summary_path)
    print(
        "  Summary SHA-256:",
        result.chunking_summary_sha256,
    )


if __name__ == "__main__":
    main()
