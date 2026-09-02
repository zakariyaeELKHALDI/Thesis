"""Deterministic corpus-record, page-audit and summary exports."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable

import pymupdf

from geotech_rag.corpus_boundaries import (
    EXPECTED_PAGE_COUNT,
    FIRST_KNOWLEDGE_PDF_PAGE,
)
from geotech_rag.corpus_extraction import (
    EXPECTED_CLEANED_CHARACTER_COUNT,
    EXPECTED_CLEANED_LINE_COUNT,
    EXPECTED_EMPTY_RETAINED_REGION_PAGES,
    EXPECTED_STRUCTURAL_REMOVAL_COUNTS,
    LINE_REGION_COORDINATE_TOLERANCE,
    PUBLISHER_COPYRIGHT_TEXT,
    RETRIEVAL_PAGE_STATES,
    expected_printed_page_number,
    extract_validated_corpus_pages,
)
from geotech_rag.corpus_manifest import (
    CorpusSourceValidation,
    calculate_file_sha256,
    load_corpus_manifest,
    validate_corpus_manifest,
)


RECORD_SCHEMA_VERSION = "1.0"
PAGE_AUDIT_SCHEMA_VERSION = "1.0"
EXTRACTION_SUMMARY_SCHEMA_VERSION = "1.0"

COORDINATE_UNIT = "pdf_point"
COORDINATE_ORIGIN = "top_left"
PARSER_LIBRARY = "PyMuPDF"

REGIONS_RELATIVE_PATH = Path(
    "data/interim/corpus/regions.jsonl"
)
PAGE_AUDIT_RELATIVE_PATH = Path(
    "data/interim/audit/pages.jsonl"
)
EXTRACTION_SUMMARY_RELATIVE_PATH = Path(
    "data/interim/audit/extraction-summary.json"
)

EXPECTED_RETRIEVAL_RECORD_COUNT = 670
EXPECTED_PAGE_STATE_COUNTS = {
    "fully_retained": 644,
    "partially_retained": 32,
    "fully_excluded": 37,
    "outside_knowledge_scope": 57,
}

PAGE_STATES = frozenset(
    EXPECTED_PAGE_STATE_COUNTS
)
REMOVAL_REASONS = frozenset(
    EXPECTED_STRUCTURAL_REMOVAL_COUNTS
)


class CorpusExportError(RuntimeError):
    """Raised when an export violates the frozen contract."""


@dataclass(frozen=True)
class CorpusExportResult:
    """Paths, fingerprints and counts from one completed export."""

    regions_path: Path
    page_audit_path: Path
    extraction_summary_path: Path
    regions_sha256: str
    page_audit_sha256: str
    retrieval_record_count: int
    page_audit_record_count: int


def _require_integer(
    value: Any,
    description: str,
) -> int:
    """Return an integer while rejecting booleans."""

    if (
        not isinstance(value, int)
        or isinstance(value, bool)
    ):
        raise CorpusExportError(
            f"{description} must be an integer."
        )

    return value


def _require_nonnegative_integer(
    value: Any,
    description: str,
) -> int:
    """Return one nonnegative integer."""

    converted_value = _require_integer(
        value,
        description,
    )

    if converted_value < 0:
        raise CorpusExportError(
            f"{description} cannot be negative."
        )

    return converted_value


def _require_finite_number(
    value: Any,
    description: str,
) -> float:
    """Return one finite numeric value."""

    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not isfinite(value)
    ):
        raise CorpusExportError(
            f"{description} must be a finite number."
        )

    return float(value)


def _normalise_bbox(
    bbox: Any,
    description: str,
) -> list[float]:
    """Validate and convert a four-coordinate bounding box."""

    if (
        not isinstance(bbox, (list, tuple))
        or len(bbox) != 4
    ):
        raise CorpusExportError(
            f"{description} must contain four coordinates."
        )

    converted_bbox = [
        _require_finite_number(
            coordinate,
            f"{description} coordinate",
        )
        for coordinate in bbox
    ]
    x0, y0, x1, y1 = converted_bbox

    if x0 > x1 or y0 > y1:
        raise CorpusExportError(
            f"{description} coordinates are reversed."
        )

    return converted_bbox


def _normalise_direction(
    direction: Any,
    description: str,
) -> list[float]:
    """Validate and convert one writing-direction vector."""

    if (
        not isinstance(direction, (list, tuple))
        or len(direction) != 2
    ):
        raise CorpusExportError(
            f"{description} must contain two coordinates."
        )

    return [
        _require_finite_number(
            coordinate,
            f"{description} coordinate",
        )
        for coordinate in direction
    ]


def _normalise_intervals(
    intervals: Any,
    page_height: float,
    description: str,
) -> list[list[float]]:
    """Validate ordered half-open vertical intervals."""

    if not isinstance(intervals, (list, tuple)):
        raise CorpusExportError(
            f"{description} must be an array."
        )

    converted_intervals = []
    previous_y1 = None

    for interval_index, interval in enumerate(
        intervals
    ):
        if (
            not isinstance(interval, (list, tuple))
            or len(interval) != 2
        ):
            raise CorpusExportError(
                f"{description} interval "
                f"{interval_index} must contain two "
                "coordinates."
            )

        y0 = _require_finite_number(
            interval[0],
            f"{description} interval {interval_index} y0",
        )
        y1 = _require_finite_number(
            interval[1],
            f"{description} interval {interval_index} y1",
        )

        if y0 < 0.0 or y0 >= y1 or y1 > page_height:
            raise CorpusExportError(
                f"{description} interval {interval_index} "
                "lies outside its physical page."
            )

        if (
            previous_y1 is not None
            and y0 < previous_y1
        ):
            raise CorpusExportError(
                f"{description} intervals are not ordered."
            )

        converted_intervals.append([y0, y1])
        previous_y1 = y1

    return converted_intervals


def _validate_page_identity(
    page: dict[str, Any],
) -> tuple[int, int]:
    """Validate the zero-based and one-based page identifiers."""

    page_index = _require_nonnegative_integer(
        page.get("pdf_page_index"),
        "PDF page index",
    )
    pdf_page_number = _require_integer(
        page.get("pdf_page_number"),
        "PDF page number",
    )

    if pdf_page_number != page_index + 1:
        raise CorpusExportError(
            "PDF page index and number do not agree."
        )

    printed_page_number = page.get(
        "printed_page_number"
    )

    if pdf_page_number < FIRST_KNOWLEDGE_PDF_PAGE:
        if printed_page_number is not None:
            raise CorpusExportError(
                "A page before printed page 1 has a "
                "printed page number."
            )
    else:
        expected_value = int(
            expected_printed_page_number(
                pdf_page_number
            )
        )

        if printed_page_number != expected_value:
            raise CorpusExportError(
                "Printed page number does not match "
                f"PDF page {pdf_page_number}."
            )

    page_state = page.get("page_state")

    if page_state not in PAGE_STATES:
        raise CorpusExportError(
            f"Invalid page state: {page_state!r}."
        )

    return page_index, pdf_page_number


def _validate_page_order(
    cleaned_pages: list[dict[str, Any]],
) -> None:
    """Require pages to appear once in increasing order."""

    previous_page_index = None

    for page in cleaned_pages:
        page_index, _ = _validate_page_identity(
            page
        )

        if (
            previous_page_index is not None
            and page_index != previous_page_index + 1
        ):
            raise CorpusExportError(
                "Cleaned pages are not contiguous and "
                "ordered."
            )

        previous_page_index = page_index


def _validate_page_dimensions(
    page: dict[str, Any],
) -> tuple[float, float]:
    """Validate positive physical page dimensions."""

    page_width = _require_finite_number(
        page.get("page_width"),
        "Page width",
    )
    page_height = _require_finite_number(
        page.get("page_height"),
        "Page height",
    )

    if page_width <= 0.0 or page_height <= 0.0:
        raise CorpusExportError(
            "Page dimensions must be positive."
        )

    return page_width, page_height


def _export_line(
    line: dict[str, Any],
    line_index: int,
    page_width: float,
    retained_y0: float,
    retained_y1: float,
    pdf_page_number: int,
) -> dict[str, Any]:
    """Validate and convert one retained positioned line."""

    line_text = line.get("text")

    if not isinstance(line_text, str) or not line_text:
        raise CorpusExportError(
            "A retained line is empty on PDF page "
            f"{pdf_page_number}."
        )

    if line_text == PUBLISHER_COPYRIGHT_TEXT:
        raise CorpusExportError(
            "The exact publisher line remains on PDF "
            f"page {pdf_page_number}."
        )

    bbox = _normalise_bbox(
        line.get("bbox"),
        f"Line bbox on PDF page {pdf_page_number}",
    )
    x0, y0, x1, y1 = bbox
    boundary_excess = max(
        0.0,
        -x0,
        retained_y0 - y0,
        x1 - page_width,
        y1 - retained_y1,
    )

    if boundary_excess > LINE_REGION_COORDINATE_TOLERANCE:
        raise CorpusExportError(
            "Line bbox exceeds its retained region by "
            f"{boundary_excess:.9f} points on PDF page "
            f"{pdf_page_number}."
        )

    writing_mode = _require_integer(
        line.get("writing_mode"),
        "Line writing mode",
    )

    return {
        "line_index": line_index,
        "text": line_text,
        "bbox": bbox,
        "block_number": _require_nonnegative_integer(
            line.get("block_number"),
            "Line block number",
        ),
        "line_number": _require_nonnegative_integer(
            line.get("line_number"),
            "Line number",
        ),
        "span_count": _require_nonnegative_integer(
            line.get("span_count"),
            "Line span count",
        ),
        "direction": _normalise_direction(
            line.get("direction"),
            "Line direction",
        ),
        "writing_mode": writing_mode,
    }


def build_retrieval_region_records(
    cleaned_pages: list[dict[str, Any]],
    source_id: str,
    source_sha256: str,
) -> list[dict[str, Any]]:
    """Convert cleaned pages into non-empty region records."""

    if not isinstance(source_id, str) or not source_id:
        raise CorpusExportError(
            "Source identifier must be a non-empty string."
        )

    _validate_sha256(
        source_sha256,
        "Source fingerprint",
    )

    _validate_page_order(cleaned_pages)
    region_records = []
    record_ids = set()

    for page in cleaned_pages:
        _, pdf_page_number = _validate_page_identity(
            page
        )
        page_width, page_height = (
            _validate_page_dimensions(page)
        )
        retained_intervals = _normalise_intervals(
            page.get("retained_regions"),
            page_height,
            "Retained regions",
        )
        lines_by_region: dict[
            int,
            list[dict[str, Any]],
        ] = defaultdict(list)

        page_lines = page.get("lines")

        if not isinstance(page_lines, list):
            raise CorpusExportError(
                "Cleaned page lines must be an array."
            )

        reconstructed_page_text = "\n".join(
            line.get("text", "")
            for line in page_lines
        )

        if reconstructed_page_text != page.get("text"):
            raise CorpusExportError(
                "Cleaned page text differs from its "
                f"line reconstruction on PDF page "
                f"{pdf_page_number}."
            )

        for line in page_lines:
            region_index = _require_nonnegative_integer(
                line.get("region_index"),
                "Line region index",
            )

            if region_index >= len(retained_intervals):
                raise CorpusExportError(
                    "A retained line refers to a missing "
                    f"region on PDF page {pdf_page_number}."
                )

            lines_by_region[region_index].append(line)

        if (
            page_lines
            and page["page_state"]
            not in RETRIEVAL_PAGE_STATES
        ):
            raise CorpusExportError(
                "An excluded page state contains retrieval "
                f"lines on PDF page {pdf_page_number}."
            )

        for region_index, (
            retained_y0,
            retained_y1,
        ) in enumerate(retained_intervals):
            source_lines = lines_by_region.get(
                region_index,
                [],
            )

            if not source_lines:
                continue

            exported_lines = [
                _export_line(
                    line=line,
                    line_index=line_index,
                    page_width=page_width,
                    retained_y0=retained_y0,
                    retained_y1=retained_y1,
                    pdf_page_number=pdf_page_number,
                )
                for line_index, line in enumerate(
                    source_lines
                )
            ]
            region_text = "\n".join(
                line["text"]
                for line in exported_lines
            )
            record_id = (
                f"{source_id}:"
                f"pdf-{pdf_page_number:04d}:"
                f"region-{region_index:02d}"
            )

            if record_id in record_ids:
                raise CorpusExportError(
                    f"Duplicate region identifier: {record_id}."
                )

            printed_page_number = page[
                "printed_page_number"
            ]

            if not isinstance(
                printed_page_number,
                int,
            ) or isinstance(
                printed_page_number,
                bool,
            ):
                raise CorpusExportError(
                    "A retrieval record has no integer "
                    f"printed page number on PDF page "
                    f"{pdf_page_number}."
                )

            region_record = {
                "record_schema_version": (
                    RECORD_SCHEMA_VERSION
                ),
                "record_id": record_id,
                "source_id": source_id,
                "source_sha256": source_sha256,
                "pdf_page_index": page[
                    "pdf_page_index"
                ],
                "pdf_page_number": pdf_page_number,
                "printed_page_number": (
                    printed_page_number
                ),
                "page_state": page["page_state"],
                "page_width": page_width,
                "page_height": page_height,
                "region_index": region_index,
                "region_bbox": [
                    0.0,
                    retained_y0,
                    page_width,
                    retained_y1,
                ],
                "coordinate_unit": COORDINATE_UNIT,
                "coordinate_origin": (
                    COORDINATE_ORIGIN
                ),
                "text": region_text,
                "line_count": len(exported_lines),
                "character_count": len(region_text),
                "lines": exported_lines,
            }

            if not region_record["text"]:
                raise CorpusExportError(
                    "An empty retrieval record was created."
                )

            region_records.append(region_record)
            record_ids.add(record_id)

    expected_order = sorted(
        region_records,
        key=lambda record: (
            record["pdf_page_index"],
            record["region_index"],
        ),
    )

    if region_records != expected_order:
        raise CorpusExportError(
            "Region records are not deterministically ordered."
        )

    return region_records


def _export_removed_line(
    line: dict[str, Any],
    pdf_page_number: int,
    page_width: float,
    retained_intervals: list[list[float]],
) -> dict[str, Any]:
    """Convert one removed line into its audit representation."""

    removal_reason = line.get("removal_reason")

    if removal_reason not in REMOVAL_REASONS:
        raise CorpusExportError(
            "Invalid structural removal reason on PDF "
            f"page {pdf_page_number}: "
            f"{removal_reason!r}."
        )

    line_text = line.get("text")

    if not isinstance(line_text, str) or not line_text:
        raise CorpusExportError(
            "A removed structural line is empty."
        )

    region_index = _require_nonnegative_integer(
        line.get("region_index"),
        "Removed-line region index",
    )

    if region_index >= len(retained_intervals):
        raise CorpusExportError(
            "A removed structural line refers to a "
            f"missing region on PDF page "
            f"{pdf_page_number}."
        )

    bbox = _normalise_bbox(
        line.get("bbox"),
        "Removed-line bbox",
    )
    x0, y0, x1, y1 = bbox
    retained_y0, retained_y1 = (
        retained_intervals[region_index]
    )
    boundary_excess = max(
        0.0,
        -x0,
        retained_y0 - y0,
        x1 - page_width,
        y1 - retained_y1,
    )

    if boundary_excess > LINE_REGION_COORDINATE_TOLERANCE:
        raise CorpusExportError(
            "Removed-line bbox exceeds its retained "
            f"region on PDF page {pdf_page_number}."
        )

    return {
        "text": line_text,
        "bbox": bbox,
        "region_index": region_index,
        "block_number": _require_nonnegative_integer(
            line.get("block_number"),
            "Removed-line block number",
        ),
        "line_number": _require_nonnegative_integer(
            line.get("line_number"),
            "Removed-line number",
        ),
        "span_count": _require_nonnegative_integer(
            line.get("span_count"),
            "Removed-line span count",
        ),
        "direction": _normalise_direction(
            line.get("direction"),
            "Removed-line direction",
        ),
        "writing_mode": _require_integer(
            line.get("writing_mode"),
            "Removed-line writing mode",
        ),
        "removal_reason": removal_reason,
    }


def build_page_audit_records(
    cleaned_pages: list[dict[str, Any]],
    region_records: list[dict[str, Any]],
    source_id: str,
    source_sha256: str,
) -> list[dict[str, Any]]:
    """Convert every cleaned page into one audit record."""

    _validate_page_order(cleaned_pages)
    emitted_records_by_page: dict[
        int,
        list[dict[str, Any]],
    ] = defaultdict(list)
    emitted_record_ids = set()

    for record in region_records:
        if (
            record.get("source_id") != source_id
            or record.get("source_sha256")
            != source_sha256
        ):
            raise CorpusExportError(
                "A region record has conflicting source "
                "provenance."
            )

        record_id = record.get("record_id")

        if record_id in emitted_record_ids:
            raise CorpusExportError(
                f"Duplicate region identifier: {record_id}."
            )

        emitted_record_ids.add(record_id)

        emitted_records_by_page[
            record["pdf_page_index"]
        ].append(record)

    page_audit_records = []

    for page in cleaned_pages:
        page_index, pdf_page_number = (
            _validate_page_identity(page)
        )
        page_width, page_height = (
            _validate_page_dimensions(page)
        )
        excluded_intervals = _normalise_intervals(
            page.get("excluded_regions"),
            page_height,
            "Excluded regions",
        )
        retained_intervals = _normalise_intervals(
            page.get("retained_regions"),
            page_height,
            "Retained regions",
        )
        removed_source_lines = page.get(
            "removed_structural_lines"
        )

        if not isinstance(removed_source_lines, list):
            raise CorpusExportError(
                "Removed structural lines must be an array."
            )

        removed_lines = [
            _export_removed_line(
                line,
                pdf_page_number,
                page_width,
                retained_intervals,
            )
            for line in removed_source_lines
        ]
        raw_line_count = _require_nonnegative_integer(
            page.get("raw_line_count"),
            "Raw line count",
        )
        cleaned_line_count = (
            _require_nonnegative_integer(
                page.get("line_count"),
                "Cleaned line count",
            )
        )
        removed_line_count = (
            _require_nonnegative_integer(
                page.get(
                    "removed_structural_line_count"
                ),
                "Removed structural line count",
            )
        )

        if removed_line_count != len(removed_lines):
            raise CorpusExportError(
                "Removed structural line count differs "
                f"from its records on PDF page "
                f"{pdf_page_number}."
            )

        if (
            raw_line_count
            != cleaned_line_count + removed_line_count
        ):
            raise CorpusExportError(
                "Raw line count does not equal cleaned "
                "plus removed lines on PDF page "
                f"{pdf_page_number}."
            )

        emitted_records = sorted(
            emitted_records_by_page.get(
                page_index,
                [],
            ),
            key=lambda record: record["region_index"],
        )
        emitted_page_text = "\n".join(
            record["text"]
            for record in emitted_records
        )

        if emitted_page_text != page.get("text"):
            raise CorpusExportError(
                "Emitted region text does not reconstruct "
                f"PDF page {pdf_page_number}."
            )

        if (
            sum(
                record["line_count"]
                for record in emitted_records
            )
            != cleaned_line_count
        ):
            raise CorpusExportError(
                "Emitted region line counts do not match "
                f"PDF page {pdf_page_number}."
            )

        cleaned_character_count = (
            _require_nonnegative_integer(
                page.get("character_count"),
                "Cleaned character count",
            )
        )

        if cleaned_character_count != len(
            page.get("text", "")
        ):
            raise CorpusExportError(
                "Cleaned character count differs from "
                f"page text on PDF page {pdf_page_number}."
            )

        page_audit_records.append(
            {
                "page_audit_schema_version": (
                    PAGE_AUDIT_SCHEMA_VERSION
                ),
                "source_id": source_id,
                "source_sha256": source_sha256,
                "pdf_page_index": page_index,
                "pdf_page_number": pdf_page_number,
                "printed_page_number": page[
                    "printed_page_number"
                ],
                "page_width": page_width,
                "page_height": page_height,
                "coordinate_unit": COORDINATE_UNIT,
                "coordinate_origin": (
                    COORDINATE_ORIGIN
                ),
                "page_state": page["page_state"],
                "excluded_y_intervals": (
                    excluded_intervals
                ),
                "retained_y_intervals": (
                    retained_intervals
                ),
                "raw_line_count": raw_line_count,
                "raw_character_count": (
                    _require_nonnegative_integer(
                        page.get("raw_character_count"),
                        "Raw character count",
                    )
                ),
                "cleaned_line_count": (
                    cleaned_line_count
                ),
                "cleaned_character_count": (
                    cleaned_character_count
                ),
                "removed_structural_line_count": (
                    removed_line_count
                ),
                "removed_structural_lines": (
                    removed_lines
                ),
                "emitted_region_count": len(
                    emitted_records
                ),
                "emitted_record_ids": [
                    record["record_id"]
                    for record in emitted_records
                ],
            }
        )

    page_indices = {
        page["pdf_page_index"]
        for page in cleaned_pages
    }
    unexpected_page_indices = (
        set(emitted_records_by_page)
        - page_indices
    )

    if unexpected_page_indices:
        raise CorpusExportError(
            "Region records refer to missing page audits."
        )

    return page_audit_records


def serialise_json_lines(
    records: Iterable[dict[str, Any]],
) -> bytes:
    """Serialise JSONL with stable keys and final newlines."""

    serialised_lines = []

    for record in records:
        try:
            serialised_lines.append(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
        except (TypeError, ValueError) as error:
            raise CorpusExportError(
                "A JSONL record is not deterministically "
                "serialisable."
            ) from error

    return "".join(serialised_lines).encode(
        "utf-8"
    )


def serialise_json_object(
    record: dict[str, Any],
) -> bytes:
    """Serialise one deterministic JSON object."""

    try:
        serialised_record = json.dumps(
            record,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise CorpusExportError(
            "The JSON summary is not deterministically "
            "serialisable."
        ) from error

    return (serialised_record + "\n").encode(
        "utf-8"
    )


def _relative_path(
    path: Path,
    project_root: Path,
) -> str:
    """Return a repository-relative POSIX path."""

    try:
        return path.resolve().relative_to(
            project_root.resolve()
        ).as_posix()
    except ValueError as error:
        raise CorpusExportError(
            f"Path lies outside the project root: {path}."
        ) from error


def build_extraction_summary(
    validation_result: CorpusSourceValidation,
    project_root: Path,
    region_records: list[dict[str, Any]],
    page_audit_records: list[dict[str, Any]],
    regions_path: Path,
    regions_sha256: str,
    page_audit_path: Path,
    page_audit_sha256: str,
) -> dict[str, Any]:
    """Build and validate the source-specific summary."""

    _validate_sha256(
        regions_sha256,
        "Region-export fingerprint",
    )
    _validate_sha256(
        page_audit_sha256,
        "Page-audit fingerprint",
    )

    if (
        validation_result.parser_distribution
        != PARSER_LIBRARY
    ):
        raise CorpusExportError(
            "Validated parser distribution is not "
            f"{PARSER_LIBRARY}."
        )

    if (
        validation_result.pdf_page_count
        != EXPECTED_PAGE_COUNT
    ):
        raise CorpusExportError(
            "Validated PDF page count differs from the "
            "frozen source."
        )

    for record in region_records:
        if (
            record.get("source_id")
            != validation_result.source_id
            or record.get("source_sha256")
            != validation_result.sha256
            or record.get("record_schema_version")
            != RECORD_SCHEMA_VERSION
        ):
            raise CorpusExportError(
                "A retrieval record conflicts with the "
                "validated source or record schema."
            )

    for page in page_audit_records:
        if (
            page.get("source_id")
            != validation_result.source_id
            or page.get("source_sha256")
            != validation_result.sha256
            or page.get("page_audit_schema_version")
            != PAGE_AUDIT_SCHEMA_VERSION
        ):
            raise CorpusExportError(
                "A page-audit record conflicts with the "
                "validated source or audit schema."
            )

    if (
        len(page_audit_records)
        != EXPECTED_PAGE_COUNT
    ):
        raise CorpusExportError(
            "The page audit does not contain exactly "
            f"{EXPECTED_PAGE_COUNT} records."
        )

    for expected_index, page in enumerate(
        page_audit_records
    ):
        if (
            page["pdf_page_index"] != expected_index
            or page["pdf_page_number"]
            != expected_index + 1
        ):
            raise CorpusExportError(
                "Page-audit records are not complete and "
                "ordered."
            )

    page_state_counts = Counter(
        page["page_state"]
        for page in page_audit_records
    )
    ordered_page_state_counts = {
        page_state: page_state_counts.get(
            page_state,
            0,
        )
        for page_state in EXPECTED_PAGE_STATE_COUNTS
    }

    if (
        ordered_page_state_counts
        != EXPECTED_PAGE_STATE_COUNTS
    ):
        raise CorpusExportError(
            "Page-state counts differ from the frozen "
            "audit."
        )

    structural_counts = Counter(
        line["removal_reason"]
        for page in page_audit_records
        for line in page[
            "removed_structural_lines"
        ]
    )
    ordered_structural_counts = {
        removal_reason: structural_counts.get(
            removal_reason,
            0,
        )
        for removal_reason in sorted(
            REMOVAL_REASONS
        )
    }

    if structural_counts != EXPECTED_STRUCTURAL_REMOVAL_COUNTS:
        raise CorpusExportError(
            "Structural-removal counts differ from the "
            "frozen audit."
        )

    orientation_counts = Counter(
        (
            tuple(line["direction"]),
            line["writing_mode"],
        )
        for record in region_records
        for line in record["lines"]
    )
    line_orientation_counts = [
        {
            "direction": list(direction),
            "writing_mode": writing_mode,
            "line_count": line_count,
        }
        for (
            direction,
            writing_mode,
        ), line_count in sorted(
            orientation_counts.items(),
            key=lambda item: (
                item[0][0][0],
                item[0][0][1],
                item[0][1],
            ),
        )
    ]
    retrieval_line_count = sum(
        record["line_count"]
        for record in region_records
    )
    retrieval_character_count = sum(
        record["character_count"]
        for record in region_records
    )
    empty_region_pages = [
        page["pdf_page_number"]
        for page in page_audit_records
        if (
            page["retained_y_intervals"]
            and page["raw_line_count"] > 0
            and page["cleaned_line_count"] == 0
        )
    ]

    frozen_totals = {
        "retrieval record count": (
            EXPECTED_RETRIEVAL_RECORD_COUNT,
            len(region_records),
        ),
        "retrieval line count": (
            EXPECTED_CLEANED_LINE_COUNT,
            retrieval_line_count,
        ),
        "retrieval character count": (
            EXPECTED_CLEANED_CHARACTER_COUNT,
            retrieval_character_count,
        ),
        "empty retained-region pages": (
            list(
                EXPECTED_EMPTY_RETAINED_REGION_PAGES
            ),
            empty_region_pages,
        ),
    }

    for total_name, (
        expected_value,
        actual_value,
    ) in frozen_totals.items():
        if actual_value != expected_value:
            raise CorpusExportError(
                f"Unexpected {total_name}: expected "
                f"{expected_value!r}, found "
                f"{actual_value!r}."
            )

    if (
        sum(
            item["line_count"]
            for item in line_orientation_counts
        )
        != retrieval_line_count
    ):
        raise CorpusExportError(
            "Orientation counts do not sum to the "
            "retrieval line count."
        )

    return {
        "extraction_summary_schema_version": (
            EXTRACTION_SUMMARY_SCHEMA_VERSION
        ),
        "manifest_relative_path": _relative_path(
            validation_result.manifest_path,
            project_root,
        ),
        "source_id": validation_result.source_id,
        "source_sha256": validation_result.sha256,
        "source_size_bytes": (
            validation_result.size_bytes
        ),
        "pdf_page_count": (
            validation_result.pdf_page_count
        ),
        "parser_library": PARSER_LIBRARY,
        "parser_version": (
            validation_result.parser_version
        ),
        "record_schema_version": (
            RECORD_SCHEMA_VERSION
        ),
        "page_audit_schema_version": (
            PAGE_AUDIT_SCHEMA_VERSION
        ),
        "page_state_counts": (
            ordered_page_state_counts
        ),
        "structural_removal_counts": (
            ordered_structural_counts
        ),
        "line_orientation_counts": (
            line_orientation_counts
        ),
        "retrieval_record_count": len(
            region_records
        ),
        "retrieval_line_count": retrieval_line_count,
        "retrieval_character_count": (
            retrieval_character_count
        ),
        "empty_retained_region_pdf_pages": (
            empty_region_pages
        ),
        "regions_relative_path": _relative_path(
            regions_path,
            project_root,
        ),
        "regions_sha256": regions_sha256,
        "page_audit_relative_path": _relative_path(
            page_audit_path,
            project_root,
        ),
        "page_audit_sha256": page_audit_sha256,
    }


def _validate_sha256(
    value: str,
    description: str,
) -> None:
    """Validate a lowercase hexadecimal SHA-256 value."""

    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(
            character not in "0123456789abcdef"
            for character in value
        )
    ):
        raise CorpusExportError(
            f"{description} is not a valid SHA-256."
        )


def _validate_serialised_round_trip(
    serialised_records: bytes,
    records: list[dict[str, Any]],
    description: str,
) -> None:
    """Confirm that JSONL recovers the exact Python values."""

    if serialised_records and not serialised_records.endswith(
        b"\n"
    ):
        raise CorpusExportError(
            f"{description} has no final newline."
        )

    recovered_records = [
        json.loads(line)
        for line in serialised_records.decode(
            "utf-8"
        ).splitlines()
    ]

    if recovered_records != records:
        raise CorpusExportError(
            f"{description} failed its JSON round trip."
        )


def _authorised_output_paths(
    manifest: dict[str, Any],
    project_root: Path,
) -> tuple[Path, Path, Path]:
    """Resolve the three fixed manifest-authorised paths."""

    configured_directory = manifest.get(
        "corpus",
        {},
    ).get("processed_output_directory")

    if configured_directory != "data/interim/":
        raise CorpusExportError(
            "Manifest processed-output directory differs "
            "from the frozen data/interim/ boundary."
        )

    paths = (
        project_root / REGIONS_RELATIVE_PATH,
        project_root / PAGE_AUDIT_RELATIVE_PATH,
        project_root
        / EXTRACTION_SUMMARY_RELATIVE_PATH,
    )

    for path in paths:
        _relative_path(path, project_root)

    return paths


def _publish_staged_file(
    staged_path: Path,
    destination_path: Path,
) -> None:
    """Atomically replace one destination with a staged file."""

    destination_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    os.replace(
        staged_path,
        destination_path,
    )


def export_validated_corpus(
    manifest_path: str | Path,
    project_root: str | Path | None = None,
    overwrite: bool = False,
) -> CorpusExportResult:
    """Validate, extract and publish all three corpus exports."""

    resolved_manifest_path = Path(
        manifest_path
    ).resolve()
    resolved_project_root = (
        Path(project_root).resolve()
        if project_root is not None
        else resolved_manifest_path.parent.parent
    )
    manifest = load_corpus_manifest(
        resolved_manifest_path
    )
    validation_result = validate_corpus_manifest(
        resolved_manifest_path,
        project_root=resolved_project_root,
    )
    (
        regions_path,
        page_audit_path,
        summary_path,
    ) = _authorised_output_paths(
        manifest,
        resolved_project_root,
    )
    destination_paths = (
        regions_path,
        page_audit_path,
        summary_path,
    )
    existing_paths = [
        path
        for path in destination_paths
        if path.exists()
    ]

    if existing_paths and not overwrite:
        raise CorpusExportError(
            "Corpus exports already exist. Use overwrite "
            "only after confirming that regeneration is "
            "intended: "
            + ", ".join(
                str(path)
                for path in existing_paths
            )
        )

    document = pymupdf.open(
        validation_result.source_path
    )

    try:
        cleaned_pages = extract_validated_corpus_pages(
            document
        )
    finally:
        document.close()

    region_records = build_retrieval_region_records(
        cleaned_pages,
        validation_result.source_id,
        validation_result.sha256,
    )
    page_audit_records = build_page_audit_records(
        cleaned_pages,
        region_records,
        validation_result.source_id,
        validation_result.sha256,
    )
    regions_bytes = serialise_json_lines(
        region_records
    )
    page_audit_bytes = serialise_json_lines(
        page_audit_records
    )
    _validate_serialised_round_trip(
        regions_bytes,
        region_records,
        "Region export",
    )
    _validate_serialised_round_trip(
        page_audit_bytes,
        page_audit_records,
        "Page-audit export",
    )

    processed_root = (
        resolved_project_root / "data/interim"
    )
    processed_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    with TemporaryDirectory(
        prefix=".corpus-export-",
        dir=processed_root,
    ) as temporary_directory:
        staging_root = Path(temporary_directory)
        staged_regions_path = (
            staging_root / "corpus/regions.jsonl"
        )
        staged_page_audit_path = (
            staging_root / "audit/pages.jsonl"
        )
        staged_summary_path = (
            staging_root
            / "audit/extraction-summary.json"
        )
        staged_regions_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        staged_page_audit_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        staged_regions_path.write_bytes(
            regions_bytes
        )
        staged_page_audit_path.write_bytes(
            page_audit_bytes
        )

        regions_fingerprint = calculate_file_sha256(
            staged_regions_path
        )
        page_audit_fingerprint = (
            calculate_file_sha256(
                staged_page_audit_path
            )
        )
        _validate_sha256(
            regions_fingerprint,
            "Region-export fingerprint",
        )
        _validate_sha256(
            page_audit_fingerprint,
            "Page-audit fingerprint",
        )
        summary = build_extraction_summary(
            validation_result=validation_result,
            project_root=resolved_project_root,
            region_records=region_records,
            page_audit_records=page_audit_records,
            regions_path=regions_path,
            regions_sha256=regions_fingerprint,
            page_audit_path=page_audit_path,
            page_audit_sha256=(
                page_audit_fingerprint
            ),
        )
        summary_bytes = serialise_json_object(
            summary
        )
        staged_summary_path.write_bytes(
            summary_bytes
        )

        # Publish the summary last. Its two fingerprints then act as
        # a consistency marker for the already completed data files.
        _publish_staged_file(
            staged_regions_path,
            regions_path,
        )
        _publish_staged_file(
            staged_page_audit_path,
            page_audit_path,
        )
        _publish_staged_file(
            staged_summary_path,
            summary_path,
        )

    if (
        calculate_file_sha256(regions_path)
        != regions_fingerprint
        or calculate_file_sha256(page_audit_path)
        != page_audit_fingerprint
    ):
        raise CorpusExportError(
            "Published export fingerprints differ from "
            "their completed staged files."
        )

    recovered_summary = json.loads(
        summary_path.read_text(encoding="utf-8")
    )

    if recovered_summary != summary:
        raise CorpusExportError(
            "Published extraction summary failed its "
            "JSON round trip."
        )

    return CorpusExportResult(
        regions_path=regions_path,
        page_audit_path=page_audit_path,
        extraction_summary_path=summary_path,
        regions_sha256=regions_fingerprint,
        page_audit_sha256=page_audit_fingerprint,
        retrieval_record_count=len(region_records),
        page_audit_record_count=len(
            page_audit_records
        ),
    )


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the production export command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Validate the authorised textbook and create "
            "the three deterministic interim exports."
        )
    )
    parser.add_argument(
        "--manifest",
        default="configs/corpus-manifest.json",
        help="Corpus-manifest path.",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Repository root containing the manifest.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace all existing derived exports.",
    )
    return parser


def main() -> None:
    """Run the deterministic production export."""

    arguments = _build_argument_parser().parse_args()
    result = export_validated_corpus(
        manifest_path=arguments.manifest,
        project_root=arguments.project_root,
        overwrite=arguments.overwrite,
    )

    print("Corpus export: PASSED")
    print(
        "  Retrieval records:",
        result.retrieval_record_count,
    )
    print(
        "  Page-audit records:",
        result.page_audit_record_count,
    )
    print(
        "  Regions path:",
        result.regions_path,
    )
    print(
        "  Regions SHA-256:",
        result.regions_sha256,
    )
    print(
        "  Page-audit path:",
        result.page_audit_path,
    )
    print(
        "  Page-audit SHA-256:",
        result.page_audit_sha256,
    )
    print(
        "  Summary path:",
        result.extraction_summary_path,
    )


if __name__ == "__main__":
    main()
