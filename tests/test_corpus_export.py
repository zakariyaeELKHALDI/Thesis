"""Tests for deterministic production corpus exports."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pymupdf
import pytest

from geotech_rag import (
    load_corpus_manifest,
    validate_corpus_manifest,
)
from geotech_rag.corpus_export import (
    COORDINATE_ORIGIN,
    COORDINATE_UNIT,
    EXPECTED_RETRIEVAL_RECORD_COUNT,
    EXTRACTION_SUMMARY_RELATIVE_PATH,
    PAGE_AUDIT_RELATIVE_PATH,
    PAGE_AUDIT_SCHEMA_VERSION,
    RECORD_SCHEMA_VERSION,
    REGIONS_RELATIVE_PATH,
    CorpusExportError,
    build_extraction_summary,
    build_page_audit_records,
    build_retrieval_region_records,
    serialise_json_lines,
    serialise_json_object,
)
from geotech_rag.corpus_extraction import (
    EXPECTED_CLEANED_CHARACTER_COUNT,
    EXPECTED_CLEANED_LINE_COUNT,
    EXPECTED_EMPTY_RETAINED_REGION_PAGES,
    EXPECTED_STRUCTURAL_REMOVAL_COUNTS,
    PUBLISHER_COPYRIGHT_TEXT,
    extract_validated_corpus_pages,
)


SOURCE_ID = "synthetic-source"
SOURCE_SHA256 = "a" * 64


def _line_record(
    text: str,
    bbox: tuple[float, float, float, float],
    region_index: int,
    block_number: int,
    line_number: int,
    direction: tuple[float, float] = (1.0, 0.0),
) -> dict:
    """Create one internal positioned-line record."""

    return {
        "text": text,
        "bbox": bbox,
        "region_index": region_index,
        "block_number": block_number,
        "line_number": line_number,
        "span_count": 1,
        "direction": direction,
        "writing_mode": 0,
    }


def _synthetic_cleaned_pages() -> list[dict]:
    """Create retained and excluded pages for unit tests."""

    first_line = _line_record(
        text="First technical line",
        bbox=(20.0, 100.0, 200.0, 120.0),
        region_index=0,
        block_number=0,
        line_number=0,
    )
    second_line = _line_record(
        text="Rotated formula \x04",
        bbox=(250.0, 400.0, 270.0, 500.0),
        region_index=1,
        block_number=1,
        line_number=0,
        direction=(0.0, -1.0),
    )
    removed_line = {
        **_line_record(
            text=PUBLISHER_COPYRIGHT_TEXT,
            bbox=(500.0, 100.0, 520.0, 250.0),
            region_index=0,
            block_number=2,
            line_number=0,
            direction=(0.0, -1.0),
        ),
        "removal_reason": (
            "remove_publisher_copyright"
        ),
    }
    page_text = "\n".join(
        [
            first_line["text"],
            second_line["text"],
        ]
    )

    return [
        {
            "pdf_page_index": 24,
            "pdf_page_number": 25,
            "printed_page_number": 1,
            "page_state": "partially_retained",
            "page_width": 600.0,
            "page_height": 700.0,
            "excluded_regions": [
                (250.0, 350.0)
            ],
            "retained_regions": [
                (0.0, 250.0),
                (350.0, 700.0),
            ],
            "raw_line_count": 3,
            "raw_character_count": 70,
            "lines": [
                first_line,
                second_line,
            ],
            "text": page_text,
            "line_count": 2,
            "character_count": len(page_text),
            "removed_structural_lines": [
                removed_line
            ],
            "removed_structural_line_count": 1,
        },
        {
            "pdf_page_index": 25,
            "pdf_page_number": 26,
            "printed_page_number": 2,
            "page_state": "fully_excluded",
            "page_width": 600.0,
            "page_height": 700.0,
            "excluded_regions": [
                (0.0, 700.0)
            ],
            "retained_regions": [],
            "raw_line_count": 0,
            "raw_character_count": 0,
            "lines": [],
            "text": "",
            "line_count": 0,
            "character_count": 0,
            "removed_structural_lines": [],
            "removed_structural_line_count": 0,
        },
    ]


def test_record_builders_follow_frozen_schema() -> None:
    """Region and page records contain the exact fields."""

    cleaned_pages = _synthetic_cleaned_pages()
    region_records = build_retrieval_region_records(
        cleaned_pages,
        SOURCE_ID,
        SOURCE_SHA256,
    )
    page_audits = build_page_audit_records(
        cleaned_pages,
        region_records,
        SOURCE_ID,
        SOURCE_SHA256,
    )

    assert len(region_records) == 2
    assert [
        record["record_id"]
        for record in region_records
    ] == [
        "synthetic-source:pdf-0025:region-00",
        "synthetic-source:pdf-0025:region-01",
    ]
    assert region_records[0]["region_bbox"] == [
        0.0,
        0.0,
        600.0,
        250.0,
    ]
    assert region_records[1]["lines"][0][
        "direction"
    ] == [0.0, -1.0]
    assert region_records[1]["text"] == (
        "Rotated formula \x04"
    )
    assert region_records[1]["character_count"] == len(
        region_records[1]["text"]
    )
    assert set(region_records[0]) == {
        "record_schema_version",
        "record_id",
        "source_id",
        "source_sha256",
        "pdf_page_index",
        "pdf_page_number",
        "printed_page_number",
        "page_state",
        "page_width",
        "page_height",
        "region_index",
        "region_bbox",
        "coordinate_unit",
        "coordinate_origin",
        "text",
        "line_count",
        "character_count",
        "lines",
    }
    assert set(region_records[0]["lines"][0]) == {
        "line_index",
        "text",
        "bbox",
        "block_number",
        "line_number",
        "span_count",
        "direction",
        "writing_mode",
    }
    assert region_records[0][
        "record_schema_version"
    ] == RECORD_SCHEMA_VERSION
    assert region_records[0][
        "coordinate_unit"
    ] == COORDINATE_UNIT
    assert region_records[0][
        "coordinate_origin"
    ] == COORDINATE_ORIGIN

    assert len(page_audits) == 2
    assert page_audits[0][
        "page_audit_schema_version"
    ] == PAGE_AUDIT_SCHEMA_VERSION
    assert page_audits[0][
        "emitted_region_count"
    ] == 2
    assert page_audits[0][
        "emitted_record_ids"
    ] == [
        "synthetic-source:pdf-0025:region-00",
        "synthetic-source:pdf-0025:region-01",
    ]
    assert page_audits[1][
        "emitted_region_count"
    ] == 0
    assert page_audits[1][
        "emitted_record_ids"
    ] == []
    assert set(page_audits[0]) == {
        "page_audit_schema_version",
        "source_id",
        "source_sha256",
        "pdf_page_index",
        "pdf_page_number",
        "printed_page_number",
        "page_width",
        "page_height",
        "coordinate_unit",
        "coordinate_origin",
        "page_state",
        "excluded_y_intervals",
        "retained_y_intervals",
        "raw_line_count",
        "raw_character_count",
        "cleaned_line_count",
        "cleaned_character_count",
        "removed_structural_line_count",
        "removed_structural_lines",
        "emitted_region_count",
        "emitted_record_ids",
    }
    assert set(
        page_audits[0][
            "removed_structural_lines"
        ][0]
    ) == {
        "text",
        "bbox",
        "region_index",
        "block_number",
        "line_number",
        "span_count",
        "direction",
        "writing_mode",
        "removal_reason",
    }


def test_missing_line_region_is_rejected() -> None:
    """A line cannot refer to a region that is absent."""

    cleaned_pages = _synthetic_cleaned_pages()
    cleaned_pages[0]["lines"][0][
        "region_index"
    ] = 2

    with pytest.raises(
        CorpusExportError,
        match="missing region",
    ):
        build_retrieval_region_records(
            cleaned_pages,
            SOURCE_ID,
            SOURCE_SHA256,
        )


def test_line_outside_region_is_rejected() -> None:
    """The derived coordinate tolerance is enforced."""

    cleaned_pages = _synthetic_cleaned_pages()
    cleaned_pages[0]["lines"][0]["bbox"] = (
        20.0,
        -0.01,
        200.0,
        120.0,
    )

    with pytest.raises(
        CorpusExportError,
        match="exceeds its retained region",
    ):
        build_retrieval_region_records(
            cleaned_pages,
            SOURCE_ID,
            SOURCE_SHA256,
        )


def test_exact_publisher_line_cannot_enter_retrieval() -> None:
    """The frozen publisher label is rejected after cleaning."""

    cleaned_pages = _synthetic_cleaned_pages()
    cleaned_pages[0]["lines"][0]["text"] = (
        PUBLISHER_COPYRIGHT_TEXT
    )
    cleaned_pages[0]["text"] = "\n".join(
        line["text"]
        for line in cleaned_pages[0]["lines"]
    )

    with pytest.raises(
        CorpusExportError,
        match="exact publisher line",
    ):
        build_retrieval_region_records(
            cleaned_pages,
            SOURCE_ID,
            SOURCE_SHA256,
        )


def test_json_serialisation_is_stable_and_reversible() -> None:
    """Key order, UTF-8 and control characters are stable."""

    records = [
        {
            "z_control": "\x04",
            "a_unicode": "©",
        }
    ]
    first_result = serialise_json_lines(records)
    second_result = serialise_json_lines(records)

    assert first_result == second_result
    assert first_result.endswith(b"\n")
    assert first_result == (
        '{"a_unicode":"©","z_control":"\\u0004"}\n'
    ).encode("utf-8")
    assert [
        json.loads(line)
        for line in first_result.decode(
            "utf-8"
        ).splitlines()
    ] == records

    summary_bytes = serialise_json_object(
        records[0]
    )
    assert summary_bytes.endswith(b"\n")
    assert json.loads(
        summary_bytes.decode("utf-8")
    ) == records[0]


def test_nonfinite_json_number_is_rejected() -> None:
    """NaN and infinity cannot enter an export."""

    with pytest.raises(
        CorpusExportError,
        match="not deterministically serialisable",
    ):
        serialise_json_lines(
            [{"invalid": float("nan")}]
        )

    with pytest.raises(
        CorpusExportError,
        match="not deterministically serialisable",
    ):
        serialise_json_object(
            {"invalid": float("inf")}
        )


def test_repository_source_reproduces_export_contract() -> None:
    """The private source must reproduce all three contracts."""

    repository_root = Path(
        __file__
    ).resolve().parents[1]
    manifest_path = (
        repository_root
        / "configs"
        / "corpus-manifest.json"
    )
    manifest = load_corpus_manifest(
        manifest_path
    )
    local_source_path = (
        repository_root
        / manifest["sources"][0][
            "relative_path"
        ]
    )

    if not local_source_path.is_file():
        pytest.skip(
            "Private local corpus source is unavailable."
        )

    validation_result = validate_corpus_manifest(
        manifest_path,
        project_root=repository_root,
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
    page_audits = build_page_audit_records(
        cleaned_pages,
        region_records,
        validation_result.source_id,
        validation_result.sha256,
    )
    regions_bytes = serialise_json_lines(
        region_records
    )
    page_audit_bytes = serialise_json_lines(
        page_audits
    )
    summary = build_extraction_summary(
        validation_result=validation_result,
        project_root=repository_root,
        region_records=region_records,
        page_audit_records=page_audits,
        regions_path=(
            repository_root / REGIONS_RELATIVE_PATH
        ),
        regions_sha256=sha256(
            regions_bytes
        ).hexdigest(),
        page_audit_path=(
            repository_root
            / PAGE_AUDIT_RELATIVE_PATH
        ),
        page_audit_sha256=sha256(
            page_audit_bytes
        ).hexdigest(),
    )

    assert len(region_records) == (
        EXPECTED_RETRIEVAL_RECORD_COUNT
    )
    assert len(page_audits) == 770
    assert summary["retrieval_record_count"] == 670
    assert summary["retrieval_line_count"] == (
        EXPECTED_CLEANED_LINE_COUNT
    )
    assert summary[
        "retrieval_character_count"
    ] == EXPECTED_CLEANED_CHARACTER_COUNT
    assert summary[
        "empty_retained_region_pdf_pages"
    ] == list(
        EXPECTED_EMPTY_RETAINED_REGION_PAGES
    )
    assert summary[
        "structural_removal_counts"
    ] == dict(
        EXPECTED_STRUCTURAL_REMOVAL_COUNTS
    )
    assert sum(
        item["line_count"]
        for item in summary[
            "line_orientation_counts"
        ]
    ) == EXPECTED_CLEANED_LINE_COUNT
    assert summary["regions_relative_path"] == (
        REGIONS_RELATIVE_PATH.as_posix()
    )
    assert summary[
        "page_audit_relative_path"
    ] == PAGE_AUDIT_RELATIVE_PATH.as_posix()
    assert (
        repository_root
        / EXTRACTION_SUMMARY_RELATIVE_PATH
    ).as_posix().endswith(
        "data/interim/audit/"
        "extraction-summary.json"
    )
    assert all(
        record["text"]
        == "\n".join(
            line["text"]
            for line in record["lines"]
        )
        for record in region_records
    )
    assert all(
        record["page_state"]
        in {
            "fully_retained",
            "partially_retained",
        }
        for record in region_records
    )
    assert page_audits[86][
        "emitted_region_count"
    ] == 0
    assert page_audits[86][
        "pdf_page_number"
    ] == 87
    assert serialise_json_lines(
        region_records
    ) == regions_bytes
    assert serialise_json_lines(
        page_audits
    ) == page_audit_bytes
