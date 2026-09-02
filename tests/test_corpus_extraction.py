"""Tests for positioned extraction and structural cleaning."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path

import pymupdf
import pytest

from geotech_rag import (
    load_corpus_manifest,
    validate_corpus_manifest,
)
from geotech_rag.corpus_extraction import (
    EXPECTED_CLEANED_CHARACTER_COUNT,
    EXPECTED_CLEANED_LINE_COUNT,
    EXPECTED_EMPTY_RETAINED_REGION_PAGES,
    EXPECTED_STRUCTURAL_REMOVAL_COUNTS,
    PUBLISHER_COPYRIGHT_TEXT,
    CorpusExtractionError,
    classify_structural_line,
    extract_positioned_lines_from_page,
    extract_validated_corpus_pages,
    remove_structural_lines_from_page,
    trim_text_edges,
)


def _line_record(
    text: str,
    bbox: tuple[float, float, float, float],
) -> dict:
    """Create one complete synthetic positioned line."""

    return {
        "text": text,
        "bbox": bbox,
        "region_index": 0,
        "block_number": 0,
        "line_number": 0,
        "span_count": 1,
        "direction": (1.0, 0.0),
        "writing_mode": 0,
    }


def _full_page_geometry(
    width: float,
    height: float,
) -> dict:
    """Create full-page geometry for a synthetic PDF."""

    return {
        "pdf_page_index": 0,
        "pdf_page_number": 1,
        "page_width": width,
        "page_height": height,
        "state": "fully_retained",
        "excluded_y_intervals": [],
        "retained_y_intervals": [
            (0.0, height)
        ],
    }


def test_edge_trimming_preserves_formula_controls() -> None:
    """Only ordinary edge whitespace may be removed."""

    assert trim_text_edges(
        " \t\x04 formula\x05 \r\n"
    ) == "\x04 formula\x05"


def test_structural_classification_uses_exact_rules() -> None:
    """Only the three audited structural cases are removed."""

    assert classify_structural_line(
        _line_record(
            "Chapter heading",
            (50.0, 10.0, 200.0, 30.0),
        ),
        pdf_page_number=25,
        page_width=600.0,
    ) == "remove_running_header"

    assert classify_structural_line(
        _line_record(
            "1",
            (550.0, 620.0, 570.0, 630.0),
        ),
        pdf_page_number=25,
        page_width=600.0,
    ) == "remove_bottom_page_number"

    assert classify_structural_line(
        _line_record(
            PUBLISHER_COPYRIGHT_TEXT,
            (200.0, 100.0, 220.0, 300.0),
        ),
        pdf_page_number=25,
        page_width=600.0,
    ) == "remove_publisher_copyright"

    # A broader publisher substring must remain untouched.
    assert classify_structural_line(
        _line_record(
            "Published by Cengage Learning",
            (100.0, 100.0, 300.0, 120.0),
        ),
        pdf_page_number=25,
        page_width=600.0,
    ) == "retain"


def test_extraction_adds_orientation_metadata() -> None:
    """Horizontal and rotated lines preserve their directions."""

    document = pymupdf.open()

    try:
        page = document.new_page(
            width=300.0,
            height=300.0,
        )
        page.insert_text(
            (40.0, 100.0),
            "Horizontal text",
        )
        page.insert_text(
            (200.0, 250.0),
            "Rotated text",
            rotate=90,
        )

        extracted_page = (
            extract_positioned_lines_from_page(
                document,
                _full_page_geometry(
                    300.0,
                    300.0,
                ),
            )
        )
    finally:
        document.close()

    assert extracted_page["line_count"] == 2
    assert all(
        len(line["direction"]) == 2
        for line in extracted_page["lines"]
    )
    assert all(
        line["writing_mode"] == 0
        for line in extracted_page["lines"]
    )
    assert {
        tuple(
            round(value, 6)
            for value in line["direction"]
        )
        for line in extracted_page["lines"]
    } == {
        (1.0, 0.0),
        (0.0, -1.0),
    }


def test_page_dimension_mismatch_is_rejected() -> None:
    """Geometry from another page cannot be reused."""

    document = pymupdf.open()

    try:
        document.new_page(
            width=300.0,
            height=300.0,
        )
        invalid_geometry = (
            _full_page_geometry(
                301.0,
                300.0,
            )
        )

        with pytest.raises(
            CorpusExtractionError,
            match="Page-width mismatch",
        ):
            extract_positioned_lines_from_page(
                document,
                invalid_geometry,
            )
    finally:
        document.close()


def test_cleaning_preserves_audit_and_input() -> None:
    """Removed lines remain auditable without mutating input."""

    raw_page = {
        "pdf_page_index": 24,
        "pdf_page_number": 25,
        "printed_page_number": 1,
        "page_state": "fully_retained",
        "page_width": 600.0,
        "page_height": 658.0,
        "excluded_regions": [],
        "retained_regions": [(0.0, 658.0)],
        "lines": [
            _line_record(
                "Header",
                (50.0, 10.0, 200.0, 30.0),
            ),
            _line_record(
                "1",
                (550.0, 620.0, 570.0, 630.0),
            ),
            _line_record(
                PUBLISHER_COPYRIGHT_TEXT,
                (200.0, 100.0, 220.0, 300.0),
            ),
            _line_record(
                "Retained technical content",
                (50.0, 100.0, 250.0, 120.0),
            ),
        ],
        "text": "temporary raw text",
        "line_count": 4,
        "character_count": 18,
    }
    original_page = deepcopy(raw_page)

    cleaned_page = (
        remove_structural_lines_from_page(
            raw_page
        )
    )

    assert raw_page == original_page
    assert cleaned_page["line_count"] == 1
    assert cleaned_page["text"] == (
        "Retained technical content"
    )
    assert cleaned_page["raw_line_count"] == 4
    assert (
        cleaned_page[
            "removed_structural_line_count"
        ]
        == 3
    )
    assert {
        line["removal_reason"]
        for line in cleaned_page[
            "removed_structural_lines"
        ]
    } == {
        "remove_running_header",
        "remove_bottom_page_number",
        "remove_publisher_copyright",
    }


def test_repository_source_reproduces_extraction_audit() -> None:
    """The private source must reproduce all frozen totals."""

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

    validation_result = (
        validate_corpus_manifest(
            manifest_path,
            project_root=repository_root,
        )
    )
    document = pymupdf.open(
        validation_result.source_path
    )

    try:
        cleaned_pages = (
            extract_validated_corpus_pages(
                document
            )
        )
    finally:
        document.close()

    assert len(cleaned_pages) == 770
    assert sum(
        page["line_count"]
        for page in cleaned_pages
    ) == EXPECTED_CLEANED_LINE_COUNT
    assert sum(
        page["character_count"]
        for page in cleaned_pages
    ) == EXPECTED_CLEANED_CHARACTER_COUNT
    assert sum(
        bool(page["lines"])
        for page in cleaned_pages
    ) == 670

    removal_counts = Counter(
        line["removal_reason"]
        for page in cleaned_pages
        for line in page[
            "removed_structural_lines"
        ]
    )
    assert (
        removal_counts
        == EXPECTED_STRUCTURAL_REMOVAL_COUNTS
    )

    emptied_pages = tuple(
        page["pdf_page_number"]
        for page in cleaned_pages
        if (
            page["raw_line_count"]
            and not page["line_count"]
        )
    )
    assert (
        emptied_pages
        == EXPECTED_EMPTY_RETAINED_REGION_PAGES
    )
    assert all(
        line["text"]
        != PUBLISHER_COPYRIGHT_TEXT
        for page in cleaned_pages
        for line in page["lines"]
    )
    assert all(
        (
            len(line["direction"]) == 2
            and isinstance(
                line["writing_mode"],
                int,
            )
        )
        for page in cleaned_pages
        for line in page["lines"]
    )
