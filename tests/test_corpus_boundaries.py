"""Tests for source-specific textbook boundaries."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pymupdf
import pytest

from geotech_rag import (
    load_corpus_manifest,
    validate_corpus_manifest,
)
from geotech_rag.corpus_boundaries import (
    CorpusBoundaryError,
    EXPECTED_PROBLEM_PAGES,
    EXPECTED_REFERENCE_PAGES,
    build_validated_knowledge_page_geometry,
    classify_heading_match,
    collect_exact_heading_matches,
    pair_problem_reference_headings,
)


def _heading_match(
    heading: str,
    page_index: int,
    y0: float,
) -> dict:
    """Create one synthetic positioned heading."""

    return {
        "heading": heading,
        "pdf_page_index": page_index,
        "pdf_page_number": page_index + 1,
        "bbox": (
            10.0,
            y0,
            100.0,
            y0 + 10.0,
        ),
        "block_number": 0,
        "line_number": 0,
    }


def test_heading_classification_uses_page_and_position() -> None:
    """Contents, headers and body matches remain distinct."""

    assert (
        classify_heading_match(
            _heading_match(
                "Problems",
                14,
                200.0,
            )
        )
        == "contents"
    )
    assert (
        classify_heading_match(
            _heading_match(
                "Problems",
                24,
                20.0,
            )
        )
        == "running_header"
    )
    assert (
        classify_heading_match(
            _heading_match(
                "Problems",
                24,
                100.0,
            )
        )
        == "body"
    )


def test_exact_heading_collection_preserves_position() -> None:
    """Detected headings retain page and bounding-box data."""

    document = pymupdf.open()

    try:
        page = document.new_page(
            width=200.0,
            height=300.0,
        )
        page.insert_text(
            (40.0, 100.0),
            "Problems",
        )

        matches = collect_exact_heading_matches(
            document,
            ("Problems",),
        )
    finally:
        document.close()

    assert len(matches) == 1
    assert matches[0]["heading"] == "Problems"
    assert matches[0]["pdf_page_index"] == 0
    assert matches[0]["pdf_page_number"] == 1
    assert len(matches[0]["bbox"]) == 4


def test_problem_pairing_uses_next_available_reference() -> None:
    """Each problem section uses the next later reference."""

    earlier_reference = _heading_match(
        "References",
        0,
        100.0,
    )
    problem_start = _heading_match(
        "Problems",
        1,
        100.0,
    )
    later_reference = _heading_match(
        "References",
        2,
        100.0,
    )

    pairs, unpaired = (
        pair_problem_reference_headings(
            [
                earlier_reference,
                problem_start,
                later_reference,
            ]
        )
    )

    assert len(pairs) == 1
    assert (
        pairs[0]["problem_start"]
        == problem_start
    )
    assert (
        pairs[0]["reference_start"]
        == later_reference
    )
    assert unpaired == [earlier_reference]


def test_problem_without_later_reference_is_rejected() -> None:
    """An unsafe unmatched problem boundary must fail."""

    with pytest.raises(
        CorpusBoundaryError,
        match="No later References",
    ):
        pair_problem_reference_headings(
            [
                _heading_match(
                    "Problems",
                    2,
                    100.0,
                ),
                _heading_match(
                    "References",
                    1,
                    100.0,
                ),
            ]
        )


def test_wrong_source_page_count_is_rejected() -> None:
    """A different PDF cannot use the frozen boundaries."""

    document = pymupdf.open()

    try:
        document.new_page()

        with pytest.raises(
            CorpusBoundaryError,
            match="Expected 770 PDF pages",
        ):
            build_validated_knowledge_page_geometry(
                document
            )
    finally:
        document.close()


def test_repository_source_reproduces_boundary_audit() -> None:
    """The private source must reproduce the notebook audit."""

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

    # A public clone can run every synthetic test without
    # possessing or distributing the copyrighted textbook.
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
        page_geometry = (
            build_validated_knowledge_page_geometry(
                document
            )
        )
    finally:
        document.close()

    assert len(page_geometry) == 770
    assert Counter(
        page["state"]
        for page in page_geometry
    ) == Counter(
        {
            "fully_retained": 644,
            "partially_retained": 32,
            "fully_excluded": 37,
            "outside_knowledge_scope": 57,
        }
    )

    actual_boundary_pages = {
        page["pdf_page_number"]
        for page in page_geometry
        if page["state"]
        == "partially_retained"
    }
    expected_boundary_pages = {
        *EXPECTED_PROBLEM_PAGES,
        *EXPECTED_REFERENCE_PAGES,
    }

    assert (
        actual_boundary_pages
        == expected_boundary_pages
    )
    assert page_geometry[23]["state"] == (
        "outside_knowledge_scope"
    )
    assert page_geometry[24]["state"] == (
        "fully_retained"
    )
    assert page_geometry[737]["state"] == (
        "outside_knowledge_scope"
    )
