"""Tests for vertical intervals and PDF page geometry."""

from __future__ import annotations

import pymupdf
import pytest

from geotech_rag.corpus_geometry import (
    CorpusGeometryError,
    build_page_geometry,
    complement_vertical_intervals,
    make_exclusion_interval,
    merge_vertical_intervals,
)


def _heading_match(
    page_index: int,
    y0: float,
) -> dict:
    """Create the minimum heading record used by tests."""

    return {
        "pdf_page_index": page_index,
        "pdf_page_number": page_index + 1,
        "bbox": (10.0, y0, 100.0, y0 + 10.0),
    }


def test_merge_combines_overlapping_and_touching_ranges() -> None:
    """Connected exclusions must become one interval."""

    assert merge_vertical_intervals(
        [
            (30.0, 40.0),
            (10.0, 20.0),
            (20.0, 35.0),
            (50.0, 60.0),
        ]
    ) == [
        (10.0, 40.0),
        (50.0, 60.0),
    ]


def test_complement_returns_all_retained_gaps() -> None:
    """Retained intervals must fill every exclusion gap."""

    assert complement_vertical_intervals(
        [
            (10.0, 20.0),
            (30.0, 40.0),
        ],
        page_height=50.0,
    ) == [
        (0.0, 10.0),
        (20.0, 30.0),
        (40.0, 50.0),
    ]


def test_nonfinite_interval_is_rejected() -> None:
    """NaN cannot enter deterministic page geometry."""

    with pytest.raises(
        CorpusGeometryError,
        match="finite number",
    ):
        merge_vertical_intervals(
            [(0.0, float("nan"))]
        )


def test_reversed_document_interval_is_rejected() -> None:
    """An exclusion cannot end before it starts."""

    with pytest.raises(
        CorpusGeometryError,
        match="Invalid exclusion interval",
    ):
        make_exclusion_interval(
            section_id="invalid",
            content_type="problems",
            start_match=_heading_match(2, 100.0),
            end_match=_heading_match(1, 200.0),
        )


def test_multi_page_exclusion_builds_expected_geometry() -> None:
    """Boundary pages stay partial and the middle is removed."""

    document = pymupdf.open()

    try:
        for _ in range(3):
            document.new_page(
                width=200.0,
                height=300.0,
            )

        geometry = build_page_geometry(
            document,
            [
                {
                    "section_id": "synthetic",
                    "start_page_index": 0,
                    "end_page_index": 2,
                    "start_y": 100.0,
                    "end_y": 200.0,
                }
            ],
        )
    finally:
        document.close()

    assert [
        page["state"] for page in geometry
    ] == [
        "partially_retained",
        "fully_excluded",
        "partially_retained",
    ]
    assert geometry[0][
        "retained_y_intervals"
    ] == [(0.0, 100.0)]
    assert geometry[1][
        "retained_y_intervals"
    ] == []
    assert geometry[2][
        "retained_y_intervals"
    ] == [(200.0, 300.0)]


def test_exclusion_outside_physical_page_is_rejected() -> None:
    """Coordinates beyond the source page must fail."""

    document = pymupdf.open()

    try:
        document.new_page(
            width=200.0,
            height=300.0,
        )

        with pytest.raises(
            CorpusGeometryError,
            match="Invalid page exclusion",
        ):
            build_page_geometry(
                document,
                [
                    {
                        "section_id": "outside",
                        "start_page_index": 0,
                        "end_page_index": 0,
                        "start_y": 100.0,
                        "end_y": 301.0,
                    }
                ],
            )
    finally:
        document.close()
