"""Pure vertical-interval and PDF page-geometry operations."""

from __future__ import annotations

from collections import defaultdict
from math import isclose, isfinite
from typing import Any

import pymupdf


COORDINATE_TOLERANCE = 1e-6


class CorpusGeometryError(ValueError):
    """Raised when page or interval geometry is invalid."""


def _require_finite_number(
    value: Any,
    description: str,
) -> float:
    """Return one finite numeric coordinate."""

    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not isfinite(value)
    ):
        raise CorpusGeometryError(
            f"{description} must be a finite number."
        )

    return float(value)


def _normalise_interval(
    interval: Any,
    description: str,
) -> tuple[float, float]:
    """Validate and return one ordered vertical interval."""

    if not isinstance(interval, (list, tuple)) or len(interval) != 2:
        raise CorpusGeometryError(
            f"{description} must contain two coordinates."
        )

    y0 = _require_finite_number(
        interval[0],
        f"{description} upper coordinate",
    )
    y1 = _require_finite_number(
        interval[1],
        f"{description} lower coordinate",
    )

    if y0 >= y1:
        raise CorpusGeometryError(
            f"{description} must satisfy y0 < y1."
        )

    return (y0, y1)


def merge_vertical_intervals(
    intervals: list[tuple[float, float]],
    tolerance: float = COORDINATE_TOLERANCE,
) -> list[tuple[float, float]]:
    """Merge overlapping or touching vertical intervals."""

    validated_tolerance = _require_finite_number(
        tolerance,
        "Coordinate tolerance",
    )

    if validated_tolerance < 0.0:
        raise CorpusGeometryError(
            "Coordinate tolerance cannot be negative."
        )

    if not intervals:
        return []

    ordered_intervals = sorted(
        _normalise_interval(
            interval,
            f"Vertical interval {interval_index}",
        )
        for interval_index, interval in enumerate(
            intervals
        )
    )
    merged_intervals = [ordered_intervals[0]]

    for y0, y1 in ordered_intervals[1:]:
        previous_y0, previous_y1 = (
            merged_intervals[-1]
        )

        if y0 <= previous_y1 + validated_tolerance:
            merged_intervals[-1] = (
                previous_y0,
                max(previous_y1, y1),
            )
        else:
            merged_intervals.append((y0, y1))

    return merged_intervals


def complement_vertical_intervals(
    excluded_intervals: list[tuple[float, float]],
    page_height: float,
    tolerance: float = COORDINATE_TOLERANCE,
) -> list[tuple[float, float]]:
    """Calculate retained page intervals outside exclusions."""

    validated_page_height = _require_finite_number(
        page_height,
        "Page height",
    )
    validated_tolerance = _require_finite_number(
        tolerance,
        "Coordinate tolerance",
    )

    if validated_page_height <= 0.0:
        raise CorpusGeometryError(
            "Page height must be positive."
        )

    if validated_tolerance < 0.0:
        raise CorpusGeometryError(
            "Coordinate tolerance cannot be negative."
        )

    merged_exclusions = merge_vertical_intervals(
        excluded_intervals,
        tolerance=validated_tolerance,
    )

    for interval_index, (y0, y1) in enumerate(
        merged_exclusions
    ):
        if (
            y0 < -validated_tolerance
            or y1
            > validated_page_height
            + validated_tolerance
        ):
            raise CorpusGeometryError(
                "Excluded interval "
                f"{interval_index} lies outside the page."
            )

    retained_intervals = []
    cursor_y = 0.0

    for excluded_y0, excluded_y1 in merged_exclusions:
        if excluded_y0 > cursor_y + validated_tolerance:
            retained_intervals.append(
                (cursor_y, excluded_y0)
            )

        cursor_y = max(cursor_y, excluded_y1)

    if cursor_y < validated_page_height - validated_tolerance:
        retained_intervals.append(
            (cursor_y, validated_page_height)
        )

    return retained_intervals


def make_exclusion_interval(
    section_id: str,
    content_type: str,
    start_match: dict[str, Any],
    end_match: dict[str, Any],
) -> dict[str, Any]:
    """Create one validated half-open document interval."""

    if not section_id.strip():
        raise CorpusGeometryError(
            "Section identifier cannot be empty."
        )

    if not content_type.strip():
        raise CorpusGeometryError(
            "Content type cannot be empty."
        )

    start_page_index = start_match.get(
        "pdf_page_index"
    )
    end_page_index = end_match.get(
        "pdf_page_index"
    )

    if (
        not isinstance(start_page_index, int)
        or isinstance(start_page_index, bool)
        or not isinstance(end_page_index, int)
        or isinstance(end_page_index, bool)
    ):
        raise CorpusGeometryError(
            "Boundary page indices must be integers."
        )

    try:
        start_y_value = start_match["bbox"][1]
        end_y_value = end_match["bbox"][1]
    except (KeyError, IndexError, TypeError) as error:
        raise CorpusGeometryError(
            "Boundary matches must contain bounding boxes."
        ) from error

    start_y = _require_finite_number(
        start_y_value,
        "Start boundary y-coordinate",
    )
    end_y = _require_finite_number(
        end_y_value,
        "End boundary y-coordinate",
    )

    if (start_page_index, start_y) >= (
        end_page_index,
        end_y,
    ):
        raise CorpusGeometryError(
            f"Invalid exclusion interval: {section_id}"
        )

    return {
        "section_id": section_id,
        "content_type": content_type,
        "start_page_index": start_page_index,
        "start_pdf_page_number": start_page_index + 1,
        "start_y": start_y,
        "end_page_index": end_page_index,
        "end_pdf_page_number": end_page_index + 1,
        "end_y": end_y,
    }


def build_page_geometry(
    pdf_document: pymupdf.Document,
    section_intervals: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert document exclusions into page-level geometry."""

    raw_page_exclusions: defaultdict[
        int,
        list[tuple[float, float]],
    ] = defaultdict(list)

    for section in section_intervals:
        section_id = section.get("section_id")
        start_page_index = section.get(
            "start_page_index"
        )
        end_page_index = section.get(
            "end_page_index"
        )

        if not isinstance(section_id, str) or not section_id:
            raise CorpusGeometryError(
                "Every exclusion requires a section identifier."
            )

        if (
            not isinstance(start_page_index, int)
            or isinstance(start_page_index, bool)
            or not isinstance(end_page_index, int)
            or isinstance(end_page_index, bool)
            or not 0
            <= start_page_index
            <= end_page_index
            < pdf_document.page_count
        ):
            raise CorpusGeometryError(
                "Section page indices are outside the PDF: "
                f"{section_id}"
            )

        start_y = _require_finite_number(
            section.get("start_y"),
            f"{section_id} start coordinate",
        )
        end_y = _require_finite_number(
            section.get("end_y"),
            f"{section_id} end coordinate",
        )

        for page_index in range(
            start_page_index,
            end_page_index + 1,
        ):
            page = pdf_document.load_page(page_index)
            page_height = float(page.rect.height)

            excluded_y0 = (
                start_y
                if page_index == start_page_index
                else 0.0
            )
            excluded_y1 = (
                end_y
                if page_index == end_page_index
                else page_height
            )

            if not (
                0.0
                <= excluded_y0
                < excluded_y1
                <= page_height
            ):
                raise CorpusGeometryError(
                    "Invalid page exclusion for "
                    f"{section_id} on PDF page "
                    f"{page_index + 1}: "
                    f"({excluded_y0}, {excluded_y1})"
                )

            raw_page_exclusions[page_index].append(
                (excluded_y0, excluded_y1)
            )

    page_inventory = []

    for page_index in range(pdf_document.page_count):
        page = pdf_document.load_page(page_index)
        page_width = float(page.rect.width)
        page_height = float(page.rect.height)
        excluded_intervals = merge_vertical_intervals(
            raw_page_exclusions.get(page_index, [])
        )
        retained_intervals = complement_vertical_intervals(
            excluded_intervals,
            page_height,
        )

        if not excluded_intervals:
            page_state = "fully_retained"
        elif (
            len(excluded_intervals) == 1
            and isclose(
                excluded_intervals[0][0],
                0.0,
                rel_tol=0.0,
                abs_tol=COORDINATE_TOLERANCE,
            )
            and isclose(
                excluded_intervals[0][1],
                page_height,
                rel_tol=0.0,
                abs_tol=COORDINATE_TOLERANCE,
            )
        ):
            page_state = "fully_excluded"
        else:
            page_state = "partially_retained"

        page_inventory.append(
            {
                "pdf_page_index": page_index,
                "pdf_page_number": page_index + 1,
                "page_width": page_width,
                "page_height": page_height,
                "state": page_state,
                "excluded_y_intervals": excluded_intervals,
                "retained_y_intervals": retained_intervals,
            }
        )

    return page_inventory
