"""Positioned text extraction and structural-line cleaning."""

from __future__ import annotations

from collections import Counter
from math import isclose, isfinite
from typing import Any

import pymupdf

from geotech_rag.corpus_boundaries import (
    EXPECTED_PAGE_COUNT,
    FIRST_KNOWLEDGE_PDF_PAGE,
    build_validated_knowledge_page_geometry,
)
from geotech_rag.corpus_geometry import (
    COORDINATE_TOLERANCE,
)


TEXT_EDGE_WHITESPACE = " \t\r\n"
RUNNING_HEADER_BOTTOM_Y = 40.0
PRINTED_PAGE_NUMBER_BOTTOM_Y = 608.0
PRINTED_PAGE_NUMBER_EDGE_WIDTH = 60.0
PRINTED_TO_PDF_PAGE_OFFSET = (
    FIRST_KNOWLEDGE_PDF_PAGE - 1
)

# The exact observed publisher label is removed. A broader substring
# rule could delete legitimate references to the publisher.
PUBLISHER_COPYRIGHT_TEXT = "© Cengage Learning 2014"

HEADING_COORDINATE_DECIMAL_PLACES = 2
HEADING_COORDINATE_HALF_ROUNDING_UNIT = (
    0.5
    * 10
    ** (-HEADING_COORDINATE_DECIMAL_PLACES)
)
LINE_REGION_COORDINATE_TOLERANCE = (
    HEADING_COORDINATE_HALF_ROUNDING_UNIT
    + COORDINATE_TOLERANCE
)

RETRIEVAL_PAGE_STATES = frozenset(
    {
        "fully_retained",
        "partially_retained",
    }
)

EXPECTED_RAW_LINE_COUNT = 41_672
EXPECTED_CLEANED_LINE_COUNT = 39_934
EXPECTED_CLEANED_CHARACTER_COUNT = 843_350
EXPECTED_PAGES_WITH_TEXT = 670
EXPECTED_EMPTY_RETAINED_REGION_PAGES = (
    87,
    144,
    370,
    445,
    509,
    727,
)
EXPECTED_STRUCTURAL_REMOVAL_COUNTS = Counter(
    {
        "remove_running_header": 1_263,
        "remove_bottom_page_number": 39,
        "remove_publisher_copyright": 436,
    }
)


class CorpusExtractionError(RuntimeError):
    """Raised when positioned extraction violates its contract."""


def trim_text_edges(text: str) -> str:
    """Remove only ordinary layout whitespace from line edges."""

    if not isinstance(text, str):
        raise CorpusExtractionError(
            "Extracted line text must be a string."
        )

    return text.strip(TEXT_EDGE_WHITESPACE)


def expected_printed_page_number(
    pdf_page_number: int,
) -> str:
    """Return the printed number expected on one PDF page."""

    if (
        not isinstance(pdf_page_number, int)
        or isinstance(pdf_page_number, bool)
    ):
        raise CorpusExtractionError(
            "PDF page number must be an integer."
        )

    return str(
        pdf_page_number
        - PRINTED_TO_PDF_PAGE_OFFSET
    )


def _validate_direction(
    direction: Any,
    pdf_page_number: int,
) -> tuple[float, float]:
    """Validate one PyMuPDF line-direction vector."""

    if (
        not isinstance(direction, (list, tuple))
        or len(direction) != 2
    ):
        raise CorpusExtractionError(
            "Line direction must contain two coordinates "
            f"on PDF page {pdf_page_number}."
        )

    converted_direction = tuple(
        float(coordinate)
        for coordinate in direction
    )

    if not all(
        isfinite(coordinate)
        for coordinate in converted_direction
    ):
        raise CorpusExtractionError(
            "Line direction must contain finite numbers "
            f"on PDF page {pdf_page_number}."
        )

    return converted_direction


def _validate_line_bbox(
    line_bbox: Any,
    retained_y0: float,
    retained_y1: float,
    page_width: float,
    pdf_page_number: int,
) -> tuple[float, float, float, float]:
    """Validate one line against its retained page region."""

    if (
        not isinstance(line_bbox, (list, tuple))
        or len(line_bbox) != 4
    ):
        raise CorpusExtractionError(
            "Line bounding box must contain four "
            f"coordinates on PDF page {pdf_page_number}."
        )

    converted_bbox = tuple(
        float(coordinate)
        for coordinate in line_bbox
    )

    if not all(
        isfinite(coordinate)
        for coordinate in converted_bbox
    ):
        raise CorpusExtractionError(
            "Line bounding box contains a non-finite "
            f"coordinate on PDF page {pdf_page_number}."
        )

    line_x0, line_y0, line_x1, line_y1 = (
        converted_bbox
    )

    if (
        line_x0
        > line_x1 + COORDINATE_TOLERANCE
        or line_y0
        > line_y1 + COORDINATE_TOLERANCE
    ):
        raise CorpusExtractionError(
            "Line bounding-box coordinates are reversed "
            f"on PDF page {pdf_page_number}."
        )

    boundary_excess = max(
        0.0,
        -line_x0,
        retained_y0 - line_y0,
        line_x1 - page_width,
        line_y1 - retained_y1,
    )

    if (
        boundary_excess
        > LINE_REGION_COORDINATE_TOLERANCE
    ):
        raise CorpusExtractionError(
            "Line bounding box exceeds its retained "
            f"region by {boundary_excess:.9f} points "
            f"on PDF page {pdf_page_number}."
        )

    return converted_bbox


def _validate_page_record(
    pdf_document: pymupdf.Document,
    page_record: dict[str, Any],
) -> pymupdf.Page:
    """Validate page identifiers and physical dimensions."""

    page_index = page_record.get(
        "pdf_page_index"
    )
    pdf_page_number = page_record.get(
        "pdf_page_number"
    )

    if (
        not isinstance(page_index, int)
        or isinstance(page_index, bool)
        or not 0
        <= page_index
        < pdf_document.page_count
    ):
        raise CorpusExtractionError(
            "PDF page index is outside the document: "
            f"{page_index!r}"
        )

    if (
        not isinstance(pdf_page_number, int)
        or isinstance(pdf_page_number, bool)
        or pdf_page_number != page_index + 1
    ):
        raise CorpusExtractionError(
            "PDF page index and number do not agree "
            f"for page index {page_index}."
        )

    source_page = pdf_document.load_page(
        page_index
    )

    if not isclose(
        float(source_page.rect.width),
        float(page_record.get("page_width")),
        rel_tol=0.0,
        abs_tol=COORDINATE_TOLERANCE,
    ):
        raise CorpusExtractionError(
            "Page-width mismatch on PDF page "
            f"{pdf_page_number}."
        )

    if not isclose(
        float(source_page.rect.height),
        float(page_record.get("page_height")),
        rel_tol=0.0,
        abs_tol=COORDINATE_TOLERANCE,
    ):
        raise CorpusExtractionError(
            "Page-height mismatch on PDF page "
            f"{pdf_page_number}."
        )

    return source_page


def extract_positioned_lines_from_page(
    pdf_document: pymupdf.Document,
    page_record: dict[str, Any],
) -> dict[str, Any]:
    """Extract positioned text from one page's retained regions."""

    source_page = _validate_page_record(
        pdf_document,
        page_record,
    )
    page_index = page_record["pdf_page_index"]
    pdf_page_number = page_record[
        "pdf_page_number"
    ]
    page_width = float(
        page_record["page_width"]
    )
    positioned_lines = []

    for region_index, (
        retained_y0,
        retained_y1,
    ) in enumerate(
        page_record["retained_y_intervals"]
    ):
        retained_y0 = float(retained_y0)
        retained_y1 = float(retained_y1)
        retained_clip = pymupdf.Rect(
            0.0,
            retained_y0,
            source_page.rect.width,
            retained_y1,
        )
        region_dictionary = source_page.get_text(
            "dict",
            clip=retained_clip,
            sort=True,
        )

        for block_order, block in enumerate(
            region_dictionary.get(
                "blocks",
                [],
            )
        ):
            # Image blocks remain outside the text-only baseline.
            if block.get("type") != 0:
                continue

            block_number = block.get(
                "number",
                block_order,
            )

            for line_number, source_line in enumerate(
                block.get("lines", [])
            ):
                spans = source_line.get(
                    "spans",
                    [],
                )
                raw_line_text = "".join(
                    span.get("text", "")
                    for span in spans
                )
                line_text = trim_text_edges(
                    raw_line_text
                )

                if not line_text:
                    continue

                line_bbox = _validate_line_bbox(
                    source_line.get("bbox"),
                    retained_y0,
                    retained_y1,
                    page_width,
                    pdf_page_number,
                )
                direction = _validate_direction(
                    source_line.get(
                        "dir",
                        (1.0, 0.0),
                    ),
                    pdf_page_number,
                )
                writing_mode = source_line.get(
                    "wmode",
                    0,
                )

                if (
                    not isinstance(writing_mode, int)
                    or isinstance(writing_mode, bool)
                ):
                    raise CorpusExtractionError(
                        "Line writing mode must be an "
                        f"integer on PDF page "
                        f"{pdf_page_number}."
                    )

                positioned_lines.append(
                    {
                        "text": line_text,
                        "bbox": line_bbox,
                        "region_index": region_index,
                        "block_number": block_number,
                        "line_number": line_number,
                        "span_count": len(spans),
                        "direction": direction,
                        "writing_mode": writing_mode,
                    }
                )

    page_text = "\n".join(
        line["text"]
        for line in positioned_lines
    )

    printed_page_number = None

    if pdf_page_number >= FIRST_KNOWLEDGE_PDF_PAGE:
        printed_page_number = int(
            expected_printed_page_number(
                pdf_page_number
            )
        )

    return {
        "pdf_page_index": page_index,
        "pdf_page_number": pdf_page_number,
        "printed_page_number": printed_page_number,
        "page_state": page_record["state"],
        "page_width": page_width,
        "page_height": float(
            page_record["page_height"]
        ),
        "excluded_regions": list(
            page_record[
                "excluded_y_intervals"
            ]
        ),
        "retained_regions": list(
            page_record[
                "retained_y_intervals"
            ]
        ),
        "lines": positioned_lines,
        "text": page_text,
        "line_count": len(positioned_lines),
        "character_count": len(page_text),
    }


def is_running_header_line(
    line_record: dict[str, Any],
) -> bool:
    """Identify a line contained in the running-header band."""

    return (
        float(line_record["bbox"][3])
        <= RUNNING_HEADER_BOTTOM_Y
    )


def is_bottom_printed_page_number(
    line_record: dict[str, Any],
    pdf_page_number: int,
    page_width: float,
) -> bool:
    """Identify an expected number at the outer page edge."""

    line_x0, line_y0, line_x1, _ = (
        line_record["bbox"]
    )
    is_in_bottom_zone = (
        line_y0
        >= PRINTED_PAGE_NUMBER_BOTTOM_Y
    )
    is_at_left_edge = (
        line_x1
        <= PRINTED_PAGE_NUMBER_EDGE_WIDTH
    )
    is_at_right_edge = (
        line_x0
        >= page_width
        - PRINTED_PAGE_NUMBER_EDGE_WIDTH
    )
    has_expected_value = (
        line_record["text"]
        == expected_printed_page_number(
            pdf_page_number
        )
    )

    return (
        is_in_bottom_zone
        and (
            is_at_left_edge
            or is_at_right_edge
        )
        and has_expected_value
    )


def classify_structural_line(
    line_record: dict[str, Any],
    pdf_page_number: int,
    page_width: float,
) -> str:
    """Classify one line as retained or structural noise."""

    if is_running_header_line(line_record):
        return "remove_running_header"

    if is_bottom_printed_page_number(
        line_record=line_record,
        pdf_page_number=pdf_page_number,
        page_width=page_width,
    ):
        return "remove_bottom_page_number"

    if (
        line_record["text"]
        == PUBLISHER_COPYRIGHT_TEXT
    ):
        return "remove_publisher_copyright"

    return "retain"


def remove_structural_lines_from_page(
    raw_page_record: dict[str, Any],
) -> dict[str, Any]:
    """Remove validated noise while preserving an audit record."""

    retained_lines = []
    removed_structural_lines = []

    for line_record in raw_page_record["lines"]:
        structural_decision = (
            classify_structural_line(
                line_record=line_record,
                pdf_page_number=raw_page_record[
                    "pdf_page_number"
                ],
                page_width=raw_page_record[
                    "page_width"
                ],
            )
        )

        if structural_decision == "retain":
            retained_lines.append(
                dict(line_record)
            )
        else:
            removed_structural_lines.append(
                {
                    **line_record,
                    "removal_reason": (
                        structural_decision
                    ),
                }
            )

    cleaned_page_text = "\n".join(
        line["text"]
        for line in retained_lines
    )
    cleaned_page_record = {
        key: value
        for key, value in raw_page_record.items()
        if key
        not in {
            "lines",
            "text",
            "line_count",
            "character_count",
        }
    }
    cleaned_page_record.update(
        {
            "raw_line_count": raw_page_record[
                "line_count"
            ],
            "raw_character_count": (
                raw_page_record[
                    "character_count"
                ]
            ),
            "lines": retained_lines,
            "text": cleaned_page_text,
            "line_count": len(retained_lines),
            "character_count": len(
                cleaned_page_text
            ),
            "removed_structural_lines": (
                removed_structural_lines
            ),
            "removed_structural_line_count": len(
                removed_structural_lines
            ),
        }
    )

    return cleaned_page_record


def _validate_cleaned_pages(
    raw_pages: list[dict[str, Any]],
    cleaned_pages: list[dict[str, Any]],
) -> None:
    """Validate the frozen source-specific extraction totals."""

    if (
        len(raw_pages) != EXPECTED_PAGE_COUNT
        or len(cleaned_pages) != EXPECTED_PAGE_COUNT
    ):
        raise CorpusExtractionError(
            "Extraction must contain exactly "
            f"{EXPECTED_PAGE_COUNT} ordered pages."
        )

    for page_index, page in enumerate(
        cleaned_pages
    ):
        if (
            page["pdf_page_index"] != page_index
            or page["pdf_page_number"]
            != page_index + 1
        ):
            raise CorpusExtractionError(
                "Cleaned pages are not in source order."
            )

        reconstructed_text = "\n".join(
            line["text"]
            for line in page["lines"]
        )

        if reconstructed_text != page["text"]:
            raise CorpusExtractionError(
                "Cleaned line reconstruction differs on "
                f"PDF page {page['pdf_page_number']}."
            )

        if (
            page["text"]
            and page["page_state"]
            not in RETRIEVAL_PAGE_STATES
        ):
            raise CorpusExtractionError(
                "Excluded page state contains retrieval "
                f"text on PDF page "
                f"{page['pdf_page_number']}."
            )

    raw_line_count = sum(
        page["line_count"]
        for page in raw_pages
    )
    cleaned_line_count = sum(
        page["line_count"]
        for page in cleaned_pages
    )
    cleaned_character_count = sum(
        page["character_count"]
        for page in cleaned_pages
    )
    pages_with_text = sum(
        bool(page["lines"])
        for page in cleaned_pages
    )
    structural_removal_counts = Counter(
        line["removal_reason"]
        for page in cleaned_pages
        for line in page[
            "removed_structural_lines"
        ]
    )
    emptied_page_numbers = tuple(
        cleaned_page["pdf_page_number"]
        for raw_page, cleaned_page in zip(
            raw_pages,
            cleaned_pages,
        )
        if (
            raw_page["lines"]
            and not cleaned_page["lines"]
        )
    )

    expected_totals = {
        "raw line count": (
            EXPECTED_RAW_LINE_COUNT,
            raw_line_count,
        ),
        "cleaned line count": (
            EXPECTED_CLEANED_LINE_COUNT,
            cleaned_line_count,
        ),
        "cleaned character count": (
            EXPECTED_CLEANED_CHARACTER_COUNT,
            cleaned_character_count,
        ),
        "pages with text": (
            EXPECTED_PAGES_WITH_TEXT,
            pages_with_text,
        ),
        "empty retained-region pages": (
            EXPECTED_EMPTY_RETAINED_REGION_PAGES,
            emptied_page_numbers,
        ),
        "structural removal counts": (
            EXPECTED_STRUCTURAL_REMOVAL_COUNTS,
            structural_removal_counts,
        ),
    }

    for total_name, (
        expected_value,
        actual_value,
    ) in expected_totals.items():
        if actual_value != expected_value:
            raise CorpusExtractionError(
                f"Unexpected {total_name}: expected "
                f"{expected_value!r}, found "
                f"{actual_value!r}."
            )

    for page in cleaned_pages:
        for line in page["lines"]:
            if (
                line["text"]
                == PUBLISHER_COPYRIGHT_TEXT
            ):
                raise CorpusExtractionError(
                    "Publisher copyright text remains "
                    "in the retrieval corpus on PDF "
                    f"page {page['pdf_page_number']}."
                )

            _validate_direction(
                line["direction"],
                page["pdf_page_number"],
            )

            if (
                not isinstance(
                    line["writing_mode"],
                    int,
                )
                or isinstance(
                    line["writing_mode"],
                    bool,
                )
            ):
                raise CorpusExtractionError(
                    "A retained line has an invalid "
                    "writing mode on PDF page "
                    f"{page['pdf_page_number']}."
                )


def extract_validated_corpus_pages(
    pdf_document: pymupdf.Document,
) -> list[dict[str, Any]]:
    """Extract and validate all 770 source-page records."""

    page_geometry = (
        build_validated_knowledge_page_geometry(
            pdf_document
        )
    )
    raw_pages = [
        extract_positioned_lines_from_page(
            pdf_document,
            page_record,
        )
        for page_record in page_geometry
    ]
    cleaned_pages = [
        remove_structural_lines_from_page(
            raw_page_record
        )
        for raw_page_record in raw_pages
    ]

    _validate_cleaned_pages(
        raw_pages,
        cleaned_pages,
    )

    return cleaned_pages
