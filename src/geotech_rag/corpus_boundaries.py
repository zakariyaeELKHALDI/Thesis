"""Source-specific heading boundaries for the textbook corpus."""

from __future__ import annotations

from collections import Counter, defaultdict
from math import isclose
from typing import Any

import pymupdf

from geotech_rag.corpus_geometry import (
    COORDINATE_TOLERANCE,
    build_page_geometry,
    make_exclusion_interval,
)


TARGET_HEADINGS = (
    "Problems",
    "References",
    "Answers to Selected Problems",
    "Index",
)

CONTENTS_MATCH_FIRST_PAGE = 15
CONTENTS_MATCH_LAST_PAGE = 23
RUNNING_HEADER_Y_LIMIT = 40.0

EXPECTED_PAGE_COUNT = 770
FIRST_KNOWLEDGE_PDF_PAGE = 25
EXPECTED_APPENDIX_START_PAGE = 730
EXPECTED_ANSWER_START_PAGE = 738
EXPECTED_INDEX_START_PAGE = 746
LAST_KNOWLEDGE_PDF_PAGE = (
    EXPECTED_ANSWER_START_PAGE - 1
)

EXPECTED_CLASSIFIED_COUNTS = Counter(
    {
        ("Problems", "contents"): 15,
        ("Problems", "running_header"): 25,
        ("Problems", "body"): 16,
        ("References", "contents"): 17,
        ("References", "running_header"): 13,
        ("References", "body"): 18,
        (
            "Answers to Selected Problems",
            "contents",
        ): 1,
        (
            "Answers to Selected Problems",
            "running_header",
        ): 7,
        (
            "Answers to Selected Problems",
            "body",
        ): 1,
        ("Index", "contents"): 1,
        ("Index", "running_header"): 4,
        ("Index", "body"): 1,
    }
)

EXPECTED_PROBLEM_PAGES = (
    87,
    114,
    144,
    166,
    216,
    261,
    291,
    324,
    370,
    445,
    509,
    569,
    598,
    662,
    697,
    727,
)

EXPECTED_REFERENCE_PAGES = (
    90,
    116,
    145,
    169,
    219,
    265,
    294,
    328,
    376,
    451,
    513,
    573,
    600,
    666,
    700,
    728,
)

EXPECTED_UNPAIRED_REFERENCE_PAGES = (
    37,
    737,
)

EXPECTED_KNOWLEDGE_STATE_COUNTS = Counter(
    {
        "fully_retained": 644,
        "fully_excluded": 37,
        "partially_retained": 32,
        "outside_knowledge_scope": 57,
    }
)


class CorpusBoundaryError(RuntimeError):
    """Raised when source boundaries differ from the audit."""


def document_position(
    match: dict[str, Any],
) -> tuple[int, float, float]:
    """Return a sortable position inside the complete PDF."""

    return (
        int(match["pdf_page_index"]),
        float(match["bbox"][1]),
        float(match["bbox"][0]),
    )


def collect_exact_heading_matches(
    pdf_document: pymupdf.Document,
    target_headings: tuple[str, ...] = TARGET_HEADINGS,
) -> list[dict[str, Any]]:
    """Find exact heading lines and preserve their coordinates."""

    matches = []
    target_set = set(target_headings)

    for page_index in range(
        pdf_document.page_count
    ):
        page = pdf_document.load_page(
            page_index
        )
        lines = defaultdict(list)

        for word in page.get_text(
            "words",
            sort=False,
        ):
            block_number = word[5]
            line_number = word[6]
            lines[
                (
                    block_number,
                    line_number,
                )
            ].append(word)

        for (
            block_number,
            line_number,
        ), line_words in lines.items():
            ordered_words = sorted(
                line_words,
                key=lambda item: item[7],
            )
            line_text = " ".join(
                word[4]
                for word in ordered_words
            )
            normalised_text = " ".join(
                line_text.split()
            )

            if normalised_text not in target_set:
                continue

            coordinates = (
                min(
                    word[0]
                    for word in ordered_words
                ),
                min(
                    word[1]
                    for word in ordered_words
                ),
                max(
                    word[2]
                    for word in ordered_words
                ),
                max(
                    word[3]
                    for word in ordered_words
                ),
            )

            matches.append(
                {
                    "heading": normalised_text,
                    "pdf_page_index": page_index,
                    "pdf_page_number": (
                        page_index + 1
                    ),
                    "bbox": tuple(
                        round(coordinate, 2)
                        for coordinate
                        in coordinates
                    ),
                    "block_number": (
                        block_number
                    ),
                    "line_number": line_number,
                }
            )

    return sorted(
        matches,
        key=document_position,
    )


def classify_heading_match(
    match: dict[str, Any],
) -> str:
    """Classify a heading by its page and vertical position."""

    pdf_page_number = int(
        match["pdf_page_number"]
    )
    y0 = float(match["bbox"][1])

    if (
        CONTENTS_MATCH_FIRST_PAGE
        <= pdf_page_number
        <= CONTENTS_MATCH_LAST_PAGE
    ):
        return "contents"

    if y0 < RUNNING_HEADER_Y_LIMIT:
        return "running_header"

    return "body"


def pair_problem_reference_headings(
    body_matches: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Pair each Problems heading with the next References."""

    problem_starts = sorted(
        (
            match
            for match in body_matches
            if match["heading"] == "Problems"
        ),
        key=document_position,
    )
    available_references = sorted(
        (
            match
            for match in body_matches
            if match["heading"]
            == "References"
        ),
        key=document_position,
    )
    problem_reference_pairs = []

    for problem_start in problem_starts:
        reference_start = next(
            (
                reference
                for reference
                in available_references
                if (
                    document_position(
                        reference
                    )
                    > document_position(
                        problem_start
                    )
                )
            ),
            None,
        )

        if reference_start is None:
            raise CorpusBoundaryError(
                "No later References heading was "
                "found for Problems on PDF page "
                f"{problem_start['pdf_page_number']}."
            )

        problem_reference_pairs.append(
            {
                "problem_start": (
                    problem_start
                ),
                "reference_start": (
                    reference_start
                ),
            }
        )
        available_references.remove(
            reference_start
        )

    return (
        problem_reference_pairs,
        available_references,
    )


def _find_unique_level_one_toc_entry(
    pdf_document: pymupdf.Document,
    title_prefix: str,
) -> list[Any]:
    """Find one top-level PDF bookmark by title prefix."""

    matching_entries = [
        entry
        for entry in pdf_document.get_toc(
            simple=True
        )
        if (
            entry[0] == 1
            and entry[1].strip().startswith(
                title_prefix
            )
        )
    ]

    if len(matching_entries) != 1:
        raise CorpusBoundaryError(
            "Expected one level-one TOC entry "
            f"beginning with {title_prefix!r}, "
            f"found {len(matching_entries)}."
        )

    return matching_entries[0]


def _validate_document_scope(
    pdf_document: pymupdf.Document,
) -> None:
    """Cross-check knowledge boundaries against bookmarks."""

    expected_toc_pages = {
        "Ch 1:": FIRST_KNOWLEDGE_PDF_PAGE,
        "Appendix - A": (
            EXPECTED_APPENDIX_START_PAGE
        ),
        "Answers to Selected Problems": (
            EXPECTED_ANSWER_START_PAGE
        ),
        "Index": EXPECTED_INDEX_START_PAGE,
    }

    for title_prefix, expected_page in (
        expected_toc_pages.items()
    ):
        toc_entry = (
            _find_unique_level_one_toc_entry(
                pdf_document,
                title_prefix,
            )
        )

        if toc_entry[2] != expected_page:
            raise CorpusBoundaryError(
                f"TOC boundary "
                f"{title_prefix!r} moved from "
                f"PDF page {expected_page} to "
                f"{toc_entry[2]}."
            )


def _discover_problem_exclusions(
    pdf_document: pymupdf.Document,
) -> list[dict[str, Any]]:
    """Discover and validate all problem intervals."""

    heading_matches = (
        collect_exact_heading_matches(
            pdf_document
        )
    )
    classified_heading_matches = [
        {
            **match,
            "candidate_type": (
                classify_heading_match(
                    match
                )
            ),
        }
        for match in heading_matches
    ]
    classified_counts = Counter(
        (
            match["heading"],
            match["candidate_type"],
        )
        for match
        in classified_heading_matches
    )

    if (
        classified_counts
        != EXPECTED_CLASSIFIED_COUNTS
    ):
        raise CorpusBoundaryError(
            "The classified heading counts "
            "differ from the audited source."
        )

    body_matches = [
        match
        for match
        in classified_heading_matches
        if match["candidate_type"] == "body"
    ]

    (
        problem_reference_pairs,
        unpaired_references,
    ) = pair_problem_reference_headings(
        body_matches
    )

    answer_matches = [
        match
        for match in body_matches
        if (
            match["heading"]
            == "Answers to Selected Problems"
        )
    ]
    index_matches = [
        match
        for match in body_matches
        if match["heading"] == "Index"
    ]

    actual_problem_pages = tuple(
        pair["problem_start"][
            "pdf_page_number"
        ]
        for pair
        in problem_reference_pairs
    )
    actual_reference_pages = tuple(
        pair["reference_start"][
            "pdf_page_number"
        ]
        for pair
        in problem_reference_pairs
    )
    actual_unpaired_reference_pages = (
        tuple(
            reference["pdf_page_number"]
            for reference
            in unpaired_references
        )
    )

    if (
        actual_problem_pages
        != EXPECTED_PROBLEM_PAGES
    ):
        raise CorpusBoundaryError(
            "The Problems boundary pages "
            "differ from the audited source."
        )

    if (
        actual_reference_pages
        != EXPECTED_REFERENCE_PAGES
    ):
        raise CorpusBoundaryError(
            "The References boundary pages "
            "differ from the audited source."
        )

    if (
        actual_unpaired_reference_pages
        != EXPECTED_UNPAIRED_REFERENCE_PAGES
    ):
        raise CorpusBoundaryError(
            "The unpaired References pages "
            "differ from the audited source."
        )

    if (
        len(answer_matches) != 1
        or answer_matches[0][
            "pdf_page_number"
        ]
        != EXPECTED_ANSWER_START_PAGE
    ):
        raise CorpusBoundaryError(
            "The selected-answer boundary "
            "differs from the audited source."
        )

    if (
        len(index_matches) != 1
        or index_matches[0][
            "pdf_page_number"
        ]
        != EXPECTED_INDEX_START_PAGE
    ):
        raise CorpusBoundaryError(
            "The Index boundary differs "
            "from the audited source."
        )

    return [
        make_exclusion_interval(
            section_id=(
                f"problems_{pair_number:02d}"
            ),
            content_type="problems",
            start_match=pair[
                "problem_start"
            ],
            end_match=pair[
                "reference_start"
            ],
        )
        for pair_number, pair in enumerate(
            problem_reference_pairs,
            start=1,
        )
    ]


def _validate_final_geometry(
    page_geometry: list[dict[str, Any]],
    problem_exclusions: list[
        dict[str, Any]
    ],
) -> None:
    """Validate the final source-specific partition."""

    if len(page_geometry) != EXPECTED_PAGE_COUNT:
        raise CorpusBoundaryError(
            "The page geometry does not "
            f"contain {EXPECTED_PAGE_COUNT} "
            "records."
        )

    page_state_counts = Counter(
        page["state"]
        for page in page_geometry
    )

    if (
        page_state_counts
        != EXPECTED_KNOWLEDGE_STATE_COUNTS
    ):
        raise CorpusBoundaryError(
            "The final page-state counts "
            "differ from the audited source."
        )

    expected_boundary_pages = {
        *EXPECTED_PROBLEM_PAGES,
        *EXPECTED_REFERENCE_PAGES,
    }
    actual_boundary_pages = {
        page["pdf_page_number"]
        for page in page_geometry
        if page["state"]
        == "partially_retained"
    }

    if (
        actual_boundary_pages
        != expected_boundary_pages
    ):
        raise CorpusBoundaryError(
            "The partially retained pages "
            "differ from the audited source."
        )

    page_geometry_by_index = {
        page["pdf_page_index"]: page
        for page in page_geometry
    }

    for page_index, page in enumerate(
        page_geometry
    ):
        if (
            page["pdf_page_index"]
            != page_index
            or page["pdf_page_number"]
            != page_index + 1
        ):
            raise CorpusBoundaryError(
                "The page geometry is not "
                "in source order."
            )

        excluded_height = sum(
            y1 - y0
            for y0, y1 in page[
                "excluded_y_intervals"
            ]
        )
        retained_height = sum(
            y1 - y0
            for y0, y1 in page[
                "retained_y_intervals"
            ]
        )

        if not isclose(
            (
                excluded_height
                + retained_height
            ),
            page["page_height"],
            rel_tol=0.0,
            abs_tol=COORDINATE_TOLERANCE,
        ):
            raise CorpusBoundaryError(
                "Retained and excluded "
                "geometry does not cover "
                f"PDF page "
                f"{page['pdf_page_number']}."
            )

    for interval in problem_exclusions:
        start_page = page_geometry_by_index[
            interval["start_page_index"]
        ]
        end_page = page_geometry_by_index[
            interval["end_page_index"]
        ]

        start_boundary_found = any(
            isclose(
                retained_y1,
                interval["start_y"],
                rel_tol=0.0,
                abs_tol=(
                    COORDINATE_TOLERANCE
                ),
            )
            for _, retained_y1
            in start_page[
                "retained_y_intervals"
            ]
        )
        end_boundary_found = any(
            isclose(
                retained_y0,
                interval["end_y"],
                rel_tol=0.0,
                abs_tol=(
                    COORDINATE_TOLERANCE
                ),
            )
            for retained_y0, _
            in end_page[
                "retained_y_intervals"
            ]
        )

        if not (
            start_boundary_found
            and end_boundary_found
        ):
            raise CorpusBoundaryError(
                "An exclusion is missing "
                "its exact boundary: "
                f"{interval['section_id']}"
            )


def build_validated_knowledge_page_geometry(
    pdf_document: pymupdf.Document,
) -> list[dict[str, Any]]:
    """Build the validated retrieval-corpus geometry."""

    if (
        pdf_document.page_count
        != EXPECTED_PAGE_COUNT
    ):
        raise CorpusBoundaryError(
            f"Expected {EXPECTED_PAGE_COUNT} "
            f"PDF pages, found "
            f"{pdf_document.page_count}."
        )

    _validate_document_scope(pdf_document)
    problem_exclusions = (
        _discover_problem_exclusions(
            pdf_document
        )
    )
    page_geometry = build_page_geometry(
        pdf_document,
        problem_exclusions,
    )

    for page_record in page_geometry:
        pdf_page_number = page_record[
            "pdf_page_number"
        ]

        if not (
            FIRST_KNOWLEDGE_PDF_PAGE
            <= pdf_page_number
            <= LAST_KNOWLEDGE_PDF_PAGE
        ):
            page_record["state"] = (
                "outside_knowledge_scope"
            )
            page_record[
                "excluded_y_intervals"
            ] = [
                (
                    0.0,
                    page_record[
                        "page_height"
                    ],
                )
            ]
            page_record[
                "retained_y_intervals"
            ] = []

    _validate_final_geometry(
        page_geometry,
        problem_exclusions,
    )

    return page_geometry
