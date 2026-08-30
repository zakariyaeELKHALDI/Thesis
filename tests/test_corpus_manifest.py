"""Tests for corpus-manifest structure and source integrity."""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
from typing import Any
import json

import pymupdf
import pytest

from geotech_rag import (
    ManifestValidationError,
    SourceIntegrityError,
    calculate_file_sha256,
    load_corpus_manifest,
    validate_corpus_manifest,
)


def _write_json(
    target_path: Path,
    content: dict[str, Any],
) -> None:
    """Write deterministic JSON used by one test."""

    target_path.write_text(
        json.dumps(
            content,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _read_json(
    source_path: Path,
) -> dict[str, Any]:
    """Read a mutable test-manifest dictionary."""

    return json.loads(
        source_path.read_text(
            encoding="utf-8"
        )
    )


def _create_synthetic_project(
    temporary_root: Path,
) -> tuple[Path, Path]:
    """
    Create one isolated project with a one-page synthetic PDF.

    No textbook text or other copyrighted source content is used.
    """

    source_directory = (
        temporary_root
        / "data"
        / "raw"
        / "corpus"
    )
    configuration_directory = (
        temporary_root / "configs"
    )

    source_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    configuration_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_path = (
        source_directory / "synthetic-source.pdf"
    )

    # Produce a real PDF so PyMuPDF page validation is exercised.
    synthetic_document = pymupdf.open()

    try:
        synthetic_page = (
            synthetic_document.new_page()
        )
        synthetic_page.insert_text(
            (72.0, 72.0),
            "Synthetic corpus validation page.",
        )
        synthetic_document.save(
            str(source_path)
        )
    finally:
        synthetic_document.close()

    manifest_path = (
        configuration_directory
        / "corpus-manifest.json"
    )

    manifest = {
        "manifest_schema_version": "1.0",
        "corpus": {
            "role": "retrieval_corpus",
            "source_document_count": 1,
            "processed_output_directory": (
                "data/interim/"
            ),
        },
        "sources": [
            {
                "source_id": "synthetic-source",
                "relative_path": (
                    source_path.relative_to(
                        temporary_root
                    ).as_posix()
                ),
                "media_type": "application/pdf",
                "sha256": calculate_file_sha256(
                    source_path
                ),
                "size_bytes": (
                    source_path.stat().st_size
                ),
                "pdf_page_count": 1,
                "repository_policy": {
                    "tracked_by_git": False,
                    "copyrighted_source_text_in_manifest": (
                        False
                    ),
                },
            }
        ],
        "pipeline_boundaries": {
            "only_raw_corpus_input_directory": (
                "data/raw/corpus/"
            ),
            "direct_indexing_of_raw_pdf_allowed": (
                False
            ),
            "excluded_from_retrieval_corpus": [
                "chapter problem regions",
                "answers to selected problems",
                "index",
            ],
            "evaluation_questions_directory": (
                "data/raw/evaluation/questions/"
            ),
            "ground_truth_directory": (
                "data/raw/ground_truth/"
            ),
            "ground_truth_available_during_generation": (
                False
            ),
        },
        "extraction_decision": {
            "production_library": "PyMuPDF",
            "package_version": version(
                "PyMuPDF"
            ),
            "formula_reconstruction_in_baseline": (
                False
            ),
        },
    }

    _write_json(
        manifest_path,
        manifest,
    )

    return (
        manifest_path,
        source_path,
    )


def test_valid_synthetic_manifest_passes(
    tmp_path: Path,
) -> None:
    """A matching manifest and synthetic PDF must pass."""

    manifest_path, source_path = (
        _create_synthetic_project(tmp_path)
    )

    validation_result = (
        validate_corpus_manifest(
            manifest_path
        )
    )

    assert (
        validation_result.source_id
        == "synthetic-source"
    )
    assert (
        validation_result.source_path
        == source_path.resolve()
    )
    assert validation_result.pdf_page_count == 1
    assert (
        validation_result.sha256
        == calculate_file_sha256(source_path)
    )


def test_modified_source_fails_integrity_check(
    tmp_path: Path,
) -> None:
    """Changing source bytes after manifest creation must fail."""

    manifest_path, source_path = (
        _create_synthetic_project(tmp_path)
    )

    # Add bytes after the manifest fingerprint was calculated.
    with source_path.open("ab") as source_file:
        source_file.write(
            b"\nmodified-after-manifest"
        )

    with pytest.raises(
        SourceIntegrityError,
        match="SHA-256",
    ):
        validate_corpus_manifest(
            manifest_path
        )


def test_source_outside_raw_corpus_is_rejected(
    tmp_path: Path,
) -> None:
    """A source path outside data/raw/corpus must fail."""

    manifest_path, _ = (
        _create_synthetic_project(tmp_path)
    )
    manifest = _read_json(manifest_path)

    manifest["sources"][0]["relative_path"] = (
        "data/raw/outside.pdf"
    )

    _write_json(
        manifest_path,
        manifest,
    )

    with pytest.raises(
        ManifestValidationError,
        match="outside the permitted raw corpus",
    ):
        validate_corpus_manifest(
            manifest_path
        )


def test_broadened_raw_input_directory_is_rejected(
    tmp_path: Path,
) -> None:
    """The permitted loader root cannot be broadened."""

    manifest_path, _ = (
        _create_synthetic_project(tmp_path)
    )
    manifest = _read_json(manifest_path)

    manifest[
        "pipeline_boundaries"
    ][
        "only_raw_corpus_input_directory"
    ] = "data/raw/"

    _write_json(
        manifest_path,
        manifest,
    )

    with pytest.raises(
        ManifestValidationError,
        match="raw corpus input directory",
    ):
        validate_corpus_manifest(
            manifest_path
        )


def test_required_answer_exclusion_is_enforced(
    tmp_path: Path,
) -> None:
    """Selected answers must remain excluded from retrieval."""

    manifest_path, _ = (
        _create_synthetic_project(tmp_path)
    )
    manifest = _read_json(manifest_path)

    manifest[
        "pipeline_boundaries"
    ][
        "excluded_from_retrieval_corpus"
    ].remove(
        "answers to selected problems"
    )

    _write_json(
        manifest_path,
        manifest,
    )

    with pytest.raises(
        ManifestValidationError,
        match="answers to selected problems",
    ):
        validate_corpus_manifest(
            manifest_path
        )


def test_direct_raw_pdf_indexing_is_rejected(
    tmp_path: Path,
) -> None:
    """The manifest must explicitly prohibit raw-PDF indexing."""

    manifest_path, _ = (
        _create_synthetic_project(tmp_path)
    )
    manifest = _read_json(manifest_path)

    manifest[
        "pipeline_boundaries"
    ][
        "direct_indexing_of_raw_pdf_allowed"
    ] = True

    _write_json(
        manifest_path,
        manifest,
    )

    with pytest.raises(
        ManifestValidationError,
        match=(
            "direct_indexing_of_raw_pdf_allowed"
        ),
    ):
        validate_corpus_manifest(
            manifest_path
        )


def test_declared_source_count_must_match(
    tmp_path: Path,
) -> None:
    """The declared source count must match the source list."""

    manifest_path, _ = (
        _create_synthetic_project(tmp_path)
    )
    manifest = _read_json(manifest_path)

    manifest["corpus"][
        "source_document_count"
    ] = 2

    _write_json(
        manifest_path,
        manifest,
    )

    with pytest.raises(
        ManifestValidationError,
        match="declared source count",
    ):
        validate_corpus_manifest(
            manifest_path
        )


def test_repository_manifest_matches_local_source() -> None:
    """
    Validate the real corpus when the private PDF is available.

    A clone without the copyrighted source skips only this local
    integration check. The synthetic unit tests remain available.
    """

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

    assert (
        validation_result.sha256
        == manifest["sources"][0]["sha256"]
    )
    assert (
        validation_result.size_bytes
        == manifest["sources"][0]["size_bytes"]
    )
    assert (
        validation_result.pdf_page_count
        == manifest["sources"][0][
            "pdf_page_count"
        ]
    )
