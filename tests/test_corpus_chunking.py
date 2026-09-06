"""Tests for deterministic, provenance-preserving corpus chunking."""

from __future__ import annotations

import json
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import pytest

from geotech_rag.corpus_chunking import (
    CHUNK_SCHEMA_VERSION,
    CorpusChunkingError,
    build_chunk_records,
    calculate_chunk_audit,
    export_chunked_corpus,
    load_chunking_config,
    load_region_records,
    validate_expected_chunk_audit,
)
from geotech_rag.corpus_export import serialise_json_lines


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ID = "synthetic-source"
SOURCE_SHA256 = "a" * 64


def _line(
    text: str,
    line_index: int,
    y0: float,
) -> dict:
    """Build one valid positioned line."""

    return {
        "bbox": [10.0, y0, 190.0, y0 + 10.0],
        "block_number": 0,
        "direction": [1.0, 0.0],
        "line_index": line_index,
        "line_number": line_index,
        "span_count": 1,
        "text": text,
        "writing_mode": 0,
    }


def _region_record(
    line_texts: list[str],
    *,
    pdf_page_number: int = 1,
) -> dict:
    """Build one valid synthetic retrieval-region record."""

    lines = [
        _line(
            text,
            line_index,
            20.0 + 12.0 * line_index,
        )
        for line_index, text in enumerate(line_texts)
    ]
    text = "\n".join(line_texts)
    pdf_page_index = pdf_page_number - 1

    return {
        "character_count": len(text),
        "coordinate_origin": "top_left",
        "coordinate_unit": "pdf_point",
        "line_count": len(lines),
        "lines": lines,
        "page_height": 300.0,
        "page_state": "fully_retained",
        "page_width": 200.0,
        "pdf_page_index": pdf_page_index,
        "pdf_page_number": pdf_page_number,
        "printed_page_number": pdf_page_number,
        "record_id": (
            f"{SOURCE_ID}:pdf-{pdf_page_number:04d}:region-00"
        ),
        "record_schema_version": "1.0",
        "region_bbox": [0.0, 0.0, 200.0, 300.0],
        "region_index": 0,
        "source_id": SOURCE_ID,
        "source_sha256": SOURCE_SHA256,
        "text": text,
    }


def _splitter_config(
    *,
    chunk_size: int = 10,
    chunk_overlap: int = 4,
) -> dict:
    """Build the frozen splitter configuration."""

    return {
        "add_start_index": True,
        "chunk_overlap": chunk_overlap,
        "chunk_size": chunk_size,
        "class_name": "CharacterTextSplitter",
        "embedded_line_newline_sentinel": "U+E000",
        "is_separator_regex": False,
        "keep_separator": False,
        "length_function": "len",
        "package": "langchain-text-splitters",
        "package_version": version("langchain-text-splitters"),
        "protect_embedded_line_newlines": True,
        "separator": "\n",
        "strip_whitespace": False,
    }


def _chunking_config(
    region_bytes: bytes,
    region_records: list[dict],
    chunks: list[dict],
    splitter_config: dict,
) -> dict:
    """Build one complete valid chunking configuration."""

    return {
        "chunking_config_schema_version": "1.1",
        "expected_source_specific_audit": (
            calculate_chunk_audit(chunks)
        ),
        "input": {
            "character_count": sum(
                record["character_count"]
                for record in region_records
            ),
            "line_count": sum(
                record["line_count"]
                for record in region_records
            ),
            "record_count": len(region_records),
            "record_schema_version": "1.0",
            "relative_path": (
                "data/interim/corpus/regions.jsonl"
            ),
            "sha256": sha256(region_bytes).hexdigest(),
        },
        "output": {
            "chunk_schema_version": "1.0",
            "chunks_relative_path": (
                "data/processed/corpus/chunks.jsonl"
            ),
            "summary_relative_path": (
                "data/processed/audit/chunking-summary.json"
            ),
            "summary_schema_version": "1.0",
        },
        "oversized_chunk_policy": "error",
        "splitter": splitter_config,
    }


def test_chunk_records_follow_frozen_schema_and_provenance() -> None:
    """Chunks retain exact parent character and line ranges."""

    region = _region_record(["aaa", "bbb", "ccc"])
    chunks = build_chunk_records(
        [region],
        _splitter_config(),
    )

    assert [chunk["text"] for chunk in chunks] == [
        "aaa\nbbb",
        "bbb\nccc",
    ]
    assert [
        (
            chunk["parent_character_start"],
            chunk["parent_character_end"],
        )
        for chunk in chunks
    ] == [(0, 7), (4, 11)]
    assert [
        (
            chunk["parent_line_start_index"],
            chunk["parent_line_end_index"],
        )
        for chunk in chunks
    ] == [(0, 2), (1, 3)]
    assert chunks[0]["chunk_id"].endswith("chunk-0000")
    assert chunks[1]["chunk_id"].endswith("chunk-0001")
    assert all(
        chunk["chunk_schema_version"] == CHUNK_SCHEMA_VERSION
        for chunk in chunks
    )
    assert chunks[0]["chunk_bbox"] == [
        10.0,
        20.0,
        190.0,
        42.0,
    ]
    assert chunks[1]["chunk_bbox"] == [
        10.0,
        32.0,
        190.0,
        54.0,
    ]


def test_formula_control_at_chunk_boundary_is_preserved() -> None:
    """Whitespace stripping cannot remove formula controls."""

    formula_line = "\x0b=1"
    region = _region_record(["A" * 198, formula_line])
    chunks = build_chunk_records(
        [region],
        _splitter_config(
            chunk_size=200,
            chunk_overlap=10,
        ),
    )

    assert [chunk["text"] for chunk in chunks] == [
        "A" * 198,
        formula_line,
    ]
    assert chunks[1]["parent_character_start"] == 199


def test_embedded_newline_does_not_split_positioned_line() -> None:
    """A U+000A glyph inside a line is not a separator."""

    region = _region_record(
        [
            "abc\ndef",
            "ghijkl",
        ]
    )
    chunks = build_chunk_records(
        [region],
        _splitter_config(),
    )

    assert [chunk["text"] for chunk in chunks] == [
        "abc\ndef",
        "ghijkl",
    ]
    assert [
        (
            chunk["parent_line_start_index"],
            chunk["parent_line_end_index"],
        )
        for chunk in chunks
    ] == [(0, 1), (1, 2)]
    assert [
        (
            chunk["parent_character_start"],
            chunk["parent_character_end"],
        )
        for chunk in chunks
    ] == [(0, 7), (8, 14)]


def test_oversized_positioned_line_is_rejected() -> None:
    """One indivisible line may not silently exceed chunk size."""

    region = _region_record(["x" * 201])

    with pytest.raises(
        CorpusChunkingError,
        match="oversized chunk",
    ):
        build_chunk_records(
            [region],
            _splitter_config(
                chunk_size=200,
                chunk_overlap=10,
            ),
        )


def test_inconsistent_parent_text_is_rejected() -> None:
    """Chunking cannot continue after line reconstruction fails."""

    region = _region_record(["first", "second"])
    region["text"] = "first\nchanged"
    region["character_count"] = len(region["text"])

    with pytest.raises(
        CorpusChunkingError,
        match="differs from its positioned lines",
    ):
        build_chunk_records(
            [region],
            _splitter_config(),
        )


def test_config_rejects_default_whitespace_stripping(
    tmp_path: Path,
) -> None:
    """The audited preservation setting is mandatory."""

    region_records = [_region_record(["aaa", "bbb"])]
    region_bytes = serialise_json_lines(region_records)
    splitter_config = _splitter_config()
    chunks = build_chunk_records(
        region_records,
        splitter_config,
    )
    config = _chunking_config(
        region_bytes,
        region_records,
        chunks,
        splitter_config,
    )
    config["splitter"]["strip_whitespace"] = True
    config_path = tmp_path / "chunking-config.json"
    config_path.write_text(
        json.dumps(config),
        encoding="utf-8",
    )

    with pytest.raises(
        CorpusChunkingError,
        match="strip_whitespace must equal False",
    ):
        load_chunking_config(config_path)


def test_region_fingerprint_mismatch_is_rejected(
    tmp_path: Path,
) -> None:
    """Chunking cannot consume a changed extraction export."""

    regions_path = tmp_path / "regions.jsonl"
    regions_path.write_bytes(
        serialise_json_lines([_region_record(["text"])])
    )

    with pytest.raises(
        CorpusChunkingError,
        match="fingerprint differs",
    ):
        load_region_records(regions_path, "b" * 64)


def test_export_is_deterministic_and_requires_overwrite(
    tmp_path: Path,
) -> None:
    """Publication is stable and existing outputs are protected."""

    region_records = [_region_record(["aaa", "bbb", "ccc"])]
    region_bytes = serialise_json_lines(region_records)
    splitter_config = _splitter_config()
    chunks = build_chunk_records(
        region_records,
        splitter_config,
    )
    config = _chunking_config(
        region_bytes,
        region_records,
        chunks,
        splitter_config,
    )
    regions_path = (
        tmp_path / "data/interim/corpus/regions.jsonl"
    )
    config_path = tmp_path / "configs/chunking-config.json"
    regions_path.parent.mkdir(parents=True)
    config_path.parent.mkdir(parents=True)
    regions_path.write_bytes(region_bytes)
    config_path.write_text(
        json.dumps(
            config,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    first_result = export_chunked_corpus(
        config_path=config_path,
        project_root=tmp_path,
    )
    first_chunks = first_result.chunks_path.read_bytes()
    first_summary = (
        first_result.chunking_summary_path.read_bytes()
    )

    with pytest.raises(
        CorpusChunkingError,
        match="already exists",
    ):
        export_chunked_corpus(
            config_path=config_path,
            project_root=tmp_path,
        )

    second_result = export_chunked_corpus(
        config_path=config_path,
        project_root=tmp_path,
        overwrite=True,
    )

    assert second_result.chunks_path.read_bytes() == first_chunks
    assert (
        second_result.chunking_summary_path.read_bytes()
        == first_summary
    )
    assert second_result.chunks_sha256 == sha256(
        first_chunks
    ).hexdigest()


def test_repository_regions_reproduce_frozen_chunk_audit() -> None:
    """The production export reproduces every frozen total."""

    config_path = PROJECT_ROOT / "configs/chunking-config.json"
    config = load_chunking_config(config_path)
    regions_path = PROJECT_ROOT / config["input"]["relative_path"]
    region_records = load_region_records(
        regions_path,
        config["input"]["sha256"],
    )
    chunks = build_chunk_records(
        region_records,
        config["splitter"],
    )
    actual_audit = validate_expected_chunk_audit(
        chunks,
        config["expected_source_specific_audit"],
    )

    assert actual_audit == config[
        "expected_source_specific_audit"
    ]
    assert len(region_records) == config["input"]["record_count"]
    assert chunks[0]["parent_record_id"] == region_records[0][
        "record_id"
    ]
    assert chunks[-1]["parent_record_id"] == region_records[-1][
        "record_id"
    ]
