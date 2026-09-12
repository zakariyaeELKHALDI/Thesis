"""Tests for validated, resumable embedding and FAISS index creation."""

from __future__ import annotations

import json
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import faiss
import numpy as np
import pytest
import tiktoken

from geotech_rag.corpus_export import (
    serialise_json_lines,
    serialise_json_object,
)
from geotech_rag.embedding_index import (
    EMBEDDING_INDEX_SUMMARY_SCHEMA_VERSION,
    INDEX_MAPPING_SCHEMA_VERSION,
    EmbeddingIndexError,
    build_faiss_index,
    build_index_mapping_records,
    calculate_embedding_input_audit,
    export_embedding_index,
    load_embedding_index_config,
    load_validated_chunk_records,
    normalise_embedding_matrix,
    validate_raw_embedding_matrix,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ID = "synthetic-source"
SOURCE_SHA256 = "a" * 64


class FakeEmbeddingClient:
    """Return deterministic non-zero vectors without making an API call."""

    def __init__(
        self,
        dimensions: int = 1536,
        fail_on_call: int | None = None,
        returned_dimensions: int | None = None,
    ) -> None:
        """Configure deterministic dimensions and an optional failure call."""

        self.dimensions = dimensions
        self.fail_on_call = fail_on_call
        self.returned_dimensions = (
            dimensions
            if returned_dimensions is None
            else returned_dimensions
        )
        self.calls: list[list[str]] = []

    def embed_documents(
        self,
        texts: list[str],
    ) -> np.ndarray:
        """Build vectors from text fingerprints and record every request."""

        self.calls.append(list(texts))
        if self.fail_on_call == len(self.calls):
            raise RuntimeError("Synthetic embedding request failure.")

        # Use a dense first coordinate and two text-derived coordinates so
        # every vector is non-zero and normalisation has visible work to do.
        matrix = np.zeros(
            (len(texts), self.returned_dimensions),
            dtype=np.float32,
        )
        matrix[:, 0] = 1.0
        if self.returned_dimensions > 1:
            matrix[:, 1] = np.asarray(
                [
                    (sha256(text.encode("utf-8")).digest()[0] + 1)
                    / 256.0
                    for text in texts
                ],
                dtype=np.float32,
            )
        if self.returned_dimensions > 2:
            matrix[:, 2] = np.asarray(
                [
                    (sha256(text.encode("utf-8")).digest()[1] + 1)
                    / 256.0
                    for text in texts
                ],
                dtype=np.float32,
            )
        return matrix


class UnexpectedEmbeddingClient:
    """Fail if a supposedly resumed run attempts a new API request."""

    def __init__(self) -> None:
        """Start with no unexpected calls."""

        self.calls: list[list[str]] = []

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Record and reject any unexpected embedding request."""

        self.calls.append(list(texts))
        raise AssertionError("A resumed batch made an unexpected client call.")


def _chunk_record(
    position: int,
    text: str,
) -> dict:
    """Build one complete synthetic chunk with stable provenance."""

    pdf_page_number = position + 1
    pdf_page_index = position
    parent_record_id = (
        f"{SOURCE_ID}:pdf-{pdf_page_number:04d}:region-00"
    )
    return {
        "character_count": len(text),
        "chunk_bbox": [10.0, 20.0, 190.0, 30.0],
        "chunk_id": f"{parent_record_id}:chunk-0000",
        "chunk_index": 0,
        "chunk_schema_version": "1.0",
        "coordinate_origin": "top_left",
        "coordinate_unit": "pdf_point",
        "line_count": 1,
        "page_height": 300.0,
        "page_state": "fully_retained",
        "page_width": 200.0,
        "parent_character_end": len(text),
        "parent_character_start": 0,
        "parent_line_end_index": 1,
        "parent_line_start_index": 0,
        "parent_record_id": parent_record_id,
        "parent_record_schema_version": "1.0",
        "pdf_page_index": pdf_page_index,
        "pdf_page_number": pdf_page_number,
        "printed_page_number": pdf_page_number,
        "region_bbox": [0.0, 0.0, 200.0, 300.0],
        "region_index": 0,
        "source_id": SOURCE_ID,
        "source_sha256": SOURCE_SHA256,
        "text": text,
    }


def _build_synthetic_project(
    project_root: Path,
    record_count: int,
) -> tuple[Path, list[dict]]:
    """Create exact temporary inputs and a matching production config."""

    # Read the accepted config so synthetic tests retain every frozen setting.
    config = json.loads(
        (
            PROJECT_ROOT / "configs/embedding-index-config.json"
        ).read_text(encoding="utf-8")
    )
    records = [
        _chunk_record(
            position,
            f"synthetic geotechnical chunk {position}",
        )
        for position in range(record_count)
    ]
    chunks_bytes = serialise_json_lines(records)
    chunks_sha256 = sha256(chunks_bytes).hexdigest()

    # Create the minimum chunking summary fields consumed by this stage.
    chunking_summary = {
        "chunk_count": record_count,
        "chunks_sha256": chunks_sha256,
    }
    chunking_summary_bytes = serialise_json_object(
        chunking_summary
    )

    chunks_path = (
        project_root / "data/processed/corpus/chunks.jsonl"
    )
    chunking_summary_path = (
        project_root
        / "data/processed/audit/chunking-summary.json"
    )
    config_path = (
        project_root / "configs/embedding-index-config.json"
    )
    chunks_path.parent.mkdir(parents=True)
    chunking_summary_path.parent.mkdir(parents=True)
    config_path.parent.mkdir(parents=True)
    chunks_path.write_bytes(chunks_bytes)
    chunking_summary_path.write_bytes(chunking_summary_bytes)

    # Recalculate the input identity and every source-specific audit value.
    encoding = tiktoken.get_encoding(
        config["embedding"]["token_encoding"]
    )
    token_counts = [
        len(encoding.encode(record["text"]))
        for record in records
    ]
    config["input"].update(
        {
            "chunking_summary_sha256": sha256(
                chunking_summary_bytes
            ).hexdigest(),
            "record_count": record_count,
            "sha256": chunks_sha256,
            "size_bytes": len(chunks_bytes),
        }
    )
    config["expected_source_specific_audit"] = (
        calculate_embedding_input_audit(
            records,
            token_counts,
            config["request_policy"]["batch_size"],
            config["api_constraints"][
                "maximum_tokens_per_input"
            ],
            config["embedding"]["dimensions"],
        )
    )
    config_path.write_bytes(serialise_json_object(config))
    return config_path, records


def test_repository_input_reproduces_frozen_audit_without_api() -> None:
    """The real chunk export passes every pre-embedding acceptance value."""

    config_path = (
        PROJECT_ROOT / "configs/embedding-index-config.json"
    )
    config = load_embedding_index_config(config_path)
    records, token_counts = load_validated_chunk_records(
        config,
        PROJECT_ROOT,
    )

    assert len(records) == 5113
    assert len(token_counts) == 5113
    assert sum(token_counts) == 303045
    assert records[0]["chunk_id"].endswith("chunk-0000")
    assert records[-1]["chunk_id"].endswith("chunk-0010")


def test_config_requires_unresolved_retrieval_depth(
    tmp_path: Path,
) -> None:
    """Index construction cannot silently select retrieval depth k."""

    config = json.loads(
        (
            PROJECT_ROOT / "configs/embedding-index-config.json"
        ).read_text(encoding="utf-8")
    )
    config["retrieval"]["k"] = 4
    config_path = tmp_path / "embedding-index-config.json"
    config_path.write_bytes(serialise_json_object(config))

    with pytest.raises(
        EmbeddingIndexError,
        match="must remain unresolved",
    ):
        load_embedding_index_config(config_path)


def test_chunk_fingerprint_mismatch_is_rejected(
    tmp_path: Path,
) -> None:
    """Embedding cannot consume a modified chunk export."""

    config_path, _ = _build_synthetic_project(
        tmp_path,
        record_count=2,
    )
    chunks_path = (
        tmp_path / "data/processed/corpus/chunks.jsonl"
    )
    chunks_path.write_bytes(chunks_path.read_bytes() + b"\n")
    config = load_embedding_index_config(config_path)

    with pytest.raises(
        EmbeddingIndexError,
        match="byte size differs",
    ):
        load_validated_chunk_records(config, tmp_path)


def test_vector_normalisation_and_faiss_ranking() -> None:
    """Explicit L2 conversion produces cosine-equivalent FAISS scores."""

    raw_matrix = validate_raw_embedding_matrix(
        [
            [3.0, 4.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        expected_rows=3,
        expected_dimensions=3,
    )
    normalised_matrix = normalise_embedding_matrix(
        raw_matrix,
        norm_tolerance=0.000001,
    )
    index = build_faiss_index(normalised_matrix)
    query = normalise_embedding_matrix(
        np.asarray([[3.0, 4.0, 0.0]], dtype=np.float32),
        norm_tolerance=0.000001,
    )
    scores, positions = index.search(query, 3)

    assert normalised_matrix.dtype == np.float32
    assert normalised_matrix.flags.c_contiguous
    assert np.allclose(
        np.linalg.norm(normalised_matrix, axis=1),
        1.0,
    )
    assert positions[0, 0] == 0
    assert scores[0, 0] == pytest.approx(1.0)


def test_non_finite_and_zero_vectors_are_rejected() -> None:
    """Invalid numerical responses cannot enter FAISS."""

    with pytest.raises(
        EmbeddingIndexError,
        match="non-finite",
    ):
        validate_raw_embedding_matrix(
            [[1.0, float("nan")]],
            expected_rows=1,
            expected_dimensions=2,
        )

    with pytest.raises(
        EmbeddingIndexError,
        match="zero-length",
    ):
        validate_raw_embedding_matrix(
            [[0.0, 0.0]],
            expected_rows=1,
            expected_dimensions=2,
        )


def test_index_mapping_preserves_order_and_provenance() -> None:
    """Each FAISS position maps to the matching ordered chunk."""

    records = [
        _chunk_record(0, "first"),
        _chunk_record(1, "second"),
    ]
    mapping_records = build_index_mapping_records(records)

    assert [
        mapping["index_position"]
        for mapping in mapping_records
    ] == [0, 1]
    assert [
        mapping["chunk_id"]
        for mapping in mapping_records
    ] == [
        record["chunk_id"]
        for record in records
    ]
    assert all(
        mapping["index_mapping_schema_version"]
        == INDEX_MAPPING_SCHEMA_VERSION
        for mapping in mapping_records
    )


def test_export_persists_and_resumes_without_new_client_calls(
    tmp_path: Path,
) -> None:
    """Validated staging supports deterministic local regeneration."""

    config_path, records = _build_synthetic_project(
        tmp_path,
        record_count=257,
    )
    first_client = FakeEmbeddingClient()
    first_result = export_embedding_index(
        config_path=config_path,
        project_root=tmp_path,
        embedding_client=first_client,
    )

    assert [len(call) for call in first_client.calls] == [256, 1]
    assert first_result.chunk_count == len(records)
    assert first_result.dimensions == 1536
    assert first_result.batch_count == 2

    matrix = np.load(
        first_result.embedding_matrix_path,
        allow_pickle=False,
    )
    index = faiss.read_index(str(first_result.faiss_index_path))
    mapping_records = [
        json.loads(line)
        for line in first_result.index_mapping_path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    summary = json.loads(
        first_result.summary_path.read_text(encoding="utf-8")
    )

    assert matrix.shape == (257, 1536)
    assert matrix.dtype == np.float32
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0)
    assert type(index).__name__ == "IndexFlatIP"
    assert index.d == 1536
    assert index.ntotal == 257
    assert len(mapping_records) == 257
    assert summary["summary_schema_version"] == (
        EMBEDDING_INDEX_SUMMARY_SCHEMA_VERSION
    )
    assert summary["input_chunks_sha256"] == (
        sha256(
            (
                tmp_path
                / "data/processed/corpus/chunks.jsonl"
            ).read_bytes()
        ).hexdigest()
    )

    first_output_bytes = {
        "matrix": first_result.embedding_matrix_path.read_bytes(),
        "index": first_result.faiss_index_path.read_bytes(),
        "mapping": first_result.index_mapping_path.read_bytes(),
        "summary": first_result.summary_path.read_bytes(),
    }
    with pytest.raises(
        EmbeddingIndexError,
        match="already exists",
    ):
        export_embedding_index(
            config_path=config_path,
            project_root=tmp_path,
            embedding_client=UnexpectedEmbeddingClient(),
        )

    resumed_client = UnexpectedEmbeddingClient()
    second_result = export_embedding_index(
        config_path=config_path,
        project_root=tmp_path,
        overwrite=True,
        embedding_client=resumed_client,
    )

    assert resumed_client.calls == []
    assert second_result.embedding_matrix_path.read_bytes() == (
        first_output_bytes["matrix"]
    )
    assert second_result.faiss_index_path.read_bytes() == (
        first_output_bytes["index"]
    )
    assert second_result.index_mapping_path.read_bytes() == (
        first_output_bytes["mapping"]
    )
    assert second_result.summary_path.read_bytes() == (
        first_output_bytes["summary"]
    )


def test_interrupted_export_resumes_after_last_complete_batch(
    tmp_path: Path,
) -> None:
    """A failed request leaves no final output and reuses valid staging."""

    config_path, _ = _build_synthetic_project(
        tmp_path,
        record_count=257,
    )
    failing_client = FakeEmbeddingClient(fail_on_call=2)

    with pytest.raises(
        EmbeddingIndexError,
        match="batch 1",
    ):
        export_embedding_index(
            config_path=config_path,
            project_root=tmp_path,
            embedding_client=failing_client,
        )

    staging_directory = (
        tmp_path / "data/processed/embeddings/staging"
    )
    assert (staging_directory / "batch-00000.npy").is_file()
    assert (staging_directory / "batch-00000.json").is_file()
    assert not (staging_directory / "batch-00001.npy").exists()
    assert not (
        tmp_path
        / "data/processed/embeddings/chunk-embeddings.npy"
    ).exists()
    assert not (
        tmp_path
        / "data/processed/audit/embedding-index-summary.json"
    ).exists()

    resumed_client = FakeEmbeddingClient()
    result = export_embedding_index(
        config_path=config_path,
        project_root=tmp_path,
        embedding_client=resumed_client,
    )

    assert [len(call) for call in resumed_client.calls] == [1]
    assert result.chunk_count == 257


def test_incomplete_staging_and_wrong_dimensions_are_rejected(
    tmp_path: Path,
) -> None:
    """Partial staging and malformed provider output stop publication."""

    incomplete_root = tmp_path / "incomplete"
    incomplete_config_path, _ = _build_synthetic_project(
        incomplete_root,
        record_count=1,
    )
    incomplete_staging = (
        incomplete_root / "data/processed/embeddings/staging"
    )
    incomplete_staging.mkdir(parents=True)
    with (incomplete_staging / "batch-00000.npy").open("wb") as handle:
        np.save(
            handle,
            np.ones((1, 1536), dtype=np.float32),
            allow_pickle=False,
        )

    with pytest.raises(
        EmbeddingIndexError,
        match="incomplete",
    ):
        export_embedding_index(
            config_path=incomplete_config_path,
            project_root=incomplete_root,
            embedding_client=UnexpectedEmbeddingClient(),
        )

    wrong_shape_root = tmp_path / "wrong-shape"
    wrong_shape_config_path, _ = _build_synthetic_project(
        wrong_shape_root,
        record_count=1,
    )
    with pytest.raises(
        EmbeddingIndexError,
        match="Unexpected embedding matrix shape",
    ):
        export_embedding_index(
            config_path=wrong_shape_config_path,
            project_root=wrong_shape_root,
            embedding_client=FakeEmbeddingClient(
                returned_dimensions=1535
            ),
        )

    assert not (
        wrong_shape_root
        / "data/processed/audit/embedding-index-summary.json"
    ).exists()
