"""Tests for the frozen production retrieval contract and implementation."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json

import numpy as np
import pytest

from geotech_rag.production_retrieval import (
    ProductionRetrievalConfigError,
    ProductionRetrievalError,
    ProductionRetrievalIntegrityError,
    load_production_retrieval_config,
    load_production_retriever,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/production-retrieval-config.json"
QUERY_MATRIX_PATH = (
    PROJECT_ROOT
    / "data/processed/evaluation/retrieval/v3/query-embeddings.npy"
)
V3_POOL_PATH = (
    PROJECT_ROOT
    / "data/processed/evaluation/retrieval/v3/ranked-candidate-pool.jsonl"
)

RESULT_RECORD_FIELDS = {
    "chunk_id",
    "index_mapping_schema_version",
    "index_position",
    "parent_record_id",
    "pdf_page_index",
    "pdf_page_number",
    "printed_page_number",
    "retrieval_result_schema_version",
    "retrieval_rank",
    "similarity_score",
    "source_id",
    "text",
}


class RecordingEmbeddingClient:
    """Return one controlled vector and record every requested text list."""

    def __init__(self, vector: list[float]) -> None:
        self.vector = vector
        self.calls: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self.vector]


@pytest.fixture(scope="module")
def retriever():
    """Load the actual frozen production resources once for this module."""

    return load_production_retriever(CONFIG_PATH, PROJECT_ROOT)


def _write_config(path: Path, config: dict) -> None:
    path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_repository_production_configuration_loads() -> None:
    """Require the tracked contract to pass its closed schema."""

    config = load_production_retrieval_config(CONFIG_PATH)

    assert config["configuration_id"] == (
        "baseline-production-retrieval-ada002-faiss-flatip-k30-v1"
    )
    assert config["search"]["retrieval_depth_k"] == 30
    assert config["selection_evidence"]["selected_depth_k"] == 30
    assert (
        config["query_embedding"]["benchmark_mode"][
            "api_request_required"
        ]
        is False
    )
    assert (
        config["query_embedding"]["interactive_mode"][
            "api_request_required"
        ]
        is True
    )


def test_configuration_rejects_additional_fields(tmp_path: Path) -> None:
    """Prevent silent expansion of the frozen production contract."""

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["unexpected_field"] = True
    changed_path = tmp_path / "changed-config.json"
    _write_config(changed_path, config)

    with pytest.raises(
        ProductionRetrievalConfigError,
        match="unexpected fields",
    ):
        load_production_retrieval_config(changed_path)


def test_configuration_rejects_depth_disagreement(tmp_path: Path) -> None:
    """Require selection, search and context depth to remain identical."""

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["search"]["retrieval_depth_k"] = 25
    changed_path = tmp_path / "changed-depth.json"
    _write_config(changed_path, config)

    with pytest.raises(
        ProductionRetrievalConfigError,
        match="Retrieval-depth fields are inconsistent",
    ):
        load_production_retrieval_config(changed_path)


def test_configuration_rejects_leakage_permission(tmp_path: Path) -> None:
    """Never allow relevance grades or ground truth into retrieval."""

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["leakage_controls"][
        "relevance_grades_used_by_production_search"
    ] = True
    changed_path = tmp_path / "changed-leakage.json"
    _write_config(changed_path, config)

    with pytest.raises(
        ProductionRetrievalConfigError,
        match="leakage prohibition",
    ):
        load_production_retrieval_config(changed_path)


def test_configuration_rejects_benchmark_api_request(
    tmp_path: Path,
) -> None:
    """Keep the benchmark independent of a second embedding response."""

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["query_embedding"]["benchmark_mode"][
        "api_request_required"
    ] = True
    changed_path = tmp_path / "changed-benchmark.json"
    _write_config(changed_path, config)

    with pytest.raises(
        ProductionRetrievalConfigError,
        match="Unexpected benchmark query policy",
    ):
        load_production_retrieval_config(changed_path)


def test_repository_resources_load_without_search(retriever) -> None:
    """Validate all artifacts without requiring an embedding request."""

    assert retriever.chunk_count == 5113
    assert retriever.retrieval_depth_k == 30
    assert len(retriever.benchmark_question_ids) == 20
    assert retriever.benchmark_question_ids[0] == "2.3b"
    assert retriever.benchmark_question_ids[-1] == "12.16"


def test_one_benchmark_result_preserves_contract(retriever) -> None:
    """Return thirty ordered records with text and full provenance."""

    result = retriever.retrieve_benchmark("2.3b")

    assert result.mode == "benchmark"
    assert result.query_identifier == "2.3b"
    assert result.query_token_count is None
    assert result.api_request_made is False
    assert len(result.records) == 30
    assert [record["retrieval_rank"] for record in result.records] == list(
        range(1, 31)
    )
    assert all(set(record) == RESULT_RECORD_FIELDS for record in result.records)
    assert all(record["text"] for record in result.records)
    assert all(
        result.records[position]["similarity_score"]
        >= result.records[position + 1]["similarity_score"]
        for position in range(29)
    )
    assert all("relevance_grade" not in record for record in result.records)
    assert all("judgement_note" not in record for record in result.records)


def test_benchmark_retrieval_is_deterministic(retriever) -> None:
    """Repeat one frozen-matrix search with byte-equivalent values."""

    first = retriever.retrieve_benchmark("6.6b")
    second = retriever.retrieve_benchmark("6.6b")

    assert first.query_embedding_sha256 == second.query_embedding_sha256
    assert [record["chunk_id"] for record in first.records] == [
        record["chunk_id"] for record in second.records
    ]
    assert [record["similarity_score"] for record in first.records] == [
        record["similarity_score"] for record in second.records
    ]


def test_all_benchmark_results_follow_frozen_order(retriever) -> None:
    """Search all twenty matrix rows in the configured question order."""

    results = retriever.retrieve_all_benchmark()

    assert len(results) == 20
    assert tuple(result.query_identifier for result in results) == (
        retriever.benchmark_question_ids
    )
    assert all(result.mode == "benchmark" for result in results)
    assert all(result.api_request_made is False for result in results)
    assert all(len(result.records) == 30 for result in results)


@pytest.mark.skipif(
    not V3_POOL_PATH.is_file(),
    reason="Private v3 candidate pool is not available.",
)
def test_benchmark_results_match_judged_v3_prefix(retriever) -> None:
    """Connect production top-30 results to the exact judged rankings."""

    candidates_by_question: dict[str, list[dict]] = defaultdict(list)
    for candidate in _read_jsonl(V3_POOL_PATH):
        candidates_by_question[candidate["question_id"]].append(candidate)

    for question_id in retriever.benchmark_question_ids:
        expected = sorted(
            candidates_by_question[question_id],
            key=lambda record: record["rank"],
        )[:30]
        actual = retriever.retrieve_benchmark(question_id).records

        assert len(expected) == len(actual) == 30
        assert [record["chunk_id"] for record in actual] == [
            record["chunk_id"] for record in expected
        ]
        assert [record["index_position"] for record in actual] == [
            record["index_position"] for record in expected
        ]
        assert [record["similarity_score"] for record in actual] == [
            record["similarity_score"] for record in expected
        ]


def test_interactive_mode_uses_one_injected_request(retriever) -> None:
    """Use one text input, normalise its vector and preserve search output."""

    matrix = np.load(QUERY_MATRIX_PATH, allow_pickle=False)
    client = RecordingEmbeddingClient(matrix[0].astype(float).tolist())

    interactive = retriever.retrieve_interactive(
        "A synthetic geotechnical question for interface testing.",
        embedding_client=client,
        query_identifier="interactive-test",
    )
    benchmark = retriever.retrieve_benchmark(
        retriever.benchmark_question_ids[0]
    )

    assert client.calls == [
        ["A synthetic geotechnical question for interface testing."]
    ]
    assert interactive.mode == "interactive"
    assert interactive.query_identifier == "interactive-test"
    assert interactive.query_token_count is not None
    assert interactive.query_token_count > 0
    assert interactive.api_request_made is True
    assert len(interactive.records) == 30
    assert [record["chunk_id"] for record in interactive.records] == [
        record["chunk_id"] for record in benchmark.records
    ]


def test_interactive_result_does_not_retain_query_text(retriever) -> None:
    """Keep the text out of the result while retaining its identifier hash."""

    query_text = "Synthetic private query"
    matrix = np.load(QUERY_MATRIX_PATH, allow_pickle=False)
    client = RecordingEmbeddingClient(matrix[1].astype(float).tolist())

    result = retriever.retrieve_interactive(
        query_text,
        embedding_client=client,
    )
    serialised = json.dumps(result.to_dict(), sort_keys=True)

    assert query_text not in serialised
    assert result.query_identifier == (
        "interactive-" + sha256(query_text.encode("utf-8")).hexdigest()
    )


def test_interactive_mode_rejects_empty_query(retriever) -> None:
    """Reject an empty request before building or calling a client."""

    with pytest.raises(
        ProductionRetrievalError,
        match="non-empty string",
    ):
        retriever.retrieve_interactive("   ")


@pytest.mark.parametrize(
    ("vector", "message"),
    [
        ([0.0] * 1536, "zero length"),
        ([float("nan")] + [0.0] * 1535, "non-finite"),
    ],
)
def test_interactive_mode_rejects_invalid_vector(
    retriever,
    vector: list[float],
    message: str,
) -> None:
    """Stop invalid provider responses before they reach FAISS."""

    client = RecordingEmbeddingClient(vector)

    with pytest.raises(
        ProductionRetrievalIntegrityError,
        match=message,
    ):
        retriever.retrieve_interactive(
            "Synthetic query",
            embedding_client=client,
        )


def test_loader_rejects_changed_artifact_fingerprint(
    tmp_path: Path,
) -> None:
    """Refuse search if a configured production artifact changed."""

    config = deepcopy(
        json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    )
    config["inputs"]["chunk_records"]["sha256"] = "0" * 64
    changed_path = tmp_path / "changed-artifact.json"
    _write_config(changed_path, config)

    with pytest.raises(
        ProductionRetrievalIntegrityError,
        match="chunk records fingerprint differs",
    ):
        load_production_retriever(changed_path, PROJECT_ROOT)


def test_unknown_benchmark_identifier_is_rejected(retriever) -> None:
    """Prevent accidental matrix-row selection by an unknown identifier."""

    with pytest.raises(
        ProductionRetrievalError,
        match="Unknown benchmark question identifier",
    ):
        retriever.retrieve_benchmark("not-a-question")
