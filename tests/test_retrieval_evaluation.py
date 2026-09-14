"""Tests for blinded retrieval-depth preparation and metric scoring."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json

import faiss
import numpy as np
import pytest

import geotech_rag.retrieval_evaluation as retrieval_module
from geotech_rag.retrieval_evaluation import (
    CANDIDATE_KEYS,
    JUDGEMENT_KEYS,
    RetrievalEvaluationConfigError,
    RetrievalEvaluationError,
    RetrievalEvaluationInputs,
    build_blinded_judgement_records,
    build_ranked_candidate_records,
    calculate_retrieval_metrics,
    judgement_id,
    load_retrieval_evaluation_config,
    load_retrieval_evaluation_inputs,
    prepare_retrieval_evaluation,
    score_retrieval_judgements,
    search_faiss_pool,
    wilson_interval,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_RELATIVE_PATH = Path("configs/retrieval-evaluation-config.json")


class FakeEmbeddingClient:
    """Return one frozen matrix and record ordered input texts."""

    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = matrix
        self.calls: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        return self.matrix.copy()


def _repository_config() -> dict:
    path = PROJECT_ROOT / CONFIG_RELATIVE_PATH
    if not path.is_file():
        pytest.skip("Frozen retrieval-evaluation configuration is unavailable.")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(
                record,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for record in records
        ),
        encoding="utf-8",
        newline="\n",
    )


def _question(position: int) -> dict:
    query_text = f"Synthetic geotechnical question {position}"
    return {
        "position": position,
        "query_text": query_text,
        "query_text_sha256": sha256(query_text.encode("utf-8")).hexdigest(),
        "question_id": f"q-{position:02d}",
    }


def _chunk(position: int) -> dict:
    text = f"Synthetic geotechnical evidence chunk {position}"
    pdf_page_number = position + 1
    parent_record_id = f"synthetic:pdf-{pdf_page_number:04d}:region-00"
    return {
        "character_count": len(text),
        "chunk_id": f"{parent_record_id}:chunk-0000",
        "parent_record_id": parent_record_id,
        "pdf_page_index": position,
        "pdf_page_number": pdf_page_number,
        "printed_page_number": pdf_page_number,
        "source_id": "synthetic",
        "text": text,
    }


def _mapping(chunk: dict, position: int) -> dict:
    return {
        "chunk_id": chunk["chunk_id"],
        "index_mapping_schema_version": "1.0",
        "index_position": position,
        "parent_record_id": chunk["parent_record_id"],
        "pdf_page_index": chunk["pdf_page_index"],
        "pdf_page_number": chunk["pdf_page_number"],
        "printed_page_number": chunk["printed_page_number"],
        "source_id": chunk["source_id"],
    }


def _synthetic_inputs(config: dict) -> tuple[RetrievalEvaluationInputs, np.ndarray]:
    dimensions = config["query_embedding"]["dimensions"]
    question_count = config["inputs"]["evaluation_questions"]["question_count"]
    pool_depth = config["search"]["judgement_pool_depth"]
    chunk_count = pool_depth + question_count

    questions = tuple(_question(position) for position in range(question_count))
    chunks = tuple(_chunk(position) for position in range(chunk_count))
    mappings = tuple(
        _mapping(chunk, position)
        for position, chunk in enumerate(chunks)
    )
    token_counts = tuple(position + 1 for position in range(chunk_count))

    corpus_matrix = np.zeros((chunk_count, dimensions), dtype=np.float32)
    for position in range(chunk_count):
        corpus_matrix[position, position] = 1.0
        corpus_matrix[position, -1] = (position + 1) / (chunk_count + 1)
    faiss.normalize_L2(corpus_matrix)
    index = faiss.IndexFlatIP(dimensions)
    index.add(corpus_matrix)
    query_matrix = np.ascontiguousarray(
        corpus_matrix[:question_count].copy(),
        dtype=np.float32,
    )
    return (
        RetrievalEvaluationInputs(
            config=config,
            embedding_config={},
            question_records=questions,
            chunk_records=chunks,
            chunk_token_counts=token_counts,
            mapping_records=mappings,
            faiss_index=index,
        ),
        query_matrix,
    )


def _metric_config(
    *,
    question_count: int = 2,
    candidate_depths: list[int] | None = None,
    pool_depth: int = 4,
) -> dict:
    return {
        "inputs": {
            "evaluation_questions": {
                "question_count": question_count,
            }
        },
        "search": {
            "candidate_depths": candidate_depths or [1, 2],
            "judgement_pool_depth": pool_depth,
        },
    }


def _metric_candidates(question_count: int = 2, pool_depth: int = 4) -> list[dict]:
    records = []
    for question_position in range(question_count):
        for rank in range(1, pool_depth + 1):
            records.append(
                {
                    "chunk_character_count": 100 + rank,
                    "chunk_token_count": 20 + rank,
                    "judgement_id": f"j-{question_position}-{rank}",
                    "question_id": f"q-{question_position}",
                    "rank": rank,
                }
            )
    return records


def test_repository_contract_loads_with_unresolved_selection() -> None:
    config = load_retrieval_evaluation_config(
        PROJECT_ROOT / CONFIG_RELATIVE_PATH
    )

    assert config["search"]["candidate_depths"] == [1, 2, 3, 4, 5, 8, 10]
    assert config["search"]["reference_depth"] == 4
    assert config["search"]["judgement_pool_depth"] == 20
    assert config["selection_rule"]["selected_depth_k"] is None


def test_repository_inputs_validate_without_embedding_or_search() -> None:
    config_path = PROJECT_ROOT / CONFIG_RELATIVE_PATH
    question_path = PROJECT_ROOT / "data/raw/evaluation/questions/benchmark-questions.jsonl"
    index_path = PROJECT_ROOT / "data/processed/index/chunks.faiss"
    if not question_path.is_file() or not index_path.is_file():
        pytest.skip("Local ignored benchmark or production index is unavailable.")

    inputs = load_retrieval_evaluation_inputs(config_path, PROJECT_ROOT)

    assert len(inputs.question_records) == 20
    assert len(inputs.chunk_records) == 5113
    assert len(inputs.mapping_records) == 5113
    assert inputs.faiss_index.ntotal == 5113


def test_candidate_depths_must_be_strictly_increasing(tmp_path: Path) -> None:
    config = deepcopy(_repository_config())
    config["search"]["candidate_depths"] = [1, 4, 4, 10]
    config["search"]["maximum_candidate_depth"] = 10
    path = tmp_path / "retrieval.json"
    _write_json(path, config)

    with pytest.raises(
        RetrievalEvaluationConfigError,
        match="unique and strictly increasing",
    ):
        load_retrieval_evaluation_config(path)


def test_pool_depth_must_exceed_candidate_range(tmp_path: Path) -> None:
    config = deepcopy(_repository_config())
    config["search"]["judgement_pool_depth"] = 10
    path = tmp_path / "retrieval.json"
    _write_json(path, config)

    with pytest.raises(
        RetrievalEvaluationConfigError,
        match="pool must exceed",
    ):
        load_retrieval_evaluation_config(path)


def test_selected_depth_cannot_be_set_before_scoring(tmp_path: Path) -> None:
    config = deepcopy(_repository_config())
    config["selection_rule"]["selected_depth_k"] = 4
    path = tmp_path / "retrieval.json"
    _write_json(path, config)

    with pytest.raises(
        RetrievalEvaluationConfigError,
        match="must remain unresolved",
    ):
        load_retrieval_evaluation_config(path)


def test_judgement_identifier_is_stable_and_pair_specific() -> None:
    first = judgement_id("configuration", "question", "chunk-a")
    repeated = judgement_id("configuration", "question", "chunk-a")
    second = judgement_id("configuration", "question", "chunk-b")

    assert first == repeated
    assert first != second
    assert len(first) == 64


def test_exact_faiss_pool_search_returns_valid_rankings() -> None:
    vectors = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.2, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    query = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)
    faiss.normalize_L2(vectors)
    faiss.normalize_L2(query)
    index = faiss.IndexFlatIP(3)
    index.add(vectors)

    scores, positions = search_faiss_pool(index, query, 4)

    assert positions.shape == (1, 4)
    assert positions[0, :2].tolist() == [0, 1]
    assert np.all(scores[:, :-1] >= scores[:, 1:])


def test_candidate_join_and_blinding_preserve_only_allowed_context() -> None:
    questions = [_question(0)]
    chunks = [_chunk(position) for position in range(3)]
    mappings = [_mapping(chunk, position) for position, chunk in enumerate(chunks)]
    token_counts = [10, 11, 12]
    scores = np.asarray([[0.9, 0.8, 0.7]], dtype=np.float32)
    positions = np.asarray([[2, 0, 1]], dtype=np.int64)

    candidates = build_ranked_candidate_records(
        "configuration",
        questions,
        chunks,
        token_counts,
        mappings,
        scores,
        positions,
    )
    blinded = build_blinded_judgement_records(
        "configuration",
        questions,
        chunks,
        candidates,
    )

    assert len(candidates) == 3
    assert set(candidates[0]) == CANDIDATE_KEYS
    assert candidates[0]["index_position"] == 2
    assert "text" not in candidates[0]
    assert [record["judgement_id"] for record in blinded] == sorted(
        record["judgement_id"] for record in blinded
    )
    assert all(set(record) == JUDGEMENT_KEYS for record in blinded)
    assert all("rank" not in record for record in blinded)
    assert all("similarity_score" not in record for record in blinded)
    assert all(record["relevance_grade"] is None for record in blinded)


def test_preparation_uses_one_ordered_embedding_call_and_refuses_completed_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = deepcopy(_repository_config())
    config_path = tmp_path / CONFIG_RELATIVE_PATH
    _write_json(config_path, config)
    inputs, query_matrix = _synthetic_inputs(config)
    monkeypatch.setattr(
        retrieval_module,
        "load_retrieval_evaluation_inputs",
        lambda config_path, project_root: inputs,
    )
    client = FakeEmbeddingClient(query_matrix)

    result = prepare_retrieval_evaluation(
        config_path,
        tmp_path,
        embedding_client=client,
    )

    assert len(client.calls) == 1
    assert client.calls[0] == [
        question["query_text"] for question in inputs.question_records
    ]
    assert result.question_count == 20
    assert result.pool_depth == 20
    assert result.candidate_count == 400
    assert result.query_embedding_matrix_path.is_file()
    assert result.ranked_candidate_pool_path.is_file()
    assert result.blinded_judgement_file_path.is_file()
    assert result.summary_path.is_file()

    judgement_records = [
        json.loads(line)
        for line in result.blinded_judgement_file_path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    judgement_records[0]["relevance_grade"] = 1
    judgement_records[0]["judgement_note"] = "Supporting concept."
    _write_jsonl(result.blinded_judgement_file_path, judgement_records)

    with pytest.raises(
        RetrievalEvaluationError,
        match="completed grades",
    ):
        prepare_retrieval_evaluation(
            config_path,
            tmp_path,
            embedding_client=client,
            overwrite=True,
        )
    assert len(client.calls) == 1


def test_metric_selection_uses_frozen_coverage_then_smallest_depth() -> None:
    config = _metric_config()
    candidates = _metric_candidates()
    grades = {
        "j-0-1": 0,
        "j-0-2": 2,
        "j-0-3": 0,
        "j-0-4": 0,
        "j-1-1": 2,
        "j-1-2": 0,
        "j-1-3": 0,
        "j-1-4": 0,
    }

    metrics = calculate_retrieval_metrics(config, candidates, grades)

    assert metrics["candidate_range_adequacy_gate"]["triggered"] is False
    assert metrics["selected_depth_k"] == 2
    assert metrics["selection_status"] == "selected_by_frozen_rule"
    assert metrics["candidate_depth_metrics"][0][
        "direct_evidence_hit_count"
    ] == 1
    assert metrics["candidate_depth_metrics"][1][
        "direct_evidence_hit_count"
    ] == 2


def test_metric_tie_selects_smallest_depth() -> None:
    config = _metric_config()
    candidates = _metric_candidates()
    grades = {
        record["judgement_id"]: (
            2 if record["judgement_id"] == "j-0-1" else 0
        )
        for record in candidates
    }

    metrics = calculate_retrieval_metrics(config, candidates, grades)

    assert metrics["candidate_range_adequacy_gate"]["triggered"] is False
    assert metrics["selected_depth_k"] == 1


def test_diagnostic_tail_triggers_candidate_range_expansion() -> None:
    config = _metric_config()
    candidates = _metric_candidates()
    grades = {
        record["judgement_id"]: (
            2
            if record["judgement_id"] in {"j-0-1", "j-1-3"}
            else 0
        )
        for record in candidates
    }

    metrics = calculate_retrieval_metrics(config, candidates, grades)

    assert metrics["candidate_range_adequacy_gate"]["triggered"] is True
    assert metrics["candidate_range_adequacy_gate"][
        "question_ids_triggering_gate"
    ] == ["q-1"]
    assert metrics["selected_depth_k"] is None
    assert metrics["selection_status"] == "candidate_range_expansion_required"


def test_wilson_interval_is_bounded_and_contains_observed_rate() -> None:
    lower, upper = wilson_interval(15, 20)

    assert 0.0 <= lower <= 0.75
    assert 0.75 <= upper <= 1.0


def test_scoring_rejects_incomplete_judgements_then_writes_text_free_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = deepcopy(_repository_config())
    config_path = tmp_path / CONFIG_RELATIVE_PATH
    _write_json(config_path, config)
    inputs, query_matrix = _synthetic_inputs(config)
    monkeypatch.setattr(
        retrieval_module,
        "load_retrieval_evaluation_inputs",
        lambda config_path, project_root: inputs,
    )
    preparation = prepare_retrieval_evaluation(
        config_path,
        tmp_path,
        embedding_client=FakeEmbeddingClient(query_matrix),
    )

    with pytest.raises(
        retrieval_module.RetrievalEvaluationSchemaError,
        match="must be completed",
    ):
        score_retrieval_judgements(config_path, tmp_path)

    judgements = [
        json.loads(line)
        for line in preparation.blinded_judgement_file_path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    for record in judgements:
        record["relevance_grade"] = 0
    _write_jsonl(preparation.blinded_judgement_file_path, judgements)

    result = score_retrieval_judgements(config_path, tmp_path)

    assert result.completed_judgement_count == 400
    assert result.selected_depth_k == 1
    metrics_text = result.metrics_path.read_text(encoding="utf-8")
    assert "Synthetic geotechnical question" not in metrics_text
    assert "Synthetic geotechnical evidence" not in metrics_text
    assert result.metrics_table_path.is_file()
