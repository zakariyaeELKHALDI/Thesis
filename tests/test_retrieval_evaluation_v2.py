"""Tests for the versioned retrieval-depth expansion."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json

import numpy as np
import pytest

from geotech_rag.retrieval_evaluation import (
    RetrievalEvaluationIntegrityError,
    calculate_file_sha256,
    judgement_id,
    load_retrieval_evaluation_config,
)
from geotech_rag.retrieval_evaluation_v2 import (
    PREFIX_MATCH_FIELDS,
    load_retrieval_evaluation_v2_lineage,
    load_reused_query_matrix,
    migrate_reused_judgements,
    validate_lineage_against_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V2_CONFIG_PATH = PROJECT_ROOT / "configs/retrieval-evaluation-config-v2.json"
V2_LINEAGE_PATH = PROJECT_ROOT / "configs/retrieval-evaluation-v2-lineage.json"


def _candidate(
    configuration_id: str,
    rank: int,
    chunk_id: str,
    similarity_score: float,
) -> dict:
    """Build one minimal candidate containing every frozen prefix field."""

    question_id = "q-01"
    return {
        "candidate_schema_version": "1.0",
        "chunk_character_count": 100 + rank,
        "chunk_id": chunk_id,
        "chunk_text_sha256": sha256(chunk_id.encode("utf-8")).hexdigest(),
        "chunk_token_count": 20 + rank,
        "configuration_id": configuration_id,
        "index_position": rank,
        "judgement_id": judgement_id(configuration_id, question_id, chunk_id),
        "parent_record_id": f"parent-{rank}",
        "pdf_page_index": rank,
        "pdf_page_number": rank + 1,
        "printed_page_number": rank + 1,
        "query_text_sha256": sha256(question_id.encode("utf-8")).hexdigest(),
        "question_id": question_id,
        "question_position": 0,
        "rank": rank,
        "similarity_score": similarity_score,
        "source_id": "synthetic",
    }


def _judgement(candidate: dict, grade: int | None, note: str | None) -> dict:
    """Build one blinded judgement associated with a synthetic candidate."""

    return {
        "chunk_text": f"Evidence for {candidate['chunk_id']}",
        "configuration_id": candidate["configuration_id"],
        "judgement_id": candidate["judgement_id"],
        "judgement_note": note,
        "judgement_schema_version": "1.0",
        "query_text": "Synthetic question",
        "relevance_grade": grade,
    }


def _migration_material() -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Build two reusable ranks and one new v2 rank."""

    source_candidates = [
        _candidate("v1", 1, "chunk-a", 0.9),
        _candidate("v1", 2, "chunk-b", 0.8),
    ]
    target_candidates = [
        _candidate("v2", 1, "chunk-a", 0.9),
        _candidate("v2", 2, "chunk-b", 0.8),
        _candidate("v2", 3, "chunk-c", 0.7),
    ]
    source_judgements = [
        _judgement(source_candidates[0], 2, "Direct evidence."),
        _judgement(source_candidates[1], 0, None),
    ]
    target_judgements = [
        _judgement(candidate, None, None)
        for candidate in target_candidates
    ]
    source_judgements.sort(key=lambda record: record["judgement_id"])
    target_judgements.sort(key=lambda record: record["judgement_id"])
    return (
        source_candidates,
        source_judgements,
        target_candidates,
        target_judgements,
    )


def test_repository_v2_configuration_and_lineage_align() -> None:
    """Require the tracked v2 contracts to describe the same run."""

    config = load_retrieval_evaluation_config(V2_CONFIG_PATH)
    lineage = load_retrieval_evaluation_v2_lineage(V2_LINEAGE_PATH)

    validate_lineage_against_config(
        lineage,
        config,
        V2_CONFIG_PATH,
        PROJECT_ROOT,
    )

    assert config["search"]["candidate_depths"] == [
        1,
        2,
        3,
        4,
        5,
        8,
        10,
        15,
        20,
    ]
    assert config["search"]["judgement_pool_depth"] == 30
    assert lineage["judgement_reuse"]["reused_judgement_count"] == 400
    assert lineage["judgement_reuse"]["new_judgement_count"] == 200


def test_reused_query_matrix_requires_exact_frozen_contract(tmp_path: Path) -> None:
    """Accept only an exact finite normalised float32 matrix."""

    matrix = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
        order="C",
    )
    path = tmp_path / "query-embeddings.npy"
    np.save(path, matrix, allow_pickle=False)
    expected_hash = calculate_file_sha256(path)

    loaded = load_reused_query_matrix(
        path,
        expected_sha256=expected_hash,
        expected_shape=[2, 2],
        expected_dtype="float32",
        norm_tolerance=1e-6,
    )

    assert np.array_equal(loaded, matrix)

    with pytest.raises(
        RetrievalEvaluationIntegrityError,
        match="fingerprint differs",
    ):
        load_reused_query_matrix(
            path,
            expected_sha256="0" * 64,
            expected_shape=[2, 2],
            expected_dtype="float32",
            norm_tolerance=1e-6,
        )


def test_migration_transfers_only_verified_prefix_judgements() -> None:
    """Transfer two existing grades while keeping the new tail incomplete."""

    (
        source_candidates,
        source_judgements,
        target_candidates,
        target_judgements,
    ) = _migration_material()

    migrated = migrate_reused_judgements(
        source_candidates,
        source_judgements,
        target_candidates,
        target_judgements,
        prefix_match_fields=PREFIX_MATCH_FIELDS,
        reused_rank_end=2,
        expected_reused_count=2,
        expected_new_count=1,
    )

    candidates_by_id = {
        record["judgement_id"]: record
        for record in target_candidates
    }
    records_by_rank = {
        candidates_by_id[record["judgement_id"]]["rank"]: record
        for record in migrated
    }
    assert records_by_rank[1]["relevance_grade"] == 2
    assert records_by_rank[1]["judgement_note"] == "Direct evidence."
    assert records_by_rank[2]["relevance_grade"] == 0
    assert records_by_rank[2]["judgement_note"] is None
    assert records_by_rank[3]["relevance_grade"] is None
    assert records_by_rank[3]["judgement_note"] is None


def test_migration_rejects_any_prefix_difference() -> None:
    """Prevent grade reuse when one frozen candidate field changes."""

    (
        source_candidates,
        source_judgements,
        target_candidates,
        target_judgements,
    ) = _migration_material()
    changed_candidates = deepcopy(target_candidates)
    changed_candidates[0]["similarity_score"] = 0.89

    with pytest.raises(
        RetrievalEvaluationIntegrityError,
        match="prefixes differ on similarity_score",
    ):
        migrate_reused_judgements(
            source_candidates,
            source_judgements,
            changed_candidates,
            target_judgements,
            prefix_match_fields=PREFIX_MATCH_FIELDS,
            reused_rank_end=2,
            expected_reused_count=2,
            expected_new_count=1,
        )


def test_v2_output_paths_are_separate_from_v1() -> None:
    """Require every generated v2 output path to differ from v1."""

    v1_path = PROJECT_ROOT / "configs/retrieval-evaluation-config.json"
    v1 = json.loads(v1_path.read_text(encoding="utf-8"))
    v2 = json.loads(V2_CONFIG_PATH.read_text(encoding="utf-8"))

    compared_fields = {
        "blinded_judgement_file_relative_path",
        "metrics_relative_path",
        "metrics_table_relative_path",
        "query_embedding_matrix_relative_path",
        "ranked_candidate_pool_relative_path",
        "summary_relative_path",
    }
    assert all(
        v1["outputs"][field] != v2["outputs"][field]
        for field in compared_fields
    )
