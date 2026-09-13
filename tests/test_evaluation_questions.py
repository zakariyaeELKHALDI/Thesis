from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json
import sys

import pytest

from geotech_rag.evaluation_questions import (
    EvaluationQuestionConfigError,
    EvaluationQuestionIntegrityError,
    EvaluationQuestionLeakageError,
    EvaluationQuestionSchemaError,
    calculate_file_sha256,
    calculate_question_projection_sha256,
    load_evaluation_question_config,
    load_evaluation_questions,
    main,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_RELATIVE_PATH = Path("configs/evaluation-question-config.json")
QUESTION_RELATIVE_PATH = Path(
    "data/raw/evaluation/questions/benchmark-questions.jsonl"
)


def _repository_material() -> tuple[dict, list[dict]]:
    config_path = PROJECT_ROOT / CONFIG_RELATIVE_PATH
    question_path = PROJECT_ROOT / QUESTION_RELATIVE_PATH
    if not config_path.is_file() or not question_path.is_file():
        pytest.skip("Local ignored evaluation-question material is unavailable.")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in question_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    return config, records


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


def _make_project(
    tmp_path: Path,
    *,
    config_mutator=None,
    record_mutator=None,
    update_file_fingerprint: bool = True,
    update_projection_fingerprint: bool = False,
) -> tuple[Path, Path, Path]:
    source_config, source_records = _repository_material()
    config = deepcopy(source_config)
    records = deepcopy(source_records)

    if record_mutator is not None:
        record_mutator(records)
    if config_mutator is not None:
        config_mutator(config)

    config_path = tmp_path / CONFIG_RELATIVE_PATH
    question_path = tmp_path / QUESTION_RELATIVE_PATH
    _write_jsonl(question_path, records)

    if update_file_fingerprint:
        config["expected_audit"]["questions_sha256"] = calculate_file_sha256(
            question_path
        )
    if update_projection_fingerprint:
        config["expected_audit"][
            "question_projection_sha256"
        ] = calculate_question_projection_sha256(records)
    _write_json(config_path, config)
    return tmp_path, config_path, question_path


def test_repository_questions_reproduce_frozen_contract() -> None:
    dataset = load_evaluation_questions(
        CONFIG_RELATIVE_PATH,
        project_root=PROJECT_ROOT,
    )

    assert len(dataset.records) == 20
    assert len({record["question_id"] for record in dataset.records}) == 20
    assert sum(record["contains_table"] for record in dataset.records) == 6
    assert not any(record["contains_figure"] for record in dataset.records)
    assert not any(
        record["ground_truth_available_during_generation"]
        for record in dataset.records
    )
    assert [
        record["question_id"]
        for record in dataset.records
        if record["source_label_status"] != "exact"
    ] == ["6.6b"]


def test_question_file_fingerprint_mismatch_is_rejected(tmp_path: Path) -> None:
    root, config_path, question_path = _make_project(
        tmp_path,
        update_file_fingerprint=False,
    )
    question_path.write_bytes(question_path.read_bytes() + b"\n")

    with pytest.raises(
        EvaluationQuestionIntegrityError,
        match="file fingerprint differs",
    ):
        load_evaluation_questions(config_path, project_root=root)


def test_query_text_fingerprint_mismatch_is_rejected(tmp_path: Path) -> None:
    def change_query(records: list[dict]) -> None:
        records[0]["query_text"] += " Changed."

    root, config_path, _ = _make_project(
        tmp_path,
        record_mutator=change_query,
    )

    with pytest.raises(
        EvaluationQuestionIntegrityError,
        match="Query-text fingerprint mismatch for 2.3b",
    ):
        load_evaluation_questions(config_path, project_root=root)


def test_question_order_change_is_rejected(tmp_path: Path) -> None:
    def swap_records(records: list[dict]) -> None:
        records[0], records[1] = records[1], records[0]
        records[0]["position"] = 0
        records[1]["position"] = 1

    root, config_path, _ = _make_project(
        tmp_path,
        record_mutator=swap_records,
        update_projection_fingerprint=True,
    )

    with pytest.raises(
        EvaluationQuestionSchemaError,
        match="identifiers or their order differ",
    ):
        load_evaluation_questions(config_path, project_root=root)


def test_noncontiguous_position_is_rejected(tmp_path: Path) -> None:
    def change_position(records: list[dict]) -> None:
        records[4]["position"] = 9

    root, config_path, _ = _make_project(
        tmp_path,
        record_mutator=change_position,
        update_projection_fingerprint=True,
    )

    with pytest.raises(
        EvaluationQuestionSchemaError,
        match="position differs from JSONL order",
    ):
        load_evaluation_questions(config_path, project_root=root)


def test_figure_dependency_is_rejected(tmp_path: Path) -> None:
    def expose_figure(records: list[dict]) -> None:
        records[2]["contains_figure"] = True

    root, config_path, _ = _make_project(
        tmp_path,
        record_mutator=expose_figure,
    )

    with pytest.raises(
        EvaluationQuestionSchemaError,
        match="Figure-dependent question is not allowed",
    ):
        load_evaluation_questions(config_path, project_root=root)


def test_ground_truth_exposure_is_rejected(tmp_path: Path) -> None:
    def expose_ground_truth(records: list[dict]) -> None:
        records[7]["ground_truth_available_during_generation"] = True

    root, config_path, _ = _make_project(
        tmp_path,
        record_mutator=expose_ground_truth,
    )

    with pytest.raises(
        EvaluationQuestionLeakageError,
        match="Ground truth is exposed during generation",
    ):
        load_evaluation_questions(config_path, project_root=root)


def test_source_label_discrepancy_count_is_enforced(tmp_path: Path) -> None:
    def remove_discrepancy(records: list[dict]) -> None:
        records[10]["source_label_status"] = "exact"

    root, config_path, _ = _make_project(
        tmp_path,
        record_mutator=remove_discrepancy,
        update_projection_fingerprint=True,
    )

    with pytest.raises(
        EvaluationQuestionSchemaError,
        match="Source-label discrepancy count differs",
    ):
        load_evaluation_questions(config_path, project_root=root)


def test_unknown_config_field_is_rejected(tmp_path: Path) -> None:
    config, _ = _repository_material()
    config["unexpected"] = True
    config_path = tmp_path / "config.json"
    _write_json(config_path, config)

    with pytest.raises(
        EvaluationQuestionConfigError,
        match="Configuration keys differ",
    ):
        load_evaluation_question_config(config_path)


def test_question_path_traversal_is_rejected(tmp_path: Path) -> None:
    config, _ = _repository_material()
    config["input"]["relative_path"] = "../questions.jsonl"
    config_path = tmp_path / "config.json"
    _write_json(config_path, config)

    with pytest.raises(
        EvaluationQuestionConfigError,
        match="safe project-relative path",
    ):
        load_evaluation_question_config(config_path)


def test_retrieval_depth_must_remain_unresolved(tmp_path: Path) -> None:
    config, _ = _repository_material()
    config["retrieval_evaluation"]["selected_depth_k"] = 4
    config_path = tmp_path / "config.json"
    _write_json(config_path, config)

    with pytest.raises(
        EvaluationQuestionConfigError,
        match="Retrieval depth k must remain unresolved",
    ):
        load_evaluation_question_config(config_path)


def test_cli_reports_read_only_validation(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _repository_material()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluation_questions",
            "--config",
            str(CONFIG_RELATIVE_PATH),
            "--project-root",
            str(PROJECT_ROOT),
        ],
    )

    main()

    output = capsys.readouterr().out
    assert "Evaluation-question validation: PASSED" in output
    assert "Question records: 20" in output
    assert "Ground truth accessed: False" in output
    assert "Retrieval depth k remains unresolved" in output
