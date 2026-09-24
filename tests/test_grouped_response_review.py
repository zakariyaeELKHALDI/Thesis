"""End-to-end checks with synthetic, never real, response text."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geotech_rag import grouped_response_review as review


def write_json(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return review.sha256(path)


def write_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return review.sha256(path)


def fixture_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path
    raw = root / "data/raw"
    questions = [{"question_id": f"q{i:02d}", "query_text": f"Question {i}"}
                 for i in range(20)]
    references = [{"question_id": q["question_id"], "validation_status": "validated",
                   "gradable": True, "formula_applicable": i % 2 == 0,
                   "calculation_applicable": True, "unit_applicable": i % 3 != 0,
                   "reference_solution_markdown": f"Reference $x_{i}$"}
                  for i,q in enumerate(questions)]
    responses = [{"question_id": q["question_id"],
                  "blinded_response_id": f"blind-{i:02d}-{j}",
                  "response_text": "" if i < 9 and j == 0 else f"Visible {i}-{j}"}
                 for i,q in enumerate(questions) for j in range(7)]
    qpath = raw / "evaluation/questions/benchmark-questions.jsonl"
    rpath = raw / "ground_truth/benchmark-reference-answers.jsonl"
    vpath = raw / "ground_truth/benchmark-reference-validation.json"
    response_path = raw / "evaluation/generation/v1/responses.jsonl"
    qhash = write_jsonl(qpath, questions)
    rhash = write_jsonl(rpath, references)
    response_hash = write_jsonl(response_path, responses)
    qconf = root / "configs/evaluation-question-config.json"
    qconf_hash = write_json(qconf, {"questions": 20})
    validation = {"question_file_relative_path": str(qpath.relative_to(root)),
                  "question_file_sha256": qhash, "question_config_relative_path": str(qconf.relative_to(root)),
                  "question_config_sha256": qconf_hash, "references_frozen_for_scoring": True,
                  "reference_answers_ready_for_response_scoring": True}
    vhash = write_json(vpath, validation)
    monkeypatch.setattr(review, "REFERENCE_SHA256", rhash)
    monkeypatch.setattr(review, "FROZEN_METADATA_SHA256", vhash)
    public = Path(__file__).resolve().parents[1] / "configs/generation-response-evaluation-config.json"
    config = json.loads(public.read_text())
    config["lineage"]["private_response_sha256"] = response_hash
    write_json(root / "configs/generation-response-evaluation-config.json", config)
    return root


def fill_scores(path: Path) -> None:
    notebook = json.loads(path.read_text())
    for i in range(20):
        cell = notebook["cells"][3 + i*2]
        values = review.score_cell(review.source(cell))
        for score in values.values():
            empty = score["primary_error_category"] == "no_visible_answer"
            for field in review.SECONDARY_APPLICABILITY:
                if score[field] is None:
                    score[field] = 0 if empty else 2
            score["answer_correctness"] = 0 if empty else 1
            score["explanation_clarity"] = 0 if empty else 2
            score["task_adaptability"] = 0 if empty else 2
            score["unsupported_claims"] = False
            score["primary_error_category"] = "no_visible_answer" if empty else "none"
            score["judgement_confidence"] = "high"
            score["judgement_note"] = "No visible answer." if empty else ""
            score["empty_response_confirmed_by_expert"] = empty
        cell["source"] = "scores = " + repr(values) + "\n"
    path.write_bytes(review.notebook_bytes(notebook))


def test_generate_validate_import_and_refuse_overwrite(tmp_path, monkeypatch):
    root = fixture_root(tmp_path, monkeypatch)
    review.generate(root)
    review.start(root)
    config = json.loads((root / "configs/generation-response-evaluation-config.json").read_text())
    blank, mapping, completed, judgments, metadata, _ = review._paths(root, config)
    first_hash = review.sha256(blank)
    assert all("condition_id" not in review.source(cell) for cell in json.loads(blank.read_text())["cells"])
    assert "blinded_response_id" not in blank.read_text()
    with pytest.raises(review.ReviewError, match="invalid correctness"):
        review.validate_return(root)
    fill_scores(completed)
    rows, fingerprints = review.validate_return(root)
    assert len(rows) == 140 and len({r["blinded_response_id"] for r in rows}) == 140
    assert sum(r["primary_error_category"] == "no_visible_answer" for r in rows) == 9
    with pytest.raises(review.ReviewError, match="expert-confirmed"):
        review.import_scores(root, confirmed=False)
    review.import_scores(root, confirmed=True)
    assert len(review.read_jsonl(judgments)) == 140
    assert json.loads(metadata.read_text())["completed_notebook_sha256"] == fingerprints["completed_notebook_sha256"]
    assert review.sha256(blank) == first_hash
    with pytest.raises(review.ReviewError, match="already exists"):
        review.import_scores(root, confirmed=True)


def test_rejects_mapped_label_change_and_question_text_edit(tmp_path, monkeypatch):
    root = fixture_root(tmp_path, monkeypatch)
    review.generate(root)
    review.start(root)
    config = json.loads((root / "configs/generation-response-evaluation-config.json").read_text())
    _, mapping, completed, *_ = review._paths(root, config)
    fill_scores(completed)
    labels = json.loads(mapping.read_text())
    labels["groups"][0]["labels"]["A"],labels["groups"][0]["labels"]["B"] = (
        labels["groups"][0]["labels"]["B"],labels["groups"][0]["labels"]["A"])
    mapping.write_text(json.dumps(labels))
    with pytest.raises(review.ReviewError, match="mapping changed"):
        review.validate_return(root)
    # Restore the exact mapping from the frozen synthetic inputs.
    config, questions, references, responses, validation = review.load_inputs(root)
    _, original_mapping = review.build_notebook(config, questions, references, responses, validation)
    mapping.write_bytes(review.notebook_bytes(original_mapping))
    notebook = json.loads(completed.read_text())
    notebook["cells"][2]["source"] = "Altered question\n"
    completed.write_bytes(review.notebook_bytes(notebook))
    with pytest.raises(review.ReviewError, match="frozen question/answer/rubric text changed"):
        review.validate_return(root)
