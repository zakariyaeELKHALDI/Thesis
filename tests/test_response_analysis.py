"""Meaningful paired-statistic and private-to-public aggregation checks."""

from __future__ import annotations

import json
import math

import pytest

from geotech_rag import response_analysis as analysis
from test_grouped_response_review import fixture_root, fill_scores, write_json, write_jsonl


def test_paired_binary_and_ordinal_statistics():
    assert analysis.exact_mcnemar([1]*10+[0]*10,[0]*20)["p_value"] == pytest.approx(2/2**10)
    assert analysis.exact_mcnemar([1]*10+[0]*10,[0]*10+[1]*10)["p_value"] == 1.0
    matrix = [[1,0,0] for _ in range(20)]
    assert analysis.cochran_q(matrix)["statistic"] == pytest.approx(40)
    assert analysis.cochran_q(matrix)["p_value"] == pytest.approx(math.exp(-20))
    assert analysis.cochran_q([[0,0,0] for _ in range(20)])["p_value"] == 1
    assert analysis.exact_wilcoxon([2]*20,[0]*20)["p_value"] == pytest.approx(2/2**20)
    assert analysis.exact_wilcoxon([2]*20,[2]*20)["p_value"] == 1
    assert analysis.friedman([[0,1,2] for _ in range(20)])["statistic"] == pytest.approx(40)
    assert analysis.friedman([[2,2,2] for _ in range(20)])["p_value"] == 1
    assert analysis.holm([0.01,0.02,0.2]) == pytest.approx([0.03,0.04,0.2])
    assert analysis.wilson(0,20)[0] == 0
    assert analysis.wilson(20,20)[1] == 1


def test_aggregate_only_output_and_exclusive_write(tmp_path):
    questions = [{"question_id":f"private-q{i}"} for i in range(20)]
    references = [{"question_id":q["question_id"],"problem_type":None if i>=10 else f"private-type-{i//2}",
                   "cognitive_level":None if i>=10 else "Apply","formula_applicable":i<17,
                   "calculation_applicable":True,"unit_applicable":i<18}
                  for i,q in enumerate(questions)]
    joined = []
    for i,question in enumerate(questions):
        for condition in analysis.CONDITIONS:
            empty = condition == "M2" and i<9
            score = {"answer_correctness":int(not empty),"explanation_clarity":0 if empty else 2,
                     "task_adaptability":0 if empty else 2,
                     "unsupported_claims":False,
                     "primary_error_category":"no_visible_answer" if empty else "none"}
            for name,applicable in analysis.APPLICABILITY.items():
                score[name] = (0 if empty else 2) if references[i][applicable] else "not_applicable"
            joined.append({"question_id":question["question_id"],"condition":condition,
                           "empty":empty,"score":score,"reference":references[i]})
    aggregate = analysis.compute_aggregates(questions,references,joined,{"judgments_sha256":"synthetic"})
    assert aggregate["judgments"] == 140
    assert next(row for row in aggregate["condition_summary"] if row["condition"]=="M2")["no_visible_answers"] == 9
    assert len(aggregate["binary_pairwise"]) == 10  # retrieval 1 + temperature 3 + model 6
    assert {row["stratum"] for row in aggregate["adaptability_strata"]["problem_type"]} == {
        "not_recorded_in_reference", "recorded_not_stratified"}
    config = {"tracked_outputs":{"condition_summary_table":"results/tables/generation-evaluation.csv",
              "secondary_score_table":"results/tables/generation-secondary-scores.csv",
              "pairwise_comparison_table":"results/tables/generation-pairwise-comparisons.csv",
              "evaluation_metrics":"results/metrics/generation-evaluation.json"}}
    payload = analysis.output_payload(tmp_path,config,aggregate)
    assert len(payload)==5
    assert all(b"private-q" not in body and b"private-type" not in body
               and b"judgement_note" not in body for body in payload.values())
    analysis.save_public_aggregates(payload)
    assert json.loads((tmp_path/"results/metrics/generation-evaluation.json").read_text())["score_set_status"]=="final_for_analysis"
    with pytest.raises(analysis.review.ReviewError,match="already exists"):
        analysis.save_public_aggregates(payload)


def test_private_lineage_unblinds_only_after_score_set_decision(tmp_path, monkeypatch):
    root = fixture_root(tmp_path, monkeypatch)
    reference_path = root / "data/raw/ground_truth/benchmark-reference-answers.jsonl"
    reference_rows = analysis.review.read_jsonl(reference_path)
    for i, row in enumerate(reference_rows):
        row["problem_type"] = "concept" if i < 10 else None
        row["cognitive_level"] = "Apply" if i < 10 else None
    monkeypatch.setattr(analysis.review, "REFERENCE_SHA256", write_jsonl(reference_path, reference_rows))
    response_path = root / "data/raw/evaluation/generation/v1/responses.jsonl"
    response_rows = analysis.review.read_jsonl(response_path)
    for i, row in enumerate(response_rows):
        row["condition_id"] = analysis.CONDITIONS[i % 7]
    response_hash = write_jsonl(response_path, response_rows)
    config_path = root / "configs/generation-response-evaluation-config.json"
    config = json.loads(config_path.read_text())
    config["lineage"]["private_response_sha256"] = response_hash
    write_json(config_path, config)

    analysis.review.generate(root)
    analysis.review.start(root)
    private = root / "data/raw/evaluation/generation/v1"
    completed = private / "blinded-grouped-response-review-completed.ipynb"
    fill_scores(completed)
    analysis.review.import_scores(root, confirmed=True)
    judgments = private / "evaluation-judgements.jsonl"
    checkpoint = private / "evaluation-judgement-metadata.json"
    status = private / "evaluation-expert-confirmation-status.json"
    decision = private / "evaluation-score-set-decision.json"
    judgments_hash = analysis.review.sha256(judgments)
    monkeypatch.setattr(analysis, "JUDGMENTS_SHA256", judgments_hash)
    monkeypatch.setattr(analysis, "COMPLETED_NOTEBOOK_SHA256", analysis.review.sha256(completed))
    status_hash = write_json(status, {"judgements_sha256":judgments_hash,
                                      "import_metadata_sha256":analysis.review.sha256(checkpoint),
                                      "confirmation_status":"preliminary",
                                      "final_professor_confirmation_received":False})
    monkeypatch.setattr(analysis, "PRELIMINARY_STATUS_SHA256", status_hash)
    recorded, fingerprint = analysis.record_score_set_decision(root)
    assert recorded == decision and fingerprint == analysis.review.sha256(decision)
    with pytest.raises(analysis.review.ReviewError, match="already recorded"):
        analysis.record_score_set_decision(root)
    cfg, questions, references, joined, fingerprints = analysis.load_frozen_analysis(root)
    assert len(joined) == 140
    assert len({(row["question_id"], row["condition"]) for row in joined}) == 140
    assert sum(row["empty"] for row in joined) == 9
    assert len(analysis.compute_aggregates(questions,references,joined,fingerprints)["condition_summary"]) == 7
    assert cfg == config
    decision.unlink()
    with pytest.raises(FileNotFoundError):
        analysis.load_frozen_analysis(root)
