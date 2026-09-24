"""Figure checks with synthetic aggregate-only inputs."""

from __future__ import annotations

import csv
import io
import json

import pytest

from geotech_rag import generation_result_figures as charts


def csv_body(rows: list[dict]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def synthetic_public_tables(tmp_path, monkeypatch):
    summary, secondary, errors = [], [], []
    correct_counts = [2, 3, 3, 2, 10, 11, 20]
    for condition, correct in zip(charts.CONDITIONS, correct_counts, strict=True):
        visible = 11 if condition == "M2" else 20
        summary.append({"condition": condition, "gradable_questions": 20, "correct": correct,
                        "accuracy": correct / 20, "wilson95_lower": correct / 25,
                        "wilson95_upper": min(1, correct / 20 + 0.1),
                        "visible_responses": visible, "completion_rate": visible / 20,
                        "no_visible_answers": 20 - visible, "unsupported_claims_count": 0})
        for category in charts.ERRORS:
            count = (20 - correct) if category == ("no_visible_answer" if condition == "M2" else "grounding") else 0
            errors.append({"condition": condition, "primary_error_category": category,
                           "count": count, "denominator": 20})
    for field, denominator in zip(charts.FIELDS, charts.DENOMINATORS, strict=True):
        for condition in charts.CONDITIONS:
            secondary.append({"condition": condition, "field": field,
                              "applicable_questions": denominator, "mean_score": 1.0,
                              "median_score": 1, "score_0_count": 0,
                              "score_1_count": denominator, "score_2_count": 0})
    metrics = {"fingerprints": {"judgments_sha256": charts.JUDGMENTS_SHA256},
               "score_set_status": "final_for_analysis", "professor_feedback_status": "preliminary",
               "condition_summary": summary, "secondary_summary": secondary, "error_counts": errors}
    payload = {
        "results/tables/generation-evaluation.csv": csv_body(summary),
        "results/tables/generation-secondary-scores.csv": csv_body(secondary),
        "results/tables/generation-pairwise-comparisons.csv": b"family,left_condition,right_condition\nretrieval,P0,P1\n",
        "results/tables/generation-error-distribution.csv": csv_body(errors),
        "results/metrics/generation-evaluation.json": json.dumps(metrics).encode(),
    }
    for relative, body in payload.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    monkeypatch.setattr(charts, "INPUT_SHA256", {name: charts.sha256(body)
                                                for name, body in payload.items()})
    return tmp_path, payload


def test_render_from_only_public_tables_and_refuse_overwrite(tmp_path, monkeypatch):
    root, payload = synthetic_public_tables(tmp_path, monkeypatch)
    rendered = charts.plan_figures(root)
    assert rendered == charts.plan_figures(root)
    assert len(rendered) == 7
    assert sum(path.suffix == ".png" for path in rendered) == 3
    assert sum(path.suffix == ".svg" for path in rendered) == 3
    assert all(body.startswith(b"\x89PNG") for path, body in rendered.items() if path.suffix == ".png")
    assert all(b"<svg" in body[:1000] for path, body in rendered.items() if path.suffix == ".svg")
    assert all(b"reference_solution_markdown" not in body and b"blinded_response_id" not in body
               for body in rendered.values())
    assert all(not path.exists() for path in rendered)
    charts.write_figures(rendered)
    assert all(path.read_bytes() == body for path, body in rendered.items())
    manifest = json.loads((root / charts.FOLDER / "generation-figure-manifest.json").read_text())
    assert manifest["input_sha256"] == {name: charts.sha256(body) for name, body in payload.items()}
    with pytest.raises(ValueError, match="already exists"):
        charts.write_figures(rendered)


def test_rejects_changed_aggregate_input(tmp_path, monkeypatch):
    root, _ = synthetic_public_tables(tmp_path, monkeypatch)
    summary = root / "results/tables/generation-evaluation.csv"
    summary.write_bytes(summary.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="aggregate input changed"):
        charts.plan_figures(root)
