"""Public-only figures from the frozen generation-response aggregates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "geotech-generation-9abcd8f"
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import numpy as np


INPUT_SHA256 = {
    "results/tables/generation-evaluation.csv": "8864bda3284f3addc0c849ae4ed56286a685f5635f0f47a615535fdceb3acd2c",
    "results/tables/generation-secondary-scores.csv": "aaccfb89ca8f1173a959a0b1b73885749fff6d6acd7da1e9e31bc01c2efb013c",
    "results/tables/generation-pairwise-comparisons.csv": "3b3fa163971e68b4d4fa1400aa3a1ab33c9b7a8baa123612968a9895bafa3c5a",
    "results/tables/generation-error-distribution.csv": "dacfbfb0b7c930b113c164507e26e347775af13b6e2e35f9a7754c92cb4dae19",
    "results/metrics/generation-evaluation.json": "134d28d43ae605b2eb06659b2bd4a71a9342a2cdd856fd9ad7cd5f1ed97dc253",
}
JUDGMENTS_SHA256 = "ad861d0d40bc4bfa59a0bda5b6f57abbff0aaac4b772cb9daad7adc4317ca567"
CONDITIONS = ("P0", "P1", "P2", "P3", "M1", "M2", "M3")
ERRORS = ("grounding", "conceptual", "calculation", "deficiency", "no_visible_answer", "other")
FIELDS = ("formula_integration", "calculation_correctness", "unit_correctness", "explanation_clarity", "task_adaptability")
DENOMINATORS = (17, 20, 18, 20, 20)
FOLDER = "results/figures/generation"


def require(condition: bool, explanation: str) -> None:
    if not condition:
        raise ValueError("STOP: " + explanation)


def sha256(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def load_public_tables(root: Path) -> dict:
    """Load exactly the five committed public files, never the private scores."""
    root = root.resolve()
    for relative, fingerprint in INPUT_SHA256.items():
        path = root / relative
        require(path.is_file() and sha256(path.read_bytes()) == fingerprint,
                "aggregate input changed: " + relative)

    def read_csv(relative: str) -> list[dict[str, str]]:
        with (root / relative).open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    summary = read_csv("results/tables/generation-evaluation.csv")
    scores = read_csv("results/tables/generation-secondary-scores.csv")
    errors = read_csv("results/tables/generation-error-distribution.csv")
    metrics = json.loads((root / "results/metrics/generation-evaluation.json").read_text(encoding="utf-8"))
    require(metrics["fingerprints"]["judgments_sha256"] == JUDGMENTS_SHA256
            and metrics["score_set_status"] == "final_for_analysis"
            and metrics["professor_feedback_status"] == "preliminary",
            "aggregate lineage differs")
    require([row["condition"] for row in summary] == list(CONDITIONS), "summary order differs")
    require([(row["field"], row["condition"]) for row in scores]
            == [(field, condition) for field in FIELDS for condition in CONDITIONS],
            "secondary summary order differs")
    require([(row["condition"], row["primary_error_category"]) for row in errors]
            == [(condition, category) for condition in CONDITIONS for category in ERRORS],
            "error summary order differs")
    require(len(metrics["condition_summary"]) == 7 and len(metrics["secondary_summary"]) == 35
            and len(metrics["error_counts"]) == 42, "metrics table counts differ")
    for row in summary:
        correct, visible = int(row["correct"]), int(row["visible_responses"])
        error_total = sum(int(item["count"]) for item in errors if item["condition"] == row["condition"])
        require(int(row["gradable_questions"]) == 20 and 0 <= correct <= visible <= 20
                and error_total == 20 - correct
                and int(row["no_visible_answers"]) == 20 - visible
                and abs(float(row["accuracy"]) - correct / 20) < 1e-12
                and abs(float(row["completion_rate"]) - visible / 20) < 1e-12,
                "outcome denominator differs for " + row["condition"])
    for row in scores:
        denominator = DENOMINATORS[FIELDS.index(row["field"])]
        counts = [int(row[f"score_{value}_count"]) for value in (0, 1, 2)]
        require(int(row["applicable_questions"]) == sum(counts) == denominator
                and abs(float(row["mean_score"]) - sum(i * count for i, count in enumerate(counts)) / denominator) < 1e-12,
                "secondary denominator or mean differs for " + row["field"])
    return {"summary": summary, "scores": scores, "errors": errors}


def accuracy_figure(rows: dict) -> plt.Figure:
    summary = rows["summary"]
    colors = ["#53677b"] * 4 + ["#087f8c"] * 3
    x = np.arange(7)
    accuracy = np.array([float(row["accuracy"]) * 100 for row in summary])
    lower = np.array([float(row["wilson95_lower"]) * 100 for row in summary])
    upper = np.array([float(row["wilson95_upper"]) * 100 for row in summary])
    completion = np.array([float(row["completion_rate"]) * 100 for row in summary])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5), sharey=True, layout="constrained")
    axes[0].bar(x, accuracy, color=colors, width=0.68)
    axes[0].errorbar(x, accuracy, yerr=[accuracy - lower, upper - accuracy], fmt="none",
                     ecolor="#27313c", capsize=3, linewidth=1.2)
    axes[1].bar(x, completion, color=colors, width=0.68)
    axes[1].bar(5, completion[5], color="#b5542a", width=0.68)
    for ax, title, values, column in ((axes[0], "Correct answers", accuracy, "correct"),
                                      (axes[1], "Visible answers", completion, "visible_responses")):
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(x, CONDITIONS)
        ax.set_ylim(0, 112)
        ax.set_yticks(np.arange(0, 101, 20), [f"{value}%" for value in range(0, 101, 20)])
        ax.set_xlabel("Experimental condition")
        ax.grid(axis="y", color="#dadde3", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        for i, row in enumerate(summary):
            value = values[i]
            ax.text(i, value / 2, f'{row[column]}/20', ha="center", va="center",
                    fontsize=8.5, color="white", fontweight="bold")
    axes[0].set_ylabel("Share of the 20 benchmark questions")
    fig.suptitle("Answer correctness and response completion", fontweight="bold", fontsize=14)
    return fig


def error_figure(rows: dict) -> plt.Figure:
    palette = {"correct": "#087f8c", "grounding": "#d99b37", "conceptual": "#b65848",
               "calculation": "#66558d", "deficiency": "#7e8792",
               "no_visible_answer": "#a93245", "other": "#4f9e9b"}
    names = {"correct": "Correct", "grounding": "Grounding", "conceptual": "Conceptual",
             "calculation": "Calculation", "deficiency": "Deficiency",
             "no_visible_answer": "No visible answer", "other": "Other"}
    fig, ax = plt.subplots(figsize=(10.8, 5.7), layout="constrained")
    bottom = np.zeros(7)
    for category in ("correct", *ERRORS):
        values = np.array([
            int(rows["summary"][j]["correct"]) if category == "correct" else
            int(next(record["count"] for record in rows["errors"]
                     if record["condition"] == condition and record["primary_error_category"] == category))
            for j, condition in enumerate(CONDITIONS)
        ])
        ax.bar(np.arange(7), values, bottom=bottom, width=0.7, color=palette[category],
               label=names[category], edgecolor="white", linewidth=0.5)
        for j, count in enumerate(values):
            if count >= 3:
                ax.text(j, bottom[j] + count / 2, str(count), ha="center", va="center",
                        color="white" if category in ("correct", "conceptual", "calculation",
                                                       "deficiency", "no_visible_answer") else "#202a35",
                        fontweight="bold", fontsize=8.5)
        bottom += values
    require(all(value == 20 for value in bottom), "outcome stacks differ from 20")
    ax.set_title("Correctness and primary error categories", fontweight="bold", fontsize=14)
    ax.set_ylabel("Answers out of 20")
    ax.set_xticks(np.arange(7), CONDITIONS)
    ax.set_yticks((0, 5, 10, 15, 20))
    ax.set_ylim(0, 21)
    ax.grid(axis="y", color="#dadde3", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=4, frameon=False)
    return fig


def secondary_figure(rows: dict) -> plt.Figure:
    by_pair = {(row["field"], row["condition"]): row for row in rows["scores"]}
    values = np.array([[float(by_pair[(field, condition)]["mean_score"])
                        for condition in CONDITIONS] for field in FIELDS])
    labels = ("Formula integration", "Calculation correctness", "Unit correctness",
              "Explanation clarity", "Task adaptability")
    fig, ax = plt.subplots(figsize=(10.8, 5), layout="constrained")
    image = ax.imshow(values, cmap="YlGnBu", vmin=0, vmax=2, aspect="auto")
    ax.set_xticks(np.arange(7), CONDITIONS)
    ax.set_yticks(np.arange(5), [f"{name}  (n={denominator})"
                                   for name, denominator in zip(labels, DENOMINATORS, strict=True)])
    ax.set_xticks(np.arange(-0.5, 7, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(5):
        for j in range(7):
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center",
                    color="white" if values[i, j] >= 1.25 else "#17283c", fontsize=9)
    fig.colorbar(image, ax=ax, shrink=0.8, pad=0.02, label="Mean expert score (0–2)")
    ax.set_title("Secondary scores by experimental condition", fontweight="bold", fontsize=14)
    ax.set_xlabel("Experimental condition")
    return fig


def plan_figures(root: Path) -> dict[Path, bytes]:
    """Render three PNGs and three SVGs in memory, with an aggregate-only manifest."""
    root = root.resolve()
    rows = load_public_tables(root)
    payload = {}
    for stem, renderer in (("generation-accuracy-and-completion", accuracy_figure),
                           ("generation-correctness-and-errors", error_figure),
                           ("generation-secondary-scores", secondary_figure)):
        figure = renderer(rows)
        try:
            for extension, settings in (("png", {"dpi": 220, "metadata": {"Software": "matplotlib"}}),
                                        ("svg", {"metadata": {"Date": None}})):
                target = io.BytesIO()
                figure.savefig(target, format=extension, bbox_inches="tight", **settings)
                body = target.getvalue()
                require(body.startswith(b"\x89PNG") if extension == "png"
                        else b"<svg" in body[:1000], "figure render failed")
                payload[root / FOLDER / f"{stem}.{extension}"] = body
        finally:
            plt.close(figure)
    manifest = {"source_commit": "9abcd8f", "input_sha256": INPUT_SHA256,
                "judgments_sha256": JUDGMENTS_SHA256,
                "rendering_library": f"matplotlib {matplotlib.__version__}",
                "figure_sha256": {path.name: sha256(body) for path, body in payload.items()}}
    payload[root / FOLDER / "generation-figure-manifest.json"] = (
        json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode("utf-8")
    return payload


def write_figures(payload: dict[Path, bytes]) -> None:
    """Create every public file exclusively and roll back if any link fails."""
    require(all(not path.exists() for path in payload), "a figure output already exists")
    staged: dict[Path, Path] = {}
    published: list[Path] = []
    try:
        for path, body in payload.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent,
                                             prefix=".figure-stage-", delete=False) as stream:
                staged[path] = Path(stream.name)
                stream.write(body)
                stream.flush()
                os.fsync(stream.fileno())
        for path, temporary in staged.items():
            os.link(temporary, path)
            published.append(path)
    except Exception:
        for path in published:
            path.unlink()
        raise
    finally:
        for path in staged.values():
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Figures from the frozen public evaluation aggregates")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    arguments = parser.parse_args()
    planned = plan_figures(arguments.project_root)
    for path, body in planned.items():
        print(path.relative_to(arguments.project_root.resolve()), len(body), sha256(body))
    if arguments.write:
        write_figures(planned)
        print("Seven public figure files saved; no private scoring inputs accessed.")
    else:
        print("Figure preview only; files unchanged.")
