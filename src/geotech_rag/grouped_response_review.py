"""Generate and import the private, grouped and blinded expert scoring notebook.

The generated notebook is data, never executed by this module. All private
artifacts live under Git-ignored data/raw/evaluation/generation/v1/.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import pprint
import random
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path


REFERENCE_SHA256 = "981650c4f1cdf622bffaec0f84f263105b5ff1b2f6a41ecb273577c9410cb5c8"
FROZEN_METADATA_SHA256 = "dc6fd4b4ba725a2517cd8e8bb5ed6b095584571e050dd349b5583fa225bfd356"
LABELS = tuple("ABCDEFG")
FIELDS = (
    "answer_correctness", "formula_integration", "calculation_correctness",
    "unit_correctness", "explanation_clarity", "task_adaptability",
    "unsupported_claims", "primary_error_category", "judgement_confidence",
    "judgement_note", "empty_response_confirmed_by_expert",
)
SECONDARY_APPLICABILITY = {
    "formula_integration": "formula_applicable",
    "calculation_correctness": "calculation_applicable",
    "unit_correctness": "unit_applicable",
}


class ReviewError(ValueError):
    """A private review input or score fails the frozen contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReviewError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    require(bool(lines) and all(line.strip() for line in lines), f"empty or blank JSONL: {path.name}")
    return [json.loads(line) for line in lines]


def notebook_bytes(book: dict) -> bytes:
    return (json.dumps(book, ensure_ascii=False, indent=1) + "\n").encode("utf-8")


def source(cell: dict) -> str:
    value = cell.get("source")
    require(isinstance(value, (str, list)), "invalid notebook cell source")
    if isinstance(value, list):
        require(all(isinstance(item, str) for item in value), "invalid notebook source lines")
        return "".join(value)
    return value


def md(text: str, cell_id: str) -> dict:
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str, cell_id: str) -> dict:
    return {"cell_type": "code", "id": cell_id, "metadata": {},
            "execution_count": None, "outputs": [], "source": text.splitlines(keepends=True)}


def load_inputs(root: Path) -> tuple[dict, list[dict], list[dict], list[dict], dict]:
    config = json.loads((root / "configs/generation-response-evaluation-config.json").read_text(encoding="utf-8"))
    paths = config["private_paths"]
    validation_path = root / paths["reference_validation_metadata"]
    reference_path = root / paths["reference_answers"]
    response_path = root / config["lineage"]["private_response_relative_path"]
    require(sha256(reference_path) == REFERENCE_SHA256, "approved reference fingerprint differs")
    require(sha256(validation_path) == FROZEN_METADATA_SHA256, "frozen validation fingerprint differs")
    require(sha256(response_path) == config["lineage"]["private_response_sha256"],
            "frozen response fingerprint differs")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    require(validation["references_frozen_for_scoring"] is True
            and validation["reference_answers_ready_for_response_scoring"] is True,
            "reference gate is closed")
    question_path = root / validation["question_file_relative_path"]
    require(sha256(question_path) == validation["question_file_sha256"], "question fingerprint differs")
    question_config_path = root / validation["question_config_relative_path"]
    require(sha256(question_config_path) == validation["question_config_sha256"],
            "question configuration fingerprint differs")
    questions = read_jsonl(question_path)
    references = read_jsonl(reference_path)
    responses = read_jsonl(response_path)
    ids = [q["question_id"] for q in questions]
    require(len(ids) == len(set(ids)) == 20, "20 distinct benchmark questions required")
    require([r["question_id"] for r in references] == ids, "reference question order differs")
    require(all(r["validation_status"] == "validated" and r["gradable"] is True
                for r in references), "unvalidated reference")
    require(len(responses) == 140, "140 generated responses required")
    require(len({r["blinded_response_id"] for r in responses}) == 140,
            "duplicate blinded response identifier")
    require(all(isinstance(r["response_text"], str) and r["question_id"] in ids
                for r in responses), "invalid response text or question")
    require(all(sum(r["question_id"] == q for r in responses) == 7 for q in ids),
            "every question requires seven responses")
    require(sum(not r["response_text"].strip() for r in responses) == 9,
            "expected nine responses without visible answer")
    return config, questions, references, responses, validation


def fence(text: str) -> str:
    """Literal local display prevents untrusted response Markdown loading remote images."""
    maximum = max((len(m.group(0)) for m in re.finditer(r"`+", text)), default=0)
    marker = "`" * max(3, maximum + 1)
    return f"{marker}text\n{text}\n{marker}"


def initial_scores(reference: dict, response_by_label: dict[str, dict]) -> dict:
    scores = {}
    for label, response in response_by_label.items():
        empty = not response["response_text"].strip()
        record = {field: None for field in FIELDS}
        for score, applicability in SECONDARY_APPLICABILITY.items():
            if reference[applicability] is False:
                record[score] = "not_applicable"
        if empty:
            record.update(answer_correctness=0, explanation_clarity=0,
                          task_adaptability=0, unsupported_claims=False,
                          primary_error_category="no_visible_answer")
            for score, applicability in SECONDARY_APPLICABILITY.items():
                record[score] = 0 if reference[applicability] else "not_applicable"
        scores[label] = record
    return scores


def build_notebook(config: dict, questions: list[dict], references: list[dict],
                   responses: list[dict], validation: dict) -> tuple[dict, dict]:
    seed = hashlib.sha256((config["lineage"]["private_response_sha256"] +
                           REFERENCE_SHA256 + validation["question_file_sha256"]).encode()).digest()
    question_order = [q["question_id"] for q in questions]
    cells = [md("# Private grouped response review\n\n"
                "We score 20 benchmark questions in the frozen question order. Each question "
                "shows its approved reference and seven answers labelled A to G. We record "
                "a separate judgment for every answer in the scoring cell below its question. "
                "We edit the scoring cells and save the notebook without running them. "
                "After confirming the scores with Professor Peña Olarte, we keep the original "
                "blank notebook and return the completed copy for validation.\n\n"
                "This private notebook and its answer texts remain outside Git. "
                "The answer labels are randomized within each question; no experimental "
                "configuration or model identity is shown.\n", "review-introduction")]
    scale = config["secondary_judgements"]
    lines = ["## Baseline study criteria\n\n",
             "Tophel et al. (2025, Section 2.1, DOI 10.1007/s10791-025-09580-8) "
             "evaluate four aspects of AI tutor responses:\n\n",
             "- **Accuracy of content:** whether the response gives technically correct information.\n",
             "- **Formula integration capability:** whether relevant geotechnical formulae are selected "
             "and applied correctly.\n",
             "- **Clarity and utility of explanations:** whether a learner can understand and "
             "follow the explanation.\n",
             "- **Adaptability to various problem types:** whether the method and presentation fit "
             "the task, including conceptual and multi-step problems.\n\n",
             "The paper reports error types but does not specify a complete per-answer numerical rubric. "
             "We use the frozen scoring rules below to operationalise these criteria consistently.\n\n",
             "## Frozen scoring guide\n\n",
             "**Answer correctness:** 1 means all materially required results are correct "
             "with an acceptable method; 0 means a material result or requirement is absent "
             "or wrong. Equivalent methods and frozen question tolerances apply.\n\n"]
    for field in ("formula_integration", "calculation_correctness", "unit_correctness",
                  "explanation_clarity", "task_adaptability"):
        lines.append(f"**{field.replace('_',' ').capitalize()}**: " + "; ".join(
            f"{k}: {v}" for k,v in scale[field]["scale"].items()) + "\n\n")
    lines.extend(["**Unsupported claims:** True or False.\n\n",
                  "## Error types for incorrect answers\n\n",
                  "The baseline study reports **grounding** (wrong equation or constraint), "
                  "**conceptual** (missing or incorrect concept), **calculation** "
                  "(algebraic or arithmetic mistake), and **deficiency** (an inability to "
                  "handle a needed part of the task, including visual inputs in the paper). "
                  "Our text-only benchmark applies the frozen operational definitions below. "
                  "We assign one primary category to every incorrect answer.\n\n"])
    for category, description in scale["primary_error_category"]["rules"].items():
        lines.append(f"- **{category}:** {description}\n")
    lines.extend(["\nCorrect answers use **none**; incorrect answers use a category other than **none**. "
                  "A response without visible answer text uses **no_visible_answer** and still "
                  "receives a score.\n\n",
                  "**Confidence:** low, medium, or high. A private note is required for an "
                  "incorrect answer, any score of 1 on a 0–2 scale, the other error "
                  "category, or low confidence. Empty answers have fixed scores under "
                  "the protocol and require explicit expert confirmation.\n"])
    cells.append(md("".join(lines), "review-rubric"))
    mapping = {"schema_version": "1.0", "reference_sha256": REFERENCE_SHA256,
               "validation_sha256": FROZEN_METADATA_SHA256,
               "response_sha256": config["lineage"]["private_response_sha256"],
               "question_sha256": validation["question_file_sha256"], "groups": []}
    for index, (question, reference) in enumerate(zip(questions, references, strict=True), 1):
        question_id = question["question_id"]
        group = [r for r in responses if r["question_id"] == question_id]
        rng = random.Random(hashlib.sha256(seed + question_id.encode()).digest())
        rng.shuffle(group)
        by_label = dict(zip(LABELS, group, strict=True))
        mapping["groups"].append({"question_id": question_id,
                                   "labels": {label: r["blinded_response_id"]
                                              for label,r in by_label.items()}})
        section = [f"# Question {index} of 20\n\n", "## Benchmark question\n\n",
                   question["query_text"], "\n\n## Approved reference solution\n\n",
                   reference["reference_solution_markdown"],
                   "\n\n## Approved grading requirements\n\n"]
        for name in ("required_final_results", "accepted_equivalent_methods", "required_formulae",
                     "accepted_units", "numeric_tolerances"):
            value = reference.get(name, [])
            section.append(f"**{name.replace('_', ' ').capitalize()}**\n\n"
                           f"{fence(json.dumps(value, ensure_ascii=False, indent=2))}\n\n")
        section.append("**Applicability:** formula " + str(reference["formula_applicable"]) +
                       ", calculation " + str(reference["calculation_applicable"]) +
                       ", units " + str(reference["unit_applicable"]) + ".\n")
        for label, response in by_label.items():
            section.append(f"\n## Answer {label}\n\n")
            section.append(fence(response["response_text"]) if response["response_text"].strip()
                           else "**No visible answer was returned.**")
            section.append("\n")
        cells.append(md("".join(section), f"question-{index:02d}"))
        scores = initial_scores(reference, by_label)
        scoring = (f"# Question {index}: we record our scores below and save the notebook.\n"
                   "# We retain each label A–G and every field; None means unscored.\n"
                   "scores = " + pprint.pformat(scores, sort_dicts=False, width=110) + "\n")
        cells.append(code(scoring, f"scores-{index:02d}"))
    notebook = {"cells": cells, "metadata": {"private_grouped_review_schema": "1.0",
                 "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
                "nbformat": 4, "nbformat_minor": 5}
    require([q["question_id"] for q in questions] == question_order, "question order mutated")
    mapping["blank_notebook_sha256"] = hashlib.sha256(notebook_bytes(notebook)).hexdigest()
    return notebook, mapping


def _paths(root: Path, config: dict) -> tuple[Path, Path, Path, Path, Path, Path]:
    raw = config["private_paths"]
    return tuple(root / raw[name] for name in (
        "grouped_review_notebook", "grouped_review_mapping",
        "completed_grouped_review_notebook", "blinded_judgements",
        "judgement_checkpoint_metadata", "judgement_correction_log"))


def create_exclusive(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as target:
        target.write(content)
        target.flush()
        os.fsync(target.fileno())


def generate(root: Path) -> None:
    config, questions, references, responses, validation = load_inputs(root)
    blank, mapping = build_notebook(config, questions, references, responses, validation)
    blank_path, map_path, completed_path, judgements, meta, _ = _paths(root, config)
    require(not any(p.exists() for p in (blank_path, map_path, completed_path, judgements, meta)),
            "private review or judgment artifact already exists; refusing to replace it")
    # Never write a mapping until the frozen notebook has been safely written.
    create_exclusive(blank_path, notebook_bytes(blank))
    try:
        create_exclusive(map_path, notebook_bytes(mapping))
    except Exception:
        blank_path.unlink()
        raise
    print("Blank private notebook:", blank_path.relative_to(root))
    print("Private label mapping:", map_path.relative_to(root))
    print("Question groups: 20; answers: 140; empty answers: 9")
    print("Blank notebook SHA-256:", sha256(blank_path))
    print("Mapping SHA-256:", sha256(map_path))


def start(root: Path) -> None:
    config, questions, references, responses, validation = load_inputs(root)
    blank, mapping_path, completed, judgments, _, _ = _paths(root, config)
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    expected_blank, expected_mapping = build_notebook(config, questions, references, responses, validation)
    require(mapping == expected_mapping, "private answer-label mapping changed")
    require(blank.read_bytes() == notebook_bytes(expected_blank), "blank notebook changed")
    require(not judgments.exists(), "judgments were already imported")
    create_exclusive(completed, blank.read_bytes())
    print("Editable private scoring notebook:", completed.relative_to(root))
    print("The blank original and mapping are unchanged.")


def score_cell(text: str) -> dict:
    try:
        tree = ast.parse(text, mode="exec")
        statements = [node for node in tree.body if not isinstance(node, ast.Expr)
                      or not isinstance(node.value, ast.Constant)
                      or not isinstance(node.value.value, str)]
        require(len(statements) == 1 and isinstance(statements[0], ast.Assign),
                "scoring cell must contain one literal scores assignment")
        assignment = statements[0]
        require(len(assignment.targets) == 1 and isinstance(assignment.targets[0], ast.Name)
                and assignment.targets[0].id == "scores", "unexpected scoring cell code")
        return ast.literal_eval(assignment.value)
    except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError) as exc:
        raise ReviewError(f"invalid scoring cell literal: {exc}") from exc


def valid_score(value: object, allowed: tuple) -> bool:
    return any(type(value) is type(item) and value == item for item in allowed)


def validate_score(score: dict, reference: dict, empty: bool, position: str) -> None:
    require(isinstance(score, dict) and set(score) == set(FIELDS), f"{position}: missing or extra fields")
    require(valid_score(score["answer_correctness"], (0,1)), f"{position}: invalid correctness")
    for name, applicability in SECONDARY_APPLICABILITY.items():
        allowed = (0,1,2) if reference[applicability] else ("not_applicable",)
        require(valid_score(score[name], allowed), f"{position}: invalid {name} applicability or value")
    for name in ("explanation_clarity", "task_adaptability"):
        require(valid_score(score[name], (0,1,2)), f"{position}: invalid {name}")
    require(type(score["unsupported_claims"]) is bool, f"{position}: unsupported_claims must be Boolean")
    require(score["primary_error_category"] in ("none", "grounding", "conceptual", "calculation",
            "deficiency", "no_visible_answer", "other"), f"{position}: invalid error category")
    require(score["judgement_confidence"] in ("low", "medium", "high"), f"{position}: invalid confidence")
    require((score["answer_correctness"] == 1) == (score["primary_error_category"] == "none"),
            f"{position}: error category conflicts with correctness")
    require(type(score["judgement_note"]) is str, f"{position}: note must be a string")
    note_required = (score["answer_correctness"] == 0 or
                     any(score[name] == 1 for name in (*SECONDARY_APPLICABILITY,
                                                         "explanation_clarity", "task_adaptability")) or
                     score["primary_error_category"] == "other" or
                     score["judgement_confidence"] == "low")
    require(not note_required or bool(score["judgement_note"].strip()), f"{position}: private note required")
    require(type(score["empty_response_confirmed_by_expert"]) is bool,
            f"{position}: empty-response confirmation must be True or False")
    require(score["empty_response_confirmed_by_expert"] is empty,
            f"{position}: empty-response confirmation differs from frozen response")
    if empty:
        require(score["answer_correctness"] == 0 and score["explanation_clarity"] == 0
                and score["task_adaptability"] == 0 and not score["unsupported_claims"]
                and score["primary_error_category"] == "no_visible_answer",
                f"{position}: empty-answer policy differs")
        for name, applicability in SECONDARY_APPLICABILITY.items():
            require(score[name] == (0 if reference[applicability] else "not_applicable"),
                    f"{position}: empty-answer {name} differs")
    else:
        require(score["primary_error_category"] != "no_visible_answer",
                f"{position}: visible answer marked empty")


def validate_return(root: Path) -> tuple[list[dict], dict]:
    config, questions, references, responses, validation = load_inputs(root)
    blank_path, mapping_path, completed_path, _, _, _ = _paths(root, config)
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    expected_blank, expected_mapping = build_notebook(config, questions, references, responses, validation)
    require(mapping == expected_mapping, "private answer-label mapping changed")
    require(sha256(blank_path) == mapping["blank_notebook_sha256"], "blank notebook changed")
    require(blank_path.read_bytes() == notebook_bytes(expected_blank), "blank notebook differs from frozen inputs")
    require(mapping["reference_sha256"] == REFERENCE_SHA256 and
            mapping["response_sha256"] == config["lineage"]["private_response_sha256"] and
            mapping["validation_sha256"] == FROZEN_METADATA_SHA256 and
            mapping["question_sha256"] == validation["question_file_sha256"],
            "private mapping input fingerprints differ")
    blank = json.loads(blank_path.read_text(encoding="utf-8"))
    completed = json.loads(completed_path.read_text(encoding="utf-8"))
    require(completed.get("nbformat") == 4 and completed.get("nbformat_minor",0) >= 5,
            "completed notebook format changed")
    require(completed.get("metadata",{}).get("private_grouped_review_schema") == "1.0",
            "private notebook marker missing")
    require(completed.get("metadata",{}).get("kernelspec") == blank["metadata"]["kernelspec"],
            "notebook execution environment marker changed")
    old_cells, new_cells = blank["cells"], completed["cells"]
    require(len(old_cells) == len(new_cells) == 42, "notebook must retain 42 cells")
    require(len(mapping["groups"]) == 20, "mapping must contain 20 groups")
    by_id = {r["blinded_response_id"]: r for r in responses}
    seen = set()
    result = []
    for index, (old, new) in enumerate(zip(old_cells, new_cells, strict=True)):
        require(old["id"] == new.get("id") and old["cell_type"] == new.get("cell_type"),
                f"cell {index+1}: ID or type changed")
        require(new.get("metadata") == old["metadata"], f"cell {index+1}: metadata changed")
        if index < 2 or index % 2 == 0:
            require(source(new) == source(old), f"cell {index+1}: frozen question/answer/rubric text changed")
            continue
        require(new.get("outputs") == [] and new.get("execution_count") is None,
                f"cell {index+1}: clear saved outputs and execution count")
        group_index = (index - 3) // 2
        question = questions[group_index]
        reference = references[group_index]
        mapping_group = mapping["groups"][group_index]
        require(mapping_group["question_id"] == question["question_id"] and
                set(mapping_group["labels"]) == set(LABELS), f"question {group_index+1}: mapping differs")
        scores = score_cell(source(new))
        require(isinstance(scores, dict) and set(scores) == set(LABELS),
                f"question {group_index+1}: exactly A–G required")
        for label in LABELS:
            identifier = mapping_group["labels"][label]
            require(identifier in by_id and identifier not in seen and
                    by_id[identifier]["question_id"] == question["question_id"],
                    f"question {group_index+1}, {label}: identifier differs or repeats")
            seen.add(identifier)
            response = by_id[identifier]
            score = scores[label]
            validate_score(score, reference, not response["response_text"].strip(),
                           f"question {group_index+1}, answer {label}")
            result.append({"blinded_response_id": identifier, "question_id": question["question_id"],
                           **score})
    require(len(seen) == len(result) == 140, "140 distinct judgments required")
    return result, {"blank_notebook_sha256": sha256(blank_path),
                    "completed_notebook_sha256": sha256(completed_path),
                    "mapping_sha256": sha256(mapping_path),
                    "reference_sha256": REFERENCE_SHA256,
                    "response_sha256": config["lineage"]["private_response_sha256"],
                    "question_sha256": validation["question_file_sha256"],
                    "judgment_count": len(result)}


def import_scores(root: Path, confirmed: bool) -> None:
    require(confirmed, "explicit --expert-confirmed required after Professor Peña Olarte confirms scores")
    rows, metadata = validate_return(root)
    config = json.loads((root / "configs/generation-response-evaluation-config.json").read_text(encoding="utf-8"))
    _, _, _, destination, metadata_path, corrections = _paths(root, config)
    require(not destination.exists() and not metadata_path.exists(), "judgment import already exists")
    require(not corrections.exists(), "existing correction log requires separate review")
    payload = ("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True)
                         for row in rows) + "\n").encode("utf-8")
    metadata["judgments_sha256"] = hashlib.sha256(payload).hexdigest()
    metadata["expert_confirmation_reported_by_researcher"] = True
    metadata["imported_on_utc"] = datetime.now(timezone.utc).isoformat()
    # Link a fully fsynced temporary file into place without overwriting an existing judgment set.
    with tempfile.NamedTemporaryFile(mode="wb", dir=destination.parent,
                                     prefix=".judgments-stage-", delete=False) as handle:
        temp = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temp, destination)
    finally:
        temp.unlink(missing_ok=True)
    create_exclusive(metadata_path, (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode())
    print("Imported blinded judgments:", destination.relative_to(root))
    print("Judgments: 140; SHA-256:", metadata["judgments_sha256"])
    print("Private import metadata:", metadata_path.relative_to(root))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("generate", "start", "validate", "import"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--expert-confirmed", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    if args.operation == "generate":
        generate(root)
    elif args.operation == "start":
        start(root)
    elif args.operation == "validate":
        rows, fingerprints = validate_return(root)
        print("Validated question groups: 20; blinded judgments:", len(rows))
        print("Completed notebook SHA-256:", fingerprints["completed_notebook_sha256"])
        print("No judgment file written.")
    else:
        import_scores(root, args.expert_confirmed)


if __name__ == "__main__":
    try:
        main()
    except (ReviewError, FileNotFoundError, FileExistsError, KeyError, json.JSONDecodeError) as exc:
        raise SystemExit(f"STOP: {exc}") from exc
