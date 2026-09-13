"""Validate the frozen retrieval-evaluation question dataset.

This module is deliberately read-only. It validates the locally stored,
Git-ignored question JSONL against the committed configuration before a later
retrieval or generation stage is allowed to use the questions.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
import argparse
import json
import re


CONFIG_SCHEMA_VERSION = "1.0"
QUESTION_SCHEMA_VERSION = "1.0"
EXPECTED_CONFIGURATION_ID = "tophel-2025-evaluation-questions-v1"
EXPECTED_QUESTION_PATH = (
    "data/raw/evaluation/questions/benchmark-questions.jsonl"
)
EXPECTED_GROUND_TRUTH_PATH = "data/raw/ground_truth/"
EXPECTED_DATASET_ID = "tophel-2025-challenging-20-reconstruction-v1"
EXPECTED_TRANSCRIPTION_MODE = "manual_text_only_standalone_reconstruction"

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
QUESTION_ID_PATTERN = re.compile(r"^[1-9][0-9]*\.[1-9][0-9]*(?:[a-z])?$")
PROBLEM_ID_PATTERN = re.compile(r"^[1-9][0-9]*\.[1-9][0-9]*$")
TOPIC_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")

CONFIG_KEYS = {
    "benchmark",
    "configuration_id",
    "evaluation_question_config_schema_version",
    "expected_audit",
    "input",
    "leakage_controls",
    "retrieval_evaluation",
    "source_textbook",
}
BENCHMARK_KEYS = {
    "base_paper",
    "dataset_id",
    "question_ids",
    "question_text_availability",
    "supporting_paper",
}
BASE_PAPER_KEYS = {
    "citation",
    "doi",
    "reported_full_question_bank_count",
    "reported_selected_question_count",
}
SUPPORTING_PAPER_KEYS = {"citation", "doi"}
EXPECTED_AUDIT_KEYS = {
    "chapter_counts",
    "contains_figure_count",
    "contains_table_count",
    "question_count",
    "question_projection_sha256",
    "questions_sha256",
    "source_label_discrepancy_count",
}
INPUT_KEYS = {"question_schema_version", "relative_path"}
LEAKAGE_CONTROL_KEYS = {
    "add_evaluation_questions_to_corpus_index",
    "ground_truth_available_during_generation",
    "ground_truth_relative_path",
    "question_file_tracked_by_git",
}
RETRIEVAL_EVALUATION_KEYS = {
    "candidate_depths",
    "selected_depth_k",
    "selection_status",
}
SOURCE_TEXTBOOK_KEYS = {
    "citation",
    "expected_pdf_sha256",
    "transcription_method",
}
QUESTION_RECORD_KEYS = {
    "benchmark_dataset_id",
    "chapter",
    "contains_figure",
    "contains_table",
    "ground_truth_available_during_generation",
    "pdf_page_numbers",
    "position",
    "printed_page_numbers",
    "query_text",
    "query_text_sha256",
    "question_id",
    "question_schema_version",
    "source_label_status",
    "source_note",
    "textbook_part",
    "textbook_problem",
    "topic",
    "transcription_mode",
}
SOURCE_LABEL_STATUSES = {
    "exact",
    "benchmark_label_not_printed_in_textbook",
}


class EvaluationQuestionError(RuntimeError):
    """Base error for evaluation-question validation failures."""


class EvaluationQuestionConfigError(EvaluationQuestionError):
    """Raised when the committed configuration is invalid."""


class EvaluationQuestionIntegrityError(EvaluationQuestionError):
    """Raised when a file or projection fingerprint differs."""


class EvaluationQuestionSchemaError(EvaluationQuestionError):
    """Raised when a question record violates the frozen schema."""


class EvaluationQuestionLeakageError(EvaluationQuestionError):
    """Raised when a leakage-control boundary is violated."""


@dataclass(frozen=True)
class EvaluationQuestionDataset:
    """Validated question records and their frozen provenance."""

    configuration_path: Path
    question_path: Path
    records: tuple[dict[str, Any], ...]
    questions_sha256: str
    question_projection_sha256: str


def calculate_file_sha256(path: Path) -> str:
    """Return the SHA-256 fingerprint of a file without modifying it."""

    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    """Serialise a JSON-compatible value for deterministic fingerprints."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def calculate_question_projection_sha256(
    records: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> str:
    """Fingerprint the ordered fields that define benchmark identity."""

    projection = [
        {
            "chapter": record["chapter"],
            "pdf_page_numbers": record["pdf_page_numbers"],
            "position": record["position"],
            "printed_page_numbers": record["printed_page_numbers"],
            "query_text_sha256": record["query_text_sha256"],
            "question_id": record["question_id"],
            "source_label_status": record["source_label_status"],
            "textbook_part": record["textbook_part"],
            "textbook_problem": record["textbook_problem"],
        }
        for record in records
    ]
    return sha256(canonical_json_bytes(projection)).hexdigest()


def _require_mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvaluationQuestionConfigError(f"{context} must be a JSON object.")
    return value


def _require_exact_keys(
    value: dict[str, Any],
    expected_keys: set[str],
    context: str,
    error_type: type[EvaluationQuestionError] = EvaluationQuestionConfigError,
) -> None:
    actual_keys = set(value)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise error_type(
            f"{context} keys differ; missing={missing}, unexpected={unexpected}."
        )


def _require_non_empty_string(
    value: object,
    context: str,
    error_type: type[EvaluationQuestionError] = EvaluationQuestionConfigError,
) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise error_type(f"{context} must be a non-empty, trimmed string.")
    return value


def _require_integer(
    value: object,
    context: str,
    *,
    minimum: int = 0,
    error_type: type[EvaluationQuestionError] = EvaluationQuestionConfigError,
) -> int:
    if type(value) is not int or value < minimum:
        raise error_type(f"{context} must be an integer of at least {minimum}.")
    return value


def _validate_sha256(
    value: object,
    context: str,
    error_type: type[EvaluationQuestionError] = EvaluationQuestionConfigError,
) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise error_type(f"{context} must be a lowercase SHA-256 fingerprint.")
    return value


def _validate_relative_path(value: object, context: str) -> str:
    relative_path = _require_non_empty_string(value, context)
    candidate = Path(relative_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise EvaluationQuestionConfigError(
            f"{context} must be a safe project-relative path."
        )
    if candidate.as_posix() != relative_path.rstrip("/"):
        raise EvaluationQuestionConfigError(
            f"{context} must use normalised forward-slash notation."
        )
    return relative_path


def _resolve_within_project(root: Path, relative_path: str) -> Path:
    resolved_root = root.resolve()
    resolved_path = (resolved_root / relative_path).resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as error:
        raise EvaluationQuestionConfigError(
            "The evaluation-question path escapes the project root."
        ) from error
    return resolved_path


def _validate_positive_page_list(
    value: object,
    context: str,
) -> list[int]:
    if not isinstance(value, list) or not value:
        raise EvaluationQuestionSchemaError(
            f"{context} must be a non-empty integer list."
        )
    pages = [
        _require_integer(
            page,
            f"{context}[{position}]",
            minimum=1,
            error_type=EvaluationQuestionSchemaError,
        )
        for position, page in enumerate(value)
    ]
    if pages != list(range(pages[0], pages[-1] + 1)):
        raise EvaluationQuestionSchemaError(
            f"{context} must contain a contiguous ascending page span."
        )
    return pages


def load_evaluation_question_config(path: Path) -> dict[str, Any]:
    """Load and strictly validate the committed question configuration."""

    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise EvaluationQuestionConfigError(
            f"Unable to read evaluation-question config: {path}"
        ) from error

    try:
        config = _require_mapping(json.loads(raw_text), "Configuration")
    except json.JSONDecodeError as error:
        raise EvaluationQuestionConfigError(
            f"Evaluation-question config is not valid JSON: {path}"
        ) from error

    _require_exact_keys(config, CONFIG_KEYS, "Configuration")
    if config["evaluation_question_config_schema_version"] != CONFIG_SCHEMA_VERSION:
        raise EvaluationQuestionConfigError(
            "Unsupported evaluation-question configuration schema version."
        )
    if config["configuration_id"] != EXPECTED_CONFIGURATION_ID:
        raise EvaluationQuestionConfigError("Unexpected configuration identifier.")

    benchmark = _require_mapping(config["benchmark"], "benchmark")
    _require_exact_keys(benchmark, BENCHMARK_KEYS, "benchmark")
    if benchmark["dataset_id"] != EXPECTED_DATASET_ID:
        raise EvaluationQuestionConfigError("Unexpected benchmark dataset identifier.")
    if benchmark["question_text_availability"] != (
        "not_published_in_base_paper_reconstructed_from_cited_textbook"
    ):
        raise EvaluationQuestionConfigError(
            "Unexpected benchmark question-text availability statement."
        )
    question_ids = benchmark["question_ids"]
    if not isinstance(question_ids, list) or not question_ids:
        raise EvaluationQuestionConfigError(
            "benchmark.question_ids must be a non-empty list."
        )
    for position, question_id in enumerate(question_ids):
        question_id = _require_non_empty_string(
            question_id,
            f"benchmark.question_ids[{position}]",
        )
        if QUESTION_ID_PATTERN.fullmatch(question_id) is None:
            raise EvaluationQuestionConfigError(
                f"Invalid benchmark question identifier: {question_id}."
            )
    if len(question_ids) != len(set(question_ids)):
        raise EvaluationQuestionConfigError(
            "Benchmark question identifiers must be unique."
        )

    base_paper = _require_mapping(benchmark["base_paper"], "benchmark.base_paper")
    _require_exact_keys(base_paper, BASE_PAPER_KEYS, "benchmark.base_paper")
    _require_non_empty_string(base_paper["citation"], "benchmark.base_paper.citation")
    _require_non_empty_string(base_paper["doi"], "benchmark.base_paper.doi")
    _require_integer(
        base_paper["reported_full_question_bank_count"],
        "benchmark.base_paper.reported_full_question_bank_count",
        minimum=1,
    )
    selected_count = _require_integer(
        base_paper["reported_selected_question_count"],
        "benchmark.base_paper.reported_selected_question_count",
        minimum=1,
    )
    if selected_count != len(question_ids):
        raise EvaluationQuestionConfigError(
            "Reported selected count differs from the question identifier list."
        )

    supporting_paper = _require_mapping(
        benchmark["supporting_paper"],
        "benchmark.supporting_paper",
    )
    _require_exact_keys(
        supporting_paper,
        SUPPORTING_PAPER_KEYS,
        "benchmark.supporting_paper",
    )
    _require_non_empty_string(
        supporting_paper["citation"],
        "benchmark.supporting_paper.citation",
    )
    _require_non_empty_string(
        supporting_paper["doi"],
        "benchmark.supporting_paper.doi",
    )

    expected_audit = _require_mapping(config["expected_audit"], "expected_audit")
    _require_exact_keys(expected_audit, EXPECTED_AUDIT_KEYS, "expected_audit")
    question_count = _require_integer(
        expected_audit["question_count"],
        "expected_audit.question_count",
        minimum=1,
    )
    if question_count != len(question_ids):
        raise EvaluationQuestionConfigError(
            "Expected question count differs from the identifier list."
        )
    for key in (
        "contains_figure_count",
        "contains_table_count",
        "source_label_discrepancy_count",
    ):
        _require_integer(expected_audit[key], f"expected_audit.{key}")
    _validate_sha256(
        expected_audit["questions_sha256"],
        "expected_audit.questions_sha256",
    )
    _validate_sha256(
        expected_audit["question_projection_sha256"],
        "expected_audit.question_projection_sha256",
    )
    chapter_counts = _require_mapping(
        expected_audit["chapter_counts"],
        "expected_audit.chapter_counts",
    )
    if not chapter_counts:
        raise EvaluationQuestionConfigError(
            "expected_audit.chapter_counts must not be empty."
        )
    for chapter, count in chapter_counts.items():
        if not isinstance(chapter, str) or not chapter.isdigit() or int(chapter) < 1:
            raise EvaluationQuestionConfigError(
                "Chapter-count keys must be positive integer strings."
            )
        _require_integer(count, f"expected_audit.chapter_counts[{chapter}]", minimum=1)
    if sum(chapter_counts.values()) != question_count:
        raise EvaluationQuestionConfigError(
            "Expected chapter counts do not sum to the question count."
        )

    input_config = _require_mapping(config["input"], "input")
    _require_exact_keys(input_config, INPUT_KEYS, "input")
    if input_config["question_schema_version"] != QUESTION_SCHEMA_VERSION:
        raise EvaluationQuestionConfigError(
            "Unsupported evaluation-question record schema version."
        )
    input_path = _validate_relative_path(
        input_config["relative_path"],
        "input.relative_path",
    )
    if input_path != EXPECTED_QUESTION_PATH:
        raise EvaluationQuestionConfigError(
            "Evaluation questions must remain in the frozen raw evaluation path."
        )

    leakage = _require_mapping(config["leakage_controls"], "leakage_controls")
    _require_exact_keys(leakage, LEAKAGE_CONTROL_KEYS, "leakage_controls")
    for key in (
        "add_evaluation_questions_to_corpus_index",
        "ground_truth_available_during_generation",
        "question_file_tracked_by_git",
    ):
        if leakage[key] is not False:
            raise EvaluationQuestionLeakageError(
                f"leakage_controls.{key} must remain false."
            )
    ground_truth_path = _validate_relative_path(
        leakage["ground_truth_relative_path"],
        "leakage_controls.ground_truth_relative_path",
    )
    if ground_truth_path.rstrip("/") + "/" != EXPECTED_GROUND_TRUTH_PATH:
        raise EvaluationQuestionLeakageError(
            "Ground truth must remain in its frozen separate directory."
        )

    retrieval = _require_mapping(
        config["retrieval_evaluation"],
        "retrieval_evaluation",
    )
    _require_exact_keys(
        retrieval,
        RETRIEVAL_EVALUATION_KEYS,
        "retrieval_evaluation",
    )
    if retrieval["candidate_depths"] is not None:
        raise EvaluationQuestionConfigError(
            "Candidate retrieval depths must remain unresolved at this stage."
        )
    if retrieval["selected_depth_k"] is not None:
        raise EvaluationQuestionConfigError(
            "Retrieval depth k must remain unresolved at this stage."
        )
    if retrieval["selection_status"] != (
        "unresolved_pending_retrieval_evaluation"
    ):
        raise EvaluationQuestionConfigError(
            "Unexpected retrieval-depth selection status."
        )

    textbook = _require_mapping(config["source_textbook"], "source_textbook")
    _require_exact_keys(textbook, SOURCE_TEXTBOOK_KEYS, "source_textbook")
    _require_non_empty_string(textbook["citation"], "source_textbook.citation")
    _validate_sha256(
        textbook["expected_pdf_sha256"],
        "source_textbook.expected_pdf_sha256",
    )
    if textbook["transcription_method"] != EXPECTED_TRANSCRIPTION_MODE:
        raise EvaluationQuestionConfigError("Unexpected transcription method.")

    return config


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise EvaluationQuestionIntegrityError(
            f"Unable to read evaluation-question JSONL: {path}"
        ) from error
    if not raw_lines:
        raise EvaluationQuestionSchemaError("Evaluation-question JSONL is empty.")

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw_lines, start=1):
        if not line:
            raise EvaluationQuestionSchemaError(
                f"Blank JSONL line at line {line_number}."
            )
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise EvaluationQuestionSchemaError(
                f"Invalid JSON at question line {line_number}."
            ) from error
        if not isinstance(record, dict):
            raise EvaluationQuestionSchemaError(
                f"Question line {line_number} must contain a JSON object."
            )
        records.append(record)
    return records


def _validate_question_record(
    record: dict[str, Any],
    *,
    line_number: int,
) -> None:
    context = f"Question line {line_number}"
    _require_exact_keys(
        record,
        QUESTION_RECORD_KEYS,
        context,
        EvaluationQuestionSchemaError,
    )
    if record["question_schema_version"] != QUESTION_SCHEMA_VERSION:
        raise EvaluationQuestionSchemaError(
            f"{context} has an unsupported schema version."
        )
    if record["benchmark_dataset_id"] != EXPECTED_DATASET_ID:
        raise EvaluationQuestionSchemaError(
            f"{context} has an unexpected benchmark dataset identifier."
        )

    position = _require_integer(
        record["position"],
        f"{context}.position",
        error_type=EvaluationQuestionSchemaError,
    )
    question_id = _require_non_empty_string(
        record["question_id"],
        f"{context}.question_id",
        EvaluationQuestionSchemaError,
    )
    if QUESTION_ID_PATTERN.fullmatch(question_id) is None:
        raise EvaluationQuestionSchemaError(
            f"{context}.question_id has an invalid format."
        )
    problem = _require_non_empty_string(
        record["textbook_problem"],
        f"{context}.textbook_problem",
        EvaluationQuestionSchemaError,
    )
    if PROBLEM_ID_PATTERN.fullmatch(problem) is None:
        raise EvaluationQuestionSchemaError(
            f"{context}.textbook_problem has an invalid format."
        )
    part = record["textbook_part"]
    if part is not None and (
        not isinstance(part, str)
        or len(part) != 1
        or not part.isascii()
        or not part.islower()
    ):
        raise EvaluationQuestionSchemaError(
            f"{context}.textbook_part must be one lowercase letter or null."
        )
    if question_id != problem + (part or ""):
        raise EvaluationQuestionSchemaError(
            f"{context} question identifier does not match its problem and part."
        )

    _require_integer(
        record["chapter"],
        f"{context}.chapter",
        minimum=1,
        error_type=EvaluationQuestionSchemaError,
    )
    topic = _require_non_empty_string(
        record["topic"],
        f"{context}.topic",
        EvaluationQuestionSchemaError,
    )
    if TOPIC_PATTERN.fullmatch(topic) is None:
        raise EvaluationQuestionSchemaError(
            f"{context}.topic must use lowercase snake_case."
        )
    _validate_positive_page_list(
        record["pdf_page_numbers"],
        f"{context}.pdf_page_numbers",
    )
    _validate_positive_page_list(
        record["printed_page_numbers"],
        f"{context}.printed_page_numbers",
    )

    query_text = _require_non_empty_string(
        record["query_text"],
        f"{context}.query_text",
        EvaluationQuestionSchemaError,
    )
    query_hash = _validate_sha256(
        record["query_text_sha256"],
        f"{context}.query_text_sha256",
        EvaluationQuestionSchemaError,
    )
    actual_query_hash = sha256(query_text.encode("utf-8")).hexdigest()
    if actual_query_hash != query_hash:
        raise EvaluationQuestionIntegrityError(
            f"Query-text fingerprint mismatch for {question_id}."
        )

    for key in ("contains_figure", "contains_table"):
        if type(record[key]) is not bool:
            raise EvaluationQuestionSchemaError(f"{context}.{key} must be boolean.")
    if record["contains_figure"] is not False:
        raise EvaluationQuestionSchemaError(
            f"Figure-dependent question is not allowed: {question_id}."
        )
    if record["ground_truth_available_during_generation"] is not False:
        raise EvaluationQuestionLeakageError(
            f"Ground truth is exposed during generation for {question_id}."
        )
    if record["transcription_mode"] != EXPECTED_TRANSCRIPTION_MODE:
        raise EvaluationQuestionSchemaError(
            f"Unexpected transcription mode for {question_id}."
        )
    if record["source_label_status"] not in SOURCE_LABEL_STATUSES:
        raise EvaluationQuestionSchemaError(
            f"Unexpected source-label status for {question_id}."
        )
    source_note = record["source_note"]
    if source_note is not None:
        _require_non_empty_string(
            source_note,
            f"{context}.source_note",
            EvaluationQuestionSchemaError,
        )
    if position != line_number - 1:
        raise EvaluationQuestionSchemaError(
            f"Question position differs from JSONL order at {question_id}."
        )


def _validate_dataset_audit(
    records: list[dict[str, Any]],
    config: dict[str, Any],
) -> str:
    expected = config["expected_audit"]
    expected_ids = config["benchmark"]["question_ids"]
    actual_ids = [record["question_id"] for record in records]

    if len(records) != expected["question_count"]:
        raise EvaluationQuestionSchemaError(
            "Question record count differs from the frozen audit."
        )
    if len(actual_ids) != len(set(actual_ids)):
        raise EvaluationQuestionSchemaError("Question identifiers are not unique.")
    if actual_ids != expected_ids:
        raise EvaluationQuestionSchemaError(
            "Question identifiers or their order differ from the frozen benchmark."
        )
    if [record["position"] for record in records] != list(range(len(records))):
        raise EvaluationQuestionSchemaError(
            "Question positions must be unique and contiguous."
        )

    chapter_counts: dict[str, int] = {}
    for record in records:
        chapter = str(record["chapter"])
        chapter_counts[chapter] = chapter_counts.get(chapter, 0) + 1
    if chapter_counts != expected["chapter_counts"]:
        raise EvaluationQuestionSchemaError(
            "Question chapter counts differ from the frozen audit."
        )
    if sum(bool(record["contains_table"]) for record in records) != expected[
        "contains_table_count"
    ]:
        raise EvaluationQuestionSchemaError(
            "Linearised-table count differs from the frozen audit."
        )
    if sum(bool(record["contains_figure"]) for record in records) != expected[
        "contains_figure_count"
    ]:
        raise EvaluationQuestionSchemaError(
            "Figure-dependency count differs from the frozen audit."
        )
    discrepancy_count = sum(
        record["source_label_status"] != "exact" for record in records
    )
    if discrepancy_count != expected["source_label_discrepancy_count"]:
        raise EvaluationQuestionSchemaError(
            "Source-label discrepancy count differs from the frozen audit."
        )

    projection_sha256 = calculate_question_projection_sha256(records)
    if projection_sha256 != expected["question_projection_sha256"]:
        raise EvaluationQuestionIntegrityError(
            "Question projection fingerprint differs from the frozen audit."
        )
    return projection_sha256


def load_evaluation_questions(
    config_path: Path,
    *,
    project_root: Path,
) -> EvaluationQuestionDataset:
    """Validate and load the frozen text-only evaluation questions."""

    root = project_root.resolve()
    resolved_config_path = config_path
    if not resolved_config_path.is_absolute():
        resolved_config_path = root / resolved_config_path
    resolved_config_path = resolved_config_path.resolve()
    try:
        resolved_config_path.relative_to(root)
    except ValueError as error:
        raise EvaluationQuestionConfigError(
            "The configuration path must remain within the project root."
        ) from error

    config = load_evaluation_question_config(resolved_config_path)
    question_path = _resolve_within_project(root, config["input"]["relative_path"])
    if not question_path.is_file():
        raise EvaluationQuestionIntegrityError(
            f"Evaluation-question file is missing: {question_path}"
        )

    actual_questions_sha256 = calculate_file_sha256(question_path)
    expected_questions_sha256 = config["expected_audit"]["questions_sha256"]
    if actual_questions_sha256 != expected_questions_sha256:
        raise EvaluationQuestionIntegrityError(
            "Evaluation-question file fingerprint differs from the frozen config."
        )

    records = _load_jsonl(question_path)
    for line_number, record in enumerate(records, start=1):
        _validate_question_record(record, line_number=line_number)
    projection_sha256 = _validate_dataset_audit(records, config)

    return EvaluationQuestionDataset(
        configuration_path=resolved_config_path,
        question_path=question_path,
        records=tuple(records),
        questions_sha256=actual_questions_sha256,
        question_projection_sha256=projection_sha256,
    )


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the local retrieval-evaluation questions against their "
            "committed contract."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/evaluation-question-config.json"),
        help="Project-relative path to the evaluation-question configuration.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Repository root containing configs and data directories.",
    )
    return parser


def main() -> None:
    arguments = _build_argument_parser().parse_args()
    dataset = load_evaluation_questions(
        arguments.config,
        project_root=arguments.project_root,
    )

    print("Evaluation-question validation: PASSED")
    print("  Question records:", len(dataset.records))
    print("  Unique identifiers:", len({r["question_id"] for r in dataset.records}))
    print("  Linearised tables:", sum(bool(r["contains_table"]) for r in dataset.records))
    print("  Figure-dependent questions:", sum(bool(r["contains_figure"]) for r in dataset.records))
    print(
        "  Source-label discrepancies:",
        sum(r["source_label_status"] != "exact" for r in dataset.records),
    )
    print("  Questions SHA-256:", dataset.questions_sha256)
    print(
        "  Question projection SHA-256:",
        dataset.question_projection_sha256,
    )
    print("  Ground truth accessed: False")
    print("  Retrieval depth k remains unresolved")


if __name__ == "__main__":
    main()
