"""Offline preparation for the frozen benchmark generation evaluation.

This module validates the generation contract, reconstructs the 140 requests
from frozen benchmark questions and production retrieval, and performs a
conservative context-window audit.  It never sends an API request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

import tiktoken


class GenerationEvaluationError(RuntimeError):
    """Base error for generation-evaluation preparation."""


class GenerationConfigError(GenerationEvaluationError):
    """Raised when the frozen generation contract is invalid."""


class GenerationIntegrityError(GenerationEvaluationError):
    """Raised when a frozen input fails an integrity check."""


class GenerationContextLimitError(GenerationEvaluationError):
    """Raised when a planned request exceeds its model context window."""


class BenchmarkRetriever(Protocol):
    """Minimal production-retrieval interface required by preparation."""

    @property
    def benchmark_question_ids(self) -> tuple[str, ...]: ...

    @property
    def retrieval_depth_k(self) -> int: ...

    def retrieve_benchmark(self, question_id: str) -> Any: ...


@dataclass(frozen=True)
class PreparedGenerationRequest:
    """One deterministic private request-manifest record."""

    record_schema_version: str
    configuration_id: str
    condition_id: str
    blinded_response_id: str
    question_id: str
    question_text_sha256: str
    retrieved_evidence_sha256: str | None
    prompt_sha256: str
    requested_model_id: str
    retrieval_enabled: bool
    retrieval_depth_k: int
    temperature: float | None
    top_p: float | None
    reasoning_effort: str | None
    endpoint: str
    system_prompt: str
    user_prompt: str
    logical_max_output_tokens: int
    output_token_parameter: str
    tokenizer_strategy: str
    estimated_input_tokens: int
    reserved_output_tokens: int
    estimated_total_tokens: int
    model_context_window_tokens: int
    context_window_passed: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record."""

        return asdict(self)


@dataclass(frozen=True)
class GenerationPreparationResult:
    """Text-free summary of one preparation run."""

    configuration_id: str
    question_count: int
    condition_count: int
    request_count: int
    retrieval_request_count: int
    zero_context_request_count: int
    maximum_estimated_input_tokens: int
    maximum_estimated_total_tokens: int
    minimum_context_headroom_tokens: int
    manifest_path: Path | None
    manifest_sha256: str | None
    api_request_made: bool = False
    paid_generation_performed: bool = False


MODEL_CONTEXT_WINDOWS: Mapping[str, int] = {
    "gpt-4-0613": 8_192,
    "gpt-4.1-2025-04-14": 1_047_576,
    "gpt-5-2025-08-07": 400_000,
    "gpt-6-astra": 1_050_000,
}

LEGACY_OUTPUT_MODELS = frozenset({"gpt-4-0613"})
REQUIRED_TOP_LEVEL_KEYS = frozenset(
    {
        "benchmark",
        "comparison_groups",
        "conditions",
        "configuration_id",
        "context_assembly",
        "cost_boundary",
        "decision_status",
        "evaluation_boundary",
        "execution_controls",
        "future_tracked_outputs",
        "generation_evaluation_config_schema_version",
        "leakage_controls",
        "model_availability_preflight",
        "model_parameter_policy",
        "private_outputs",
        "production_retrieval",
        "prompt",
        "protocol",
        "request_policy",
        "response_record_contract",
    }
)


def calculate_file_sha256(path: str | Path) -> str:
    """Calculate a file SHA-256 fingerprint."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json_line(record: Mapping[str, Any]) -> str:
    return json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GenerationIntegrityError(f"Could not load {label}.") from error
    if not isinstance(value, dict):
        raise GenerationIntegrityError(f"{label} must be a JSON object.")
    return value


def _resolve_relative(root: Path, relative: str, label: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise GenerationConfigError(f"{label} must be a safe relative path.")
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise GenerationConfigError(f"{label} escapes the project root.") from error
    return resolved


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GenerationConfigError(message)


def load_generation_evaluation_config(path: str | Path) -> dict[str, Any]:
    """Load and validate the frozen generation-evaluation contract."""

    config_path = Path(path)
    config = _load_json(config_path, "generation configuration")
    _require(
        set(config) == REQUIRED_TOP_LEVEL_KEYS,
        "Unexpected generation-configuration top-level schema.",
    )
    _require(
        config["generation_evaluation_config_schema_version"] == "1.0",
        "Unexpected generation-configuration schema version.",
    )
    _require(
        config["decision_status"] == "frozen_before_paid_generation",
        "Unexpected generation decision status.",
    )

    benchmark = config["benchmark"]
    conditions = config["conditions"]
    _require(isinstance(benchmark, dict), "benchmark must be a mapping.")
    _require(isinstance(conditions, list), "conditions must be a list.")
    _require(benchmark["question_count"] == 20, "Expected 20 questions.")
    _require(benchmark["condition_count"] == 7, "Expected seven conditions.")
    _require(len(conditions) == 7, "Expected seven condition records.")
    _require(
        benchmark["planned_response_count"] == 140,
        "Expected 140 planned responses.",
    )

    condition_ids = [condition.get("condition_id") for condition in conditions]
    _require(len(set(condition_ids)) == 7, "Condition identifiers must be unique.")
    _require(
        condition_ids == ["P0", "P1", "P2", "P3", "M1", "M2", "M3"],
        "Unexpected condition order.",
    )
    for condition in conditions:
        model_id = condition.get("model_id")
        _require(model_id in MODEL_CONTEXT_WINDOWS, "Unknown model context window.")
        if condition["condition_id"] in {"M2", "M3"}:
            _require(condition["temperature"] is None, "Reasoning temperature must be null.")
            _require(condition["top_p"] is None, "Reasoning top_p must be null.")
            _require(condition["reasoning_effort"] == "low", "Unexpected reasoning effort.")
        else:
            _require(condition["reasoning_effort"] is None, "Legacy reasoning must be null.")
            _require(isinstance(condition["temperature"], (int, float)), "Missing temperature.")
            _require(condition["top_p"] == 1.0, "Unexpected top_p.")

    assembly = config["context_assembly"]
    _require(assembly["context_record_field"] == "text", "Unexpected context field.")
    _require(assembly["retrieved_record_count"] == 30, "Expected 30 retrieved records.")
    _require(assembly["truncation_enabled"] is False, "Context truncation must be disabled.")
    _require(assembly["deduplication_enabled"] is False, "Context deduplication must be disabled.")
    _require(assembly["preserve_faiss_returned_order"] is True, "FAISS order must be preserved.")

    controls = config["execution_controls"]
    _require(controls["paid_generation_enabled"] is False, "Paid generation must remain disabled.")
    cost = config["cost_boundary"]
    _require(cost["generation_must_refuse_while_values_are_null"] is True, "Null-cost refusal required.")
    _require(cost["authorized_maximum_cost_usd"] is None, "Cost authorization must still be null.")
    _require(cost["estimated_maximum_cost_usd"] is None, "Cost estimate must still be null.")

    leakage = config["leakage_controls"]
    for key in (
        "answer_keys_available_during_generation",
        "ground_truth_available_during_generation",
        "model_generated_relevance_advice_available",
        "relevance_grades_available_during_generation",
        "relevance_notes_available_during_generation",
    ):
        _require(leakage[key] is False, f"Leakage control {key} must be false.")
    return config


def _verify_linked_file(
    root: Path,
    relative_path: str,
    expected_sha256: str,
    label: str,
) -> Path:
    path = _resolve_relative(root, relative_path, label)
    if not path.is_file():
        raise GenerationIntegrityError(f"Missing {label}: {relative_path}")
    actual = calculate_file_sha256(path)
    if actual != expected_sha256:
        raise GenerationIntegrityError(f"Wrong SHA-256 for {label}.")
    return path


def _load_questions(
    path: Path,
    expected_ids: Sequence[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise GenerationIntegrityError(f"Question line {line_number} is not an object.")
        records.append(value)
    if len(records) != len(expected_ids):
        raise GenerationIntegrityError("Unexpected benchmark question count.")
    identifiers = [record.get("question_id") for record in records]
    if identifiers != list(expected_ids):
        raise GenerationIntegrityError("Benchmark question order changed.")
    for record in records:
        text = record.get("query_text")
        if not isinstance(text, str) or not text.strip():
            raise GenerationIntegrityError("A benchmark query_text is invalid.")
        if record.get("query_text_sha256") != _text_sha256(text):
            raise GenerationIntegrityError("A benchmark query_text hash is invalid.")
        if record.get("ground_truth_available_during_generation") is not False:
            raise GenerationIntegrityError("Ground truth is available during generation.")
    return records


def _result_payload(result: Any) -> dict[str, Any]:
    if hasattr(result, "to_dict"):
        payload = result.to_dict()
    elif isinstance(result, dict):
        payload = result
    else:
        raise GenerationIntegrityError("Unsupported retrieval-result type.")
    if not isinstance(payload, dict):
        raise GenerationIntegrityError("Retrieval result is not a mapping.")
    return payload


def assemble_context(
    records: Sequence[Mapping[str, Any]],
    assembly: Mapping[str, Any],
) -> tuple[str, str]:
    """Assemble all retrieved texts in their returned order."""

    expected_count = int(assembly["retrieved_record_count"])
    if len(records) != expected_count:
        raise GenerationIntegrityError("Unexpected retrieved-record count.")
    field = assembly["context_record_field"]
    prefix = assembly["record_prefix_template"]
    separator = assembly["record_separator"]
    parts: list[str] = []
    previous_rank = 0
    for ordinal, record in enumerate(records, 1):
        text = record.get(field)
        rank = record.get("retrieval_rank")
        if not isinstance(text, str) or not text:
            raise GenerationIntegrityError("Retrieved context text is missing.")
        if rank != ordinal or rank <= previous_rank:
            raise GenerationIntegrityError("Retrieved order or rank is invalid.")
        previous_rank = rank
        parts.append(f"{prefix.format(ordinal=ordinal)}\n{text}")
    context = separator.join(parts)
    return context, _text_sha256(context)


def _render_exact(template: str, replacements: Mapping[str, str]) -> str:
    rendered = template
    for name, value in replacements.items():
        placeholder = "{" + name + "}"
        if rendered.count(placeholder) != 1:
            raise GenerationIntegrityError(f"Unexpected placeholder count for {name}.")
        rendered = rendered.replace(placeholder, value)
    if "{question}" in rendered or "{context}" in rendered:
        raise GenerationIntegrityError("An unresolved prompt placeholder remains.")
    return rendered


def _encoding_for_model(model_id: str) -> tuple[list[Any], str]:
    if model_id == "gpt-6-astra":
        encodings = [
            tiktoken.get_encoding("cl100k_base"),
            tiktoken.get_encoding("o200k_base"),
        ]
        return encodings, "max_of_cl100k_base_and_o200k_base"
    try:
        encoding = tiktoken.encoding_for_model(model_id)
    except KeyError as error:
        raise GenerationIntegrityError(f"No tokenizer mapping for {model_id}.") from error
    return [encoding], f"tiktoken_{encoding.name}"


def estimate_chat_input_tokens(
    model_id: str,
    system_prompt: str,
    user_prompt: str,
) -> tuple[int, str]:
    """Conservatively estimate two-message Chat Completions input tokens."""

    encodings, strategy = _encoding_for_model(model_id)
    estimates = []
    for encoding in encodings:
        content_tokens = len(encoding.encode(system_prompt, disallowed_special=()))
        content_tokens += len(encoding.encode(user_prompt, disallowed_special=()))
        # Conservative structural allowance for two roles, message framing,
        # separators and assistant priming. Exact billed usage is stored later.
        estimates.append(content_tokens + 24)
    return max(estimates), strategy


def _blind_response_id(configuration_id: str, condition_id: str, question_id: str) -> str:
    material = "\x1f".join((configuration_id, condition_id, question_id))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _prepare_records(
    config: Mapping[str, Any],
    prompt: Mapping[str, Any],
    questions: Sequence[Mapping[str, Any]],
    retriever: BenchmarkRetriever,
) -> list[PreparedGenerationRequest]:
    expected_ids = tuple(record["question_id"] for record in questions)
    if tuple(retriever.benchmark_question_ids) != expected_ids:
        raise GenerationIntegrityError("Retriever benchmark order changed.")
    if retriever.retrieval_depth_k != config["production_retrieval"]["retrieval_depth_k"]:
        raise GenerationIntegrityError("Retriever depth changed.")

    retrieval_by_question: dict[str, dict[str, Any]] = {}
    for question_id in expected_ids:
        payload = _result_payload(retriever.retrieve_benchmark(question_id))
        if payload.get("query_identifier") != question_id:
            raise GenerationIntegrityError("Retrieval question identifier changed.")
        if payload.get("api_request_made") is not False:
            raise GenerationIntegrityError("Retrieval unexpectedly made an API request.")
        retrieval_by_question[question_id] = payload

    request_policy = config["request_policy"]
    max_output = int(request_policy["logical_max_output_tokens"])
    prepared: list[PreparedGenerationRequest] = []
    for condition in config["conditions"]:
        for question in questions:
            question_id = question["question_id"]
            query_text = question["query_text"]
            evidence_hash: str | None = None
            if condition["retrieval_enabled"]:
                payload = retrieval_by_question[question_id]
                context, evidence_hash = assemble_context(
                    payload["records"],
                    config["context_assembly"],
                )
                user_prompt = _render_exact(
                    prompt["rag_user_template"],
                    {"context": context, "question": query_text},
                )
            else:
                user_prompt = _render_exact(
                    prompt["zero_context_user_template"],
                    {"question": query_text},
                )
            model_id = condition["model_id"]
            input_tokens, tokenizer_strategy = estimate_chat_input_tokens(
                model_id,
                prompt["system_prompt"],
                user_prompt,
            )
            context_window = MODEL_CONTEXT_WINDOWS[model_id]
            total = input_tokens + max_output
            if total > context_window:
                raise GenerationContextLimitError(
                    f"Condition {condition['condition_id']} question {question_id} "
                    "exceeds its context window."
                )
            output_parameter = (
                request_policy["legacy_output_token_parameter"]
                if model_id in LEGACY_OUTPUT_MODELS
                else request_policy["modern_output_token_parameter"]
            )
            prepared.append(
                PreparedGenerationRequest(
                    record_schema_version="1.0",
                    configuration_id=config["configuration_id"],
                    condition_id=condition["condition_id"],
                    blinded_response_id=_blind_response_id(
                        config["configuration_id"], condition["condition_id"], question_id
                    ),
                    question_id=question_id,
                    question_text_sha256=question["query_text_sha256"],
                    retrieved_evidence_sha256=evidence_hash,
                    prompt_sha256=config["prompt"]["sha256"],
                    requested_model_id=model_id,
                    retrieval_enabled=condition["retrieval_enabled"],
                    retrieval_depth_k=condition["retrieval_depth_k"],
                    temperature=condition["temperature"],
                    top_p=condition["top_p"],
                    reasoning_effort=condition["reasoning_effort"],
                    endpoint=request_policy["endpoint"],
                    system_prompt=prompt["system_prompt"],
                    user_prompt=user_prompt,
                    logical_max_output_tokens=max_output,
                    output_token_parameter=output_parameter,
                    tokenizer_strategy=tokenizer_strategy,
                    estimated_input_tokens=input_tokens,
                    reserved_output_tokens=max_output,
                    estimated_total_tokens=total,
                    model_context_window_tokens=context_window,
                    context_window_passed=True,
                )
            )
    return prepared


def _write_manifest(path: Path, records: Iterable[PreparedGenerationRequest]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise GenerationIntegrityError("Request manifest already exists; overwrite refused.")
    payload = "".join(_canonical_json_line(record.to_dict()) + "\n" for record in records)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)
    return calculate_file_sha256(path)


def prepare_generation_requests(
    config_path: str | Path,
    project_root: str | Path,
    *,
    write_manifest: bool = False,
    retriever: BenchmarkRetriever | None = None,
) -> GenerationPreparationResult:
    """Validate and prepare all requests without any generation API call."""

    root = Path(project_root).resolve()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config = load_generation_evaluation_config(config_file)
    prompt_path = _verify_linked_file(
        root, config["prompt"]["relative_path"], config["prompt"]["sha256"], "prompt"
    )
    _verify_linked_file(
        root, config["protocol"]["relative_path"], config["protocol"]["sha256"], "protocol"
    )
    production_config_path = _verify_linked_file(
        root,
        config["production_retrieval"]["configuration_relative_path"],
        config["production_retrieval"]["configuration_sha256"],
        "production retrieval configuration",
    )
    _verify_linked_file(
        root,
        config["production_retrieval"]["implementation_relative_path"],
        config["production_retrieval"]["implementation_sha256"],
        "production retrieval implementation",
    )
    prompt = _load_json(prompt_path, "generation prompt")
    if prompt.get("prompt_id") != config["prompt"]["prompt_id"]:
        raise GenerationIntegrityError("Prompt identifier changed.")

    production_config = _load_json(production_config_path, "production retrieval configuration")
    benchmark_mode = production_config["query_embedding"]["benchmark_mode"]
    questions_path = _verify_linked_file(
        root,
        benchmark_mode["questions_relative_path"],
        benchmark_mode["questions_sha256"],
        "benchmark questions",
    )
    question_config_path = _verify_linked_file(
        root,
        benchmark_mode["question_configuration_relative_path"],
        benchmark_mode["question_configuration_sha256"],
        "question configuration",
    )
    question_config = _load_json(question_config_path, "question configuration")
    expected_ids = question_config["benchmark"]["question_ids"]
    questions = _load_questions(questions_path, expected_ids)

    if retriever is None:
        from geotech_rag.production_retrieval import load_production_retriever

        retriever = load_production_retriever(production_config_path, root)
    records = _prepare_records(config, prompt, questions, retriever)
    if len(records) != config["benchmark"]["planned_response_count"]:
        raise GenerationIntegrityError("Unexpected prepared-request count.")

    manifest_path: Path | None = None
    manifest_sha256: str | None = None
    if write_manifest:
        manifest_path = _resolve_relative(
            root,
            config["private_outputs"]["request_manifest_relative_path"],
            "request manifest",
        )
        manifest_sha256 = _write_manifest(manifest_path, records)

    maximum_input = max(record.estimated_input_tokens for record in records)
    maximum_total = max(record.estimated_total_tokens for record in records)
    minimum_headroom = min(
        record.model_context_window_tokens - record.estimated_total_tokens
        for record in records
    )
    retrieval_count = sum(record.retrieval_enabled for record in records)
    return GenerationPreparationResult(
        configuration_id=config["configuration_id"],
        question_count=len(questions),
        condition_count=len(config["conditions"]),
        request_count=len(records),
        retrieval_request_count=retrieval_count,
        zero_context_request_count=len(records) - retrieval_count,
        maximum_estimated_input_tokens=maximum_input,
        maximum_estimated_total_tokens=maximum_total,
        minimum_context_headroom_tokens=minimum_headroom,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
    )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Write the private ignored request manifest after implementation commit.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    result = prepare_generation_requests(
        args.config,
        args.project_root,
        write_manifest=args.write_manifest,
    )
    print("Generation request preparation: PASSED")
    print("  Configuration ID:", result.configuration_id)
    print("  Questions:", result.question_count)
    print("  Conditions:", result.condition_count)
    print("  Planned requests:", result.request_count)
    print("  Retrieval requests:", result.retrieval_request_count)
    print("  Zero-context requests:", result.zero_context_request_count)
    print("  Maximum estimated input tokens:", result.maximum_estimated_input_tokens)
    print("  Maximum estimated total tokens:", result.maximum_estimated_total_tokens)
    print("  Minimum context headroom tokens:", result.minimum_context_headroom_tokens)
    print("  Manifest written:", result.manifest_path is not None)
    if result.manifest_path is not None:
        print("  Manifest:", result.manifest_path)
        print("  Manifest SHA-256:", result.manifest_sha256)
    print("  API request made: False")
    print("  Paid generation performed: False")
    print("  Question or chunk text displayed: False")
    print("  Ground truth accessed: False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
