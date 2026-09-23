from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

import geotech_rag.generation_execution as execution_module
from geotech_rag.generation_execution import (
    REQUIRED_RESPONSE_FIELDS,
    CheckpointPaths,
    ExecutionCostLimitError,
    ExecutionIntegrityError,
    ExecutionRequestError,
    _atomic_write_json,
    _atomic_write_jsonl,
    _execute_checkpointed_requests,
    _load_or_initialize_checkpoint,
    _response_record_from_completion,
    _retry_after_seconds,
    build_chat_completion_request,
    known_usage_cost,
    load_execution_enablement,
    load_manifest,
    maximum_request_cost,
    reserve_attempt_cost,
    retry_delay_seconds,
    retryable_status,
    rounded_usd,
    validate_authorization,
    validate_manifest,
)


def _condition(
    condition_id: str,
    model_id: str,
    *,
    retrieval_enabled: bool,
    temperature: float | None,
    top_p: float | None,
    reasoning_effort: str | None,
) -> dict:
    """Create one synthetic frozen condition."""

    return {
        "condition_id": condition_id,
        "model_id": model_id,
        "retrieval_enabled": retrieval_enabled,
        "retrieval_depth_k": (
            30 if retrieval_enabled else 0
        ),
        "temperature": temperature,
        "top_p": top_p,
        "reasoning_effort": reasoning_effort,
    }


def _configuration() -> dict:
    """Create the seven-condition public contract."""

    return {
        "configuration_id": (
            "benchmark-generation-evaluation-v1"
        ),
        "conditions": [
            _condition(
                "P0",
                "gpt-4-0613",
                retrieval_enabled=False,
                temperature=0.1,
                top_p=1.0,
                reasoning_effort=None,
            ),
            _condition(
                "P1",
                "gpt-4-0613",
                retrieval_enabled=True,
                temperature=0.1,
                top_p=1.0,
                reasoning_effort=None,
            ),
            _condition(
                "P2",
                "gpt-4-0613",
                retrieval_enabled=True,
                temperature=0.5,
                top_p=1.0,
                reasoning_effort=None,
            ),
            _condition(
                "P3",
                "gpt-4-0613",
                retrieval_enabled=True,
                temperature=1.0,
                top_p=1.0,
                reasoning_effort=None,
            ),
            _condition(
                "M1",
                "gpt-4.1-2025-04-14",
                retrieval_enabled=True,
                temperature=0.1,
                top_p=1.0,
                reasoning_effort=None,
            ),
            _condition(
                "M2",
                "gpt-5-2025-08-07",
                retrieval_enabled=True,
                temperature=None,
                top_p=None,
                reasoning_effort="low",
            ),
            _condition(
                "M3",
                "gpt-6-astra",
                retrieval_enabled=True,
                temperature=None,
                top_p=None,
                reasoning_effort="low",
            ),
        ],
    }


def _blinded_id(
    condition_id: str,
    question_id: str,
) -> str:
    """Reproduce the deterministic blinded-ID method."""

    # Join the three identifiers with the frozen separator.
    material = "\x1f".join(
        (
            "benchmark-generation-evaluation-v1",
            condition_id,
            question_id,
        )
    )

    # Hash the identifier material without private text.
    return hashlib.sha256(
        material.encode("utf-8")
    ).hexdigest()


def _manifest_record(
    condition: dict,
    question_number: int,
) -> dict:
    """Create one request using synthetic non-benchmark text."""

    # Create the common synthetic question identifier.
    question_id = f"q{question_number:02d}"
    model_id = condition["model_id"]

    # Match the frozen model-specific output parameter.
    output_parameter = (
        "max_tokens"
        if model_id == "gpt-4-0613"
        else "max_completion_tokens"
    )

    return {
        "blinded_response_id": _blinded_id(
            condition["condition_id"],
            question_id,
        ),
        "condition_id": condition["condition_id"],
        "configuration_id": (
            "benchmark-generation-evaluation-v1"
        ),
        "context_window_passed": True,
        "endpoint": "chat.completions",
        "estimated_input_tokens": 100,
        "estimated_total_tokens": 2100,
        "logical_max_output_tokens": 2000,
        "model_context_window_tokens": 8192,
        "output_token_parameter": output_parameter,
        "prompt_sha256": "a" * 64,
        "question_id": question_id,
        "question_text_sha256": "b" * 64,
        "reasoning_effort": (
            condition["reasoning_effort"]
        ),
        "record_schema_version": "1.0",
        "requested_model_id": model_id,
        "reserved_output_tokens": 2000,
        "retrieval_depth_k": (
            condition["retrieval_depth_k"]
        ),
        "retrieval_enabled": (
            condition["retrieval_enabled"]
        ),
        "retrieved_evidence_sha256": (
            "c" * 64
            if condition["retrieval_enabled"]
            else None
        ),
        "system_prompt": (
            "Synthetic tutor instructions"
        ),
        "temperature": condition["temperature"],
        "tokenizer_strategy": "test-tokenizer",
        "top_p": condition["top_p"],
        "user_prompt": (
            "Synthetic question without "
            "benchmark content"
        ),
    }


def _manifest_records() -> list[dict]:
    """Create 140 records in condition-major order."""

    return [
        _manifest_record(
            condition,
            question_number,
        )
        for condition in _configuration()[
            "conditions"
        ]
        for question_number in range(1, 21)
    ]


def _request_policy() -> dict:
    """Create the frozen request fields used here."""

    return {
        "api_response_storage_requested": False,
        "n": 1,
        "retry_initial_seconds": 2,
        "retry_max_seconds": 8,
        "retry_multiplier": 2,
        "retryable_http_statuses": [
            408,
            409,
            429,
            500,
            502,
            503,
            504,
        ],
        "stream": False,
        "timeout_seconds": 120,
    }


def _pricing() -> dict:
    """Create the four frozen model rates."""

    return {
        "model_rates": {
            "gpt-4-0613": {
                (
                    "input_usd_per_million_tokens"
                ): 30.0,
                (
                    "output_usd_per_million_tokens"
                ): 60.0,
            },
            "gpt-4.1-2025-04-14": {
                (
                    "input_usd_per_million_tokens"
                ): 2.0,
                (
                    "output_usd_per_million_tokens"
                ): 8.0,
            },
            "gpt-5-2025-08-07": {
                (
                    "input_usd_per_million_tokens"
                ): 1.25,
                (
                    "output_usd_per_million_tokens"
                ): 10.0,
            },
            "gpt-6-astra": {
                (
                    "input_usd_per_million_tokens"
                ): 10.0,
                (
                    "output_usd_per_million_tokens"
                ): 50.0,
            },
        }
    }


def _authorization() -> dict:
    """Create a minimal valid $25 authorization."""

    return {
        "authorization_schema_version": "1.0",
        "authorization_scope": {
            "extra_requests_authorized": False,
            (
                "maximum_completed_response_count"
            ): 140,
            "model_substitution_authorized": False,
            "parameter_substitution_authorized": (
                False
            ),
            "planned_request_count": 140,
            "unplanned_conditions_authorized": False,
        },
        "authorized_maximum_cost_usd": 25.0,
        "budget_authorization_status": "AUTHORIZED",
        "execution_authorization_status": (
            "NOT_ENABLED"
        ),
        "execution_controls": {
            "paid_generation_enabled": False,
            "runtime_hard_cost_cap_usd": 25.0,
        },
        (
            "theoretical_full_retry_exposure_authorized"
        ): False,
    }


def _enablement() -> dict:
    """Create the future fail-closed switch schema."""

    return {
        "authorization": {},
        "configuration": {},
        "configuration_id": (
            "benchmark-generation-evaluation-v1"
        ),
        "enabled_on_utc_date": "2026-09-23",
        "execution_enablement_schema_version": "1.0",
        "execution_status": "ENABLED",
        "implementation": {
            "commit": "d" * 40,
            "tests_passed": True,
        },
        "manifest": {},
        "pricing_recheck": {},
        "runtime_controls": {
            "explicit_execute_flag_required": True,
            (
                "model_availability_preflight_required"
            ): True,
            "paid_generation_enabled": True,
            "parallel_request_count": 1,
            "require_clean_worktree": True,
        },
        "state_before_enablement": {
            "api_request_made": False,
            (
                "generation_response_evidence_created"
            ): False,
            "ground_truth_accessed": False,
            "paid_generation_performed": False,
        },
    }


class _StatusError(Exception):
    """Synthetic structured HTTP error."""

    def __init__(self, status_code: int) -> None:
        # Store only the status used by retry classification.
        super().__init__("synthetic status error")
        self.status_code = status_code


def test_loads_valid_enablement(
    tmp_path: Path,
) -> None:
    """A complete explicit switch passes validation."""

    # Save a synthetic enablement record.
    path = tmp_path / "enablement.json"
    path.write_text(
        json.dumps(_enablement()),
        encoding="utf-8",
    )

    # Confirm that the enabled state is loaded.
    assert (
        load_execution_enablement(path)[
            "execution_status"
        ]
        == "ENABLED"
    )


def test_rejects_disabled_enablement(
    tmp_path: Path,
) -> None:
    """A disabled switch fails before client creation."""

    # Disable paid generation in the synthetic record.
    value = _enablement()
    value["runtime_controls"][
        "paid_generation_enabled"
    ] = False

    # Save and validate the disabled record.
    path = tmp_path / "enablement.json"
    path.write_text(
        json.dumps(value),
        encoding="utf-8",
    )

    with pytest.raises(
        ExecutionIntegrityError,
        match="disabled",
    ):
        load_execution_enablement(path)


def test_validates_separate_budget_authorization() -> None:
    """Budget approval returns the exact hard cap."""

    assert (
        validate_authorization(_authorization())
        == Decimal("25.0")
    )


def test_rejects_budget_record_that_enables_execution() -> None:
    """The budget record cannot be the live switch."""

    # Illegally enable execution in the old record.
    value = _authorization()
    value["execution_controls"][
        "paid_generation_enabled"
    ] = True

    with pytest.raises(
        ExecutionIntegrityError,
        match="disabled",
    ):
        validate_authorization(value)


def test_loads_and_validates_manifest(
    tmp_path: Path,
) -> None:
    """The private manifest retains all 140 records."""

    # Save synthetic private records as JSONL.
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        "".join(
            json.dumps(record) + "\n"
            for record in _manifest_records()
        ),
        encoding="utf-8",
    )

    # Load and validate without displaying prompt text.
    records = load_manifest(path)
    validate_manifest(
        records,
        _configuration(),
    )

    assert len(records) == 140


def test_rejects_manifest_parameter_change() -> None:
    """A post-preparation temperature change fails."""

    # Change P1 after manifest preparation.
    records = _manifest_records()
    records[20]["temperature"] = 0.9

    with pytest.raises(
        ExecutionIntegrityError,
        match="temperature",
    ):
        validate_manifest(
            records,
            _configuration(),
        )


def test_rejects_duplicate_blinded_identifier() -> None:
    """Every response identifier must be unique."""

    # Copy one blinded identifier to another request.
    records = _manifest_records()
    records[1]["blinded_response_id"] = records[0][
        "blinded_response_id"
    ]

    with pytest.raises(
        ExecutionIntegrityError,
        match="Duplicate",
    ):
        validate_manifest(
            records,
            _configuration(),
        )


def test_rejects_changed_question_order() -> None:
    """All conditions must use the same question order."""

    # Exchange two M1 question identifiers.
    records = _manifest_records()
    records[80]["question_id"], records[81][
        "question_id"
    ] = (
        records[81]["question_id"],
        records[80]["question_id"],
    )

    with pytest.raises(
        ExecutionIntegrityError,
        match="Question order",
    ):
        validate_manifest(
            records,
            _configuration(),
        )


def test_builds_legacy_request() -> None:
    """GPT-4 uses sampling and max_tokens."""

    # Build the P0 request.
    record = _manifest_records()[0]
    request = build_chat_completion_request(
        record,
        _request_policy(),
    )

    # Confirm exact legacy request controls.
    assert request["model"] == "gpt-4-0613"
    assert request["max_tokens"] == 2000
    assert "max_completion_tokens" not in request
    assert request["temperature"] == 0.1
    assert request["top_p"] == 1.0
    assert "reasoning_effort" not in request
    assert request["store"] is False
    assert request["stream"] is False
    assert request["n"] == 1
    assert request["timeout"] == 120


def test_builds_reasoning_request() -> None:
    """GPT-5 uses reasoning without sampling fields."""

    # Index 100 is the first M2 record.
    record = _manifest_records()[100]
    request = build_chat_completion_request(
        record,
        _request_policy(),
    )

    # Confirm exact reasoning-model controls.
    assert (
        request["model"]
        == "gpt-5-2025-08-07"
    )
    assert (
        request["max_completion_tokens"]
        == 2000
    )
    assert "max_tokens" not in request
    assert request["reasoning_effort"] == "low"
    assert "temperature" not in request
    assert "top_p" not in request


def test_calculates_request_and_usage_costs() -> None:
    """Reserved and returned tokens use exact rates."""

    # Use a synthetic GPT-4 request.
    record = _manifest_records()[0]

    # 100 input tokens and 2,000 reserved output
    # tokens cost 0.003 plus 0.12 USD.
    assert (
        maximum_request_cost(
            record,
            _pricing(),
        )
        == Decimal("0.123")
    )

    # 100 input and 50 actual output tokens
    # cost 0.003 plus 0.003 USD.
    assert (
        known_usage_cost(
            "gpt-4-0613",
            100,
            50,
            _pricing(),
        )
        == Decimal("0.006")
    )


def test_cost_reservation_accepts_exact_cap() -> None:
    """A projection equal to the cap is permitted."""

    assert (
        reserve_attempt_cost(
            Decimal("24.94"),
            Decimal("0.06"),
            Decimal("25.0"),
        )
        == Decimal("25.00")
    )


def test_cost_reservation_rejects_cap_breach() -> None:
    """A projected breach fails before API contact."""

    with pytest.raises(
        ExecutionCostLimitError,
        match="exceed",
    ):
        reserve_attempt_cost(
            Decimal("24.95"),
            Decimal("0.06"),
            Decimal("25.0"),
        )


def test_cost_reporting_rounds_half_up() -> None:
    """Cost reports retain six decimal places."""

    assert (
        rounded_usd(
            Decimal("17.9526795")
        )
        == Decimal("17.952680")
    )


@pytest.mark.parametrize(
    ("attempt_count", "expected_delay"),
    [
        (1, 2.0),
        (2, 4.0),
        (3, 8.0),
        (4, 8.0),
    ],
)
def test_retry_delay_follows_policy(
    attempt_count: int,
    expected_delay: float,
) -> None:
    """Backoff follows the frozen 2, 4, 8 pattern."""

    assert (
        retry_delay_seconds(
            attempt_count,
            _request_policy(),
        )
        == expected_delay
    )


def test_retry_delay_honours_retry_after() -> None:
    """A longer server delay takes priority."""

    assert (
        retry_delay_seconds(
            1,
            _request_policy(),
            9.0,
        )
        == 9.0
    )


def test_only_frozen_statuses_are_retryable() -> None:
    """No extra retry category is added silently."""

    # Status 429 is frozen as retryable.
    assert (
        retryable_status(
            _StatusError(429),
            _request_policy(),
        )
        == 429
    )

    # Status 400 is not frozen as retryable.
    assert (
        retryable_status(
            _StatusError(400),
            _request_policy(),
        )
        is None
    )

class _CheckpointClock:
    """Return predictable timezone-aware timestamps."""

    def __init__(self) -> None:
        # Start at the frozen experiment date.
        self.value = datetime(
            2026,
            9,
            23,
            tzinfo=timezone.utc,
        )

    def __call__(self) -> datetime:
        # Return the current value and advance one second.
        current = self.value
        self.value += timedelta(seconds=1)
        return current


def _checkpoint_paths(
    tmp_path: Path,
) -> CheckpointPaths:
    """Create isolated private checkpoint paths."""

    return CheckpointPaths(
        checkpoint=(
            tmp_path
            / "responses-checkpoint.jsonl"
        ),
        metadata=(
            tmp_path
            / "responses-checkpoint-metadata.json"
        ),
        responses=(
            tmp_path
            / "responses.jsonl"
        ),
    )


def _synthetic_completion(
    response_id: str = "response-1",
    *,
    content: str | None = "Synthetic answer",
    refusal: str | None = None,
    input_tokens: int = 100,
    output_tokens: int = 50,
    total_tokens: int = 150,
) -> SimpleNamespace:
    """Create one synthetic SDK completion object."""

    # Build a message containing content or refusal.
    message = SimpleNamespace(
        content=content,
        refusal=refusal,
    )

    # Build the one requested choice.
    choice = SimpleNamespace(
        message=message,
    )

    # Build exact usage metadata.
    usage = SimpleNamespace(
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
        total_tokens=total_tokens,
    )

    # Return the shape used by the OpenAI SDK.
    return SimpleNamespace(
        choices=[choice],
        id=response_id,
        model="gpt-4-0613",
        system_fingerprint="fp-synthetic",
        usage=usage,
    )


def _synthetic_response_record(
    request: dict,
    *,
    retry_count: int = 0,
) -> dict:
    """Convert a synthetic completion to a response record."""

    return _response_record_from_completion(
        request,
        _synthetic_completion(),
        "2026-09-23T10:00:00Z",
        "2026-09-23T10:00:01Z",
        retry_count,
    )


def _initialize_checkpoint(
    tmp_path: Path,
) -> tuple[
    list[dict],
    dict,
    CheckpointPaths,
]:
    """Initialize a one-request synthetic checkpoint."""

    # Use one synthetic request because this helper
    # tests checkpoint mechanics, not manifest size.
    requests = _manifest_records()[:1]
    paths = _checkpoint_paths(tmp_path)

    # Create the empty checkpoint pair.
    responses, metadata = (
        _load_or_initialize_checkpoint(
            paths,
            requests,
            _pricing(),
            (
                "benchmark-generation-"
                "evaluation-v1"
            ),
            "1" * 64,
            "2" * 64,
            "3" * 64,
            3,
            _CheckpointClock(),
        )
    )

    return responses, metadata, paths


def test_initializes_atomic_checkpoint_pair(
    tmp_path: Path,
) -> None:
    """Checkpoint and metadata start together."""

    responses, metadata, paths = (
        _initialize_checkpoint(tmp_path)
    )

    # The initial checkpoint contains no response.
    assert responses == []
    assert (
        paths.checkpoint.read_text(
            encoding="utf-8"
        )
        == ""
    )

    # Metadata records only text-free lineage.
    assert metadata[
        "checkpoint_metadata_schema_version"
    ] == "1.0"
    assert metadata[
        "completed_response_count"
    ] == 0
    assert metadata[
        "conservative_committed_cost_usd"
    ] == "0"
    assert metadata[
        "known_usage_cost_usd"
    ] == "0"

    # No temporary file remains after replacement.
    assert not list(
        tmp_path.glob("*.tmp")
    )


def test_converts_successful_completion_to_contract() -> None:
    """A successful SDK object becomes one exact record."""

    request = _manifest_records()[0]

    response = _response_record_from_completion(
        request,
        _synthetic_completion(),
        "2026-09-23T10:00:00Z",
        "2026-09-23T10:00:01Z",
        0,
    )

    # Require exactly the frozen response fields.
    assert (
        set(response)
        == REQUIRED_RESPONSE_FIELDS
    )

    # Confirm identity, usage and response status.
    assert (
        response["blinded_response_id"]
        == request["blinded_response_id"]
    )
    assert response["response_id"] == "response-1"
    assert (
        response["returned_model_id"]
        == "gpt-4-0613"
    )
    assert response["input_token_count"] == 100
    assert response["output_token_count"] == 50
    assert response["total_token_count"] == 150
    assert (
        response["completion_status"]
        == "completed"
    )
    assert (
        response["response_text"]
        == "Synthetic answer"
    )


@pytest.mark.parametrize(
    (
        "content",
        "refusal",
        "expected_status",
        "expected_text",
    ),
    [
        (
            None,
            "Synthetic refusal",
            "refused",
            "Synthetic refusal",
        ),
        (
            None,
            None,
            "empty",
            "",
        ),
    ],
)
def test_preserves_refusal_and_empty_completion(
    content: str | None,
    refusal: str | None,
    expected_status: str,
    expected_text: str,
) -> None:
    """HTTP-successful outcomes are not regenerated."""

    request = _manifest_records()[0]

    response = _response_record_from_completion(
        request,
        _synthetic_completion(
            content=content,
            refusal=refusal,
        ),
        "2026-09-23T10:00:00Z",
        "2026-09-23T10:00:01Z",
        0,
    )

    assert (
        response["completion_status"]
        == expected_status
    )
    assert (
        response["response_text"]
        == expected_text
    )


def test_rejects_inconsistent_api_usage() -> None:
    """Returned token counts must reconcile exactly."""

    request = _manifest_records()[0]

    # The total intentionally differs from input plus output.
    completion = _synthetic_completion(
        input_tokens=100,
        output_tokens=50,
        total_tokens=151,
    )

    with pytest.raises(
        ExecutionRequestError,
        match="reconcile",
    ):
        _response_record_from_completion(
            request,
            completion,
            "2026-09-23T10:00:00Z",
            "2026-09-23T10:00:01Z",
            0,
        )


def test_rejects_checkpoint_response_change(
    tmp_path: Path,
) -> None:
    """A response cannot move to another condition."""

    _, metadata, paths = (
        _initialize_checkpoint(tmp_path)
    )
    request = _manifest_records()[0]
    response = _synthetic_response_record(
        request
    )

    # Corrupt the saved experimental condition.
    response["condition_id"] = "M1"

    # Save enough matching metadata for loading to reach
    # response-contract validation first.
    metadata["attempt_counts"][
        request["blinded_response_id"]
    ] = 1
    metadata[
        "conservative_committed_cost_usd"
    ] = "0.123"
    metadata["last_event"] = "attempt_reserved"

    _atomic_write_jsonl(
        paths.checkpoint,
        [response],
    )
    _atomic_write_json(
        paths.metadata,
        metadata,
    )

    with pytest.raises(
        ExecutionIntegrityError,
        match="condition_id",
    ):
        _load_or_initialize_checkpoint(
            paths,
            [request],
            _pricing(),
            (
                "benchmark-generation-"
                "evaluation-v1"
            ),
            "1" * 64,
            "2" * 64,
            "3" * 64,
            3,
            _CheckpointClock(),
        )


def test_reconciles_checkpoint_after_interruption(
    tmp_path: Path,
) -> None:
    """A saved response can repair stale metadata."""

    _, metadata, paths = (
        _initialize_checkpoint(tmp_path)
    )
    request = _manifest_records()[0]
    response = _synthetic_response_record(
        request
    )

    # Reproduce interruption after the response checkpoint
    # was replaced but before metadata was updated.
    metadata["attempt_counts"][
        request["blinded_response_id"]
    ] = 1
    metadata[
        "conservative_committed_cost_usd"
    ] = "0.123"
    metadata["last_event"] = "attempt_reserved"

    _atomic_write_json(
        paths.metadata,
        metadata,
    )
    _atomic_write_jsonl(
        paths.checkpoint,
        [response],
    )

    # Reloading must reconcile the one-record difference.
    responses, repaired = (
        _load_or_initialize_checkpoint(
            paths,
            [request],
            _pricing(),
            (
                "benchmark-generation-"
                "evaluation-v1"
            ),
            "1" * 64,
            "2" * 64,
            "3" * 64,
            3,
            _CheckpointClock(),
        )
    )

    assert len(responses) == 1
    assert (
        repaired["completed_response_count"]
        == 1
    )
    assert (
        repaired["known_usage_cost_usd"]
        == "0.006"
    )
    assert (
        repaired["last_event"]
        == (
            "checkpoint_reconciled_"
            "after_interruption"
        )
    )


def test_rejects_changed_checkpoint_cost(
    tmp_path: Path,
) -> None:
    """Known usage is recalculated from saved records."""

    _, metadata, paths = (
        _initialize_checkpoint(tmp_path)
    )
    request = _manifest_records()[0]
    response = _synthetic_response_record(
        request
    )

    # Save an otherwise complete checkpoint.
    metadata["attempt_counts"][
        request["blinded_response_id"]
    ] = 1
    metadata[
        "completed_response_count"
    ] = 1
    metadata[
        "conservative_committed_cost_usd"
    ] = "0.123"

    # Intentionally falsify the known usage cost.
    metadata["known_usage_cost_usd"] = "0.007"
    metadata["last_event"] = (
        "response_checkpointed"
    )

    _atomic_write_jsonl(
        paths.checkpoint,
        [response],
    )
    _atomic_write_json(
        paths.metadata,
        metadata,
    )

    with pytest.raises(
        ExecutionIntegrityError,
        match="known usage cost",
    ):
        _load_or_initialize_checkpoint(
            paths,
            [request],
            _pricing(),
            (
                "benchmark-generation-"
                "evaluation-v1"
            ),
            "1" * 64,
            "2" * 64,
            "3" * 64,
            3,
            _CheckpointClock(),
        )


def test_reads_numeric_retry_after_header() -> None:
    """A numeric server delay is preserved."""

    # Create only the structured response headers used
    # by the retry helper.
    error = _StatusError(429)
    error.response = SimpleNamespace(
        headers={
            "retry-after": "3.5",
        }
    )

    assert (
        _retry_after_seconds(error)
        == 3.5
    )

class _RetryStatusError(Exception):
    """Synthetic HTTP error with structured metadata."""

    def __init__(
        self,
        status_code: int,
        *,
        retry_after: str | None = None,
        message: str = "synthetic failure",
    ) -> None:
        # Preserve the supplied text only inside the exception.
        super().__init__(message)

        # Expose the same structured fields read by the
        # execution loop.
        self.status_code = status_code
        headers = (
            {}
            if retry_after is None
            else {
                "retry-after": retry_after,
            }
        )
        self.response = SimpleNamespace(
            headers=headers
        )


class _SimulatedProcessStop(BaseException):
    """Simulate interruption outside normal exceptions."""


class _FakeCompletions:
    """Deterministic in-memory completion endpoint."""

    def __init__(
        self,
        outcomes: list[object],
    ) -> None:
        # Copy outcomes so tests cannot mutate the source.
        self.outcomes = list(outcomes)

        # Retain request dictionaries for later assertions.
        self.calls: list[dict] = []

    def create(self, **kwargs):
        """Return or raise the next synthetic outcome."""

        # Record exactly what the runner would transmit.
        self.calls.append(kwargs)

        if not self.outcomes:
            raise AssertionError(
                "No synthetic completion remains."
            )

        # Remove outcomes in deterministic order.
        outcome = self.outcomes.pop(0)

        if isinstance(outcome, BaseException):
            raise outcome

        return outcome


class _FakeClient:
    """Minimal injected client accepted by the loop."""

    def __init__(
        self,
        outcomes: list[object],
    ) -> None:
        # Expose client.chat.completions.create.
        self.completions = _FakeCompletions(
            outcomes
        )
        self.chat = SimpleNamespace(
            completions=self.completions
        )


def _execution_policy() -> dict:
    """Add the frozen attempt ceiling to test policy."""

    # Copy the existing request-policy fixture.
    policy = _request_policy()

    # Permit no more than three total attempts.
    policy["maximum_attempt_count"] = 3

    return policy


def _run_execution_core(
    tmp_path: Path,
    requests: list[dict],
    client: _FakeClient,
    *,
    hard_cap: Decimal = Decimal("25"),
    sleep_fn=lambda seconds: None,
):
    """Run the private core using synthetic inputs."""

    return _execute_checkpointed_requests(
        client=client,
        requests=requests,
        request_policy=_execution_policy(),
        pricing=_pricing(),
        hard_cap=hard_cap,
        paths=_checkpoint_paths(tmp_path),
        configuration_id=(
            "benchmark-generation-evaluation-v1"
        ),
        manifest_sha256="1" * 64,
        authorization_sha256="2" * 64,
        pricing_snapshot_sha256="3" * 64,
        now_fn=_CheckpointClock(),
        sleep_fn=sleep_fn,
    )


def test_execution_checkpoints_each_success(
    tmp_path: Path,
) -> None:
    """Sequential successes create complete outputs."""

    # Use two synthetic P0 request records.
    requests = _manifest_records()[:2]

    # Return one synthetic completion per request.
    client = _FakeClient(
        [
            _synthetic_completion(
                "response-1"
            ),
            _synthetic_completion(
                "response-2"
            ),
        ]
    )

    result = _run_execution_core(
        tmp_path,
        requests,
        client,
    )

    # Confirm sequential completion and conservative cost.
    assert result.completed_response_count == 2
    assert (
        result.generation_attempt_count_this_run
        == 2
    )
    assert (
        result.conservative_committed_cost_usd
        == Decimal("0.246")
    )
    assert (
        result.known_usage_cost_usd
        == Decimal("0.012")
    )

    # Confirm exactly two transmitted fake requests.
    assert len(client.completions.calls) == 2
    assert all(
        call["store"] is False
        for call in client.completions.calls
    )
    assert all(
        call["stream"] is False
        for call in client.completions.calls
    )

    # Confirm checkpoint and final private output agree.
    paths = _checkpoint_paths(tmp_path)
    checkpoint_records = [
        json.loads(line)
        for line in paths.checkpoint.read_text(
            encoding="utf-8"
        ).splitlines()
    ]

    assert len(checkpoint_records) == 2
    assert (
        set(checkpoint_records[0])
        == REQUIRED_RESPONSE_FIELDS
    )
    assert (
        paths.responses.read_bytes()
        == paths.checkpoint.read_bytes()
    )

    # Atomic writes must leave no temporary file.
    assert not list(
        tmp_path.glob("*.tmp")
    )


def test_retryable_failure_uses_second_attempt(
    tmp_path: Path,
) -> None:
    """A frozen 429 status retries and records it."""

    waits: list[float] = []

    # The first attempt fails with Retry-After 3,
    # and the second returns a response.
    client = _FakeClient(
        [
            _RetryStatusError(
                429,
                retry_after="3",
            ),
            _synthetic_completion(
                "response-1"
            ),
        ]
    )

    result = _run_execution_core(
        tmp_path,
        _manifest_records()[:1],
        client,
        sleep_fn=waits.append,
    )

    # Retry-After is longer than the local 2 seconds.
    assert waits == [3.0]
    assert (
        result.generation_attempt_count_this_run
        == 2
    )

    # Both possible attempts remain reserved.
    assert (
        result.conservative_committed_cost_usd
        == Decimal("0.246")
    )

    response = json.loads(
        _checkpoint_paths(tmp_path)
        .checkpoint.read_text(
            encoding="utf-8"
        )
        .splitlines()[0]
    )

    assert response["retry_count"] == 1


def test_interrupted_attempt_resumes_pending_request(
    tmp_path: Path,
) -> None:
    """A process interruption preserves earlier success."""

    requests = _manifest_records()[:2]

    # Complete the first request, then simulate the
    # process ending during the second request.
    first_client = _FakeClient(
        [
            _synthetic_completion(
                "response-1"
            ),
            _SimulatedProcessStop(),
        ]
    )

    with pytest.raises(
        _SimulatedProcessStop
    ):
        _run_execution_core(
            tmp_path,
            requests,
            first_client,
        )

    # Only the first successful response is saved.
    saved_before_resume = (
        _checkpoint_paths(tmp_path)
        .checkpoint.read_text(
            encoding="utf-8"
        )
        .splitlines()
    )

    assert len(saved_before_resume) == 1
    assert (
        json.loads(
            saved_before_resume[0]
        )["response_id"]
        == "response-1"
    )

    # Resume with one completion for the pending request.
    second_client = _FakeClient(
        [
            _synthetic_completion(
                "response-2"
            )
        ]
    )

    result = _run_execution_core(
        tmp_path,
        requests,
        second_client,
    )

    assert (
        result.previously_completed_response_count
        == 1
    )
    assert result.completed_response_count == 2
    assert (
        result.generation_attempt_count_this_run
        == 1
    )

    # The interrupted second request used attempt two.
    saved_after_resume = (
        _checkpoint_paths(tmp_path)
        .checkpoint.read_text(
            encoding="utf-8"
        )
        .splitlines()
    )

    assert (
        json.loads(
            saved_after_resume[0]
        )["response_id"]
        == "response-1"
    )
    assert (
        json.loads(
            saved_after_resume[1]
        )["response_id"]
        == "response-2"
    )
    assert (
        json.loads(
            saved_after_resume[1]
        )["retry_count"]
        == 1
    )

    # The interrupted possible charge remains reserved.
    assert (
        result.conservative_committed_cost_usd
        == Decimal("0.369")
    )


def test_non_retryable_failure_blocks_resume(
    tmp_path: Path,
) -> None:
    """An unfrozen status cannot be retried later."""

    requests = _manifest_records()[:1]

    # Include private-looking text to confirm that raw
    # exception messages are not stored.
    first_client = _FakeClient(
        [
            _RetryStatusError(
                400,
                message=(
                    "PRIVATE_TEXT_MUST_NOT_BE_SAVED"
                ),
            )
        ]
    )

    with pytest.raises(
        ExecutionRequestError,
        match="stopped",
    ):
        _run_execution_core(
            tmp_path,
            requests,
            first_client,
        )

    # Inspect only the private metadata test file.
    metadata_text = (
        _checkpoint_paths(tmp_path)
        .metadata.read_text(
            encoding="utf-8"
        )
    )

    assert (
        "PRIVATE_TEXT_MUST_NOT_BE_SAVED"
        not in metadata_text
    )
    assert '"status_code": 400' in metadata_text

    # Restarting cannot silently retry the blocked request.
    second_client = _FakeClient(
        [
            _synthetic_completion(
                "must-not-run"
            )
        ]
    )

    with pytest.raises(
        ExecutionRequestError,
        match="blocked",
    ):
        _run_execution_core(
            tmp_path,
            requests,
            second_client,
        )

    assert second_client.completions.calls == []


def test_cost_cap_blocks_before_fake_call(
    tmp_path: Path,
) -> None:
    """Cost is reserved before client transmission."""

    client = _FakeClient(
        [
            _synthetic_completion(
                "must-not-run"
            )
        ]
    )

    # One synthetic GPT-4 request needs 0.123 USD.
    with pytest.raises(
        ExecutionCostLimitError,
        match="exceed",
    ):
        _run_execution_core(
            tmp_path,
            _manifest_records()[:1],
            client,
            hard_cap=Decimal("0.122999"),
        )

    # The client method was never entered.
    assert client.completions.calls == []

    metadata = json.loads(
        _checkpoint_paths(tmp_path)
        .metadata.read_text(
            encoding="utf-8"
        )
    )

    assert (
        metadata["last_event"]
        == "cost_limit_blocked"
    )


@pytest.mark.parametrize(
    (
        "content",
        "refusal",
        "expected_status",
        "expected_text",
    ),
    [
        (
            None,
            "Synthetic refusal",
            "refused",
            "Synthetic refusal",
        ),
        (
            None,
            None,
            "empty",
            "",
        ),
    ],
)
def test_successful_non_answer_is_not_regenerated(
    tmp_path: Path,
    content: str | None,
    refusal: str | None,
    expected_status: str,
    expected_text: str,
) -> None:
    """Refusal and empty output remain one outcome."""

    client = _FakeClient(
        [
            _synthetic_completion(
                "response-1",
                content=content,
                refusal=refusal,
            )
        ]
    )

    _run_execution_core(
        tmp_path,
        _manifest_records()[:1],
        client,
    )

    response = json.loads(
        _checkpoint_paths(tmp_path)
        .responses.read_text(
            encoding="utf-8"
        )
        .splitlines()[0]
    )

    assert (
        response["completion_status"]
        == expected_status
    )
    assert (
        response["response_text"]
        == expected_text
    )
    assert len(client.completions.calls) == 1


def test_completed_run_is_idempotent(
    tmp_path: Path,
) -> None:
    """A completed response cannot be regenerated."""

    requests = _manifest_records()[:1]

    # Complete the request in the first invocation.
    _run_execution_core(
        tmp_path,
        requests,
        _FakeClient(
            [
                _synthetic_completion(
                    "response-1"
                )
            ]
        ),
    )

    # Supply another outcome that must remain unused.
    second_client = _FakeClient(
        [
            _synthetic_completion(
                "must-not-run"
            )
        ]
    )

    result = _run_execution_core(
        tmp_path,
        requests,
        second_client,
    )

    assert (
        result.previously_completed_response_count
        == 1
    )
    assert (
        result.generation_attempt_count_this_run
        == 0
    )
    assert second_client.completions.calls == []


def test_three_retryable_failures_exhaust_policy(
    tmp_path: Path,
) -> None:
    """Three failed attempts create a terminal block."""

    waits: list[float] = []

    client = _FakeClient(
        [
            _RetryStatusError(429),
            _RetryStatusError(429),
            _RetryStatusError(429),
        ]
    )

    with pytest.raises(
        ExecutionRequestError,
        match="stopped",
    ):
        _run_execution_core(
            tmp_path,
            _manifest_records()[:1],
            client,
            sleep_fn=waits.append,
        )

    # No wait follows the final exhausted attempt.
    assert waits == [2.0, 4.0]
    assert len(client.completions.calls) == 3

    metadata = json.loads(
        _checkpoint_paths(tmp_path)
        .metadata.read_text(
            encoding="utf-8"
        )
    )

    blinded_id = _manifest_records()[0][
        "blinded_response_id"
    ]

    assert (
        metadata["attempt_counts"][
            blinded_id
        ]
        == 3
    )
    assert (
        metadata[
            "conservative_committed_cost_usd"
        ]
        == "0.369"
    )
    assert (
        metadata[
            "terminal_blocked_response_id"
        ]
        == blinded_id
    )


def test_malformed_success_blocks_regeneration(
    tmp_path: Path,
) -> None:
    """A malformed successful object becomes terminal."""

    # A successful object with no choice violates the
    # response contract after one attempted request.
    malformed = SimpleNamespace(
        choices=[],
        id="response-1",
        model="gpt-4-0613",
        system_fingerprint=None,
        usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        ),
    )

    requests = _manifest_records()[:1]
    first_client = _FakeClient([malformed])

    with pytest.raises(
        ExecutionRequestError,
        match="exactly one choice",
    ):
        _run_execution_core(
            tmp_path,
            requests,
            first_client,
        )

    # A later process is not allowed to regenerate it.
    second_client = _FakeClient(
        [
            _synthetic_completion(
                "must-not-run"
            )
        ]
    )

    with pytest.raises(
        ExecutionRequestError,
        match="blocked",
    ):
        _run_execution_core(
            tmp_path,
            requests,
            second_client,
        )

    assert second_client.completions.calls == []

def _complete_enablement() -> dict:
    """Create the exact future nested enablement."""

    # Deep-copy the existing basic fixture.
    value = json.loads(
        json.dumps(_enablement())
    )

    # Link the separate $25 authorization.
    value["authorization"] = {
        "authorized_maximum_cost_usd": 25.0,
        "relative_path": (
            "configs/"
            "generation-cost-authorization.json"
        ),
        "sha256": "a" * 64,
    }

    # Link the frozen generation configuration.
    value["configuration"] = {
        "relative_path": (
            "configs/"
            "generation-evaluation-config.json"
        ),
        "sha256": "b" * 64,
    }

    # Link the committed tested implementation.
    value["implementation"] = {
        "commit": "c" * 40,
        "execution_test_count": 49,
        "full_suite_test_count": 167,
        "relative_path": (
            "src/geotech_rag/"
            "generation_execution.py"
        ),
        "sha256": "d" * 64,
        "tests_passed": True,
    }

    # Link the private 140-request manifest.
    value["manifest"] = {
        "record_count": 140,
        "relative_path": (
            "data/processed/evaluation/"
            "generation/v1/"
            "request-manifest.jsonl"
        ),
        "sha256": "e" * 64,
    }

    # Record same-day official Astra pricing evidence.
    value["pricing_recheck"] = {
        "checked_on_utc_date": "2026-09-23",
        (
            "gpt_6_astra_input_"
            "usd_per_million_tokens"
        ): 10.0,
        (
            "gpt_6_astra_output_"
            "usd_per_million_tokens"
        ): 50.0,
        "model_rates_unchanged": True,
        (
            "official_openai_"
            "documentation_used"
        ): True,
        "snapshot_relative_path": (
            "configs/"
            "generation-pricing-snapshot.json"
        ),
        "snapshot_sha256": "f" * 64,
        "source_domain": "developers.openai.com",
    }

    return value


class _FakeModels:
    """Synthetic model-list endpoint."""

    def __init__(
        self,
        model_ids: list[str],
        *,
        failure: Exception | None = None,
    ) -> None:
        # Keep only public model identifiers.
        self.model_ids = model_ids
        self.failure = failure
        self.call_count = 0

    def list(self):
        """Return the synthetic model list."""

        self.call_count += 1

        if self.failure is not None:
            raise self.failure

        return SimpleNamespace(
            data=[
                SimpleNamespace(id=model_id)
                for model_id in self.model_ids
            ]
        )


class _ModelOnlyClient:
    """Client exposing only the model-list endpoint."""

    def __init__(
        self,
        model_ids: list[str],
        *,
        failure: Exception | None = None,
    ) -> None:
        self.models = _FakeModels(
            model_ids,
            failure=failure,
        )


def _all_required_model_ids() -> list[str]:
    """Return the four distinct frozen model IDs."""

    return [
        "gpt-4-0613",
        "gpt-4.1-2025-04-14",
        "gpt-5-2025-08-07",
        "gpt-6-astra",
    ]


def test_accepts_complete_same_day_enablement() -> None:
    """Exact same-day pricing evidence passes."""

    execution_module._validate_enablement_contract(
        _complete_enablement(),
        "benchmark-generation-evaluation-v1",
        _authorization(),
        _pricing(),
        "2026-09-23",
    )


def test_rejects_expired_enablement_date() -> None:
    """Yesterday's switch cannot authorize today's run."""

    with pytest.raises(
        ExecutionIntegrityError,
        match="current UTC date",
    ):
        (
            execution_module
            ._validate_enablement_contract(
                _complete_enablement(),
                (
                    "benchmark-generation-"
                    "evaluation-v1"
                ),
                _authorization(),
                _pricing(),
                "2026-09-24",
            )
        )


def test_rejects_changed_astra_price() -> None:
    """A changed rate requires a new cost boundary."""

    value = _complete_enablement()
    value["pricing_recheck"][
        (
            "gpt_6_astra_output_"
            "usd_per_million_tokens"
        )
    ] = 55.0

    with pytest.raises(
        ExecutionIntegrityError,
        match="output pricing changed",
    ):
        (
            execution_module
            ._validate_enablement_contract(
                value,
                (
                    "benchmark-generation-"
                    "evaluation-v1"
                ),
                _authorization(),
                _pricing(),
                "2026-09-23",
            )
        )


def test_model_preflight_accepts_all_models() -> None:
    """All four public model IDs pass preflight."""

    client = _ModelOnlyClient(
        _all_required_model_ids()
    )

    required = (
        execution_module
        .preflight_required_models(
            client,
            _configuration(),
        )
    )

    assert required == tuple(
        _all_required_model_ids()
    )
    assert client.models.call_count == 1


def test_model_preflight_rejects_missing_model() -> None:
    """Missing Astra stops without substitution."""

    client = _ModelOnlyClient(
        _all_required_model_ids()[:-1]
    )

    with pytest.raises(
        execution_module.ExecutionAuthorizationError,
        match="unavailable",
    ):
        (
            execution_module
            .preflight_required_models(
                client,
                _configuration(),
            )
        )

    assert client.models.call_count == 1


def test_model_preflight_hides_raw_failure() -> None:
    """Model-list exception text is not propagated."""

    client = _ModelOnlyClient(
        [],
        failure=RuntimeError(
            "PRIVATE_ERROR_TEXT"
        ),
    )

    with pytest.raises(
        execution_module.ExecutionAuthorizationError,
    ) as caught:
        (
            execution_module
            .preflight_required_models(
                client,
                _configuration(),
            )
        )

    assert (
        "PRIVATE_ERROR_TEXT"
        not in str(caught.value)
    )


def test_absent_action_flag_fails_before_client_use() -> None:
    """No file or client is used without the flag."""

    # The object intentionally has no client interface.
    client = object()

    with pytest.raises(
        execution_module.ExecutionAuthorizationError,
        match="action flag",
    ):
        (
            execution_module
            .execute_generation_with_client(
                client=client,
                config_path=(
                    "does-not-exist.json"
                ),
                enablement_path=(
                    "also-does-not-exist.json"
                ),
                project_root=".",
                explicit_execute_paid_generation=False,
            )
        )


def test_model_preflight_occurs_before_manifest_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing model stops before prompt text is read."""

    # Return only the public configuration needed
    # by the model preflight.
    synthetic_inputs = SimpleNamespace(
        configuration=_configuration(),
    )

    monkeypatch.setattr(
        execution_module,
        "load_execution_inputs",
        lambda *args, **kwargs: synthetic_inputs,
    )

    manifest_opened = False

    def fail_if_manifest_opened(path):
        # Record an ordering failure if this is called.
        nonlocal manifest_opened
        manifest_opened = True
        raise AssertionError(
            "Manifest was opened before preflight."
        )

    monkeypatch.setattr(
        execution_module,
        "load_manifest",
        fail_if_manifest_opened,
    )

    # Omit Astra from the public model list.
    client = _ModelOnlyClient(
        _all_required_model_ids()[:-1]
    )

    with pytest.raises(
        execution_module.ExecutionAuthorizationError,
        match="unavailable",
    ):
        (
            execution_module
            .execute_generation_with_client(
                client=client,
                config_path="synthetic-config.json",
                enablement_path=(
                    "synthetic-enablement.json"
                ),
                project_root=tmp_path,
                explicit_execute_paid_generation=True,
            )
        )

    assert manifest_opened is False


def test_dirty_worktree_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Any public uncommitted file blocks execution."""

    # Return a successful Git command containing one
    # untracked public file.
    monkeypatch.setattr(
        execution_module,
        "_git_run",
        lambda root, arguments: SimpleNamespace(
            returncode=0,
            stdout="?? public-file.txt\n",
            stderr="",
        ),
    )

    with pytest.raises(
        ExecutionIntegrityError,
        match="clean",
    ):
        execution_module._require_clean_worktree(
            tmp_path
        )


def test_changed_implementation_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Post-commit runner changes invalidate enablement."""

    # Create an implementation path inside the project.
    implementation_path = (
        tmp_path
        / "src"
        / "geotech_rag"
        / "generation_execution.py"
    )
    implementation_path.parent.mkdir(
        parents=True
    )
    implementation_path.write_text(
        "synthetic implementation\n",
        encoding="utf-8",
    )

    # The tracked-file check is tested separately by
    # the real execution audit, so isolate history here.
    monkeypatch.setattr(
        execution_module,
        "_require_tracked",
        lambda *args, **kwargs: None,
    )

    # The commit is an ancestor, but the implementation
    # differs from that commit.
    results = iter(
        [
            SimpleNamespace(
                returncode=0,
                stdout="",
                stderr="",
            ),
            SimpleNamespace(
                returncode=1,
                stdout="",
                stderr="",
            ),
        ]
    )

    monkeypatch.setattr(
        execution_module,
        "_git_run",
        lambda root, arguments: next(results),
    )

    with pytest.raises(
        ExecutionIntegrityError,
        match="changed",
    ):
        (
            execution_module
            ._validate_implementation_git_state(
                tmp_path,
                implementation_path,
                "c" * 40,
            )
        )


def test_cli_refuses_before_gate_or_client(
    monkeypatch,
    capsys,
):
    """The absent action flag must stop all later work."""

    def unexpected_call(*args, **kwargs):
        raise AssertionError(
            "The execution gate must not be called."
        )

    # If the CLI incorrectly continues, either patched
    # boundary will fail the test immediately.
    monkeypatch.setattr(
        execution_module,
        "load_execution_inputs",
        unexpected_call,
    )

    status = execution_module.main(
        [],
        client_factory=unexpected_call,
    )

    output = capsys.readouterr().out

    assert status == 2
    assert "Generation execution: REFUSED" in output
    assert "OpenAI client created: False" in output
    assert "API request made: False" in output


def test_cli_invalid_enablement_blocks_before_client(
    monkeypatch,
    capsys,
):
    """A failed public gate must prevent client creation."""

    state = {
        "client_created": False,
    }

    def reject_enablement(*args, **kwargs):
        raise execution_module.ExecutionAuthorizationError(
            "Synthetic public enablement rejection."
        )

    def forbidden_client_factory():
        state["client_created"] = True
        raise AssertionError(
            "The client must not be created."
        )

    monkeypatch.setattr(
        execution_module,
        "load_execution_inputs",
        reject_enablement,
    )

    status = execution_module.main(
        ["--execute-paid-generation"],
        client_factory=forbidden_client_factory,
    )

    output = capsys.readouterr().out

    assert status == 1
    assert state["client_created"] is False
    assert "Generation execution: BLOCKED" in output
    assert (
        "Synthetic public enablement rejection."
        in output
    )


def test_create_openai_client_disables_sdk_retries(
    monkeypatch,
):
    """The real-client helper must freeze SDK retries at zero."""

    constructor_arguments = {}

    class FakeOpenAI:
        """A constructor-only replacement for the SDK client."""

        def __init__(self, **kwargs):
            constructor_arguments.update(kwargs)

    # Replace the imported OpenAI package only inside this
    # test. No real client or credential access can occur.
    fake_openai_module = type(
        "FakeOpenAIModule",
        (),
        {
            "OpenAI": FakeOpenAI,
        },
    )()

    monkeypatch.setitem(
        __import__("sys").modules,
        "openai",
        fake_openai_module,
    )

    client = (
        execution_module._create_openai_client()
    )

    assert isinstance(client, FakeOpenAI)
    assert constructor_arguments == {
        "max_retries": 0,
    }


def test_cli_sanitizes_client_factory_failure(
    monkeypatch,
    capsys,
):
    """Credential or constructor details must not be printed."""

    private_failure_detail = (
        "SYNTHETIC_PRIVATE_CREDENTIAL_DETAIL"
    )

    # The public gate succeeds, allowing the test to reach
    # the injected constructor without reading real files.
    monkeypatch.setattr(
        execution_module,
        "load_execution_inputs",
        lambda *args, **kwargs: object(),
    )

    def failing_client_factory():
        raise RuntimeError(
            private_failure_detail
        )

    status = execution_module.main(
        ["--execute-paid-generation"],
        client_factory=failing_client_factory,
    )

    output = capsys.readouterr().out

    assert status == 1
    assert "Generation execution: BLOCKED" in output
    assert (
        "The OpenAI client could not be created"
        in output
    )
    assert private_failure_detail not in output


def test_cli_success_reports_only_operational_evidence(
    monkeypatch,
    capsys,
):
    """A successful CLI run must remain text-free."""

    observed = {}
    private_text = (
        "SYNTHETIC_PRIVATE_BENCHMARK_TEXT"
    )

    class FakeClient:
        """A client marker containing text that must not leak."""

        hidden_text = private_text

    fake_client = FakeClient()

    def fake_load_execution_inputs(
        config_path,
        enablement_path,
        project_root,
    ):
        observed["preliminary_gate"] = (
            config_path,
            enablement_path,
            project_root,
        )
        return object()

    def fake_execute_generation_with_client(
        **kwargs,
    ):
        observed["execution_arguments"] = kwargs

        return execution_module.ExecutionRunResult(
            planned_response_count=140,
            previously_completed_response_count=20,
            completed_response_count=140,
            generation_attempt_count_this_run=120,
            conservative_committed_cost_usd=(
                execution_module.Decimal(
                    "17.952680"
                )
            ),
            known_usage_cost_usd=(
                execution_module.Decimal(
                    "12.345678"
                )
            ),
            response_sha256=("f" * 64),
        )

    monkeypatch.setattr(
        execution_module,
        "load_execution_inputs",
        fake_load_execution_inputs,
    )
    monkeypatch.setattr(
        execution_module,
        "execute_generation_with_client",
        fake_execute_generation_with_client,
    )

    status = execution_module.main(
        [
            "--config",
            "synthetic-config.json",
            "--enablement",
            "synthetic-enablement.json",
            "--project-root",
            "synthetic-root",
            "--execute-paid-generation",
        ],
        client_factory=lambda: fake_client,
    )

    output = capsys.readouterr().out

    assert status == 0
    assert observed["preliminary_gate"] == (
        "synthetic-config.json",
        "synthetic-enablement.json",
        "synthetic-root",
    )

    execution_arguments = observed[
        "execution_arguments"
    ]

    assert execution_arguments["client"] is fake_client
    assert (
        execution_arguments["config_path"]
        == "synthetic-config.json"
    )
    assert (
        execution_arguments["enablement_path"]
        == "synthetic-enablement.json"
    )
    assert (
        execution_arguments["project_root"]
        == "synthetic-root"
    )
    assert (
        execution_arguments[
            "explicit_execute_paid_generation"
        ]
        is True
    )

    assert "Generation execution: PASSED" in output
    assert "Planned response count: 140" in output
    assert (
        "Previously completed response count: 20"
        in output
    )
    assert "Completed response count: 140" in output
    assert "Generation attempts this run: 120" in output
    assert "$17.952680" in output
    assert "$12.345678" in output
    assert ("f" * 64) in output

    # The CLI must not print client contents, questions,
    # retrieved context or generated responses.
    assert private_text not in output
