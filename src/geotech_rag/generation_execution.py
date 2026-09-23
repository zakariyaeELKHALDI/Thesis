"""Guarded execution utilities for the frozen generation benchmark.

The offline preparation module remains unchanged. This module validates the
separate budget authorization and future execution-enablement record, builds
exact Chat Completions requests, and provides deterministic cost and retry
helpers.

Importing this module never creates an API client or request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from geotech_rag.generation_evaluation import (
    load_generation_evaluation_config,
)


class GenerationExecutionError(RuntimeError):
    """Base error for guarded benchmark execution."""


class ExecutionAuthorizationError(
    GenerationExecutionError
):
    """Raised when paid execution is not explicitly enabled."""


class ExecutionIntegrityError(GenerationExecutionError):
    """Raised when an execution input fails an integrity check."""


class ExecutionCostLimitError(GenerationExecutionError):
    """Raised before an attempt could exceed the cost ceiling."""


class ExecutionRequestError(GenerationExecutionError):
    """Raised when a request violates the frozen contract."""


EXPECTED_CONDITION_IDS = (
    "P0",
    "P1",
    "P2",
    "P3",
    "M1",
    "M2",
    "M3",
)

REQUIRED_RESPONSE_FIELDS = frozenset(
    {
        "record_schema_version",
        "configuration_id",
        "condition_id",
        "blinded_response_id",
        "question_id",
        "retrieved_evidence_sha256",
        "prompt_sha256",
        "requested_model_id",
        "returned_model_id",
        "temperature",
        "top_p",
        "reasoning_effort",
        "response_id",
        "request_started_at_utc",
        "response_received_at_utc",
        "input_token_count",
        "output_token_count",
        "total_token_count",
        "system_fingerprint",
        "retry_count",
        "completion_status",
        "response_text",
    }
)

REQUIRED_ENABLEMENT_KEYS = frozenset(
    {
        "authorization",
        "configuration",
        "configuration_id",
        "enabled_on_utc_date",
        "execution_enablement_schema_version",
        "execution_status",
        "implementation",
        "manifest",
        "pricing_recheck",
        "runtime_controls",
        "state_before_enablement",
    }
)


def calculate_file_sha256(path: str | Path) -> str:
    """Calculate a stable SHA-256 file fingerprint."""

    # Create the fingerprint object.
    digest = hashlib.sha256()

    # Process the file in blocks to avoid unnecessary memory use.
    with Path(path).open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    # Return the completed hexadecimal fingerprint.
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    """Raise a stable integrity error for a failed condition."""

    if not condition:
        raise ExecutionIntegrityError(message)


def _load_json(
    path: Path,
    label: str,
) -> dict[str, Any]:
    """Load one JSON object using a stable error boundary."""

    try:
        # Read and parse the requested JSON file.
        value = json.loads(
            path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ExecutionIntegrityError(
            f"Could not load {label}."
        ) from error

    # Execution contracts must always be JSON objects.
    if not isinstance(value, dict):
        raise ExecutionIntegrityError(
            f"{label} must be a JSON object."
        )

    return value


def _resolve_relative(
    root: Path,
    relative: str,
    label: str,
) -> Path:
    """Resolve a repository-relative path without allowing escape."""

    # Reject absolute and parent-traversal paths.
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ExecutionIntegrityError(
            f"{label} must be a safe relative path."
        )

    # Resolve the final path against the project root.
    resolved = (root / path).resolve()

    # Confirm that the result remains inside the project.
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ExecutionIntegrityError(
            f"{label} escapes the project root."
        ) from error

    return resolved


def _verify_file(
    root: Path,
    relative_path: str,
    expected_sha256: str,
    label: str,
) -> Path:
    """Resolve a linked file and verify its frozen fingerprint."""

    # Resolve the safe project-relative path.
    path = _resolve_relative(
        root,
        relative_path,
        label,
    )

    # Require the linked artifact to exist.
    if not path.is_file():
        raise ExecutionIntegrityError(
            f"Missing {label}."
        )

    # Reject any content change after authorization.
    if calculate_file_sha256(path) != expected_sha256:
        raise ExecutionIntegrityError(
            f"Wrong SHA-256 for {label}."
        )

    return path


def _decimal(
    value: object,
    label: str,
) -> Decimal:
    """Convert a JSON value to an exact non-negative decimal."""

    try:
        # Convert through text to avoid binary float arithmetic.
        converted = Decimal(str(value))
    except Exception as error:
        raise ExecutionIntegrityError(
            f"Invalid decimal value for {label}."
        ) from error

    # Cost values must be finite and non-negative.
    if not converted.is_finite() or converted < 0:
        raise ExecutionIntegrityError(
            f"Invalid decimal value for {label}."
        )

    return converted


def rounded_usd(value: Decimal) -> Decimal:
    """Round cost evidence to six decimal places."""

    return value.quantize(
        Decimal("0.000001"),
        rounding=ROUND_HALF_UP,
    )


def load_execution_enablement(
    path: str | Path,
) -> dict[str, Any]:
    """Load and validate the future fail-closed execution switch."""

    # Load the enablement record without creating a client.
    enablement = _load_json(
        Path(path),
        "execution enablement",
    )

    # Reject missing or unexpected top-level fields.
    _require(
        set(enablement) == REQUIRED_ENABLEMENT_KEYS,
        (
            "Unexpected execution-enablement "
            "top-level schema."
        ),
    )

    # Require the first frozen schema version.
    _require(
        (
            enablement[
                "execution_enablement_schema_version"
            ]
            == "1.0"
        ),
        (
            "Unexpected execution-enablement "
            "schema version."
        ),
    )

    # The later enablement record must explicitly switch execution on.
    _require(
        enablement["execution_status"] == "ENABLED",
        "Paid execution is not enabled.",
    )

    # Validate the fail-closed runtime controls.
    controls = enablement["runtime_controls"]
    _require(
        isinstance(controls, dict),
        "runtime_controls must be a mapping.",
    )
    _require(
        controls.get("paid_generation_enabled") is True,
        "Paid execution is disabled.",
    )
    _require(
        (
            controls.get(
                "explicit_execute_flag_required"
            )
            is True
        ),
        "Explicit execution flag is required.",
    )
    _require(
        controls.get("require_clean_worktree") is True,
        "Clean worktree verification is required.",
    )
    _require(
        controls.get("parallel_request_count") == 1,
        "Requests must remain sequential.",
    )
    _require(
        (
            controls.get(
                "model_availability_preflight_required"
            )
            is True
        ),
        "Model availability preflight is required.",
    )

    # Require committed and tested implementation evidence.
    implementation = enablement["implementation"]
    _require(
        isinstance(implementation, dict),
        "implementation must be a mapping.",
    )
    _require(
        implementation.get("tests_passed") is True,
        "Execution tests did not pass.",
    )
    _require(
        (
            isinstance(
                implementation.get("commit"),
                str,
            )
            and len(implementation["commit"]) == 40
        ),
        "Implementation commit must be a full Git hash.",
    )

    # Confirm that enablement was created before any generation.
    before = enablement["state_before_enablement"]
    _require(
        isinstance(before, dict),
        (
            "state_before_enablement must be "
            "a mapping."
        ),
    )

    for key in (
        "api_request_made",
        "generation_response_evidence_created",
        "ground_truth_accessed",
        "paid_generation_performed",
    ):
        _require(
            before.get(key) is False,
            (
                "Unexpected pre-enablement state "
                f"for {key}."
            ),
        )

    return enablement


def validate_authorization(
    authorization: Mapping[str, Any],
) -> Decimal:
    """Validate budget approval without enabling execution."""

    # Require the frozen authorization schema.
    _require(
        (
            authorization.get(
                "authorization_schema_version"
            )
            == "1.0"
        ),
        "Unexpected authorization schema version.",
    )

    # Require explicit researcher budget approval.
    _require(
        (
            authorization.get(
                "budget_authorization_status"
            )
            == "AUTHORIZED"
        ),
        "Generation cost is not authorized.",
    )

    # The authorization file must not also enable execution.
    _require(
        (
            authorization.get(
                "execution_authorization_status"
            )
            == "NOT_ENABLED"
        ),
        (
            "The budget record must not enable "
            "execution."
        ),
    )

    # Validate the authorization controls.
    controls = authorization.get(
        "execution_controls"
    )
    _require(
        isinstance(controls, dict),
        (
            "Authorization controls must be "
            "a mapping."
        ),
    )
    _require(
        controls.get("paid_generation_enabled")
        is False,
        (
            "The budget record must keep paid "
            "generation disabled."
        ),
    )

    # The unrealistic full retry exposure stays unauthorized.
    _require(
        (
            authorization.get(
                "theoretical_full_retry_exposure_authorized"
            )
            is False
        ),
        (
            "Full retry exposure must remain "
            "unauthorized."
        ),
    )

    # Validate the authorized experimental scope.
    scope = authorization.get(
        "authorization_scope"
    )
    _require(
        isinstance(scope, dict),
        "Authorization scope must be a mapping.",
    )
    _require(
        scope.get("planned_request_count") == 140,
        (
            "Expected authorization for 140 "
            "planned requests."
        ),
    )
    _require(
        (
            scope.get(
                "maximum_completed_response_count"
            )
            == 140
        ),
        (
            "Expected at most 140 completed "
            "responses."
        ),
    )
    _require(
        scope.get("extra_requests_authorized")
        is False,
        "Extra requests must remain unauthorized.",
    )
    _require(
        scope.get("model_substitution_authorized")
        is False,
        (
            "Model substitution must remain "
            "unauthorized."
        ),
    )
    _require(
        (
            scope.get(
                "parameter_substitution_authorized"
            )
            is False
        ),
        (
            "Parameter substitution must remain "
            "unauthorized."
        ),
    )
    _require(
        (
            scope.get(
                "unplanned_conditions_authorized"
            )
            is False
        ),
        (
            "Unplanned conditions must remain "
            "unauthorized."
        ),
    )

    # Load the exact approved hard cap.
    hard_cap = _decimal(
        authorization.get(
            "authorized_maximum_cost_usd"
        ),
        "authorized maximum cost",
    )
    _require(
        hard_cap == Decimal("25.0"),
        "Unexpected authorized maximum cost.",
    )

    # Confirm the runtime cap matches the approval.
    runtime_cap = _decimal(
        controls.get(
            "runtime_hard_cost_cap_usd"
        ),
        "runtime hard cap",
    )
    _require(
        runtime_cap == hard_cap,
        (
            "Runtime and authorized cost caps "
            "differ."
        ),
    )

    return hard_cap


def load_manifest(
    path: str | Path,
) -> tuple[dict[str, Any], ...]:
    """Load the private manifest without displaying prompt text."""

    records: list[dict[str, Any]] = []

    try:
        # Read the private JSONL file into individual lines.
        lines = Path(path).read_text(
            encoding="utf-8"
        ).splitlines()
    except OSError as error:
        raise ExecutionIntegrityError(
            (
                "Could not load the private "
                "request manifest."
            )
        ) from error

    # Parse each non-empty manifest line.
    for line_number, line in enumerate(
        lines,
        1,
    ):
        if not line.strip():
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ExecutionIntegrityError(
                (
                    f"Manifest line {line_number} "
                    "is invalid JSON."
                )
            ) from error

        if not isinstance(record, dict):
            raise ExecutionIntegrityError(
                (
                    f"Manifest line {line_number} "
                    "is not an object."
                )
            )

        records.append(record)

    # The frozen benchmark requires exactly 140 requests.
    _require(
        len(records) == 140,
        (
            "Expected exactly 140 private "
            "request records."
        ),
    )

    return tuple(records)


def validate_manifest(
    records: Sequence[Mapping[str, Any]],
    configuration: Mapping[str, Any],
) -> None:
    """Cross-check every request against the public contract."""

    # Index the seven frozen conditions in their saved order.
    conditions = {
        condition["condition_id"]: condition
        for condition in configuration["conditions"]
    }

    _require(
        tuple(conditions) == EXPECTED_CONDITION_IDS,
        "Unexpected condition order.",
    )

    # Prepare text-free counters and ordering checks.
    counts = {
        condition_id: 0
        for condition_id in EXPECTED_CONDITION_IDS
    }
    blinded_ids: set[str] = set()
    question_ids_by_condition: dict[
        str,
        list[str],
    ] = {
        condition_id: []
        for condition_id in EXPECTED_CONDITION_IDS
    }

    # Validate every private request without printing content.
    for record in records:
        condition_id = record.get("condition_id")
        _require(
            condition_id in conditions,
            "Unknown manifest condition.",
        )
        condition = conditions[str(condition_id)]
        counts[str(condition_id)] += 1

        # Verify the public configuration lineage.
        _require(
            (
                record.get("configuration_id")
                == configuration["configuration_id"]
            ),
            (
                "Manifest configuration identifier "
                "changed."
            ),
        )
        _require(
            (
                record.get("requested_model_id")
                == condition["model_id"]
            ),
            "Manifest model identifier changed.",
        )

        # Verify all experimental parameters.
        for key in (
            "retrieval_enabled",
            "retrieval_depth_k",
            "temperature",
            "top_p",
            "reasoning_effort",
        ):
            _require(
                record.get(key) == condition[key],
                f"Manifest field {key} changed.",
            )

        # Verify the frozen endpoint and token controls.
        _require(
            (
                record.get("endpoint")
                == "chat.completions"
            ),
            "Unexpected endpoint.",
        )
        _require(
            (
                record.get("context_window_passed")
                is True
            ),
            "Context audit did not pass.",
        )
        _require(
            (
                record.get(
                    "logical_max_output_tokens"
                )
                == 2000
            ),
            "Unexpected output-token allowance.",
        )
        _require(
            (
                record.get("reserved_output_tokens")
                == 2000
            ),
            "Unexpected reserved output tokens.",
        )
        _require(
            (
                isinstance(
                    record.get(
                        "estimated_input_tokens"
                    ),
                    int,
                )
                and record[
                    "estimated_input_tokens"
                ]
                > 0
            ),
            (
                "Invalid estimated input-token "
                "count."
            ),
        )

        # Require the two prepared prompt messages.
        _require(
            (
                isinstance(
                    record.get("system_prompt"),
                    str,
                )
                and isinstance(
                    record.get("user_prompt"),
                    str,
                )
            ),
            "Manifest prompt text is invalid.",
        )

        # Require one unique blinded response identifier.
        blinded_id = record.get(
            "blinded_response_id"
        )
        _require(
            (
                isinstance(blinded_id, str)
                and len(blinded_id) == 64
            ),
            "Invalid blinded response identifier.",
        )
        _require(
            blinded_id not in blinded_ids,
            (
                "Duplicate blinded response "
                "identifier."
            ),
        )
        blinded_ids.add(blinded_id)

        # Retain only question identifiers for order checks.
        question_id = record.get("question_id")
        _require(
            (
                isinstance(question_id, str)
                and bool(question_id)
            ),
            "Invalid question identifier.",
        )
        question_ids_by_condition[
            str(condition_id)
        ].append(question_id)

    # Require twenty requests for each condition.
    _require(
        set(counts.values()) == {20},
        "Expected 20 requests per condition.",
    )

    # Require identical question order in all conditions.
    reference_order = question_ids_by_condition[
        EXPECTED_CONDITION_IDS[0]
    ]

    for condition_id in EXPECTED_CONDITION_IDS[1:]:
        _require(
            (
                question_ids_by_condition[
                    condition_id
                ]
                == reference_order
            ),
            (
                "Question order differs between "
                "conditions."
            ),
        )


def build_chat_completion_request(
    record: Mapping[str, Any],
    request_policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one exact SDK request from a prepared record."""

    # Only the frozen Chat Completions endpoint is valid.
    _require(
        (
            record.get("endpoint")
            == "chat.completions"
        ),
        "Unexpected endpoint.",
    )

    # Build the common request shared by all models.
    kwargs: dict[str, Any] = {
        "model": record["requested_model_id"],
        "messages": [
            {
                "role": "system",
                "content": record[
                    "system_prompt"
                ],
            },
            {
                "role": "user",
                "content": record[
                    "user_prompt"
                ],
            },
        ],
        "n": request_policy["n"],
        "stream": request_policy["stream"],
        "store": request_policy[
            "api_response_storage_requested"
        ],
        "timeout": request_policy[
            "timeout_seconds"
        ],
    }

    # Apply the model-specific output-token parameter.
    output_parameter = record[
        "output_token_parameter"
    ]
    _require(
        output_parameter
        in {
            "max_tokens",
            "max_completion_tokens",
        },
        "Unexpected output-token parameter.",
    )
    kwargs[output_parameter] = record[
        "logical_max_output_tokens"
    ]

    # Apply sampling controls only to non-reasoning models.
    if record["reasoning_effort"] is None:
        _require(
            record["temperature"] is not None,
            "Legacy temperature is missing.",
        )
        _require(
            record["top_p"] is not None,
            "Legacy top_p is missing.",
        )
        kwargs["temperature"] = record[
            "temperature"
        ]
        kwargs["top_p"] = record["top_p"]

    # Apply reasoning effort without temperature or top_p.
    else:
        _require(
            record["temperature"] is None,
            (
                "Reasoning temperature must be "
                "omitted."
            ),
        )
        _require(
            record["top_p"] is None,
            "Reasoning top_p must be omitted.",
        )
        kwargs["reasoning_effort"] = record[
            "reasoning_effort"
        ]

    return kwargs


def maximum_request_cost(
    record: Mapping[str, Any],
    pricing: Mapping[str, Any],
) -> Decimal:
    """Calculate the conservative maximum for one attempt."""

    # Locate the exact rate for the requested model.
    model_id = record["requested_model_id"]
    rates = pricing.get("model_rates", {})

    _require(
        model_id in rates,
        (
            "No frozen pricing exists for the "
            "requested model."
        ),
    )

    # Load both rates as exact decimal values.
    input_rate = _decimal(
        rates[model_id][
            "input_usd_per_million_tokens"
        ],
        "input token rate",
    )
    output_rate = _decimal(
        rates[model_id][
            "output_usd_per_million_tokens"
        ],
        "output token rate",
    )

    # Use the conservative prepared token counts.
    input_tokens = _decimal(
        record["estimated_input_tokens"],
        "estimated input tokens",
    )
    output_tokens = _decimal(
        record["reserved_output_tokens"],
        "reserved output tokens",
    )

    # Return the unrounded cost for later accumulation.
    return (
        input_tokens * input_rate
        + output_tokens * output_rate
    ) / Decimal("1000000")


def known_usage_cost(
    requested_model_id: str,
    input_token_count: int,
    output_token_count: int,
    pricing: Mapping[str, Any],
) -> Decimal:
    """Estimate known cost from returned API usage."""

    # Locate the approved rate for the requested model.
    rates = pricing.get("model_rates", {})
    _require(
        requested_model_id in rates,
        "No frozen pricing exists for usage.",
    )

    # Convert both token rates to exact decimals.
    input_rate = _decimal(
        rates[requested_model_id][
            "input_usd_per_million_tokens"
        ],
        "input token rate",
    )
    output_rate = _decimal(
        rates[requested_model_id][
            "output_usd_per_million_tokens"
        ],
        "output token rate",
    )

    # Calculate the known usage cost without rounding.
    return (
        Decimal(input_token_count) * input_rate
        + Decimal(output_token_count) * output_rate
    ) / Decimal("1000000")


def reserve_attempt_cost(
    committed_cost: Decimal,
    request_cost: Decimal,
    hard_cap: Decimal,
) -> Decimal:
    """Reserve a whole worst-case attempt before API contact."""

    # Add the complete possible request cost before sending.
    projected = committed_cost + request_cost

    # Refuse before the authorization could be exceeded.
    if projected > hard_cap:
        raise ExecutionCostLimitError(
            (
                "The next request attempt would "
                "exceed the authorized cost ceiling."
            )
        )

    return projected


def retry_delay_seconds(
    completed_attempt_count: int,
    request_policy: Mapping[str, Any],
    retry_after_seconds: float | None = None,
) -> float:
    """Calculate frozen backoff and honour Retry-After."""

    # At least one failed attempt must exist before waiting.
    if completed_attempt_count < 1:
        raise ExecutionRequestError(
            (
                "Completed attempt count must be "
                "positive."
            )
        )

    # Calculate the frozen exponential delay.
    initial = float(
        request_policy["retry_initial_seconds"]
    )
    multiplier = float(
        request_policy["retry_multiplier"]
    )
    maximum = float(
        request_policy["retry_max_seconds"]
    )
    delay = min(
        initial
        * multiplier
        ** (completed_attempt_count - 1),
        maximum,
    )

    # Honour a longer non-negative server instruction.
    if retry_after_seconds is not None:
        if retry_after_seconds < 0:
            raise ExecutionRequestError(
                "Retry-After cannot be negative."
            )
        delay = max(
            delay,
            retry_after_seconds,
        )

    return delay


def retryable_status(
    error: BaseException,
    request_policy: Mapping[str, Any],
) -> int | None:
    """Return a permitted retryable HTTP status."""

    # Read only the structured status, not private error text.
    status = getattr(
        error,
        "status_code",
        None,
    )

    # Permit only statuses already frozen in the config.
    if (
        isinstance(status, int)
        and status
        in request_policy[
            "retryable_http_statuses"
        ]
    ):
        return status

    return None

@dataclass(frozen=True)
class CheckpointPaths:
    """Private paths used by the resumable execution core."""

    checkpoint: Path
    metadata: Path
    responses: Path


CHECKPOINT_METADATA_KEYS = frozenset(
    {
        "attempt_counts",
        "authorization_sha256",
        "checkpoint_metadata_schema_version",
        "completed_response_count",
        "configuration_id",
        "conservative_committed_cost_usd",
        "known_usage_cost_usd",
        "last_event",
        "last_failure",
        "manifest_sha256",
        "pricing_snapshot_sha256",
        "terminal_blocked_response_id",
        "updated_at_utc",
    }
)


def _canonical_json_line(
    record: Mapping[str, Any],
) -> str:
    """Serialize one private record deterministically."""

    return json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _timestamp(
    now_fn: Callable[[], datetime],
) -> str:
    """Return an explicit UTC timestamp."""

    # Obtain time through an injectable clock for testing.
    value = now_fn()

    # Reject ambiguous local or naive timestamps.
    if (
        value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ExecutionIntegrityError(
            (
                "Execution clock must return a "
                "timezone-aware value."
            )
        )

    # Normalize the timestamp to UTC.
    return (
        value.astimezone(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _decimal_text(value: Decimal) -> str:
    """Serialize an exact decimal without exponent notation."""

    return format(value, "f")


def _atomic_write_text(
    path: Path,
    payload: str,
) -> None:
    """Replace one private file atomically."""

    # Create only the required private parent directory.
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Write beside the final destination.
    temporary = path.with_name(
        path.name + ".tmp"
    )

    with temporary.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        # Save and flush the complete new content.
        handle.write(payload)
        handle.flush()

        # Ask the operating system to persist the bytes.
        os.fsync(handle.fileno())

    # Atomically replace the destination.
    temporary.replace(path)


def _atomic_write_json(
    path: Path,
    record: Mapping[str, Any],
) -> None:
    """Write deterministic private JSON atomically."""

    # Use readable and stable metadata formatting.
    payload = (
        json.dumps(
            record,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    _atomic_write_text(
        path,
        payload,
    )


def _atomic_write_jsonl(
    path: Path,
    records: Sequence[Mapping[str, Any]],
) -> None:
    """Write a deterministic private JSONL checkpoint."""

    # Serialize each record independently.
    payload = "".join(
        _canonical_json_line(record) + "\n"
        for record in records
    )

    _atomic_write_text(
        path,
        payload,
    )


def _load_jsonl(
    path: Path,
    label: str,
) -> list[dict[str, Any]]:
    """Load private JSONL without displaying saved text."""

    try:
        # Read the complete private file.
        lines = path.read_text(
            encoding="utf-8"
        ).splitlines()
    except OSError as error:
        raise ExecutionIntegrityError(
            f"Could not load {label}."
        ) from error

    records: list[dict[str, Any]] = []

    # Parse each non-empty JSONL record.
    for line_number, line in enumerate(
        lines,
        1,
    ):
        if not line.strip():
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ExecutionIntegrityError(
                (
                    f"{label} line {line_number} "
                    "is invalid JSON."
                )
            ) from error

        if not isinstance(record, dict):
            raise ExecutionIntegrityError(
                (
                    f"{label} line {line_number} "
                    "is not an object."
                )
            )

        records.append(record)

    return records


def _initial_checkpoint_metadata(
    configuration_id: str,
    manifest_sha256: str,
    authorization_sha256: str,
    pricing_snapshot_sha256: str,
    now_fn: Callable[[], datetime],
) -> dict[str, Any]:
    """Create text-free metadata before any attempt."""

    return {
        "attempt_counts": {},
        "authorization_sha256": (
            authorization_sha256
        ),
        "checkpoint_metadata_schema_version": (
            "1.0"
        ),
        "completed_response_count": 0,
        "configuration_id": configuration_id,
        "conservative_committed_cost_usd": (
            "0"
        ),
        "known_usage_cost_usd": "0",
        "last_event": "initialized",
        "last_failure": None,
        "manifest_sha256": manifest_sha256,
        "pricing_snapshot_sha256": (
            pricing_snapshot_sha256
        ),
        "terminal_blocked_response_id": None,
        "updated_at_utc": _timestamp(now_fn),
    }


def _validate_response_record(
    response: Mapping[str, Any],
    request: Mapping[str, Any],
    maximum_attempt_count: int,
) -> None:
    """Validate one saved response against its request."""

    # Require the exact private response schema.
    _require(
        set(response) == REQUIRED_RESPONSE_FIELDS,
        "Unexpected response-record schema.",
    )

    # Require every experiment identifier and parameter
    # to match the prepared request.
    for key in (
        "configuration_id",
        "condition_id",
        "blinded_response_id",
        "question_id",
        "retrieved_evidence_sha256",
        "prompt_sha256",
        "requested_model_id",
        "temperature",
        "top_p",
        "reasoning_effort",
    ):
        _require(
            response.get(key) == request.get(key),
            f"Response field {key} changed.",
        )

    _require(
        (
            response.get("record_schema_version")
            == "1.0"
        ),
        "Unexpected response schema version.",
    )

    # Validate API identity fields without requiring a
    # particular mutable returned-model value.
    _require(
        isinstance(
            response.get("returned_model_id"),
            str,
        ),
        "Returned model ID is invalid.",
    )
    _require(
        isinstance(
            response.get("response_id"),
            str,
        ),
        "Response ID is invalid.",
    )

    # Require both timestamps to be present.
    for key in (
        "request_started_at_utc",
        "response_received_at_utc",
    ):
        _require(
            (
                isinstance(response.get(key), str)
                and bool(response[key])
            ),
            f"Response field {key} is invalid.",
        )

    # Validate non-negative token-usage fields.
    for key in (
        "input_token_count",
        "output_token_count",
        "total_token_count",
    ):
        _require(
            (
                isinstance(response.get(key), int)
                and response[key] >= 0
            ),
            f"Response field {key} is invalid.",
        )

    # Require the returned usage totals to reconcile.
    _require(
        (
            response["total_token_count"]
            == response["input_token_count"]
            + response["output_token_count"]
        ),
        "Response token counts do not reconcile.",
    )

    # A service fingerprint may legitimately be absent.
    fingerprint = response.get(
        "system_fingerprint"
    )
    _require(
        (
            fingerprint is None
            or isinstance(fingerprint, str)
        ),
        "System fingerprint is invalid.",
    )

    # Retry count is zero for the first attempt.
    retry_count = response.get("retry_count")
    _require(
        (
            isinstance(retry_count, int)
            and 0
            <= retry_count
            < maximum_attempt_count
        ),
        "Response retry count is invalid.",
    )

    # Preserve refusals and empty successful responses
    # instead of selectively regenerating them.
    _require(
        response.get("completion_status")
        in {
            "completed",
            "refused",
            "empty",
        },
        "Unexpected completion status.",
    )
    _require(
        isinstance(
            response.get("response_text"),
            str,
        ),
        "Response text is invalid.",
    )


def _known_checkpoint_cost(
    responses: Sequence[Mapping[str, Any]],
    pricing: Mapping[str, Any],
) -> Decimal:
    """Recalculate known cost from saved usage."""

    total = Decimal("0")

    # Add the returned usage cost for each response.
    for response in responses:
        total += known_usage_cost(
            str(
                response["requested_model_id"]
            ),
            int(response["input_token_count"]),
            int(response["output_token_count"]),
            pricing,
        )

    return total


def _load_or_initialize_checkpoint(
    paths: CheckpointPaths,
    requests: Sequence[Mapping[str, Any]],
    pricing: Mapping[str, Any],
    configuration_id: str,
    manifest_sha256: str,
    authorization_sha256: str,
    pricing_snapshot_sha256: str,
    maximum_attempt_count: int,
    now_fn: Callable[[], datetime],
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Create or validate private checkpoint state."""

    # The checkpoint and metadata form one logical pair.
    checkpoint_exists = paths.checkpoint.exists()
    metadata_exists = paths.metadata.exists()

    if checkpoint_exists != metadata_exists:
        raise ExecutionIntegrityError(
            (
                "Checkpoint and metadata must "
                "exist together."
            )
        )

    # Initialize an empty atomic checkpoint before
    # the first possible request attempt.
    if not checkpoint_exists:
        responses: list[dict[str, Any]] = []

        metadata = _initial_checkpoint_metadata(
            configuration_id,
            manifest_sha256,
            authorization_sha256,
            pricing_snapshot_sha256,
            now_fn,
        )

        _atomic_write_jsonl(
            paths.checkpoint,
            responses,
        )
        _atomic_write_json(
            paths.metadata,
            metadata,
        )

        return responses, metadata

    # Load an existing resumable checkpoint.
    responses = _load_jsonl(
        paths.checkpoint,
        "response checkpoint",
    )

    _require(
        len(responses) <= len(requests),
        "Checkpoint contains too many responses.",
    )

    # Responses must remain a prefix of the manifest.
    for index, response in enumerate(responses):
        _validate_response_record(
            response,
            requests[index],
            maximum_attempt_count,
        )

    # Load and validate text-free checkpoint metadata.
    metadata = _load_json(
        paths.metadata,
        "checkpoint metadata",
    )

    _require(
        set(metadata) == CHECKPOINT_METADATA_KEYS,
        (
            "Unexpected checkpoint-metadata "
            "schema."
        ),
    )
    _require(
        (
            metadata[
                "checkpoint_metadata_schema_version"
            ]
            == "1.0"
        ),
        "Unexpected checkpoint schema version.",
    )
    _require(
        (
            metadata["configuration_id"]
            == configuration_id
        ),
        "Checkpoint configuration changed.",
    )
    _require(
        (
            metadata["manifest_sha256"]
            == manifest_sha256
        ),
        "Checkpoint manifest lineage changed.",
    )
    _require(
        (
            metadata["authorization_sha256"]
            == authorization_sha256
        ),
        (
            "Checkpoint authorization lineage "
            "changed."
        ),
    )
    _require(
        (
            metadata["pricing_snapshot_sha256"]
            == pricing_snapshot_sha256
        ),
        "Checkpoint pricing lineage changed.",
    )

    # Recover the narrow interruption window where
    # the checkpoint was replaced but metadata was not.
    saved_completed_count = metadata[
        "completed_response_count"
    ]

    if (
        saved_completed_count
        == len(responses) - 1
        and metadata["last_event"]
        == "attempt_reserved"
    ):
        reconciled_known_cost = (
            _known_checkpoint_cost(
                responses,
                pricing,
            )
        )
        reconciled_committed_cost = max(
            _decimal(
                metadata[
                    "conservative_committed_cost_usd"
                ],
                "checkpoint committed cost",
            ),
            reconciled_known_cost,
        )

        metadata[
            "completed_response_count"
        ] = len(responses)
        metadata[
            "known_usage_cost_usd"
        ] = _decimal_text(
            reconciled_known_cost
        )
        metadata[
            "conservative_committed_cost_usd"
        ] = _decimal_text(
            reconciled_committed_cost
        )
        metadata["last_event"] = (
            "checkpoint_reconciled_after_interruption"
        )
        metadata["last_failure"] = None
        metadata["updated_at_utc"] = _timestamp(
            now_fn
        )

        _atomic_write_json(
            paths.metadata,
            metadata,
        )
    else:
        _require(
            (
                saved_completed_count
                == len(responses)
            ),
            (
                "Checkpoint count does not "
                "reconcile."
            ),
        )

    # Validate accumulated attempt counts.
    attempt_counts = metadata["attempt_counts"]
    _require(
        isinstance(attempt_counts, dict),
        (
            "Checkpoint attempt counts must be "
            "a mapping."
        ),
    )

    valid_ids = {
        str(request["blinded_response_id"])
        for request in requests
    }

    for response_id, count in (
        attempt_counts.items()
    ):
        _require(
            response_id in valid_ids,
            (
                "Checkpoint contains an unknown "
                "attempt identifier."
            ),
        )
        _require(
            (
                isinstance(count, int)
                and 1
                <= count
                <= maximum_attempt_count
            ),
            "Checkpoint attempt count is invalid.",
        )

    # A completed response must match its exact
    # persisted number of attempts.
    for response in responses:
        response_id = str(
            response["blinded_response_id"]
        )
        _require(
            (
                attempt_counts.get(response_id)
                == response["retry_count"] + 1
            ),
            (
                "Checkpoint attempt and retry "
                "counts differ."
            ),
        )

    # Recalculate cost evidence rather than trusting
    # mutable metadata alone.
    committed = _decimal(
        metadata[
            "conservative_committed_cost_usd"
        ],
        "checkpoint committed cost",
    )
    known = _decimal(
        metadata["known_usage_cost_usd"],
        "checkpoint known cost",
    )

    _require(
        (
            known
            == _known_checkpoint_cost(
                responses,
                pricing,
            )
        ),
        (
            "Checkpoint known usage cost "
            "changed."
        ),
    )
    _require(
        known <= committed,
        (
            "Known usage cost exceeds conservative "
            "committed cost."
        ),
    )

    # A terminal failure prevents silent regeneration.
    blocked = metadata[
        "terminal_blocked_response_id"
    ]
    _require(
        (
            blocked is None
            or blocked in valid_ids
        ),
        "Checkpoint terminal block is invalid.",
    )

    return responses, metadata


def _get_value(
    value: Any,
    name: str,
    default: Any = None,
) -> Any:
    """Read a field from an SDK object or mapping."""

    # Synthetic tests may use dictionaries.
    if isinstance(value, Mapping):
        return value.get(name, default)

    # Real SDK objects expose typed attributes.
    return getattr(
        value,
        name,
        default,
    )


def _response_record_from_completion(
    request: Mapping[str, Any],
    completion: Any,
    request_started_at_utc: str,
    response_received_at_utc: str,
    retry_count: int,
) -> dict[str, Any]:
    """Convert one SDK completion to the contract."""

    # The frozen request asks for exactly one choice.
    choices = _get_value(
        completion,
        "choices",
    )

    if (
        not isinstance(choices, Sequence)
        or len(choices) != 1
    ):
        raise ExecutionRequestError(
            (
                "The API did not return exactly "
                "one choice."
            )
        )

    # Read the assistant message from the one choice.
    message = _get_value(
        choices[0],
        "message",
    )
    content = _get_value(
        message,
        "content",
    )
    refusal = _get_value(
        message,
        "refusal",
    )

    # Treat all HTTP-successful outcomes as the one
    # generated result, including refusals or empties.
    if isinstance(content, str) and content:
        response_text = content
        completion_status = "completed"
    elif isinstance(refusal, str) and refusal:
        response_text = refusal
        completion_status = "refused"
    else:
        response_text = ""
        completion_status = "empty"

    # Read exact usage returned by the API.
    usage = _get_value(
        completion,
        "usage",
    )
    input_tokens = _get_value(
        usage,
        "prompt_tokens",
    )
    output_tokens = _get_value(
        usage,
        "completion_tokens",
    )
    total_tokens = _get_value(
        usage,
        "total_tokens",
    )

    # Usage fields are required for cost evidence.
    for label, token_count in (
        ("input", input_tokens),
        ("output", output_tokens),
        ("total", total_tokens),
    ):
        if (
            not isinstance(token_count, int)
            or token_count < 0
        ):
            raise ExecutionRequestError(
                (
                    f"The API {label} token count "
                    "is invalid."
                )
            )

    if (
        total_tokens
        != input_tokens + output_tokens
    ):
        raise ExecutionRequestError(
            (
                "The API token counts do not "
                "reconcile."
            )
        )

    # Require stable response identity fields.
    response_id = _get_value(
        completion,
        "id",
    )
    returned_model = _get_value(
        completion,
        "model",
    )

    if (
        not isinstance(response_id, str)
        or not response_id
    ):
        raise ExecutionRequestError(
            "The API response ID is invalid."
        )

    if (
        not isinstance(returned_model, str)
        or not returned_model
    ):
        raise ExecutionRequestError(
            "The returned model ID is invalid."
        )

    # Build the exact private response-record contract.
    return {
        "record_schema_version": "1.0",
        "configuration_id": request[
            "configuration_id"
        ],
        "condition_id": request[
            "condition_id"
        ],
        "blinded_response_id": request[
            "blinded_response_id"
        ],
        "question_id": request[
            "question_id"
        ],
        "retrieved_evidence_sha256": request[
            "retrieved_evidence_sha256"
        ],
        "prompt_sha256": request[
            "prompt_sha256"
        ],
        "requested_model_id": request[
            "requested_model_id"
        ],
        "returned_model_id": returned_model,
        "temperature": request["temperature"],
        "top_p": request["top_p"],
        "reasoning_effort": request[
            "reasoning_effort"
        ],
        "response_id": response_id,
        "request_started_at_utc": (
            request_started_at_utc
        ),
        "response_received_at_utc": (
            response_received_at_utc
        ),
        "input_token_count": input_tokens,
        "output_token_count": output_tokens,
        "total_token_count": total_tokens,
        "system_fingerprint": _get_value(
            completion,
            "system_fingerprint",
        ),
        "retry_count": retry_count,
        "completion_status": completion_status,
        "response_text": response_text,
    }


def _retry_after_seconds(
    error: BaseException,
) -> float | None:
    """Read a numeric Retry-After header safely."""

    # OpenAI status errors expose the HTTP response.
    response = getattr(
        error,
        "response",
        None,
    )
    headers = getattr(
        response,
        "headers",
        None,
    )

    if headers is None:
        return None

    # HTTP libraries may preserve either header case.
    value = (
        headers.get("retry-after")
        or headers.get("Retry-After")
    )

    if value is None:
        return None

    try:
        # Numeric Retry-After values represent seconds.
        seconds = float(value)
    except (TypeError, ValueError):
        return None

    # Ignore invalid negative server values.
    return seconds if seconds >= 0 else None

@dataclass(frozen=True)
class ExecutionRunResult:
    """Text-free summary of one execution invocation."""

    planned_response_count: int
    previously_completed_response_count: int
    completed_response_count: int
    generation_attempt_count_this_run: int
    conservative_committed_cost_usd: Decimal
    known_usage_cost_usd: Decimal
    response_sha256: str


def _save_terminal_failure(
    metadata: dict[str, Any],
    paths: CheckpointPaths,
    response_id: str,
    failure_type: str,
    status_code: int | None,
    now_fn: Callable[[], datetime],
) -> None:
    """Persist a text-free terminal block."""

    # Record only structured failure metadata.
    metadata["last_event"] = (
        "terminal_failure"
    )
    metadata["last_failure"] = {
        "failure_type": failure_type,
        "status_code": status_code,
    }

    # Prevent this response from being regenerated
    # during a later resumed process.
    metadata[
        "terminal_blocked_response_id"
    ] = response_id
    metadata["updated_at_utc"] = _timestamp(
        now_fn
    )

    _atomic_write_json(
        paths.metadata,
        metadata,
    )


def _execute_checkpointed_requests(
    *,
    client: Any,
    requests: Sequence[Mapping[str, Any]],
    request_policy: Mapping[str, Any],
    pricing: Mapping[str, Any],
    hard_cap: Decimal,
    paths: CheckpointPaths,
    configuration_id: str,
    manifest_sha256: str,
    authorization_sha256: str,
    pricing_snapshot_sha256: str,
    now_fn: Callable[[], datetime] = (
        lambda: datetime.now(timezone.utc)
    ),
    sleep_fn: Callable[[float], None] = (
        time.sleep
    ),
) -> ExecutionRunResult:
    """Execute validated requests with atomic checkpoints."""

    # Load the frozen maximum attempt count.
    maximum_attempts = int(
        request_policy[
            "maximum_attempt_count"
        ]
    )

    # Initialize or validate all prior private state.
    responses, metadata = (
        _load_or_initialize_checkpoint(
            paths,
            requests,
            pricing,
            configuration_id,
            manifest_sha256,
            authorization_sha256,
            pricing_snapshot_sha256,
            maximum_attempts,
            now_fn,
        )
    )

    previously_completed = len(responses)

    # A terminally blocked record cannot be retried
    # merely by restarting the process.
    blocked = metadata[
        "terminal_blocked_response_id"
    ]

    if blocked is not None:
        raise ExecutionRequestError(
            (
                "Checkpoint contains a terminally "
                "blocked request."
            )
        )

    # A completed final file may only coexist with a
    # complete checkpoint.
    if (
        paths.responses.exists()
        and len(responses) < len(requests)
    ):
        raise ExecutionIntegrityError(
            (
                "Final responses exist beside an "
                "incomplete checkpoint."
            )
        )

    # Reject an already excessive resumed cost state.
    if (
        _decimal(
            metadata[
                "conservative_committed_cost_usd"
            ],
            "checkpoint committed cost",
        )
        > hard_cap
    ):
        raise ExecutionCostLimitError(
            (
                "Checkpoint cost already exceeds "
                "the authorized ceiling."
            )
        )

    generation_attempts = 0

    # Resume from the first request without a response.
    for request in requests[len(responses):]:
        blinded_id = str(
            request["blinded_response_id"]
        )

        # Continue only while the current request has
        # attempts permitted by the frozen policy.
        while True:
            prior_attempts = int(
                metadata["attempt_counts"].get(
                    blinded_id,
                    0,
                )
            )

            # Prevent process restarts from exceeding
            # the total three-attempt allowance.
            if prior_attempts >= maximum_attempts:
                _save_terminal_failure(
                    metadata,
                    paths,
                    blinded_id,
                    (
                        "maximum_attempts_"
                        "exhausted"
                    ),
                    None,
                    now_fn,
                )
                raise ExecutionRequestError(
                    (
                        "Maximum request attempts "
                        "were exhausted."
                    )
                )

            # Calculate the complete worst-case cost
            # for this individual attempt.
            committed = _decimal(
                metadata[
                    "conservative_committed_cost_usd"
                ],
                "checkpoint committed cost",
            )
            attempt_cost = maximum_request_cost(
                request,
                pricing,
            )

            try:
                # Reserve the possible cost before
                # transmitting any private prompt.
                reserved = reserve_attempt_cost(
                    committed,
                    attempt_cost,
                    hard_cap,
                )
            except ExecutionCostLimitError:
                # Persist a text-free cost stop.
                metadata["last_event"] = (
                    "cost_limit_blocked"
                )
                metadata["last_failure"] = None
                metadata["updated_at_utc"] = (
                    _timestamp(now_fn)
                )
                _atomic_write_json(
                    paths.metadata,
                    metadata,
                )
                raise

            attempt_number = prior_attempts + 1

            # Persist the attempt and its cost before
            # calling the injected client.
            metadata["attempt_counts"][
                blinded_id
            ] = attempt_number
            metadata[
                "conservative_committed_cost_usd"
            ] = _decimal_text(reserved)
            metadata["last_event"] = (
                "attempt_reserved"
            )
            metadata["last_failure"] = None
            metadata["updated_at_utc"] = (
                _timestamp(now_fn)
            )

            _atomic_write_json(
                paths.metadata,
                metadata,
            )

            # Record request timing immediately before
            # the injected client method is called.
            started = _timestamp(now_fn)
            generation_attempts += 1

            try:
                # Build only the already frozen request
                # fields and send one sequential request.
                completion = (
                    client.chat.completions.create(
                        **build_chat_completion_request(
                            request,
                            request_policy,
                        )
                    )
                )
            except Exception as error:
                # Classify only the frozen HTTP statuses.
                status = retryable_status(
                    error,
                    request_policy,
                )
                observed_status = getattr(
                    error,
                    "status_code",
                    None,
                )

                # Save no raw exception message because it
                # may contain unnecessary request details.
                metadata["last_failure"] = {
                    "failure_type": (
                        type(error).__name__
                    ),
                    "status_code": observed_status,
                }
                metadata["last_event"] = (
                    "retryable_failure"
                    if status is not None
                    else "non_retryable_failure"
                )
                metadata["updated_at_utc"] = (
                    _timestamp(now_fn)
                )

                _atomic_write_json(
                    paths.metadata,
                    metadata,
                )

                # Retry only a frozen status with attempts
                # remaining under the same cost ceiling.
                if (
                    status is not None
                    and attempt_number
                    < maximum_attempts
                ):
                    delay = retry_delay_seconds(
                        attempt_number,
                        request_policy,
                        _retry_after_seconds(error),
                    )
                    sleep_fn(delay)
                    continue

                # A non-permitted or exhausted failure
                # blocks automatic regeneration.
                _save_terminal_failure(
                    metadata,
                    paths,
                    blinded_id,
                    type(error).__name__,
                    (
                        observed_status
                        if isinstance(
                            observed_status,
                            int,
                        )
                        else None
                    ),
                    now_fn,
                )

                raise ExecutionRequestError(
                    (
                        "Generation stopped after a "
                        "non-permitted or exhausted "
                        "failure."
                    )
                ) from error

            # Record receipt time before response parsing.
            received = _timestamp(now_fn)

            try:
                # Convert the successful SDK object to the
                # exact private response contract.
                response = (
                    _response_record_from_completion(
                        request,
                        completion,
                        started,
                        received,
                        attempt_number - 1,
                    )
                )
            except ExecutionRequestError:
                # A successful but malformed API result must
                # not be regenerated automatically.
                _save_terminal_failure(
                    metadata,
                    paths,
                    blinded_id,
                    "response_contract_failure",
                    None,
                    now_fn,
                )
                raise

            # Save the successful response atomically before
            # updating the secondary metadata file.
            responses.append(response)

            _atomic_write_jsonl(
                paths.checkpoint,
                responses,
            )

            # Recalculate known cost from all stored usage.
            known_cost = _known_checkpoint_cost(
                responses,
                pricing,
            )

            # Keep the larger of reserved exposure and known
            # returned usage as the conservative commitment.
            effective_committed = max(
                _decimal(
                    metadata[
                        "conservative_committed_cost_usd"
                    ],
                    "checkpoint committed cost",
                ),
                known_cost,
            )

            metadata[
                "completed_response_count"
            ] = len(responses)
            metadata[
                "known_usage_cost_usd"
            ] = _decimal_text(known_cost)
            metadata[
                "conservative_committed_cost_usd"
            ] = _decimal_text(
                effective_committed
            )
            metadata["last_event"] = (
                "response_checkpointed"
            )
            metadata["last_failure"] = None
            metadata["updated_at_utc"] = (
                _timestamp(now_fn)
            )

            _atomic_write_json(
                paths.metadata,
                metadata,
            )

            # Stop immediately if actual returned usage
            # unexpectedly exceeds the reserved boundary.
            if effective_committed > hard_cap:
                _save_terminal_failure(
                    metadata,
                    paths,
                    blinded_id,
                    (
                        "returned_usage_exceeded_"
                        "authorized_cap"
                    ),
                    None,
                    now_fn,
                )
                raise ExecutionCostLimitError(
                    (
                        "Returned usage exceeded the "
                        "authorized cost ceiling."
                    )
                )

            # Continue to the next prepared request.
            break

    # A successful run must finish the entire sequence.
    _require(
        len(responses) == len(requests),
        (
            "Generation did not complete every "
            "planned response."
        ),
    )

    # Never overwrite an existing final response file.
    if paths.responses.exists():
        existing = _load_jsonl(
            paths.responses,
            "final private responses",
        )
        _require(
            existing == responses,
            (
                "Existing final responses differ "
                "from the checkpoint."
            ),
        )
    else:
        _atomic_write_jsonl(
            paths.responses,
            responses,
        )

    # Mark the private checkpoint as complete.
    metadata["last_event"] = (
        "generation_complete"
    )
    metadata["updated_at_utc"] = _timestamp(
        now_fn
    )

    _atomic_write_json(
        paths.metadata,
        metadata,
    )

    # Return text-free execution evidence only.
    return ExecutionRunResult(
        planned_response_count=len(requests),
        previously_completed_response_count=(
            previously_completed
        ),
        completed_response_count=len(responses),
        generation_attempt_count_this_run=(
            generation_attempts
        ),
        conservative_committed_cost_usd=(
            _decimal(
                metadata[
                    "conservative_committed_cost_usd"
                ],
                "checkpoint committed cost",
            )
        ),
        known_usage_cost_usd=_decimal(
            metadata["known_usage_cost_usd"],
            "checkpoint known cost",
        ),
        response_sha256=calculate_file_sha256(
            paths.responses
        ),
    )

@dataclass(frozen=True)
class ExecutionInputs:
    """Validated public lineage and private paths."""

    root: Path
    configuration: dict[str, Any]
    authorization: dict[str, Any]
    enablement: dict[str, Any]
    pricing: dict[str, Any]
    hard_cap: Decimal
    manifest_path: Path
    manifest_sha256: str
    authorization_sha256: str
    pricing_snapshot_sha256: str
    checkpoint_paths: CheckpointPaths


AUTHORIZATION_LINK_KEYS = frozenset(
    {
        "authorized_maximum_cost_usd",
        "relative_path",
        "sha256",
    }
)

CONFIGURATION_LINK_KEYS = frozenset(
    {
        "relative_path",
        "sha256",
    }
)

IMPLEMENTATION_LINK_KEYS = frozenset(
    {
        "commit",
        "execution_test_count",
        "full_suite_test_count",
        "relative_path",
        "sha256",
        "tests_passed",
    }
)

MANIFEST_LINK_KEYS = frozenset(
    {
        "record_count",
        "relative_path",
        "sha256",
    }
)

PRICING_RECHECK_KEYS = frozenset(
    {
        "checked_on_utc_date",
        (
            "gpt_6_astra_input_"
            "usd_per_million_tokens"
        ),
        (
            "gpt_6_astra_output_"
            "usd_per_million_tokens"
        ),
        "model_rates_unchanged",
        "official_openai_documentation_used",
        "snapshot_relative_path",
        "snapshot_sha256",
        "source_domain",
    }
)

RUNTIME_CONTROL_KEYS = frozenset(
    {
        "explicit_execute_flag_required",
        (
            "model_availability_"
            "preflight_required"
        ),
        "paid_generation_enabled",
        "parallel_request_count",
        "require_clean_worktree",
    }
)

PRE_ENABLEMENT_STATE_KEYS = frozenset(
    {
        "api_request_made",
        "generation_response_evidence_created",
        "ground_truth_accessed",
        "paid_generation_performed",
    }
)


def _require_mapping_keys(
    value: object,
    expected_keys: frozenset[str],
    label: str,
) -> Mapping[str, Any]:
    """Require one exact nested mapping schema."""

    _require(
        isinstance(value, Mapping),
        f"{label} must be a mapping.",
    )
    _require(
        set(value) == expected_keys,
        f"Unexpected {label} schema.",
    )

    return value


def _git_run(
    root: Path,
    arguments: Sequence[str],
) -> subprocess.CompletedProcess[str]:
    """Run one local Git check safely."""

    try:
        # Capture Git output so failures can be reported
        # without printing unrelated repository details.
        return subprocess.run(
            [
                "git",
                *arguments,
            ],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise ExecutionIntegrityError(
            "Could not run a required Git check."
        ) from error


def _project_relative(
    root: Path,
    path: Path,
    label: str,
) -> str:
    """Return a safe repository-relative path."""

    try:
        relative = (
            path.resolve()
            .relative_to(root.resolve())
        )
    except ValueError as error:
        raise ExecutionIntegrityError(
            f"{label} is outside the project root."
        ) from error

    return relative.as_posix()


def _require_clean_worktree(
    root: Path,
) -> None:
    """Require no public uncommitted change."""

    result = _git_run(
        root,
        [
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
    )

    _require(
        result.returncode == 0,
        "Could not inspect the Git worktree.",
    )
    _require(
        not result.stdout.strip(),
        (
            "The Git worktree must be clean "
            "before generation."
        ),
    )


def _require_tracked(
    root: Path,
    path: Path,
    label: str,
) -> None:
    """Require a public artifact to be tracked."""

    relative = _project_relative(
        root,
        path,
        label,
    )

    result = _git_run(
        root,
        [
            "ls-files",
            "--error-unmatch",
            "--",
            relative,
        ],
    )

    _require(
        result.returncode == 0,
        f"{label} must be tracked by Git.",
    )


def _require_ignored(
    root: Path,
    path: Path,
    label: str,
) -> None:
    """Require a private path to be ignored."""

    relative = _project_relative(
        root,
        path,
        label,
    )

    result = _git_run(
        root,
        [
            "check-ignore",
            "--quiet",
            "--",
            relative,
        ],
    )

    _require(
        result.returncode == 0,
        f"{label} must be ignored by Git.",
    )


def _validate_implementation_git_state(
    root: Path,
    implementation_path: Path,
    implementation_commit: str,
) -> None:
    """Require committed and unchanged runner code."""

    # The runner must be present in Git.
    _require_tracked(
        root,
        implementation_path,
        "generation implementation",
    )

    # Later enablement commits are permitted only when
    # the implementation commit remains in their history.
    ancestor = _git_run(
        root,
        [
            "merge-base",
            "--is-ancestor",
            implementation_commit,
            "HEAD",
        ],
    )

    _require(
        ancestor.returncode == 0,
        (
            "The authorized implementation commit "
            "is not an ancestor of HEAD."
        ),
    )

    # The implementation file must be unchanged since
    # the commit recorded by the enablement.
    relative = _project_relative(
        root,
        implementation_path,
        "generation implementation",
    )

    unchanged = _git_run(
        root,
        [
            "diff",
            "--quiet",
            implementation_commit,
            "--",
            relative,
        ],
    )

    _require(
        unchanged.returncode == 0,
        (
            "The generation implementation changed "
            "after its authorized commit."
        ),
    )


def _validate_enablement_contract(
    enablement: Mapping[str, Any],
    configuration_id: str,
    authorization: Mapping[str, Any],
    pricing: Mapping[str, Any],
    current_utc_date: str,
) -> None:
    """Validate the complete nested enablement."""

    # Require the same experimental configuration.
    _require(
        (
            enablement["configuration_id"]
            == configuration_id
        ),
        "Enablement configuration changed.",
    )

    # The enablement expires at the end of its UTC date.
    _require(
        (
            enablement["enabled_on_utc_date"]
            == current_utc_date
        ),
        (
            "Execution enablement must be created "
            "on the current UTC date."
        ),
    )

    # Validate authorization linkage and cap.
    authorization_link = _require_mapping_keys(
        enablement["authorization"],
        AUTHORIZATION_LINK_KEYS,
        "authorization link",
    )

    _require(
        (
            _decimal(
                authorization_link[
                    "authorized_maximum_cost_usd"
                ],
                (
                    "enablement authorized "
                    "maximum cost"
                ),
            )
            == _decimal(
                authorization[
                    "authorized_maximum_cost_usd"
                ],
                "authorization maximum cost",
            )
        ),
        (
            "Enablement and authorization cost "
            "caps differ."
        ),
    )

    # Validate exact nested schemas.
    _require_mapping_keys(
        enablement["configuration"],
        CONFIGURATION_LINK_KEYS,
        "configuration link",
    )

    implementation = _require_mapping_keys(
        enablement["implementation"],
        IMPLEMENTATION_LINK_KEYS,
        "implementation link",
    )

    _require(
        implementation["tests_passed"] is True,
        "Execution tests did not pass.",
    )

    for key in (
        "execution_test_count",
        "full_suite_test_count",
    ):
        _require(
            (
                isinstance(
                    implementation[key],
                    int,
                )
                and implementation[key] > 0
            ),
            f"Invalid {key}.",
        )

    manifest = _require_mapping_keys(
        enablement["manifest"],
        MANIFEST_LINK_KEYS,
        "manifest link",
    )

    _require(
        manifest["record_count"] == 140,
        "Enablement must link 140 requests.",
    )

    _require_mapping_keys(
        enablement["runtime_controls"],
        RUNTIME_CONTROL_KEYS,
        "runtime controls",
    )

    _require_mapping_keys(
        enablement["state_before_enablement"],
        PRE_ENABLEMENT_STATE_KEYS,
        "pre-enablement state",
    )

    # Validate the same-day mutable price recheck.
    pricing_recheck = _require_mapping_keys(
        enablement["pricing_recheck"],
        PRICING_RECHECK_KEYS,
        "pricing recheck",
    )

    _require(
        (
            pricing_recheck[
                "checked_on_utc_date"
            ]
            == current_utc_date
        ),
        (
            "Mutable pricing must be rechecked "
            "on the execution date."
        ),
    )
    _require(
        (
            pricing_recheck[
                (
                    "official_openai_"
                    "documentation_used"
                )
            ]
            is True
        ),
        (
            "Official OpenAI pricing "
            "documentation must be used."
        ),
    )
    _require(
        (
            pricing_recheck["source_domain"]
            == "developers.openai.com"
        ),
        "Unexpected pricing source domain.",
    )
    _require(
        (
            pricing_recheck[
                "model_rates_unchanged"
            ]
            is True
        ),
        (
            "Changed model pricing requires a "
            "new cost audit and authorization."
        ),
    )

    # Compare the mutable Astra rates to the
    # already authorized pricing snapshot.
    astra_rates = pricing["model_rates"][
        "gpt-6-astra"
    ]

    _require(
        (
            _decimal(
                pricing_recheck[
                    (
                        "gpt_6_astra_input_"
                        "usd_per_million_tokens"
                    )
                ],
                "rechecked Astra input rate",
            )
            == _decimal(
                astra_rates[
                    (
                        "input_usd_per_"
                        "million_tokens"
                    )
                ],
                "frozen Astra input rate",
            )
        ),
        "Rechecked Astra input pricing changed.",
    )

    _require(
        (
            _decimal(
                pricing_recheck[
                    (
                        "gpt_6_astra_output_"
                        "usd_per_million_tokens"
                    )
                ],
                "rechecked Astra output rate",
            )
            == _decimal(
                astra_rates[
                    (
                        "output_usd_per_"
                        "million_tokens"
                    )
                ],
                "frozen Astra output rate",
            )
        ),
        "Rechecked Astra output pricing changed.",
    )


def load_execution_inputs(
    config_path: str | Path,
    enablement_path: str | Path,
    project_root: str | Path,
    *,
    now_fn: Callable[[], datetime] = (
        lambda: datetime.now(timezone.utc)
    ),
) -> ExecutionInputs:
    """Validate lineage without opening prompt text."""

    # Resolve the project and require a clean public state.
    root = Path(project_root).resolve()
    _require_clean_worktree(root)

    # Execution accepts only safe project-relative inputs.
    config_file = _resolve_relative(
        root,
        str(config_path),
        "generation configuration",
    )
    enablement_file = _resolve_relative(
        root,
        str(enablement_path),
        "execution enablement",
    )

    # Both public control files must be committed.
    _require_tracked(
        root,
        config_file,
        "generation configuration",
    )
    _require_tracked(
        root,
        enablement_file,
        "execution enablement",
    )

    # Load the later explicit execution switch.
    enablement = load_execution_enablement(
        enablement_file
    )

    # Verify and load the frozen generation config.
    configuration_link = _require_mapping_keys(
        enablement["configuration"],
        CONFIGURATION_LINK_KEYS,
        "configuration link",
    )

    linked_config = _verify_file(
        root,
        str(
            configuration_link["relative_path"]
        ),
        str(configuration_link["sha256"]),
        "generation configuration",
    )

    _require(
        linked_config == config_file,
        (
            "Requested and enabled "
            "configurations differ."
        ),
    )

    configuration = (
        load_generation_evaluation_config(
            config_file
        )
    )

    # Verify and load the separate budget approval.
    authorization_link = _require_mapping_keys(
        enablement["authorization"],
        AUTHORIZATION_LINK_KEYS,
        "authorization link",
    )

    authorization_path = _verify_file(
        root,
        str(
            authorization_link["relative_path"]
        ),
        str(authorization_link["sha256"]),
        "generation cost authorization",
    )

    _require_tracked(
        root,
        authorization_path,
        "generation cost authorization",
    )

    authorization = _load_json(
        authorization_path,
        "generation cost authorization",
    )

    hard_cap = validate_authorization(
        authorization
    )

    _require(
        (
            authorization["configuration_id"]
            == configuration["configuration_id"]
        ),
        (
            "Authorization configuration "
            "identifier changed."
        ),
    )

    _require(
        (
            authorization_link["relative_path"]
            == configuration[
                "execution_controls"
            ][
                "cost_authorization_relative_path"
            ]
        ),
        (
            "Configuration and enablement "
            "authorization paths differ."
        ),
    )

    # Verify the complete authorization lineage.
    lineage = authorization.get("lineage")

    _require(
        isinstance(lineage, Mapping),
        (
            "Authorization lineage must be "
            "a mapping."
        ),
    )

    _require(
        (
            lineage[
                (
                    "generation_configuration_"
                    "relative_path"
                )
            ]
            == configuration_link["relative_path"]
        ),
        (
            "Authorization configuration path "
            "changed."
        ),
    )
    _require(
        (
            lineage[
                (
                    "generation_configuration_"
                    "sha256"
                )
            ]
            == configuration_link["sha256"]
        ),
        (
            "Authorization configuration "
            "fingerprint changed."
        ),
    )

    # Independently verify the cost and preparation audits.
    for path_key, sha_key, label in (
        (
            "cost_audit_relative_path",
            "cost_audit_sha256",
            "generation cost audit",
        ),
        (
            "preparation_audit_relative_path",
            "preparation_audit_sha256",
            "generation preparation audit",
        ),
    ):
        linked = _verify_file(
            root,
            str(lineage[path_key]),
            str(lineage[sha_key]),
            label,
        )

        _require_tracked(
            root,
            linked,
            label,
        )

    # Verify and load the pricing snapshot.
    pricing_recheck = _require_mapping_keys(
        enablement["pricing_recheck"],
        PRICING_RECHECK_KEYS,
        "pricing recheck",
    )

    _require(
        (
            lineage[
                "pricing_snapshot_relative_path"
            ]
            == pricing_recheck[
                "snapshot_relative_path"
            ]
        ),
        "Authorization pricing path changed.",
    )
    _require(
        (
            lineage["pricing_snapshot_sha256"]
            == pricing_recheck[
                "snapshot_sha256"
            ]
        ),
        (
            "Authorization pricing fingerprint "
            "changed."
        ),
    )

    pricing_path = _verify_file(
        root,
        str(
            pricing_recheck[
                "snapshot_relative_path"
            ]
        ),
        str(
            pricing_recheck["snapshot_sha256"]
        ),
        "generation pricing snapshot",
    )

    _require_tracked(
        root,
        pricing_path,
        "generation pricing snapshot",
    )

    pricing = _load_json(
        pricing_path,
        "generation pricing snapshot",
    )

    _require(
        (
            pricing["configuration_id"]
            == configuration["configuration_id"]
        ),
        (
            "Pricing configuration identifier "
            "changed."
        ),
    )

    # Calculate the current UTC date once.
    current_value = now_fn()

    _require(
        (
            current_value.tzinfo is not None
            and current_value.utcoffset()
            is not None
        ),
        (
            "Execution clock must return a "
            "timezone-aware value."
        ),
    )

    current_utc_date = (
        current_value
        .astimezone(timezone.utc)
        .date()
        .isoformat()
    )

    _validate_enablement_contract(
        enablement,
        configuration["configuration_id"],
        authorization,
        pricing,
        current_utc_date,
    )

    # Verify the committed implementation selected
    # by the future enablement record.
    implementation = _require_mapping_keys(
        enablement["implementation"],
        IMPLEMENTATION_LINK_KEYS,
        "implementation link",
    )

    implementation_path = _verify_file(
        root,
        str(implementation["relative_path"]),
        str(implementation["sha256"]),
        "generation implementation",
    )

    _require(
        (
            implementation["relative_path"]
            == (
                "src/geotech_rag/"
                "generation_execution.py"
            )
        ),
        (
            "Unexpected generation "
            "implementation path."
        ),
    )

    _validate_implementation_git_state(
        root,
        implementation_path,
        str(implementation["commit"]),
    )

    # Verify the private manifest by fingerprint
    # without opening or parsing its prompt text.
    manifest_link = _require_mapping_keys(
        enablement["manifest"],
        MANIFEST_LINK_KEYS,
        "manifest link",
    )

    _require(
        (
            lineage[
                (
                    "private_request_manifest_"
                    "relative_path"
                )
            ]
            == manifest_link["relative_path"]
        ),
        (
            "Authorization manifest path "
            "changed."
        ),
    )
    _require(
        (
            lineage[
                (
                    "private_request_manifest_"
                    "sha256"
                )
            ]
            == manifest_link["sha256"]
        ),
        (
            "Authorization manifest fingerprint "
            "changed."
        ),
    )
    _require(
        (
            manifest_link["relative_path"]
            == configuration[
                "private_outputs"
            ][
                "request_manifest_relative_path"
            ]
        ),
        (
            "Configuration and enablement "
            "manifest paths differ."
        ),
    )

    manifest_path = _verify_file(
        root,
        str(manifest_link["relative_path"]),
        str(manifest_link["sha256"]),
        "private request manifest",
    )

    _require_ignored(
        root,
        manifest_path,
        "private request manifest",
    )

    # Resolve and verify all private output paths.
    private_outputs = configuration[
        "private_outputs"
    ]

    checkpoint_paths = CheckpointPaths(
        checkpoint=_resolve_relative(
            root,
            private_outputs[
                "checkpoint_relative_path"
            ],
            "private response checkpoint",
        ),
        metadata=_resolve_relative(
            root,
            private_outputs[
                (
                    "checkpoint_metadata_"
                    "relative_path"
                )
            ],
            "private checkpoint metadata",
        ),
        responses=_resolve_relative(
            root,
            private_outputs[
                "response_relative_path"
            ],
            "private final responses",
        ),
    )

    for private_path, label in (
        (
            checkpoint_paths.checkpoint,
            "private response checkpoint",
        ),
        (
            checkpoint_paths.metadata,
            "private checkpoint metadata",
        ),
        (
            checkpoint_paths.responses,
            "private final responses",
        ),
    ):
        _require_ignored(
            root,
            private_path,
            label,
        )

    # Return paths and public evidence only.
    return ExecutionInputs(
        root=root,
        configuration=configuration,
        authorization=authorization,
        enablement=enablement,
        pricing=pricing,
        hard_cap=hard_cap,
        manifest_path=manifest_path,
        manifest_sha256=str(
            manifest_link["sha256"]
        ),
        authorization_sha256=str(
            authorization_link["sha256"]
        ),
        pricing_snapshot_sha256=str(
            pricing_recheck["snapshot_sha256"]
        ),
        checkpoint_paths=checkpoint_paths,
    )


def preflight_required_models(
    client: Any,
    configuration: Mapping[str, Any],
) -> tuple[str, ...]:
    """Verify model IDs before opening the manifest."""

    # Preserve the first frozen occurrence of each model.
    required = tuple(
        dict.fromkeys(
            condition["model_id"]
            for condition
            in configuration["conditions"]
        )
    )

    try:
        # This model-list request contains no private
        # question, context, answer or ground truth.
        listing = client.models.list()
    except Exception:
        raise ExecutionAuthorizationError(
            (
                "Model availability preflight "
                "failed."
            )
        ) from None

    data = _get_value(
        listing,
        "data",
    )

    _require(
        isinstance(data, Sequence),
        "Model-list response is invalid.",
    )

    # Extract public model identifiers only.
    available = {
        model_id
        for item in data
        if isinstance(
            (
                model_id := _get_value(
                    item,
                    "id",
                )
            ),
            str,
        )
    }

    missing = [
        model_id
        for model_id in required
        if model_id not in available
    ]

    if missing:
        raise ExecutionAuthorizationError(
            (
                "One or more frozen model "
                "identifiers are unavailable; "
                "substitution is prohibited."
            )
        )

    return required


def execute_generation_with_client(
    *,
    client: Any,
    config_path: str | Path,
    enablement_path: str | Path,
    project_root: str | Path,
    explicit_execute_paid_generation: bool,
    now_fn: Callable[[], datetime] = (
        lambda: datetime.now(timezone.utc)
    ),
    sleep_fn: Callable[[float], None] = (
        time.sleep
    ),
) -> ExecutionRunResult:
    """Run the fully gated injected-client experiment."""

    # The explicit action flag is checked before
    # any file, client method or private text is used.
    if not explicit_execute_paid_generation:
        raise ExecutionAuthorizationError(
            (
                "The explicit paid-generation "
                "action flag is absent."
            )
        )

    # Validate public lineage and ignored paths first.
    inputs = load_execution_inputs(
        config_path,
        enablement_path,
        project_root,
        now_fn=now_fn,
    )

    # Verify availability before opening private prompts.
    preflight_required_models(
        client,
        inputs.configuration,
    )

    # Only now load and validate private request text.
    records = load_manifest(
        inputs.manifest_path
    )
    validate_manifest(
        records,
        inputs.configuration,
    )

    # Run the already tested private checkpoint core.
    return _execute_checkpointed_requests(
        client=client,
        requests=records,
        request_policy=inputs.configuration[
            "request_policy"
        ],
        pricing=inputs.pricing,
        hard_cap=inputs.hard_cap,
        paths=inputs.checkpoint_paths,
        configuration_id=inputs.configuration[
            "configuration_id"
        ],
        manifest_sha256=(
            inputs.manifest_sha256
        ),
        authorization_sha256=(
            inputs.authorization_sha256
        ),
        pricing_snapshot_sha256=(
            inputs.pricing_snapshot_sha256
        ),
        now_fn=now_fn,
        sleep_fn=sleep_fn,
    )


def _create_openai_client() -> Any:
    """Create the real client with SDK retries disabled."""

    # Import locally so importing this module remains
    # free from client construction and credential access.
    try:
        from openai import OpenAI

        return OpenAI(max_retries=0)
    except Exception as error:
        # Do not expose credential values or lower-level
        # client configuration details in CLI output.
        raise ExecutionAuthorizationError(
            (
                "The OpenAI client could not be "
                "created from the configured environment."
            )
        ) from error


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the explicit paid-generation CLI."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen and fully gated "
            "benchmark generation experiment."
        )
    )

    # Use the frozen public configuration by default.
    parser.add_argument(
        "--config",
        default=(
            "configs/"
            "generation-evaluation-config.json"
        ),
        help=(
            "Path to the frozen generation "
            "configuration."
        ),
    )

    # This file will be created only after the
    # implementation itself has been committed.
    parser.add_argument(
        "--enablement",
        default=(
            "configs/"
            "generation-execution-enablement.json"
        ),
        help=(
            "Path to the committed same-day "
            "execution-enablement record."
        ),
    )

    parser.add_argument(
        "--project-root",
        default=".",
        help="Path to the Git project root.",
    )

    # A deliberate action flag is required because
    # this command can make paid external requests.
    parser.add_argument(
        "--execute-paid-generation",
        action="store_true",
        help=(
            "Explicitly authorize this invocation "
            "to perform the already enabled run."
        ),
    )

    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    client_factory: Callable[[], Any] | None = None,
) -> int:
    """Run the guarded command-line interface."""

    parser = _build_argument_parser()
    arguments = parser.parse_args(argv)

    # Refuse before reading any file, loading private
    # prompts, accessing credentials or creating a client.
    if not arguments.execute_paid_generation:
        print("Generation execution: REFUSED")
        print(
            "  Reason: explicit paid-generation "
            "flag is absent"
        )
        print("  OpenAI client created: False")
        print("  API request made: False")
        return 2

    try:
        # Validate the public enablement, Git lineage,
        # cost authorization and implementation identity
        # before the real client can be constructed.
        load_execution_inputs(
            arguments.config,
            arguments.enablement,
            arguments.project_root,
        )

        # Tests can inject a fake factory. Normal CLI use
        # selects the real zero-retry OpenAI client.
        factory = (
            _create_openai_client
            if client_factory is None
            else client_factory
        )

        try:
            client = factory()
        except GenerationExecutionError:
            raise
        except Exception as error:
            raise ExecutionAuthorizationError(
                (
                    "The OpenAI client could not be "
                    "created from the configured environment."
                )
            ) from error

        # Reuse the tested outer gate. Its second public
        # validation also protects against a file changing
        # between the first check and execution.
        result = execute_generation_with_client(
            client=client,
            config_path=arguments.config,
            enablement_path=arguments.enablement,
            project_root=arguments.project_root,
            explicit_execute_paid_generation=(
                arguments.execute_paid_generation
            ),
        )

    except GenerationExecutionError as error:
        # Errors from the execution layer are designed to
        # report control failures without private text.
        print("Generation execution: BLOCKED")
        print("  Reason:", str(error))
        print("  Execution completed: False")
        return 1

    # Report only text-free operational evidence.
    print("Generation execution: PASSED")
    print(
        "  Planned response count:",
        result.planned_response_count,
    )
    print(
        "  Previously completed response count:",
        result.previously_completed_response_count,
    )
    print(
        "  Completed response count:",
        result.completed_response_count,
    )
    print(
        "  Generation attempts this run:",
        result.generation_attempt_count_this_run,
    )
    print(
        "  Conservative committed cost:",
        (
            f"${result.conservative_committed_cost_usd:.6f}"
        ),
    )
    print(
        "  Known usage cost:",
        f"${result.known_usage_cost_usd:.6f}",
    )
    print(
        "  Response file SHA-256:",
        result.response_sha256,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
