from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from geotech_rag.generation_evaluation import (
    GenerationConfigError,
    GenerationContextLimitError,
    GenerationIntegrityError,
    _prepare_records,
    assemble_context,
    estimate_chat_input_tokens,
    load_generation_evaluation_config,
)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _condition(
    condition_id: str,
    model_id: str,
    *,
    retrieval: bool,
    temperature: float | None,
    top_p: float | None,
    reasoning: str | None,
) -> dict:
    return {
        "condition_id": condition_id,
        "model_id": model_id,
        "model_reference_type": "mutable_alias" if condition_id == "M3" else "fixed_snapshot",
        "reasoning_effort": reasoning,
        "retrieval_depth_k": 30 if retrieval else 0,
        "retrieval_enabled": retrieval,
        "study_role": "test",
        "temperature": temperature,
        "top_p": top_p,
    }


def _config() -> dict:
    return {
        "benchmark": {
            "condition_count": 7,
            "one_response_per_question_condition": True,
            "planned_response_count": 140,
            "question_count": 20,
            "question_order": "frozen_benchmark_order",
            "question_text_may_not_be_logged_publicly": True,
        },
        "comparison_groups": {},
        "conditions": [
            _condition("P0", "gpt-4-0613", retrieval=False, temperature=0.1, top_p=1.0, reasoning=None),
            _condition("P1", "gpt-4-0613", retrieval=True, temperature=0.1, top_p=1.0, reasoning=None),
            _condition("P2", "gpt-4-0613", retrieval=True, temperature=0.5, top_p=1.0, reasoning=None),
            _condition("P3", "gpt-4-0613", retrieval=True, temperature=1.0, top_p=1.0, reasoning=None),
            _condition("M1", "gpt-4.1-2025-04-14", retrieval=True, temperature=0.1, top_p=1.0, reasoning=None),
            _condition("M2", "gpt-5-2025-08-07", retrieval=True, temperature=None, top_p=None, reasoning="low"),
            _condition("M3", "gpt-6-astra", retrieval=True, temperature=None, top_p=None, reasoning="low"),
        ],
        "configuration_id": "benchmark-generation-evaluation-v1",
        "context_assembly": {
            "abort_if_context_limit_exceeded": True,
            "context_record_field": "text",
            "deduplication_enabled": False,
            "include_chunk_identifier": False,
            "include_rank": False,
            "include_relevance_grade": False,
            "include_relevance_note": False,
            "include_similarity_score": False,
            "include_source_provenance": False,
            "preserve_faiss_returned_order": True,
            "record_prefix_template": "[Retrieved excerpt {ordinal}]",
            "record_separator": "\n\n",
            "retrieved_record_count": 30,
            "truncation_enabled": False,
        },
        "cost_boundary": {
            "authorized_maximum_cost_usd": None,
            "estimated_maximum_cost_usd": None,
            "generation_must_refuse_while_values_are_null": True,
            "pricing_snapshot_relative_path": None,
        },
        "decision_status": "frozen_before_paid_generation",
        "evaluation_boundary": {},
        "execution_controls": {
            "paid_generation_enabled": False,
        },
        "future_tracked_outputs": {},
        "generation_evaluation_config_schema_version": "1.0",
        "leakage_controls": {
            "answer_keys_available_during_generation": False,
            "ground_truth_available_during_generation": False,
            "model_generated_relevance_advice_available": False,
            "only_question_prompt_and_allowed_context_transmitted": True,
            "relevance_grades_available_during_generation": False,
            "relevance_notes_available_during_generation": False,
        },
        "model_availability_preflight": {},
        "model_parameter_policy": {},
        "private_outputs": {},
        "production_retrieval": {"retrieval_depth_k": 30},
        "prompt": {"sha256": "prompt-hash"},
        "protocol": {},
        "request_policy": {
            "endpoint": "chat.completions",
            "legacy_output_token_parameter": "max_tokens",
            "logical_max_output_tokens": 2000,
            "modern_output_token_parameter": "max_completion_tokens",
        },
        "response_record_contract": {},
    }


def _write_config(tmp_path: Path, config: dict) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_loads_frozen_configuration(tmp_path: Path) -> None:
    config = load_generation_evaluation_config(_write_config(tmp_path, _config()))
    assert config["benchmark"]["planned_response_count"] == 140


def test_rejects_paid_generation_enabled(tmp_path: Path) -> None:
    config = _config()
    config["execution_controls"]["paid_generation_enabled"] = True
    with pytest.raises(GenerationConfigError, match="disabled"):
        load_generation_evaluation_config(_write_config(tmp_path, config))


def test_rejects_non_null_cost_authorization(tmp_path: Path) -> None:
    config = _config()
    config["cost_boundary"]["authorized_maximum_cost_usd"] = 20.0
    with pytest.raises(GenerationConfigError, match="null"):
        load_generation_evaluation_config(_write_config(tmp_path, config))


def test_assemble_context_preserves_all_records() -> None:
    records = [{"retrieval_rank": index, "text": f"evidence-{index}"} for index in range(1, 31)]
    context, fingerprint = assemble_context(records, _config()["context_assembly"])
    assert context.startswith("[Retrieved excerpt 1]\nevidence-1")
    assert context.endswith("[Retrieved excerpt 30]\nevidence-30")
    assert context.count("[Retrieved excerpt ") == 30
    assert fingerprint == _hash(context)


def test_assemble_context_rejects_rank_change() -> None:
    records = [{"retrieval_rank": index, "text": f"evidence-{index}"} for index in range(1, 31)]
    records[2]["retrieval_rank"] = 4
    with pytest.raises(GenerationIntegrityError, match="rank"):
        assemble_context(records, _config()["context_assembly"])


def test_assemble_context_rejects_missing_record() -> None:
    records = [{"retrieval_rank": index, "text": "evidence"} for index in range(1, 30)]
    with pytest.raises(GenerationIntegrityError, match="count"):
        assemble_context(records, _config()["context_assembly"])


@pytest.mark.parametrize(
    ("model_id", "expected_strategy"),
    [
        ("gpt-4-0613", "tiktoken_cl100k_base"),
        ("gpt-4.1-2025-04-14", "tiktoken_o200k_base"),
        ("gpt-5-2025-08-07", "tiktoken_o200k_base"),
        ("gpt-6-astra", "max_of_cl100k_base_and_o200k_base"),
    ],
)
def test_tokenizer_strategy(model_id: str, expected_strategy: str) -> None:
    count, strategy = estimate_chat_input_tokens(model_id, "system", "user")
    assert count > 24
    assert strategy == expected_strategy


class _FakeRetriever:
    benchmark_question_ids = ("q1",)
    retrieval_depth_k = 30

    def retrieve_benchmark(self, question_id: str):
        assert question_id == "q1"
        records = tuple(
            {"retrieval_rank": index, "text": f"evidence-{index}"}
            for index in range(1, 31)
        )
        return SimpleNamespace(
            to_dict=lambda: {
                "query_identifier": question_id,
                "api_request_made": False,
                "records": records,
            }
        )


def test_prepares_conditions_without_api_request() -> None:
    config = _config()
    config["benchmark"]["question_count"] = 1
    config["benchmark"]["planned_response_count"] = 7
    prompt = {
        "system_prompt": "Tutor",
        "rag_user_template": "Context: {context}\nQuestion: {question}",
        "zero_context_user_template": "Question: {question}",
    }
    question = {
        "question_id": "q1",
        "query_text": "A question",
        "query_text_sha256": _hash("A question"),
    }
    records = _prepare_records(config, prompt, [question], _FakeRetriever())
    assert len(records) == 7
    assert sum(record.retrieval_enabled for record in records) == 6
    assert records[0].retrieved_evidence_sha256 is None
    assert all(record.context_window_passed for record in records)
    assert all(record.estimated_total_tokens > 2000 for record in records)
    assert {record.output_token_parameter for record in records[:4]} == {"max_tokens"}
    assert records[4].output_token_parameter == "max_completion_tokens"


def test_context_limit_failure_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    import geotech_rag.generation_evaluation as module

    monkeypatch.setitem(module.MODEL_CONTEXT_WINDOWS, "gpt-4-0613", 25)
    config = _config()
    config["conditions"] = [config["conditions"][0]]
    prompt = {
        "system_prompt": "Tutor",
        "rag_user_template": "{context}{question}",
        "zero_context_user_template": "Question: {question}",
    }
    question = {
        "question_id": "q1",
        "query_text": "A question",
        "query_text_sha256": _hash("A question"),
    }
    with pytest.raises(GenerationContextLimitError):
        _prepare_records(config, prompt, [question], _FakeRetriever())
