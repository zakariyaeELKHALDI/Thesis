"""Fingerprint-checked analysis of the 140 frozen, blinded expert judgments.

Public output contains only aggregate, text-free results. Response text,
reference solutions, question identifiers and judgment notes remain private.
"""

from __future__ import annotations

import csv
import argparse
import io
import json
import math
import os
import statistics
import tempfile
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

from geotech_rag import grouped_response_review as review


JUDGMENTS_SHA256 = "ad861d0d40bc4bfa59a0bda5b6f57abbff0aaac4b772cb9daad7adc4317ca567"
COMPLETED_NOTEBOOK_SHA256 = "56f87d4b53d60a6dbfd2447299237259ec5492f7cbff115e993fe7ff2da39bce"
PRELIMINARY_STATUS_SHA256 = "4921c92663be4cafeec37c03341e1e18395234036363404ac8bc712ac8217655"
CONDITIONS = ("P0", "P1", "P2", "P3", "M1", "M2", "M3")
ORDINAL_FIELDS = ("formula_integration", "calculation_correctness", "unit_correctness",
                  "explanation_clarity", "task_adaptability")
APPLICABILITY = dict(review.SECONDARY_APPLICABILITY)
FAMILIES = {"retrieval": ("P0", "P1"), "temperature": ("P1", "P2", "P3"),
            "model_generation": ("P1", "M1", "M2", "M3")}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise review.ReviewError(message)


def record_score_set_decision(root: Path) -> tuple[Path, str]:
    """Append the researcher's final-for-analysis decision without changing scores."""
    private = root.resolve() / "data/raw/evaluation/generation/v1"
    judgments = private / "evaluation-judgements.jsonl"
    completed = private / "blinded-grouped-response-review-completed.ipynb"
    checkpoint = private / "evaluation-judgement-metadata.json"
    status = private / "evaluation-expert-confirmation-status.json"
    destination = private / "evaluation-score-set-decision.json"
    require(review.sha256(judgments) == JUDGMENTS_SHA256, "imported judgment fingerprint changed")
    require(review.sha256(completed) == COMPLETED_NOTEBOOK_SHA256, "completed scoring notebook changed")
    require(review.sha256(status) == PRELIMINARY_STATUS_SHA256, "preliminary feedback status changed")
    require(not destination.exists(), "score-set decision already recorded")
    metadata = json.loads(checkpoint.read_text(encoding="utf-8"))
    previous = json.loads(status.read_text(encoding="utf-8"))
    require(metadata["judgments_sha256"] == JUDGMENTS_SHA256
            and metadata["completed_notebook_sha256"] == COMPLETED_NOTEBOOK_SHA256
            and metadata["judgment_count"] == 140,
            "judgment import checkpoint differs")
    require(previous["judgements_sha256"] == JUDGMENTS_SHA256
            and previous["import_metadata_sha256"] == review.sha256(checkpoint)
            and previous["confirmation_status"] == "preliminary"
            and previous["final_professor_confirmation_received"] is False,
            "preliminary feedback record differs")
    decision = {
        "score_set_status": "final_for_analysis",
        "professor_feedback_status": "preliminary",
        "judgements_sha256": JUDGMENTS_SHA256,
        "completed_notebook_sha256": COMPLETED_NOTEBOOK_SHA256,
        "previous_status_sha256": PRELIMINARY_STATUS_SHA256,
        "recorded_on_utc_date": datetime.now(timezone.utc).date().isoformat(),
        "decision_source": "thesis_researcher_instruction",
        "future_overall_system_comments_recorded_separately_in_thesis": True,
    }
    with destination.open("xb") as stream:
        stream.write((json.dumps(decision, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    return destination, review.sha256(destination)


def load_frozen_analysis(root: Path) -> tuple[dict, list[dict], list[dict], list[dict], dict]:
    """Verify every input and pair judgments with conditions only in memory."""
    root = root.resolve()
    config, questions, references, responses, validation = review.load_inputs(root)
    private = root / "data/raw/evaluation/generation/v1"
    judgments = private / "evaluation-judgements.jsonl"
    checkpoint = private / "evaluation-judgement-metadata.json"
    status = private / "evaluation-expert-confirmation-status.json"
    decision = private / "evaluation-score-set-decision.json"
    completed = private / "blinded-grouped-response-review-completed.ipynb"
    require(review.sha256(judgments) == JUDGMENTS_SHA256, "judgment fingerprint changed")
    require(review.sha256(completed) == COMPLETED_NOTEBOOK_SHA256,
            "completed scoring notebook changed since confirmation")
    require(review.sha256(status) == PRELIMINARY_STATUS_SHA256,
            "preliminary confirmation record changed")
    checkpoint_record = json.loads(checkpoint.read_text(encoding="utf-8"))
    previous = json.loads(status.read_text(encoding="utf-8"))
    current = json.loads(decision.read_text(encoding="utf-8"))
    require(checkpoint_record["judgments_sha256"] == JUDGMENTS_SHA256 and
            checkpoint_record["completed_notebook_sha256"] == COMPLETED_NOTEBOOK_SHA256 and
            checkpoint_record["judgment_count"] == 140, "import checkpoint differs")
    require(previous["import_metadata_sha256"] == review.sha256(checkpoint)
            and previous["judgements_sha256"] == JUDGMENTS_SHA256
            and previous["confirmation_status"] == "preliminary"
            and previous["final_professor_confirmation_received"] is False,
            "preliminary confirmation lineage differs")
    require(current["previous_status_sha256"] == PRELIMINARY_STATUS_SHA256
            and current["judgements_sha256"] == JUDGMENTS_SHA256
            and current["completed_notebook_sha256"] == COMPLETED_NOTEBOOK_SHA256
            and current["score_set_status"] == "final_for_analysis"
            and current["professor_feedback_status"] == "preliminary",
            "final score-set decision differs")
    require(checkpoint_record["reference_sha256"] == review.sha256(root / config["private_paths"]["reference_answers"])
            and checkpoint_record["response_sha256"] == config["lineage"]["private_response_sha256"]
            and checkpoint_record["question_sha256"] == validation["question_file_sha256"],
            "imported source fingerprint differs")
    judgment_rows = review.read_jsonl(judgments)
    require(len(judgment_rows) == 140, "expected 140 judgments")
    response_by_id = {row["blinded_response_id"]: row for row in responses}
    reference_by_id = {row["question_id"]: row for row in references}
    seen_ids, seen_pairs, joined = set(), set(), []
    for row in judgment_rows:
        require("condition_id" not in row and "response_text" not in row
                and "requested_model_id" not in row and "returned_model_id" not in row,
                "blinded judgment contains condition or response text")
        rid = row["blinded_response_id"]
        require(rid in response_by_id and rid not in seen_ids, "unknown or duplicate blinded ID")
        seen_ids.add(rid)
        response = response_by_id[rid]
        qid, condition = response["question_id"], response["condition_id"]
        require(row["question_id"] == qid and condition in CONDITIONS,
                "question or condition differs from frozen response")
        require((qid, condition) not in seen_pairs, "duplicate question-condition pair")
        seen_pairs.add((qid, condition))
        reference = reference_by_id[qid]
        score = {name: row[name] for name in review.FIELDS}
        require(set(row) == set(review.FIELDS) | {"question_id", "blinded_response_id"},
                "judgment schema contains unexpected fields")
        review.validate_score(score, reference, not response["response_text"].strip(),
                              "private judgment")
        joined.append({"question_id": qid, "condition": condition, "score": score,
                       "empty": not response["response_text"].strip(),
                       "reference": reference})
    require(len(seen_ids) == len(seen_pairs) == 140, "complete paired design required")
    require({(q["question_id"], c) for q in questions for c in CONDITIONS} == seen_pairs,
            "question-condition matrix incomplete")
    require(sum(item["empty"] for item in joined) == 9, "nine empty responses required")
    require(len({item["condition"] for item in joined if item["empty"]})
            == config["empty_response_policy"]["observed_empty_condition_count"],
            "empty response condition count differs from frozen audit")
    fingerprints = {"judgments_sha256": JUDGMENTS_SHA256,
                    "reference_sha256": checkpoint_record["reference_sha256"],
                    "responses_sha256": checkpoint_record["response_sha256"],
                    "completed_scoring_notebook_sha256": COMPLETED_NOTEBOOK_SHA256,
                    "preliminary_confirmation_status_sha256": PRELIMINARY_STATUS_SHA256,
                    "final_score_set_decision_sha256": review.sha256(decision),
                    "import_checkpoint_sha256": review.sha256(checkpoint)}
    return config, questions, references, joined, fingerprints


def wilson(successes: int, total: int) -> tuple[float, float]:
    require(total > 0 and 0 <= successes <= total, "Wilson interval requires valid denominator")
    z = statistics.NormalDist().inv_cdf(0.975)
    p = successes / total
    denominator = 1 + z*z/total
    midpoint = (p + z*z/(2*total)) / denominator
    spread = z/denominator * math.sqrt(p*(1-p)/total + z*z/(4*total*total))
    return max(0.0, midpoint-spread), min(1.0, midpoint+spread)


def chi_square_survival(statistic: float, df: int) -> float:
    """Closed forms for the df 1, 2, 3 used in this frozen analysis."""
    require(df in (1, 2, 3) and statistic >= -1e-10, "unsupported chi-square degrees of freedom")
    x = max(0.0, statistic)
    if df == 1:
        return math.erfc(math.sqrt(x / 2))
    if df == 2:
        return math.exp(-x / 2)
    return math.erfc(math.sqrt(x / 2)) + math.sqrt(2*x/math.pi)*math.exp(-x/2)


def exact_mcnemar(left: list[int], right: list[int]) -> dict:
    require(len(left) == len(right) == 20, "McNemar requires the paired 20 questions")
    left_only = sum(a == 1 and b == 0 for a,b in zip(left,right,strict=True))
    right_only = sum(a == 0 and b == 1 for a,b in zip(left,right,strict=True))
    discordant = left_only + right_only
    p = (min(1.0, 2*sum(math.comb(discordant,k) for k in range(min(left_only,right_only)+1))
             / 2**discordant) if discordant else 1.0)
    return {"left_only_correct":left_only,"right_only_correct":right_only,
            "discordant":discordant,"p_value":p}


def cochran_q(matrix: list[list[int]]) -> dict:
    n, k = len(matrix), len(matrix[0])
    require(n == 20 and k in (3,4) and all(len(row) == k for row in matrix),
            "Cochran Q requires 20 matched questions and 3 or 4 conditions")
    columns = [sum(row[j] for row in matrix) for j in range(k)]
    rows = [sum(row) for row in matrix]
    total = sum(columns)
    denom = k*total - sum(value*value for value in rows)
    if denom == 0:
        return {"statistic":0.0,"df":k-1,"p_value":1.0,"no_discordance":True}
    statistic = (k-1)*(k*sum(value*value for value in columns)-total*total)/denom
    return {"statistic":statistic,"df":k-1,
            "p_value":chi_square_survival(statistic,k-1),"no_discordance":False}


def _average_ranks(values: list[float]) -> tuple[list[float], float]:
    """Return 1-based average ranks and the sum of t^3-t tie terms."""
    order = sorted(range(len(values)),key=values.__getitem__)
    ranks = [0.0]*len(values)
    ties = 0
    start = 0
    while start < len(order):
        end = start+1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = (start+1 + end)/2
        for idx in order[start:end]:
            ranks[idx] = average
        t = end-start
        ties += t**3-t
        start = end
    return ranks, ties


def exact_wilcoxon(left: list[int], right: list[int]) -> dict:
    require(len(left) == len(right) and len(left) <= 20, "paired Wilcoxon input differs")
    differences = [a-b for a,b in zip(left,right,strict=True) if a != b]
    n = len(differences)
    if not n:
        return {"nonzero_pairs":0,"statistic":0.0,"p_value":1.0}
    ranks,_ = _average_ranks([abs(d) for d in differences])
    doubled = [round(2*r) for r in ranks]
    positive = sum(rank for d,rank in zip(differences,doubled) if d>0)
    counts = Counter({0:1})
    for rank in doubled:
        updated = counts.copy()
        for total,count in counts.items():
            updated[total+rank] += count
        counts = updated
    denominator = 2**n
    lower = sum(c for total,c in counts.items() if total <= positive)/denominator
    upper = sum(c for total,c in counts.items() if total >= positive)/denominator
    return {"nonzero_pairs":n,"statistic":min(positive,sum(doubled)-positive)/2,
            "p_value":min(1.0,2*min(lower,upper))}


def friedman(matrix: list[list[int]]) -> dict:
    n, k = len(matrix), len(matrix[0])
    require(n > 0 and k in (3,4) and all(len(row)==k for row in matrix),
            "Friedman needs common paired applicable questions")
    ranking = [_average_ranks(row) for row in matrix]
    totals = [sum(pair[0][j] for pair in ranking) for j in range(k)]
    raw = 12/(n*k*(k+1))*sum(t*t for t in totals)-3*n*(k+1)
    correction = 1-sum(pair[1] for pair in ranking)/(n*k*(k*k-1))
    if correction <= 0:
        return {"statistic":0.0,"df":k-1,"p_value":1.0,"no_variation":True}
    statistic = max(0.0,raw/correction)
    return {"statistic":statistic,"df":k-1,
            "p_value":chi_square_survival(statistic,k-1),"no_variation":False}


def holm(raw: list[float]) -> list[float]:
    require(all(0 <= p <= 1 for p in raw), "invalid raw p value")
    result = [0.0]*len(raw)
    maximum = 0.0
    for rank,index in enumerate(sorted(range(len(raw)),key=raw.__getitem__)):
        maximum = max(maximum, min(1.0,(len(raw)-rank)*raw[index]))
        result[index] = maximum
    return result


def compute_aggregates(questions: list[dict], references: list[dict], joined: list[dict],
                       fingerprints: dict) -> dict:
    """Compute only aggregate results; no private text or per-question result is returned."""
    qids = [q["question_id"] for q in questions]
    by_pair = {(row["question_id"],row["condition"]):row for row in joined}
    require(len(qids)==20 and len(by_pair)==140, "20 by 7 matched input required")
    refs = {r["question_id"]:r for r in references}
    table = []
    for condition in CONDITIONS:
        records = [by_pair[(qid,condition)] for qid in qids]
        correct = sum(row["score"]["answer_correctness"] for row in records)
        completed = sum(not row["empty"] for row in records)
        lo,hi = wilson(correct,20)
        table.append({"condition":condition,"gradable_questions":20,"correct":correct,
                      "accuracy":correct/20,"wilson95_lower":lo,"wilson95_upper":hi,
                      "visible_responses":completed,"completion_rate":completed/20,
                      "no_visible_answers":20-completed,
                      "unsupported_claims_count":sum(row["score"]["unsupported_claims"] for row in records)})
    secondary = []
    for field in ORDINAL_FIELDS:
        for condition in CONDITIONS:
            values = [by_pair[(qid,condition)]["score"][field] for qid in qids
                      if field not in APPLICABILITY or refs[qid][APPLICABILITY[field]]]
            require(values and all(type(v) is int and 0<=v<=2 for v in values),
                    f"{field} applicable score differs")
            secondary.append({"condition":condition,"field":field,"applicable_questions":len(values),
                              "mean_score":sum(values)/len(values),"median_score":statistics.median(values),
                              **{f"score_{value}_count":values.count(value) for value in (0,1,2)}})
    errors = []
    for condition in CONDITIONS:
        group = [by_pair[(qid,condition)] for qid in qids]
        counts = Counter(row["score"]["primary_error_category"] for row in group)
        require(sum(counts[category] for category in
                    ("grounding","conceptual","calculation","deficiency","no_visible_answer","other"))
                == sum(row["score"]["answer_correctness"] == 0 for row in group),
                "primary error count differs from incorrect responses")
        for category in ("grounding","conceptual","calculation","deficiency","no_visible_answer","other"):
            errors.append({"condition":condition,"primary_error_category":category,
                           "count":counts[category],"denominator":20})
    comparisons, omnibus, ordinal = [], {}, {}
    for family,conditions in FAMILIES.items():
        if len(conditions)>2:
            matrix = [[by_pair[(qid,c)]["score"]["answer_correctness"] for c in conditions]
                      for qid in qids]
            omnibus[family] = cochran_q(matrix)
        pairs = list(combinations(conditions,2))
        p_values = []
        raw = []
        for left,right in pairs:
            outcomes = exact_mcnemar([by_pair[(qid,left)]["score"]["answer_correctness"] for qid in qids],
                                     [by_pair[(qid,right)]["score"]["answer_correctness"] for qid in qids])
            raw.append((left,right,outcomes))
            p_values.append(outcomes["p_value"])
        adjusted = holm(p_values) if len(pairs)>1 else p_values
        for (left,right,outcome),corrected in zip(raw,adjusted,strict=True):
            comparisons.append({"family":family,"left_condition":left,"right_condition":right,
                                **outcome,"holm_adjusted_p_value":corrected})
        ordinal[family] = {}
        for field in ORDINAL_FIELDS:
            eligible = [qid for qid in qids if field not in APPLICABILITY
                        or refs[qid][APPLICABILITY[field]]]
            matrix = [[by_pair[(qid,c)]["score"][field] for c in conditions] for qid in eligible]
            if len(conditions)==2:
                omnibus_ordinal = None
                pair_list = [(conditions[0],conditions[1])]
            else:
                omnibus_ordinal = friedman(matrix)
                pair_list = list(combinations(conditions,2))
            pair_results = []
            for left,right in pair_list:
                ranks = exact_wilcoxon([by_pair[(qid,left)]["score"][field] for qid in eligible],
                                       [by_pair[(qid,right)]["score"][field] for qid in eligible])
                pair_results.append({"left_condition":left,"right_condition":right,**ranks})
            adjusted = holm([item["p_value"] for item in pair_results]) if len(pair_results)>1 else [item["p_value"] for item in pair_results]
            for item, p in zip(pair_results,adjusted,strict=True):
                item["holm_adjusted_p_value"] = p
            ordinal[family][field] = {"common_applicable_questions":len(eligible),
                                      "friedman":omnibus_ordinal,"paired_wilcoxon":pair_results}
    strata = {}
    for field in ("problem_type","cognitive_level"):
        recorded_counts = Counter(refs[qid][field] for qid in qids if refs[qid][field] is not None)
        # A one- or two-question category would disclose individual scores in public results.
        pool_recorded = any(count < 3 for count in recorded_counts.values())
        def label(qid: str) -> str:
            value = refs[qid][field]
            if value is None:
                return "not_recorded_in_reference"
            return "recorded_not_stratified" if pool_recorded else value
        values = sorted({label(qid) for qid in qids})
        strata[field] = [{"stratum":value,"condition":condition,
                          "question_count":sum(label(qid)==value for qid in qids),
                          "correct":sum(by_pair[(qid,condition)]["score"]["answer_correctness"]
                                        for qid in qids if label(qid)==value),
                          "mean_adaptability":statistics.mean(
                              by_pair[(qid,condition)]["score"]["task_adaptability"]
                              for qid in qids if label(qid)==value)}
                         for value in values for condition in CONDITIONS]
    return {"score_set_status":"final_for_analysis","professor_feedback_status":"preliminary",
            "sampled_responses":False,"judgments":140,"questions_per_condition":20,
            "confidence_interval":"Wilson 95%","paired_tests_primary":True,
            "baseline_style_chi_square_performed":False,
            "baseline_style_chi_square_reason":"No original GPT-4 versus Llama-3 API pair exists in these seven frozen conditions.",
            "grouped_presentation_limitation":True,"single_domain_expert":True,
            "fingerprints":fingerprints,"condition_summary":table,"secondary_summary":secondary,
            "error_counts":errors,"binary_omnibus":omnibus,
            "binary_pairwise":comparisons,"ordinal_comparisons":ordinal,
            "adaptability_strata":strata}


def csv_bytes(records: list[dict]) -> bytes:
    columns = list(records[0])
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output,fieldnames=columns,lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    return output.getvalue().encode("utf-8")


def output_payload(root: Path, config: dict, aggregate: dict) -> dict[Path,bytes]:
    paths = config["tracked_outputs"]
    return {root/paths["condition_summary_table"]:csv_bytes(aggregate["condition_summary"]),
            root/paths["secondary_score_table"]:csv_bytes(aggregate["secondary_summary"]),
            root/paths["pairwise_comparison_table"]:csv_bytes(aggregate["binary_pairwise"]),
            root/"results/tables/generation-error-distribution.csv":csv_bytes(aggregate["error_counts"]),
            root/paths["evaluation_metrics"]:(json.dumps(aggregate,ensure_ascii=False,sort_keys=True,indent=2)+"\n").encode()}


def save_public_aggregates(payload: dict[Path,bytes]) -> None:
    """Exclusive publication; private rows, notes and question IDs are never serialized."""
    require(all(not path.exists() for path in payload),"public analysis output already exists")
    staged = {}
    published = []
    try:
        for path,content in payload.items():
            path.parent.mkdir(parents=True,exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="wb",dir=path.parent,
                                             prefix=".analysis-stage-",delete=False) as target:
                staged[path]=Path(target.name)
                target.write(content)
                target.flush()
                os.fsync(target.fileno())
        for path,temporary in staged.items():
            os.link(temporary,path)
            published.append(path)
    except Exception:
        for path in published:
            path.unlink()
        raise
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record the fixed score-set decision after the 140-score import")
    parser.add_argument("operation", choices=("record-score-set-decision",))
    parser.add_argument("--project-root", type=Path, required=True)
    arguments = parser.parse_args()
    file, fingerprint = record_score_set_decision(arguments.project_root)
    print("Private score-set decision:", file.relative_to(arguments.project_root.resolve()))
    print("Scores: final for analysis; professor feedback: preliminary")
    print("Judgment file SHA-256:", JUDGMENTS_SHA256)
    print("Decision SHA-256:", fingerprint)
