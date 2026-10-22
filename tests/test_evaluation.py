import asyncio
import json
import threading
import time

import pytest
from pydantic import BaseModel
from thki import APIError

from scrydb_rag import RagError, cli, evaluation
from scrydb_rag.benchmark import ANSWERS_FILE, read_jsonl
from scrydb_rag.evaluation import METRICS, ThkiModel, run_evaluation

from conftest import FakeClient


class Verdict(BaseModel):
    verdict: str


def schema_name(params):
    return params.get("response_format", {}).get("json_schema", {}).get("name")


def test_metric_names_of_the_cli_match():
    assert cli.METRIC_NAMES == list(METRICS)
    assert cli.DEFAULT_THRESHOLD == evaluation.DEFAULT_THRESHOLD


def test_judge_requests_the_schema_and_returns_an_instance():
    client = FakeClient(lambda messages, params: '{"verdict": "yes"}')
    judge = ThkiModel(client, "openai-gpt-oss-120b", {"reasoning_effort": "low"})
    assert judge.generate("Is it?", schema=Verdict) == Verdict(verdict="yes")
    params = client.chat_requests[0]["params"]
    assert schema_name(params) == "Verdict"
    assert params["response_format"]["json_schema"]["schema"] == Verdict.model_json_schema()
    assert params["reasoning_effort"] == "low"
    assert judge.generate("Plain text?") == '{"verdict": "yes"}'
    assert "response_format" not in client.chat_requests[1]["params"]


def test_judge_returns_text_that_does_not_fit_the_schema():
    judge = ThkiModel(FakeClient(lambda messages, params: 'Sure! {"verdict": "yes"}'))
    assert judge.generate("Is it?", schema=Verdict) == 'Sure! {"verdict": "yes"}'


def test_judge_does_without_schemas_if_the_model_rejects_them():
    def reply(messages, params):
        if "response_format" in params:
            raise APIError("400 response_format is not supported", 400)
        return '{"verdict": "no"}'

    client = FakeClient(reply)
    judge = ThkiModel(client)
    assert judge.generate("Is it?", schema=Verdict) == Verdict(verdict="no")
    assert not judge.structured_outputs
    judge.generate("Is it?", schema=Verdict)
    assert len(client.chat_requests) == 3  # the rejected request is not repeated


def test_judge_keeps_schemas_if_the_request_itself_fails():
    def reply(messages, params):
        raise APIError("400 the prompt is too long", 400)

    judge = ThkiModel(FakeClient(reply))
    with pytest.raises(APIError):
        judge.generate("Is it?", schema=Verdict)
    assert judge.structured_outputs


def test_judge_limits_concurrent_requests():
    active, peak, lock = [0], [0], threading.Lock()

    def slow_reply(messages, params):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.05)
        with lock:
            active[0] -= 1
        return "ok"

    judge = ThkiModel(FakeClient(slow_reply), max_concurrency=2)

    async def ask_many():
        return await asyncio.gather(*[judge.a_generate(f"prompt {i}") for i in range(8)])

    assert asyncio.run(ask_many()) == ["ok"] * 8
    assert peak[0] == 2
    asyncio.run(ask_many())  # a new event loop gets a new semaphore


def answer_relevancy_judge(messages, params):
    """Replies of a judge for DeepEval's answer relevancy metric."""
    return {
        "Statements": '{"statements": ["Statins do not cause breast cancer.", "The weather is nice."]}',
        "Verdicts": '{"verdicts": [{"verdict": "yes"}, {"verdict": "no", "reason": "off topic"}]}',
        "AnswerRelevancyScoreReason": '{"reason": "One of two statements is relevant."}',
    }[schema_name(params)]


def write_answers(run_dir, records):
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / ANSWERS_FILE).open("w", encoding="utf-8") as answers:
        for record in records:
            answers.write(json.dumps(record) + "\n")


def answer(query_id, text="Statins do not cause breast cancer [1]. The weather is nice."):
    return {
        "qid": query_id,
        "query": "Do statins cause breast cancer?",
        "answer": text,
        "retrieval_context": ["Statin use is not associated with breast cancer."],
    }


def test_deepeval_metric_with_the_judge(tmp_path):
    write_answers(tmp_path, [answer("q1")])
    judge = ThkiModel(FakeClient(answer_relevancy_judge), "openai-gpt-oss-120b")
    result = run_evaluation(tmp_path, judge, ["answer_relevancy"])
    assert result.summary["answer_relevancy"] == {"mean": 0.5, "pass_rate": 1.0, "evaluated": 1, "missing": 0}
    [record] = read_jsonl(result.results_path)
    assert record["score"] == 0.5
    assert record["reason"] == "One of two statements is relevant."
    assert result.results_path.name == "deepeval-openai-gpt-oss-120b.jsonl"
    summary = json.loads(result.results_path.with_suffix(".json").read_text())
    assert summary["judge"] == "openai-gpt-oss-120b"


class FakeMetric:
    """Stands in for a DeepEval metric: fails for query q2 until told otherwise."""

    calls = []
    fail = True

    def __init__(self, **kwargs):
        assert kwargs["eval_mode"] == "llm"  # all decisions are made by the judge
        self.score = self.reason = self.error = None

    async def a_measure(self, test_case, _show_indicator=True):
        FakeMetric.calls.append(test_case.name)
        if test_case.name == "q2" and FakeMetric.fail:
            raise ValueError("Evaluation LLM outputted an invalid JSON.")
        self.score, self.reason = 0.8, "fine"


def test_evaluation_is_resumed_and_retries_failures(tmp_path, monkeypatch):
    monkeypatch.setitem(METRICS, "answer_relevancy", FakeMetric)
    FakeMetric.calls, FakeMetric.fail = [], True
    write_answers(tmp_path, [answer("q1"), answer("q2"), answer("q3", text="  ")])
    judge = ThkiModel(FakeClient())

    result = run_evaluation(tmp_path, judge, ["answer_relevancy"])
    assert sorted(FakeMetric.calls) == ["q1", "q2"]
    assert result.empty == ["q3"]
    assert result.summary["answer_relevancy"]["evaluated"] == 1
    assert result.summary["answer_relevancy"]["missing"] == 1
    errors = [record["error"] for record in read_jsonl(result.results_path) if record["error"]]
    assert errors == ["ValueError: Evaluation LLM outputted an invalid JSON."]

    FakeMetric.calls, FakeMetric.fail = [], False
    result = run_evaluation(tmp_path, judge, ["answer_relevancy"])
    assert FakeMetric.calls == ["q2"]
    assert result.summary["answer_relevancy"] == {"mean": 0.8, "pass_rate": 1.0, "evaluated": 2, "missing": 0}

    FakeMetric.calls = []
    result = run_evaluation(tmp_path, judge, ["answer_relevancy"], query_ids=["q2"])
    assert FakeMetric.calls == []
    assert result.summary["answer_relevancy"]["evaluated"] == 1


def test_evaluation_needs_answers(tmp_path):
    with pytest.raises(RagError, match="no answers"):
        run_evaluation(tmp_path, ThkiModel(FakeClient()))
