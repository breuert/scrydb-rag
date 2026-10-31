import json

import pytest
from thki import APIError

from scrydb_rag import ContextError, RagError
from scrydb_rag.benchmark import (
    ANSWERS_FILE,
    CONFIG_FILE,
    RETRIEVAL_FILE,
    RUN_FILE,
    evaluate_retrieval,
    qrels_query_ids,
    read_jsonl,
    run_benchmark,
    slug,
)


def test_qrels_query_ids_keep_their_order(qrels):
    assert qrels_query_ids(qrels) == ["q1", "q2", "q3"]


def test_model_ids_of_different_platforms_get_different_slugs():
    assert slug("openai-gpt-oss-120b") == "openai-gpt-oss-120b"
    assert slug("openai/gpt-oss-120b") == "openai__gpt-oss-120b"
    assert slug("nvidia/nemotron-3.5-lightning:free") == "nvidia__nemotron-3.5-lightning-free"


def test_benchmark_answers_the_test_queries(pipeline, client, qrels, tmp_path):
    output_dir = tmp_path / "run"
    result = run_benchmark(pipeline, "toy", qrels, output_dir)

    assert [record["qid"] for record in result.records] == ["q1", "q2", "q3"]
    assert not result.failed and not result.skipped
    assert len(client.chat_requests) == 3
    assert result.records[0]["answer"] == "Statins do not raise the risk of breast cancer [1]."
    record = result.records[1]
    assert record["query"] == "Is coffee good for the liver?"
    # The context holds top_k passages, the ranking all documents up to rank 10
    assert len(record["retrieval_context"]) == 2
    assert len(record["ranking"]) == 3
    assert record["retrieval_context"][0].startswith("Coffee and the liver")
    # The query's own document (q2) is excluded
    assert record["ranking"][0][0] == "d3"
    assert "q2" not in [docid for docid, _ in record["ranking"]]

    assert len(read_jsonl(output_dir / ANSWERS_FILE)) == 3
    assert json.loads((output_dir / CONFIG_FILE).read_text())["api_base"] == "https://chat.kiconnect.nrw/api/v1"
    run_line = (output_dir / RUN_FILE).read_text().splitlines()[0].split()
    assert run_line[:4] == ["q1", "Q0", "d1", "1"] and run_line[5] == "bm25"
    metrics = json.loads((output_dir / RETRIEVAL_FILE).read_text())
    assert metrics["queries"] == 3
    assert set(metrics) == {"queries", "nDCG@10", "nDCG@2", "RR@10", "P@2", "R@2"}
    assert result.effectiveness["R@2"] == 1.0


def test_benchmark_is_resumed(pipeline, client, qrels, tmp_path):
    output_dir = tmp_path / "run"
    assert len(run_benchmark(pipeline, "toy", qrels, output_dir, limit=1).records) == 1
    result = run_benchmark(pipeline, "toy", qrels, output_dir)
    assert len(result.records) == 3
    assert len(client.chat_requests) == 3  # no query is answered twice
    run_benchmark(pipeline, "toy", qrels, output_dir)
    assert len(client.chat_requests) == 3


def test_benchmark_refuses_to_resume_with_other_settings(pipeline, client, qrels, tmp_path):
    run_benchmark(pipeline, "toy", qrels, tmp_path, limit=1)
    pipeline.params = {"temperature": 0.1}
    with pytest.raises(RagError, match="other settings \\(params\\)"):
        run_benchmark(pipeline, "toy", qrels, tmp_path)
    pipeline.params = {}
    client.api_base = "https://openrouter.ai/api/v1"
    with pytest.raises(RagError, match="other settings \\(api_base\\)"):
        run_benchmark(pipeline, "toy", qrels, tmp_path)


def test_runs_that_did_not_record_the_api_are_resumed(pipeline, qrels, tmp_path):
    run_benchmark(pipeline, "toy", qrels, tmp_path, limit=1)
    settings = json.loads((tmp_path / CONFIG_FILE).read_text())
    del settings["api_base"]
    (tmp_path / CONFIG_FILE).write_text(json.dumps(settings))
    assert len(run_benchmark(pipeline, "toy", qrels, tmp_path).records) == 3


def test_failed_queries_are_retried(pipeline, client, qrels, tmp_path):
    failures = []

    def flaky_reply(messages, params):
        if "coffee" in messages[-1]["content"] and not failures:
            failures.append(1)
            raise APIError("503 Service Unavailable", 503)
        return "An answer [1]."

    client.reply = flaky_reply
    result = run_benchmark(pipeline, "toy", qrels, tmp_path)
    assert list(result.failed) == ["q2"]
    assert [record["qid"] for record in result.records] == ["q1", "q3"]
    result = run_benchmark(pipeline, "toy", qrels, tmp_path)
    assert not result.failed
    assert [record["qid"] for record in result.records] == ["q1", "q2", "q3"]


def test_benchmark_stops_if_a_prompt_does_not_fit_into_the_context(pipeline, client, qrels, tmp_path):
    def too_long(messages, params):
        raise ContextError("the prompt did not fit into the context of qwen3:8b (4,096 tokens)")

    client.reply = too_long
    with pytest.raises(ContextError):
        run_benchmark(pipeline, "toy", qrels, tmp_path)
    assert read_jsonl(tmp_path / ANSWERS_FILE) == []


def test_queries_without_results_count_as_zero(qrels):
    records = [
        {"qid": "q1", "ranking": [["d1", 2.0], ["d2", 1.0]]},
        {"qid": "q2", "ranking": []},
    ]
    effectiveness = evaluate_retrieval(records, qrels, top_k=2)
    assert effectiveness["nDCG@10"] == pytest.approx(0.5)
    assert effectiveness["P@2"] == pytest.approx(0.25)
