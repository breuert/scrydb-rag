"""Evaluation of benchmark answers with DeepEval, judged by models of the THKI platform.

The judge is a custom DeepEval model (``ThkiModel``) that sends the prompts
of DeepEval's metrics to a chat model of the platform. The metrics need no
reference answers, which the test collections do not have:

    answer_relevancy      Do the statements of the answer address the request?
    faithfulness          Do the passages support the claims of the answer?
    contextual_relevancy  Are the statements of the passages relevant to the request?

The results are kept in the directory of the benchmark run, in two files
per judge model:

    deepeval-<judge>.jsonl  one record per query and metric: score and reason
    deepeval-<judge>.json   mean score and pass rate of each metric

Each result is appended as soon as it is available, so an interrupted
evaluation is resumed by repeating the command, which also retries the
metrics that failed.
"""

import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple, Type

# DeepEval sends usage data to its developers unless this is set before it is imported
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")

from deepeval.metrics import (  # noqa: E402
    AnswerRelevancyMetric,
    BaseMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.models import DeepEvalBaseLLM  # noqa: E402
from deepeval.test_case import LLMTestCase  # noqa: E402
from pydantic import BaseModel, ValidationError  # noqa: E402
from thki import DEFAULT_CHAT_MODEL, APIError, Client  # noqa: E402
from tqdm import tqdm  # noqa: E402

from scrydb_rag import RagError  # noqa: E402
from scrydb_rag.benchmark import ANSWERS_FILE, DEFAULT_WORKERS, read_jsonl, slug  # noqa: E402

METRICS: Dict[str, Type[BaseMetric]] = {
    "answer_relevancy": AnswerRelevancyMetric,
    "faithfulness": FaithfulnessMetric,
    "contextual_relevancy": ContextualRelevancyMetric,
}
DEFAULT_THRESHOLD = 0.5


class ThkiModel(DeepEvalBaseLLM):
    """Chat model of the THKI platform as judge of DeepEval's metrics.

    Asks for replies that are constrained to the JSON schema that a metric
    expects, and limits the number of concurrent requests, as the API answers
    too many of them with status 429.
    """

    def __init__(
        self,
        client: Client,
        model: str = DEFAULT_CHAT_MODEL,
        params: Optional[Dict[str, Any]] = None,
        max_concurrency: int = DEFAULT_WORKERS,
    ):
        """Initialize the judge.

        Args:
            client: API client
            model: ID of the chat model
            params: Additional request parameters (e.g., reasoning_effort)
            max_concurrency: Number of requests that are sent concurrently
        """
        self.client = client
        self.model_id = model
        self.params = params or {}
        self.max_concurrency = max_concurrency
        # Turned off if the model rejects JSON schemas; DeepEval then
        # extracts the JSON from the replies, as its prompts ask for it
        self.structured_outputs = True
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._semaphore: Optional[asyncio.Semaphore] = None
        super().__init__(model)

    def load_model(self) -> Client:
        return self.client

    def get_model_name(self) -> str:
        return self.model_id

    def generate(self, prompt: str, schema: Optional[Type[BaseModel]] = None) -> Any:
        """Return the reply to a prompt, as instance of the schema if one is given and the reply fits it."""
        content = self._complete(prompt, schema)
        if schema is None:
            return content
        try:
            return schema.model_validate_json(content)
        except ValidationError:
            return content  # DeepEval tries to extract the JSON from the text itself

    async def a_generate(self, prompt: str, schema: Optional[Type[BaseModel]] = None) -> Any:
        async with self._limit():
            return await asyncio.to_thread(self.generate, prompt, schema)

    def _limit(self) -> asyncio.Semaphore:
        # A semaphore belongs to the event loop that it is first used in
        loop = asyncio.get_running_loop()
        if self._loop is not loop:
            self._loop, self._semaphore = loop, asyncio.Semaphore(self.max_concurrency)
        return self._semaphore

    def _complete(self, prompt: str, schema: Optional[Type[BaseModel]]) -> str:
        messages = [{"role": "user", "content": prompt}]
        if schema is None or not self.structured_outputs:
            return self._chat(messages)
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
        }
        try:
            return self._chat(messages, response_format=response_format)
        except APIError as e:
            if e.status != 400:
                raise
        content = self._chat(messages)
        # Only now it is clear that the schema, not the request, was rejected
        self.structured_outputs = False
        return content

    def _chat(self, messages: List[Dict[str, str]], **params: Any) -> str:
        data = self.client.chat(self.model_id, messages, **{**self.params, **params})
        return (data["choices"][0].get("message") or {}).get("content") or ""


@dataclass
class EvaluationResult:
    """Outcome of an evaluation.

    Attributes:
        judge: ID of the judge model
        results_path: File with the result of each query and metric
        summary: Mean score, pass rate, and number of evaluated and missing queries, by metric
        empty: IDs of the queries with empty answers, which are not evaluated
    """

    judge: str
    results_path: Path
    summary: Dict[str, Dict[str, Any]]
    empty: List[str] = field(default_factory=list)


def make_test_case(record: Dict[str, Any]) -> LLMTestCase:
    """Return the DeepEval test case of an answer of a benchmark run."""
    return LLMTestCase(
        input=record["query"],
        actual_output=record["answer"],
        retrieval_context=record["retrieval_context"],
        name=record["qid"],
    )


async def measure(record: Dict[str, Any], name: str, judge: ThkiModel, threshold: float) -> Dict[str, Any]:
    """Evaluate an answer with a metric and return the result as record."""
    metric = METRICS[name](model=judge, threshold=threshold, include_reason=True, async_mode=True, eval_mode="llm")
    result = {"qid": record["qid"], "metric": name, "judge": judge.get_model_name()}
    try:
        await metric.a_measure(make_test_case(record), _show_indicator=False)
    except Exception as e:  # e.g., a reply of the judge that is not valid JSON
        return {**result, "score": None, "reason": None, "error": f"{type(e).__name__}: {e}"}
    if metric.score is None:
        return {**result, "score": None, "reason": None, "error": metric.error or "no score"}
    return {**result, "score": metric.score, "reason": metric.reason, "error": None}


async def measure_all(
    tasks: List[Tuple[Dict[str, Any], str]], judge: ThkiModel, threshold: float, results_path: Path
) -> None:
    """Evaluate answers with metrics and append the results to a file as they are available."""
    # Twice as many metrics as requests are in progress, so that one is ready
    # to send its next request whenever another one receives a reply
    running = asyncio.Semaphore(2 * judge.max_concurrency)

    async def limited(record: Dict[str, Any], name: str) -> Dict[str, Any]:
        async with running:
            return await measure(record, name, judge, threshold)

    with results_path.open("a", encoding="utf-8") as results:
        coroutines = [limited(record, name) for record, name in tasks]
        for future in tqdm(asyncio.as_completed(coroutines), total=len(tasks), desc="Evaluating", unit="metrics"):
            results.write(json.dumps(await future, ensure_ascii=False) + "\n")
            results.flush()


def summarize(
    results: List[Dict[str, Any]], query_ids: List[str], metrics: List[str], threshold: float
) -> Dict[str, Dict[str, Any]]:
    """Return the mean score, pass rate, and number of evaluated and missing queries of each metric."""
    scores = {(result["qid"], result["metric"]): result["score"] for result in results if result["error"] is None}
    summary = {}
    for name in metrics:
        values = [scores[query_id, name] for query_id in query_ids if (query_id, name) in scores]
        summary[name] = {
            "mean": mean(values) if values else None,
            "pass_rate": sum(value >= threshold for value in values) / len(values) if values else None,
            "evaluated": len(values),
            "missing": len(query_ids) - len(values),
        }
    return summary


def run_evaluation(
    run_dir: Path,
    judge: ThkiModel,
    metrics: Optional[List[str]] = None,
    threshold: float = DEFAULT_THRESHOLD,
    limit: Optional[int] = None,
    query_ids: Optional[List[str]] = None,
) -> EvaluationResult:
    """Evaluate the answers of a benchmark run with DeepEval.

    Args:
        run_dir: Directory of the benchmark run
        judge: Model that judges the answers
        metrics: Names of the metrics (default: all, see ``METRICS``)
        threshold: Score from which on an answer passes a metric
        limit: Number of answers to evaluate (default: all)
        query_ids: IDs of the queries whose answers are evaluated (default: all)

    Raises:
        RagError: If the directory holds no answers
    """
    metrics = metrics or list(METRICS)
    records = read_jsonl(run_dir / ANSWERS_FILE)
    if query_ids is not None:
        selected = set(query_ids)
        records = [record for record in records if record["qid"] in selected]
    records = records[:limit]
    if not records:
        raise RagError(f"no answers in {run_dir} (see 'scrydb-rag bench')")
    # An empty answer (e.g., when the reasoning used up all tokens) would
    # count as perfectly relevant and faithful, as it makes no statements
    empty = [record["qid"] for record in records if not record["answer"].strip()]
    records = [record for record in records if record["answer"].strip()]

    results_path = run_dir / f"deepeval-{slug(judge.get_model_name())}.jsonl"
    done = {(result["qid"], result["metric"]) for result in read_jsonl(results_path) if result["error"] is None}
    tasks = [(record, name) for record in records for name in metrics if (record["qid"], name) not in done]
    if tasks:
        asyncio.run(measure_all(tasks, judge, threshold, results_path))

    summary = summarize(read_jsonl(results_path), [record["qid"] for record in records], metrics, threshold)
    summary_path = results_path.with_suffix(".json")
    summary_path.write_text(
        json.dumps({"judge": judge.get_model_name(), "threshold": threshold, "metrics": summary}, indent=2) + "\n",
        encoding="utf-8",
    )
    return EvaluationResult(judge.get_model_name(), results_path, summary, empty)
