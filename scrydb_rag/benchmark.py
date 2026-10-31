"""Benchmark runs of the RAG pipeline over the queries of a test collection.

A run answers each query in the qrels of a split and keeps its results in an
output directory:

    config.json     settings of the run, which a resumed run must match
    answers.jsonl   one record per query: the retrieved documents, the
                    passages given to the model, and its answer
    run.trec        the retrieved documents as TREC run file
    retrieval.json  effectiveness of the retrieval, measured with the qrels

Each answer is appended as soon as it is generated, so an interrupted run is
resumed by repeating the command, which also retries the queries that failed.
"""

import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import ir_measures
from ir_measures import RR, P, R, ScoredDoc, nDCG
from thki import Reply, ThkiError
from tqdm import tqdm

from scrydb_rag import ContextError, RagError
from scrydb_rag.pipeline import RAGPipeline
from scrydb_rag.retrieval import Passage

CONFIG_FILE = "config.json"
ANSWERS_FILE = "answers.jsonl"
RUN_FILE = "run.trec"
RETRIEVAL_FILE = "retrieval.json"
DEFAULT_RESULTS_DIR = Path("results")
# KI:connect.nrw answers more concurrent requests per key with status 429
DEFAULT_WORKERS = 3
# Documents per query in the run file; at least 10, so that nDCG@10 can be
# compared with the results of scrydb-eval
RUN_DEPTH = 10


@dataclass
class BenchmarkResult:
    """Outcome of a benchmark run.

    Attributes:
        output_dir: Directory of the run
        records: Records of the answered queries, in the order of the qrels
        failed: Error messages of the queries that could not be answered, by query ID
        skipped: IDs of the queries in the qrels that are not stored in the index
        effectiveness: Retrieval measures, averaged over the answered queries
    """

    output_dir: Path
    records: List[Dict[str, Any]]
    failed: Dict[str, str] = field(default_factory=dict)
    skipped: List[str] = field(default_factory=list)
    effectiveness: Dict[str, float] = field(default_factory=dict)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Return the records of a JSONL file (none if it does not exist)."""
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as lines:
        return [json.loads(line) for line in lines if line.strip()]


def qrels_query_ids(path: Path) -> List[str]:
    """Return the IDs of the queries in a qrels file, in their order of appearance."""
    with path.open(encoding="utf-8") as lines:
        return list(dict.fromkeys(line.split()[0] for line in lines if line.strip()))


def slug(name: str) -> str:
    """Return a name that can be part of a file name (e.g., of a model ID).

    The slash of model IDs like "openai/gpt-oss-120b" (OpenRouter) becomes
    "__", so that they differ from IDs like "openai-gpt-oss-120b" (KI:connect.nrw).
    """
    return "".join(char if char.isalnum() or char in "._-" else "-" for char in name.replace("/", "__"))


def default_output_dir(dataset: str, pipeline: RAGPipeline) -> Path:
    """Return the directory for the results of a dataset and a pipeline configuration."""
    config = pipeline.retriever.config
    return DEFAULT_RESULTS_DIR / dataset / f"{config.method}-k{config.top_k}-{slug(pipeline.model)}"


def run_settings(pipeline: RAGPipeline, dataset: str, split: str, exclude_self: bool) -> Dict[str, Any]:
    """Return the settings that determine the results of a benchmark run."""
    config = pipeline.retriever.config
    return {
        "dataset": dataset,
        "split": split,
        "method": config.method,
        "mode": config.mode,
        "precision": config.precision,
        "rerank": config.rerank,
        "top_k": config.top_k,
        "rerank_depth": config.rerank_depth,
        "candidate_limit": config.candidate_limit,
        "exclude_self_matches": exclude_self,
        "api_base": pipeline.client.api_base,
        "model": pipeline.model,
        "params": pipeline.params,
        "system_prompt": pipeline.system_prompt,
        "max_passage_chars": pipeline.max_passage_chars,
    }


def check_settings(path: Path, settings: Dict[str, Any]) -> None:
    """Save the settings of a run, or check that they match those of the run that is resumed.

    Raises:
        RagError: If the directory holds a run with other settings
    """
    if not path.is_file():
        path.write_text(json.dumps(settings, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return
    previous = json.loads(path.read_text(encoding="utf-8"))
    # Settings that earlier versions did not record (e.g., api_base) are not checked for their runs
    changed = [key for key in previous if previous[key] != settings.get(key)]
    if changed:
        raise RagError(
            f"{path.parent} holds a run with other settings ({', '.join(changed)}); "
            "choose another directory with --output-dir"
        )


def make_record(query_id: str, query: str, ranking: List[Passage], context: List[str], reply: Reply) -> Dict[str, Any]:
    """Return the record of an answered query.

    The first len(context) documents of the ranking are the passages of the context.
    """
    return {
        "qid": query_id,
        "query": query,
        "ranking": [[passage.docid, passage.score] for passage in ranking],
        "retrieval_context": context,
        "answer": reply.content,
        "reasoning": reply.reasoning,
        "model": reply.model,
        "finish_reason": reply.finish_reason,
        "usage": reply.usage,
        "seconds": round(reply.seconds, 3),
    }


def write_run(path: Path, records: List[Dict[str, Any]], tag: str) -> None:
    """Write the rankings of records as TREC run file."""
    with path.open("w", encoding="utf-8") as run:
        for record in records:
            for rank, (docid, score) in enumerate(record["ranking"], start=1):
                run.write(f"{record['qid']} Q0 {docid} {rank} {score} {tag}\n")


def evaluate_retrieval(records: List[Dict[str, Any]], qrels_path: Path, top_k: int) -> Dict[str, float]:
    """Return retrieval measures, averaged over the queries of records.

    Queries without any retrieved document count with a score of 0, as with
    ``trec_eval -c``, as each of them is still answered from an empty context.
    """
    if not records:
        return {}
    measures = list(dict.fromkeys([nDCG @ 10, nDCG @ top_k, RR @ 10, P @ top_k, R @ top_k]))
    run = [ScoredDoc(record["qid"], docid, score) for record in records for docid, score in record["ranking"]]
    scores: Dict[Any, Dict[str, float]] = defaultdict(dict)
    for metric in ir_measures.iter_calc(measures, list(ir_measures.read_trec_qrels(str(qrels_path))), run):
        scores[metric.measure][metric.query_id] = metric.value
    return {
        str(measure): sum(scores[measure].get(record["qid"], 0.0) for record in records) / len(records)
        for measure in measures
    }


def run_benchmark(
    pipeline: RAGPipeline,
    dataset: str,
    qrels_path: Path,
    output_dir: Path,
    split: str = "test",
    limit: Optional[int] = None,
    workers: int = DEFAULT_WORKERS,
    exclude_self: bool = True,
) -> BenchmarkResult:
    """Answer the queries of a test collection and evaluate the retrieval.

    Args:
        pipeline: RAG pipeline, whose index stores the queries
        dataset: Name of the dataset
        qrels_path: Qrels of the queries to answer
        output_dir: Directory of the run (see the module docstring)
        split: Name of the split of the qrels
        limit: Number of queries to answer (default: all)
        workers: Number of answers that are generated concurrently
        exclude_self: Whether to remove documents with the ID of the query from its results

    Raises:
        RagError: If the output directory holds a run with other settings
        ContextError: If a prompt did not fit into the context of the model
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    check_settings(output_dir / CONFIG_FILE, run_settings(pipeline, dataset, split, exclude_self))

    stored = pipeline.retriever.index.queries
    query_ids = qrels_query_ids(qrels_path)
    skipped = [query_id for query_id in query_ids if query_id not in stored]
    query_ids = [query_id for query_id in query_ids if query_id in stored][:limit]

    answers_path = output_dir / ANSWERS_FILE
    answered = {record["qid"] for record in read_jsonl(answers_path)}
    pending = [query_id for query_id in query_ids if query_id not in answered]
    failed: Dict[str, str] = {}
    if pending:
        config = pipeline.retriever.config
        # Retrieval happens in this thread, which scrydb's SQLite connection belongs to
        rankings = pipeline.retriever.search_stored(pending, max(config.top_k, RUN_DEPTH), exclude_self)
        pool = ThreadPoolExecutor(workers)
        try:
            futures = {}
            for query_id in pending:
                query = stored[query_id]["text"]
                context = pipeline.context(rankings[query_id][:config.top_k])
                futures[pool.submit(pipeline.generate, query, context)] = (query_id, query, context)
            with answers_path.open("a", encoding="utf-8") as answers:
                progress = tqdm(as_completed(futures), total=len(futures), desc="Generating answers", unit="queries")
                for future in progress:
                    query_id, query, context = futures[future]
                    try:
                        reply = future.result()
                    except ContextError:
                        raise  # the context is too small, likely for the other prompts as well
                    except ThkiError as e:
                        failed[query_id] = str(e)
                        continue
                    record = make_record(query_id, query, rankings[query_id], context, reply)
                    answers.write(json.dumps(record, ensure_ascii=False) + "\n")
                    answers.flush()
        finally:
            # Without this, an interrupted run would wait for all queries to be answered
            pool.shutdown(wait=False, cancel_futures=True)

    records_by_id = {record["qid"]: record for record in read_jsonl(answers_path)}
    records = [records_by_id[query_id] for query_id in query_ids if query_id in records_by_id]
    write_run(output_dir / RUN_FILE, records, pipeline.retriever.config.method)
    effectiveness = evaluate_retrieval(records, qrels_path, pipeline.retriever.config.top_k)
    (output_dir / RETRIEVAL_FILE).write_text(
        json.dumps({"queries": len(records), **effectiveness}, indent=2) + "\n", encoding="utf-8"
    )
    return BenchmarkResult(output_dir, records, failed, skipped, effectiveness)
