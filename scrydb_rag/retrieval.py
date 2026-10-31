"""Retrieval of passages from a scrydb index."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import scrydb
from thki import Client, embed_batches

from scrydb_rag import RagError
from scrydb_rag.datasets import DEFAULT_TASK

MODES = ("lexical", "semantic", "hybrid")
PRECISIONS = ("binary", "int8", "float")
DEFAULT_TOP_K = 5
# Depths of the candidate lists, as in the runs published with scrydb-eval
DEFAULT_RERANK_DEPTH = 1000
DEFAULT_CANDIDATE_LIMIT = 1000

# Names of the precisions in the method names of scrydb-eval's run files
PRECISION_NAMES = {"binary": "hamming", "int8": "cosine_int8", "float": "cosine_float"}

# Result fields holding the score of a ranking stage, in the order in which
# scrydb's Run prefers them for the score column of run files: the field of
# the last stage comes first. Hamming distances are negated, so that higher
# scores are better for every field.
SCORE_FIELDS = (
    ("rrf_score", 1),
    ("cosine_similarity", 1),
    ("int8_similarity", 1),
    ("hamming_distance", -1),
    ("score", 1),
)


class ApiEmbedding:
    """Embeds texts with an embedding model of an OpenAI-compatible API.

    Has the interface of scrydb's ``SentenceEmbedding``, so it can be attached
    to an index with ``Index.add_model()`` for semantic search with query texts
    that have no stored embedding.
    """

    def __init__(self, client: Client, model: str, task: str = DEFAULT_TASK):
        """Initialize the embedding model.

        Args:
            client: API client
            model: ID of the embedding model
            task: Instruction for embedding queries (documents are embedded without one)
        """
        self.client = client
        self.model = model
        self.task = task

    def encode_queries(self, texts: List[str]) -> np.ndarray:
        # The instruction format of Qwen3-Embedding exactly as the stored
        # query embeddings used it (without a space after "Query:")
        return self.embed([f"Instruct: {self.task}\nQuery:{text}" for text in texts])

    def encode_documents(self, texts: List[str]) -> np.ndarray:
        return self.embed(list(texts))

    def embed(self, texts: List[str]) -> np.ndarray:
        """Return the embeddings of texts as rows of a matrix."""
        vectors = [vector for batch, _ in embed_batches(self.client, self.model, texts, {}) for vector in batch]
        return np.asarray(vectors, dtype=np.float32)


@dataclass(frozen=True)
class RetrievalConfig:
    """Retrieval method, in the vocabulary of scrydb's ``Index.search()``.

    Attributes:
        mode: "lexical" (BM25 over FTS5), "semantic" (vector search), or
            "hybrid" (reciprocal rank fusion of both)
        precision: Vector representation of semantic search ("binary", "int8",
            or "float")
        rerank: Precision of a second-stage rerank of the candidates, or None
        top_k: Number of passages that an answer is generated from
        rerank_depth: Number of first-stage candidates that are reranked
        candidate_limit: Number of candidates of each ranking that hybrid search fuses
    """

    mode: str = "lexical"
    precision: str = "binary"
    rerank: Optional[str] = None
    top_k: int = DEFAULT_TOP_K
    rerank_depth: int = DEFAULT_RERANK_DEPTH
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise RagError(f"unknown mode '{self.mode}' (expected {', '.join(MODES)})")
        if self.precision not in PRECISIONS:
            raise RagError(f"unknown precision '{self.precision}' (expected {', '.join(PRECISIONS)})")
        if self.rerank is not None and self.rerank not in PRECISIONS:
            raise RagError(f"unknown rerank precision '{self.rerank}' (expected {', '.join(PRECISIONS)})")
        if self.mode == "semantic" and self.rerank == self.precision:
            raise RagError(f"reranking with the precision of the first stage ({self.precision}) changes nothing")
        if min(self.top_k, self.rerank_depth, self.candidate_limit) < 1:
            raise RagError("the number of passages and candidates must be positive")

    @property
    def method(self) -> str:
        """Name of the method as in scrydb-eval's run files (e.g., "bm25-cosine_float")."""
        if self.mode == "lexical":
            parts = ["bm25"]
        elif self.mode == "semantic":
            parts = [PRECISION_NAMES[self.precision]]
        else:
            parts = ["rrf", PRECISION_NAMES[self.precision]]
        if self.rerank:
            parts.append(PRECISION_NAMES[self.rerank])
        return "-".join(parts)


@dataclass
class Passage:
    """Retrieved document.

    Attributes:
        docid: ID of the document
        rank: Rank in the results (starting at 1)
        score: Score of the last ranking stage (higher is better)
        text: Text of the document
    """

    docid: str
    rank: int
    score: float
    text: str

    @property
    def title(self) -> str:
        """First line of the text, which is the title for documents that have one."""
        return self.text.strip().split("\n", 1)[0]


def stage_score(result: Mapping[str, Any]) -> float:
    """Return the score of the last ranking stage of a scrydb result."""
    for name, sign in SCORE_FIELDS:
        value = result.get(name)
        if value is not None:
            return sign * value
    return 0.0


class Retriever:
    """Retrieves passages from a scrydb index."""

    def __init__(
        self,
        index_path: Path,
        config: RetrievalConfig = RetrievalConfig(),
        embedding: Optional[ApiEmbedding] = None,
    ):
        """Open the index.

        Args:
            index_path: Path of the scrydb index
            config: Retrieval method
            embedding: Model that embeds query texts for semantic search and
                reranking (without one, only lexical search takes query texts)
        """
        if not Path(index_path).is_file():
            raise RagError(f"no scrydb index at {index_path}")
        self.index = scrydb.Index.open(str(index_path))
        self.embedding = embedding
        if embedding is not None:
            self.index.add_model(embedding)
        self.config = config

    def close(self) -> None:
        self.index.close()

    def search(self, query: str, depth: Optional[int] = None) -> List[Passage]:
        """Retrieve passages for a query text.

        Args:
            query: Text of the query
            depth: Number of passages (default: top_k of the configuration)
        """
        depth = depth or self.config.top_k
        if self.embedding is None and (self.config.mode != "lexical" or self.config.rerank):
            raise RagError("no model to embed the question with (see --embed-model)")
        params = self._search_params(depth)
        results = self.index.search(query, **params)
        if not results and params["rerank"]:
            # sqlite-vec (0.1.9) finds nothing when a KNN query is restricted to
            # a single rowid, which is what scrydb's rerank does for a first stage
            # with one candidate (e.g., if only one document contains the query
            # terms). One candidate needs no rerank, so the first stage is used.
            results = self.index.search(query, **{**params, "rerank": False})
        return self._passages(results, depth)

    def search_stored(
        self, query_ids: Iterable[str], depth: Optional[int] = None, exclude_self: bool = True
    ) -> Dict[str, List[Passage]]:
        """Retrieve passages for queries that are stored in the index.

        Their stored embeddings are used instead of embedding their texts
        again, which reproduces the runs published with scrydb-eval exactly.

        Args:
            query_ids: IDs of the queries
            depth: Number of passages per query (default: top_k of the configuration)
            exclude_self: Whether to remove a document with the ID of the query
                from its results (see below)

        Returns:
            The passages of each query
        """
        depth = depth or self.config.top_k
        # One more result makes up for a removed self-match
        params = self._search_params(depth + 1 if exclude_self else depth)
        run = self.index.batch_search(queries=list(query_ids), **params)
        lost = [query_id for query_id, results in run.items() if not results]
        if lost and params["rerank"]:
            # Reranks that lost the only candidate (see search())
            run.update(self.index.batch_search(queries=lost, **{**params, "rerank": False}))
        rankings = {}
        for query_id, results in run.items():
            if exclude_self:
                # Some datasets (e.g., ArguAna) draw their queries from the
                # corpus. As in BEIR, the query's own document is not a result.
                results = [result for result in results if result["id"] != query_id]
            rankings[query_id] = self._passages(results, depth)
        return rankings

    def _search_params(self, depth: int) -> Dict[str, Any]:
        config = self.config
        top_k = depth
        if config.mode == "semantic" and config.rerank:
            # scrydb reranks the results of the first stage, which are as many
            # as requested, so all candidates are requested and cut afterwards
            top_k = max(depth, config.rerank_depth)
        return {
            "mode": config.mode,
            "top_k": top_k,
            "precision": config.precision,
            "rerank": config.rerank or False,
            "rerank_depth": config.rerank_depth,
            "candidate_limit": config.candidate_limit,
        }

    @staticmethod
    def _passages(results: List[scrydb.SearchResult], depth: int) -> List[Passage]:
        return [
            Passage(result["id"], rank, stage_score(result), str(result["document"].get("text", "")))
            for rank, result in enumerate(results[:depth], start=1)
        ]
