"""Fixtures: a toy scrydb index and a stand-in for the THKI API client."""

import threading
from pathlib import Path
from typing import Any, Callable, Dict, List

import pytest
import scrydb

from scrydb_rag.pipeline import RAGPipeline
from scrydb_rag.retrieval import RetrievalConfig, Retriever, ThkiEmbedding

# Each keyword is one dimension of the toy embeddings
KEYWORDS = ["statin", "vitamin", "coffee", "bagel", "liver", "good", "fracture", "cancer"]

DOCUMENTS = [
    {"docid": "d1", "text": "Statins and breast cancer\n\nStatin use is not associated with a higher risk of breast cancer."},
    {"docid": "d2", "text": "Vitamin D and fractures\n\nVitamin D supplements may prevent fractures in older adults."},
    {"docid": "d3", "text": "Coffee and the liver\n\nDrinking coffee is associated with a lower risk of liver cirrhosis."},
    {"docid": "d4", "text": "Bagels\n\nA bagel is a ring-shaped bread roll."},
    # The same text as query q2, under its ID, as when queries are drawn from the corpus
    {"docid": "q2", "text": "Is coffee good for the liver?"},
]
QUERIES = [
    {"qid": "q1", "text": "Do statins cause breast cancer?"},
    {"qid": "q2", "text": "Is coffee good for the liver?"},
    {"qid": "q3", "text": "bagels"},  # contained by a single document
    {"qid": "train-1", "text": "Vitamin D"},  # not in the test qrels
]
QRELS = "q1\t0\td1\t1\nq2\t0\td3\t1\nq2\t0\td2\t0\nq3\t0\td4\t1\n"


def keyword_vector(text: str) -> List[float]:
    """Return an embedding with a positive dimension for each keyword in a text."""
    lowered = text.lower()
    return [1.0 if keyword in lowered else -0.1 for keyword in KEYWORDS]


def default_reply(messages: List[Dict[str, Any]], params: Dict[str, Any]) -> str:
    return "Statins do not raise the risk of breast cancer [1]."


class FakeClient:
    """Stands in for thki.Client.

    Chat replies come from a function of the messages and request parameters
    (which can also raise errors), and texts are embedded by their keywords.
    All requests are recorded.
    """

    def __init__(self, reply: Callable[[List[Dict[str, Any]], Dict[str, Any]], str] = default_reply):
        self.reply = reply
        self.chat_requests: List[Dict[str, Any]] = []
        self.embedded: List[str] = []
        self.lock = threading.Lock()

    def models(self) -> List[Dict[str, Any]]:
        return [{"id": "openai-gpt-oss-120b"}, {"id": "qwen-qwen3-embedding-8b"}]

    def chat(self, model: str, messages: List[Dict[str, Any]], **params: Any) -> Dict[str, Any]:
        with self.lock:
            self.chat_requests.append({"model": model, "messages": messages, "params": params})
        content = self.reply(messages, params)
        return {
            "choices": [{"message": {"content": content, "reasoning": "Let me think."}, "finish_reason": "stop"}],
            "model": model,
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        }

    def embed(self, model: str, texts: List[str], **params: Any) -> Dict[str, Any]:
        with self.lock:
            self.embedded.extend(texts)
        return {"data": [{"index": i, "embedding": keyword_vector(text)} for i, text in enumerate(texts)]}


@pytest.fixture
def toy_index(tmp_path: Path) -> Path:
    """Path of a scrydb index of the toy documents and queries, with embeddings at all precisions."""
    path = tmp_path / "toy.db"
    with scrydb.Index.open(str(path)) as index:
        index.index_documents(
            [{**document, "embedding": keyword_vector(document["text"])} for document in DOCUMENTS],
            id_field="docid",
            embedding_field="embedding",
            store_int8_embeddings=True,
        )
        index.index_queries(
            [{**query, "embedding": keyword_vector(query["text"])} for query in QUERIES],
            id_field="qid",
            embedding_field="embedding",
            store_int8_embeddings=True,
        )
    return path


@pytest.fixture
def qrels(tmp_path: Path) -> Path:
    path = tmp_path / "test.trec"
    path.write_text(QRELS, encoding="utf-8")
    return path


@pytest.fixture
def client() -> FakeClient:
    return FakeClient()


@pytest.fixture
def pipeline(toy_index: Path, client: FakeClient) -> RAGPipeline:
    retriever = Retriever(toy_index, RetrievalConfig(top_k=2), ThkiEmbedding(client, task="Find documents"))
    yield RAGPipeline(retriever, client, "openai-gpt-oss-120b")
    retriever.close()
