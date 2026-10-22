"""The BEIR datasets that are indexed with scrydb and shared on Hugging Face.

Each dataset is a single SQLite file with the documents, the queries, the
FTS5 index, and the Qwen3-Embedding-8B embeddings of documents and queries,
plus the qrels of its splits as TREC files:

    https://huggingface.co/datasets/breuert/scrydb-eval

Downloads mirror the layout of that repository under the data directory,
e.g., ``data/indices/beir/nfcorpus.db`` and ``data/qrels/nfcorpus/test.trec``.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from huggingface_hub import snapshot_download

from scrydb_rag import RagError

REPO_ID = "breuert/scrydb-eval"
REPO_TYPE = "dataset"
DEFAULT_DATA_DIR = Path("data")

# Instructions the stored query embeddings were computed with (by the
# embedding scripts of scrydb-eval). Queries that are embedded at search time
# need the same instruction to match the document embeddings.
TASKS = {
    "arguana": "Given a claim, find documents that refute the claim",
    "fiqa": "Given a financial question, retrieve user replies that best answer the question",
    "nfcorpus": "Given a search query, retrieve relevant research articles that answer the query",
    "quora": "Given a question, retrieve questions that are semantically equivalent to the given question",
    "scidocs": "Given a scientific paper title, retrieve paper abstracts that are cited by the given paper",
    "scifact": "Given a scientific claim, retrieve documents that support or refute the claim",
    "trec-covid": "Given a web search query, retrieve relevant passages that answer the query",
    "webis-touche2020": "Given a question on a controversial topic, retrieve relevant arguments that support or "
    "oppose the topic",
}
# The general-purpose instruction recommended for Qwen3-Embedding
DEFAULT_TASK = "Given a web search query, retrieve relevant passages that answer the query"


@dataclass(frozen=True)
class Dataset:
    """A document collection indexed with scrydb, with the qrels of its splits if known.

    Attributes:
        name: Name of the dataset (e.g., "nfcorpus")
        index: Path of the scrydb index
        task: Instruction for embedding queries (see ``ThkiEmbedding``)
        qrels_dir: Directory with the qrels of the splits as <split>.trec
    """

    name: str
    index: Path
    task: str = DEFAULT_TASK
    qrels_dir: Optional[Path] = None

    def qrels(self, split: str = "test") -> Path:
        """Return the path of the qrels of a split.

        Raises:
            RagError: If the qrels of the split are not available
        """
        path = self.qrels_dir / f"{split}.trec" if self.qrels_dir else None
        if path is None or not path.is_file():
            available = sorted(p.stem for p in self.qrels_dir.glob("*.trec")) if self.qrels_dir else []
            hint = f"available splits: {', '.join(available)}" if available else "see --qrels"
            raise RagError(f"no qrels of the {split} split of {self.name} ({hint})")
        return path

    @property
    def downloaded(self) -> bool:
        return self.index.is_file()


def dataset(name: str, data_dir: Path = DEFAULT_DATA_DIR) -> Dataset:
    """Return one of the datasets on Hugging Face, located in the data directory."""
    return Dataset(
        name,
        data_dir / "indices" / "beir" / f"{name}.db",
        TASKS[name],
        data_dir / "qrels" / name,
    )


def resolve(name: str, data_dir: Path = DEFAULT_DATA_DIR) -> Dataset:
    """Return the dataset with a name, or the one of a scrydb index given by its path.

    Raises:
        RagError: If the name is unknown, or the dataset is not downloaded yet
    """
    if name in TASKS:
        found = dataset(name, data_dir)
        if not found.downloaded:
            raise RagError(f"{name} is not downloaded yet (run 'scrydb-rag download {name}')")
        return found
    path = Path(name)
    if path.is_file():
        # A copy of one of the datasets (e.g., downloaded with scrydb-eval) keeps its instruction
        return Dataset(path.stem, path, TASKS.get(path.stem, DEFAULT_TASK))
    raise RagError(
        f"unknown dataset '{name}' (expected the path of a scrydb index or one of: {', '.join(TASKS)})"
    )


def download(names: List[str], data_dir: Path = DEFAULT_DATA_DIR) -> List[Dataset]:
    """Download the indices and qrels of datasets from Hugging Face.

    Downloads are resumed if interrupted, and files that are complete are not
    downloaded again.
    """
    unknown = [name for name in names if name not in TASKS]
    if unknown:
        raise RagError(f"unknown dataset '{unknown[0]}' (expected one of: {', '.join(TASKS)})")
    patterns = [pattern for name in names for pattern in (f"indices/beir/{name}.db", f"qrels/{name}/*.trec")]
    snapshot_download(REPO_ID, repo_type=REPO_TYPE, local_dir=data_dir, allow_patterns=patterns)
    return [dataset(name, data_dir) for name in names]
