import pytest

from scrydb_rag import RagError
from scrydb_rag.datasets import DEFAULT_TASK, TASKS, Dataset, resolve
from scrydb_rag.retrieval import ApiEmbedding, RetrievalConfig, Retriever, stage_score

from conftest import EMBED_MODEL


@pytest.mark.parametrize(
    "config, method",
    [
        (RetrievalConfig("lexical"), "bm25"),
        (RetrievalConfig("lexical", rerank="int8"), "bm25-cosine_int8"),
        (RetrievalConfig("semantic", "binary"), "hamming"),
        (RetrievalConfig("semantic", "binary", "float"), "hamming-cosine_float"),
        (RetrievalConfig("semantic", "int8", "float"), "cosine_int8-cosine_float"),
        (RetrievalConfig("hybrid", "float"), "rrf-cosine_float"),
    ],
)
def test_method_names_follow_the_run_files_of_scrydb_eval(config, method):
    assert config.method == method


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "fuzzy"},
        {"precision": "int4"},
        {"rerank": "cosine"},
        {"mode": "semantic", "precision": "float", "rerank": "float"},
        {"top_k": 0},
    ],
)
def test_invalid_configurations_are_rejected(kwargs):
    with pytest.raises(RagError):
        RetrievalConfig(**kwargs)


def test_stage_score_prefers_the_last_stage():
    assert stage_score({"score": 3.0}) == 3.0
    assert stage_score({"score": 3.0, "cosine_similarity": 0.5}) == 0.5
    assert stage_score({"hamming_distance": 7}) == -7
    assert stage_score({"rrf_score": 0.03, "score": 3.0, "hamming_distance": 7}) == 0.03


def test_lexical_search(toy_index):
    retriever = Retriever(toy_index, RetrievalConfig(top_k=2))
    passages = retriever.search("Do statins cause breast cancer?")
    assert [passage.docid for passage in passages][:1] == ["d1"]
    assert passages[0].rank == 1
    assert passages[0].title == "Statins and breast cancer"
    assert len(passages) <= 2


def test_semantic_search_embeds_the_query_with_the_instruction(toy_index, client):
    embedding = ApiEmbedding(client, EMBED_MODEL, "Find documents")
    retriever = Retriever(toy_index, RetrievalConfig("semantic", "float"), embedding)
    passages = retriever.search("Which vitamin prevents fractures?")
    assert passages[0].docid == "d2"
    # The format of the stored query embeddings: no space after "Query:"
    assert client.embedded == ["Instruct: Find documents\nQuery:Which vitamin prevents fractures?"]


@pytest.mark.parametrize(
    "config",
    [RetrievalConfig("semantic"), RetrievalConfig("hybrid"), RetrievalConfig("lexical", rerank="float")],
)
def test_questions_need_an_embedding_model_unless_searched_lexically(toy_index, config):
    retriever = Retriever(toy_index, config)
    with pytest.raises(RagError, match="--embed-model"):
        retriever.search("Which vitamin prevents fractures?")
    assert retriever.search_stored(["q1"])["q1"][0].docid == "d1"


def test_stored_queries_use_their_stored_embeddings(toy_index, client):
    retriever = Retriever(toy_index, RetrievalConfig("semantic", "binary", "float"), ApiEmbedding(client, EMBED_MODEL))
    rankings = retriever.search_stored(["q1", "q2"])
    assert rankings["q1"][0].docid == "d1"
    assert client.embedded == []


def test_self_matches_are_excluded(toy_index):
    retriever = Retriever(toy_index, RetrievalConfig("semantic", "float", top_k=2))
    assert retriever.search_stored(["q2"], exclude_self=False)["q2"][0].docid == "q2"
    passages = retriever.search_stored(["q2"])["q2"]
    assert [passage.docid for passage in passages][:1] == ["d3"]
    # The removed self-match is made up for by one more result
    assert len(passages) == 2


@pytest.mark.parametrize("rerank", ["binary", "int8", "float"])
def test_rerank_keeps_a_single_candidate(toy_index, client, rerank):
    retriever = Retriever(toy_index, RetrievalConfig("lexical", rerank=rerank), ApiEmbedding(client, EMBED_MODEL))
    assert [passage.docid for passage in retriever.search("bagels")] == ["d4"]
    assert [passage.docid for passage in retriever.search_stored(["q3"])["q3"]] == ["d4"]


def test_semantic_rerank_gets_all_candidates(toy_index):
    retriever = Retriever(toy_index, RetrievalConfig("semantic", "binary", "float", top_k=2, rerank_depth=50))
    assert retriever._search_params(3)["top_k"] == 50
    retriever.config = RetrievalConfig("lexical", rerank="float", top_k=2, rerank_depth=50)
    assert retriever._search_params(3)["top_k"] == 3


def test_missing_index_is_reported(tmp_path):
    with pytest.raises(RagError, match="no scrydb index"):
        Retriever(tmp_path / "missing.db")


def test_datasets_are_resolved_by_name_or_path(tmp_path, toy_index):
    with pytest.raises(RagError, match="unknown dataset"):
        resolve("no-such-dataset", tmp_path)
    with pytest.raises(RagError, match="scrydb-rag download nfcorpus"):
        resolve("nfcorpus", tmp_path)
    found = resolve(str(toy_index), tmp_path)
    assert (found.name, found.index, found.task) == ("toy", toy_index, DEFAULT_TASK)
    copy = tmp_path / "scifact.db"
    copy.write_bytes(toy_index.read_bytes())
    assert resolve(str(copy), tmp_path).task == TASKS["scifact"]


def test_missing_qrels_name_the_available_splits(tmp_path, toy_index):
    (tmp_path / "test.trec").write_text("q1 0 d1 1\n", encoding="utf-8")
    found = Dataset("toy", toy_index, qrels_dir=tmp_path)
    assert found.qrels("test") == tmp_path / "test.trec"
    with pytest.raises(RagError, match="available splits: test"):
        found.qrels("dev")
    with pytest.raises(RagError, match="--qrels"):
        Dataset("toy", toy_index).qrels()
