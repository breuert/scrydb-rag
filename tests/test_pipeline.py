import pytest

from scrydb_rag.pipeline import SYSTEM_PROMPT, citations, truncate


def test_short_texts_are_not_truncated():
    assert truncate("A short passage.", 100) == "A short passage."


def test_long_texts_are_cut_at_a_word_boundary():
    text = "word " * 100
    cut = truncate(text, 42)
    assert cut.endswith("word …")
    assert len(cut) <= 42 + 2


@pytest.mark.parametrize(
    "answer, cited",
    [
        ("No citations.", []),
        ("A [2]. B [1][3].", [2, 1, 3]),
        ("A [1, 3]. B [2-4].", [1, 3, 2, 4]),
        ("GPT-OSS style 【2】 and 【1†L4-L9】.", [2, 1]),
        ("Out of range [0], [6], [2019-2021], and [1-999999999].", [1, 2, 3, 4, 5]),
    ],
)
def test_citations(answer, cited):
    assert citations(answer, 5) == cited


def test_messages_number_the_passages(pipeline):
    messages = pipeline.messages("Do statins cause cancer?", ["First passage.", "Second passage."])
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert messages[1]["role"] == "user"
    assert "[1] First passage.\n\n[2] Second passage." in messages[1]["content"]
    assert messages[1]["content"].endswith("Request: Do statins cause cancer?")
    assert "(none found)" in pipeline.messages("Anything?", [])[1]["content"]


def test_context_truncates_the_passages(pipeline):
    pipeline.max_passage_chars = 20
    passages = pipeline.retriever.search("Do statins cause breast cancer?")
    assert all(len(text) <= 22 for text in pipeline.context(passages))


def test_generate_returns_the_reply(pipeline, client):
    pipeline.params = {"temperature": 0.2}
    reply = pipeline.generate("Do statins cause breast cancer?", ["Statins are safe."])
    assert reply.content == "Statins do not raise the risk of breast cancer [1]."
    assert reply.reasoning == "Let me think."
    assert reply.finish_reason == "stop"
    assert reply.usage["completion_tokens"] == 20
    assert client.chat_requests[0]["params"] == {"temperature": 0.2}
