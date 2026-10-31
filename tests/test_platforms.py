import pytest
import requests
from thki import Client

from scrydb_rag import ContextError, RagError
from scrydb_rag.platforms import OLLAMA, OTHER, PLATFORMS, OllamaClient, ollama_params, platform

MODEL = "qwen3:8b"
# The /api/ps entry of a model that Ollama loaded with its default context length
LOADED = {"name": MODEL, "model": MODEL, "context_length": 4096}


class FakeResponse:
    def __init__(self, data):
        self.data = data

    def json(self):
        return self.data


@pytest.fixture
def ollama(monkeypatch):
    """An Ollama client whose requests to /api/ps are answered with the models in its `loaded` list."""
    client = OllamaClient("http://localhost:11434/v1", "")
    client.loaded = [LOADED]
    client.requested = []

    def get(url, **kwargs):
        client.requested.append(url)
        return FakeResponse({"models": client.loaded})

    monkeypatch.setattr(client.session, "get", get)
    return client


def reply(prompt_tokens, completion_tokens):
    return {
        "choices": [{"message": {"content": "An answer [1]."}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


def prompt(chars):
    return [{"role": "system", "content": "Answer."}, {"role": "user", "content": "x" * (chars - len("Answer."))}]


@pytest.mark.parametrize(
    "api_base, expected",
    [
        ("https://chat.kiconnect.nrw/api/v1", PLATFORMS["chat.kiconnect.nrw"]),
        ("https://openrouter.ai/api/v1/", PLATFORMS["openrouter.ai"]),
        ("http://localhost:11434/v1", OLLAMA),
        ("http://gpu-server.example.org:11434/v1", OLLAMA),
        ("http://localhost:8000/v1", OTHER),
    ],
)
def test_platforms_are_recognized_by_host_or_by_ollamas_port(api_base, expected):
    assert platform(api_base) == expected


@pytest.mark.parametrize("api_base", ["http://localhost:11434/api/v1", "http://localhost:11434"])
def test_ollama_needs_the_url_of_its_openai_compatible_api(api_base):
    with pytest.raises(RagError, match="is at http://localhost:11434/v1, not"):
        OllamaClient(api_base, "")
    assert OllamaClient("http://localhost:11434/v1/", "").server == "http://localhost:11434"


def test_max_completion_tokens_are_sent_as_max_tokens():
    params = {"max_completion_tokens": 50, "temperature": 0.2}
    assert ollama_params(params) == {"max_tokens": 50, "temperature": 0.2}
    assert params == {"max_completion_tokens": 50, "temperature": 0.2}
    assert ollama_params({"top_p": 0.9}) == {"top_p": 0.9}


def test_replies_that_fit_into_the_context_are_returned(ollama, monkeypatch):
    requests_sent = []

    def chat(self, model, messages, **params):
        requests_sent.append(params)
        return reply(1000, 500)

    monkeypatch.setattr(Client, "chat", chat)
    assert ollama.chat(MODEL, prompt(4500), max_completion_tokens=600)["usage"]["prompt_tokens"] == 1000
    assert requests_sent == [{"max_tokens": 600}]
    ollama.chat(MODEL, prompt(4500))
    # The context length of a model is only looked up once
    assert ollama.requested == ["http://localhost:11434/api/ps"]


def test_replies_that_overflow_the_context_are_rejected(ollama, monkeypatch):
    monkeypatch.setattr(Client, "chat", lambda self, model, messages, **params: reply(3000, 1500))
    with pytest.raises(ContextError, match=r"reply \(4,500 tokens\) did not fit into the context of qwen3:8b \(4,096"):
        ollama.chat(MODEL, prompt(13000))


def test_replies_to_prompts_that_ollama_cut_are_rejected(ollama, monkeypatch):
    # Ollama cuts a prompt that exceeds the context to about half of it, keeping its end
    monkeypatch.setattr(Client, "chat", lambda self, model, messages, **params: reply(2050, 100))
    with pytest.raises(ContextError, match=r"prompt \(15,000 characters\) did not fit .* cut it to 2,050 tokens"):
        ollama.chat(MODEL, prompt(15000))
    # Without the context length (e.g., for cloud models, which run elsewhere), cut prompts are recognized as well
    ollama.loaded, ollama.context_lengths = [], {}
    with pytest.raises(ContextError, match=r"did not fit into the context of qwen3:8b, so"):
        ollama.chat(MODEL, prompt(15000))


def test_streamed_replies_are_checked_when_their_usage_arrives(ollama, monkeypatch):
    def stream_chat(self, model, messages, **params):
        assert params == {"max_tokens": 600}
        yield {"choices": [{"index": 0, "delta": {"content": "An answer"}}]}
        yield {"choices": [], "usage": {"prompt_tokens": 3000, "completion_tokens": 1500}}

    monkeypatch.setattr(Client, "stream_chat", stream_chat)
    chunks = ollama.stream_chat(MODEL, prompt(13000), max_completion_tokens=600)
    assert next(chunks)["choices"][0]["delta"]["content"] == "An answer"
    with pytest.raises(ContextError):
        next(chunks)


def test_the_context_length_is_unknown_if_ollamas_own_api_fails(ollama, monkeypatch):
    def get(url, **kwargs):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(ollama.session, "get", get)
    assert ollama.context_length(MODEL) is None
