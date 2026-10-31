"""The platforms that serve the language and embedding models.

Any API that is compatible with OpenAI's chat completions and embeddings
endpoints works. For KI:connect.nrw, OpenRouter, and Ollama, scrydb-rag knows
the IDs of the default models where the platform serves them: GPT-OSS 120B,
which generates and judges the answers, and Qwen3-Embedding-8B, which
embedded the documents and queries of the scrydb-eval indices.

Ollama runs a model with a context length (4096 tokens on machines with less
than 24 GiB of VRAM) that cuts longer prompts without notice, keeping their
end, and it drops the beginning of a prompt when the reply fills the context.
As the instructions and the first passages are lost then, ``OllamaClient``
rejects such replies.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Type
from urllib.parse import urlparse

import requests
from thki import CONNECT_TIMEOUT, DEFAULT_TIMEOUT, Client

from scrydb_rag import ContextError, RagError
from scrydb_rag.benchmark import DEFAULT_WORKERS

OLLAMA_PORT = 11434
# RAG prompts have 3.8 to 5.2 characters per token (measured for passages of
# four BEIR datasets with the tokenizers of Qwen3 and Llama 3.2). Ollama cuts
# a prompt that exceeds the context to about half of it, which at least
# doubles that ratio.
CUT_CHARS_PER_TOKEN = 6.5
RAISE_CONTEXT = "raise the context length in Ollama's settings (or with OLLAMA_CONTEXT_LENGTH)"


def ollama_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Return request parameters with max_completion_tokens as max_tokens, as Ollama ignores the former."""
    if "max_completion_tokens" not in params:
        return params
    params = dict(params)
    params["max_tokens"] = params.pop("max_completion_tokens")
    return params


class OllamaClient(Client):
    """Client for the OpenAI-compatible API of an Ollama server.

    Sends max_completion_tokens as max_tokens, and rejects replies to prompts
    that did not fit into the context of the model (see the module docstring).
    """

    def __init__(self, api_base: str, *args: Any, **kwargs: Any):
        """Initialize the client (see ``Client``).

        Raises:
            RagError: If the base URL is not the one of Ollama's OpenAI-compatible API
        """
        url = urlparse(api_base)
        self.server = f"{url.scheme}://{url.netloc}"  # root of Ollama's own API (e.g., /api/ps)
        if url.path.rstrip("/") != "/v1":
            raise RagError(f"the OpenAI-compatible API of Ollama is at {self.server}/v1, not {api_base}")
        super().__init__(api_base, *args, **kwargs)
        self.context_lengths: Dict[str, int] = {}

    def chat(self, model: str, messages: List[Dict[str, Any]], **params: Any) -> Dict[str, Any]:
        data = super().chat(model, messages, **ollama_params(params))
        self.check_context(model, messages, data.get("usage") or {})
        return data

    def stream_chat(self, model: str, messages: List[Dict[str, Any]], **params: Any) -> Iterator[Dict[str, Any]]:
        for chunk in super().stream_chat(model, messages, **ollama_params(params)):
            if chunk.get("usage"):
                self.check_context(model, messages, chunk["usage"])
            yield chunk

    def context_length(self, model: str) -> Optional[int]:
        """Return the context length that Ollama runs a loaded model with (None if it is not known)."""
        if model not in self.context_lengths:
            try:
                response = self.session.get(self.server + "/api/ps", timeout=(CONNECT_TIMEOUT, self.timeout))
                loaded = response.json()["models"]
            except (requests.RequestException, ValueError, KeyError, TypeError):
                return None
            for entry in loaded:
                if model in (entry.get("name"), entry.get("model")) and entry.get("context_length"):
                    self.context_lengths[model] = entry["context_length"]
        return self.context_lengths.get(model)

    def check_context(self, model: str, messages: List[Dict[str, Any]], usage: Dict[str, Any]) -> None:
        """Check that the prompt of a reply was not cut and that the reply did not fill the context.

        Raises:
            ContextError: If Ollama dropped the beginning of the prompt
        """
        prompt, completion = usage.get("prompt_tokens") or 0, usage.get("completion_tokens") or 0
        chars = sum(len(message.get("content") or "") for message in messages)
        context = self.context_length(model)
        window = f"the context of {model}" + (f" ({context:,} tokens)" if context else "")
        if prompt and chars / prompt > CUT_CHARS_PER_TOKEN:
            raise ContextError(
                f"the prompt ({chars:,} characters) did not fit into {window}, so Ollama cut it to {prompt:,} "
                f"tokens, dropping its beginning; {RAISE_CONTEXT}"
            )
        if context and prompt + completion > context:
            raise ContextError(
                f"the prompt and the reply ({prompt + completion:,} tokens) did not fit into {window}, so Ollama "
                f"dropped the beginning of the prompt; {RAISE_CONTEXT}"
            )


@dataclass(frozen=True)
class Platform:
    """Defaults of an API.

    Attributes:
        model: ID of GPT-OSS 120B, which generates and judges the answers by
            default (None if the platform does not serve it)
        embed_model: ID of Qwen3-Embedding-8B, which embeds questions for
            semantic search (None if it is not known)
        needs_key: Whether requests need an API key
        workers: Number of requests that are sent concurrently
        timeout: Seconds to wait for a reply, which arrives as a whole unless it is streamed
        client: Class of the API client
    """

    model: Optional[str] = None
    embed_model: Optional[str] = None
    needs_key: bool = False
    workers: int = DEFAULT_WORKERS
    timeout: float = DEFAULT_TIMEOUT
    client: Type[Client] = Client


# Platforms by the host of their API
PLATFORMS = {
    "chat.kiconnect.nrw": Platform("openai-gpt-oss-120b", "qwen-qwen3-embedding-8b", needs_key=True),
    "openrouter.ai": Platform("openai/gpt-oss-120b", "qwen/qwen3-embedding-8b", needs_key=True),
}
# Ollama (on any host, with its port) serves the models that were pulled, and
# answers one request at a time unless OLLAMA_NUM_PARALLEL says otherwise. Local
# models may generate slowly: gemma4:12b took up to 3 minutes for an answer
# with its reasoning on a laptop.
OLLAMA = Platform(embed_model="qwen3-embedding:8b", workers=1, timeout=900, client=OllamaClient)
OTHER = Platform()


def platform(api_base: str) -> Platform:
    """Return the defaults of the platform of an API."""
    url = urlparse(api_base)
    if url.port == OLLAMA_PORT:
        return OLLAMA
    return PLATFORMS.get(url.hostname or "", OTHER)
