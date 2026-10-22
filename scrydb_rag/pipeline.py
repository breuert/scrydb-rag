"""Generation of answers from retrieved passages."""

import re
import time
from typing import Any, Dict, List, Optional

from thki import Client, Reply, split_message

from scrydb_rag.retrieval import Passage, Retriever

SYSTEM_PROMPT = """\
You answer requests with passages that a search engine retrieved from a document collection.

- Use only the information in the passages. If they do not contain what is needed, say so \
briefly instead of answering from your own knowledge.
- Cite the passages that support a statement by their numbers in plain square brackets, e.g., \
[1] or [2][3], directly after the statement.
- The request can be a question, a claim, an argument, or a topic: answer a question, assess \
whether the passages support or refute a claim, respond to an argument with the evidence of \
the passages, and summarize what the passages say about a topic.
- Be concise, and answer in the language of the request."""

# Longer passages are cut, so that a few very long documents cannot crowd out the others
MAX_PASSAGE_CHARS = 4000

# Citations like [1], [2, 3], or [2-4], and the 【1】 or 【1†source】 that GPT-OSS was trained to write
CITATION_RE = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\]|【(\d+(?:\s*[,–-]\s*\d+)*)(?:†[^】]*)?】")


def truncate(text: str, max_chars: int) -> str:
    """Cut a text after at most max_chars characters, at a word boundary if possible."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    if " " in cut[max_chars // 2:]:
        cut = cut[:cut.rindex(" ")]
    return cut.rstrip() + " …"


def citations(answer: str, count: int) -> List[int]:
    """Return the numbers of the passages that an answer cites, in their order of appearance.

    Args:
        answer: Text of the answer
        count: Number of passages (other numbers in brackets are ignored)
    """
    cited: Dict[int, None] = {}
    for match in CITATION_RE.finditer(answer):
        for part in re.split(r"\s*,\s*", match.group(1) or match.group(2)):
            bounds = [int(number) for number in re.split(r"\s*[–-]\s*", part)]
            for number in range(max(bounds[0], 1), min(bounds[-1], count) + 1):
                cited[number] = None
    return list(cited)


class RAGPipeline:
    """Answers requests with passages retrieved from a scrydb index."""

    def __init__(
        self,
        retriever: Retriever,
        client: Client,
        model: str,
        params: Optional[Dict[str, Any]] = None,
        system_prompt: str = SYSTEM_PROMPT,
        max_passage_chars: int = MAX_PASSAGE_CHARS,
    ):
        """Initialize the pipeline.

        Args:
            retriever: Retriever of the passages
            client: API client
            model: ID of the model that generates the answers
            params: Additional request parameters (e.g., temperature)
            system_prompt: Instructions for the model
            max_passage_chars: Number of characters after which passages are cut
        """
        self.retriever = retriever
        self.client = client
        self.model = model
        self.params = params or {}
        self.system_prompt = system_prompt
        self.max_passage_chars = max_passage_chars

    def context(self, passages: List[Passage]) -> List[str]:
        """Return the texts of passages as they are given to the model."""
        return [truncate(passage.text.strip(), self.max_passage_chars) for passage in passages]

    def messages(self, request: str, context: List[str]) -> List[Dict[str, str]]:
        """Return the chat messages that ask the model to answer a request with passages."""
        passages = "\n\n".join(f"[{number}] {text}" for number, text in enumerate(context, start=1))
        user = f"Passages:\n\n{passages or '(none found)'}\n\nRequest: {request}"
        return [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": user}]

    def generate(self, request: str, context: List[str]) -> Reply:
        """Generate the answer to a request from passages, without printing it.

        Raises:
            APIError: If the request fails or the API returns an error
        """
        start = time.monotonic()
        data = self.client.chat(self.model, self.messages(request, context), **self.params)
        choice = data["choices"][0]
        reasoning, content = split_message(choice.get("message") or {})
        return Reply(
            content=content,
            reasoning=reasoning,
            model=data.get("model") or "",
            finish_reason=choice.get("finish_reason"),
            usage=data.get("usage") or {},
            seconds=time.monotonic() - start,
        )
