"""Interactive question answering in the terminal."""

import atexit
import contextlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import List

from thki import (
    BOLD,
    DIM,
    ThkiError,
    generate,
    parse_value,
    print_error,
    report,
    resolve_model,
    styled,
    switch,
)

from scrydb_rag import RagError
from scrydb_rag.pipeline import RAGPipeline, citations
from scrydb_rag.retrieval import MODES, PRECISIONS, Passage, RetrievalConfig

HISTORY_FILE = Path.home() / ".scrydb_rag_history"
TITLE_CHARS = 100

REPL_HELP = """\
/mode [lexical|semantic|hybrid]  show or set the retrieval mode
/precision [binary|int8|float]   show or set the vector precision of semantic search
/rerank [off|binary|int8|float]  show or set the precision of a second-stage rerank
/k [N]                           show or set the number of passages per answer
/passages [on|off]               list the retrieved passages before each answer
/show N                          print passage N of the last answer as the model saw it
/model [NAME]                    show or switch the model
/models                          list the available models
/set [KEY [VALUE]]               show the request parameters, set one (e.g., /set temperature 0.2),
                                 or remove one (e.g., /set temperature)
/reasoning [on|off]              show the reasoning of the model
/usage [on|off]                  show the token usage and timing of answers
/exit                            quit (or press Ctrl+D)

Each question is answered on its own, from the passages retrieved for it.
Press Ctrl+C to stop an answer."""


def shorten(text: str, max_chars: int = TITLE_CHARS) -> str:
    """Return a text on one line, cut after max_chars characters."""
    text = " ".join(text.split())
    return text if len(text) <= max_chars else text[:max_chars - 1].rstrip() + "…"


def print_passages(passages: List[Passage]) -> None:
    """Print the ID, score, and title of retrieved passages."""
    if not passages:
        print(styled("(no passages found)", DIM, sys.stdout))
    for passage in passages:
        line = f"[{passage.rank}] {passage.docid} ({passage.score:.4g})  {shorten(passage.title)}"
        print(styled(line, DIM, sys.stdout))
    print()


def print_sources(answer: str, passages: List[Passage]) -> None:
    """Print the ID and title of the passages that an answer cites."""
    cited = citations(answer, len(passages))
    if not cited:
        return
    print(styled("\nSources:", BOLD, sys.stdout))
    for number in cited:
        passage = passages[number - 1]
        print(styled(f"[{number}] {passage.docid}  {shorten(passage.title)}", DIM, sys.stdout))


def enable_history() -> None:
    """Enable line editing and keep the history of inputs in ~/.scrydb_rag_history."""
    try:
        import readline
    except ImportError:  # not available on Windows
        return
    with contextlib.suppress(OSError):
        readline.read_history_file(str(HISTORY_FILE))
    readline.set_history_length(1000)

    def save_history() -> None:
        with contextlib.suppress(OSError):
            readline.write_history_file(str(HISTORY_FILE))

    atexit.register(save_history)


class Repl:
    """Answers questions typed in the terminal with a RAG pipeline.

    Lines starting with a slash are commands (see ``REPL_HELP``).
    """

    def __init__(
        self,
        pipeline: RAGPipeline,
        dataset: str,
        stream: bool = True,
        show_passages: bool = False,
        show_reasoning: bool = False,
        show_usage: bool = False,
    ):
        self.pipeline = pipeline
        self.dataset = dataset
        self.stream = stream
        self.show_passages = show_passages
        self.show_reasoning = show_reasoning
        self.show_usage = show_usage
        self.context: List[str] = []  # passages of the last answer, as the model saw them
        self.running = True
        self.commands = {
            "help": self.show_help,
            "mode": self.set_mode,
            "precision": self.set_precision,
            "rerank": self.set_rerank,
            "k": self.set_top_k,
            "passages": self.toggle_passages,
            "show": self.show_passage,
            "model": self.set_model,
            "models": self.list_models,
            "set": self.set_param,
            "reasoning": self.toggle_reasoning,
            "usage": self.toggle_usage,
            "exit": self.stop,
            "quit": self.stop,
        }

    @property
    def config(self) -> RetrievalConfig:
        return self.pipeline.retriever.config

    def run(self) -> None:
        """Read and answer questions until the user quits."""
        enable_history()
        documents = len(self.pipeline.retriever.index.documents)
        print(
            styled(
                f"scrydb-rag · {self.dataset} ({documents:,} documents) · {self.config.method}, "
                f"k={self.config.top_k} · {self.pipeline.model} · /help for commands · Ctrl+D to quit",
                DIM,
                sys.stdout,
            )
        )
        while self.running:
            try:
                line = input(">>> ").strip()
                if line.startswith("/"):
                    self.run_command(line[1:])
                elif line:
                    self.ask(line)
            except KeyboardInterrupt:
                print()
            except EOFError:
                print()
                break

    def run_command(self, line: str) -> None:
        """Run a command (given without the slash)."""
        name, _, arg = line.partition(" ")
        command = self.commands.get(name.lower())
        if command is None:
            print_error(f"unknown command '/{name}' (see /help)")
            return
        try:
            command(arg.strip())
        except ThkiError as e:
            print_error(str(e))

    def ask(self, question: str) -> None:
        """Retrieve passages for a question, and print the answer generated from them."""
        try:
            passages = self.pipeline.retriever.search(question)
        except ThkiError as e:  # e.g., the question could not be embedded
            print_error(str(e))
            return
        if self.show_passages:
            print_passages(passages)
        context = self.pipeline.context(passages)
        messages = self.pipeline.messages(question, context)
        try:
            reply = generate(
                self.pipeline.client, self.pipeline.model, messages, self.pipeline.params, self.stream, self.show_reasoning
            )
        except ThkiError as e:
            print_error(str(e))
            return
        self.context = context
        report(reply, self.show_usage)
        if reply.interrupted:
            print(styled("(stopped)", DIM, sys.stdout))
        print_sources(reply.content, passages)
        print()

    def show_help(self, arg: str) -> None:
        print(REPL_HELP)

    def set_mode(self, arg: str) -> None:
        if arg:
            self.pipeline.retriever.config = replace(self.config, mode=arg)
        print(f"mode: {self.config.mode} ({self.config.method}); choices: {', '.join(MODES)}")

    def set_precision(self, arg: str) -> None:
        if arg:
            self.pipeline.retriever.config = replace(self.config, precision=arg)
        print(f"precision: {self.config.precision} ({self.config.method}); choices: {', '.join(PRECISIONS)}")

    def set_rerank(self, arg: str) -> None:
        if arg:
            self.pipeline.retriever.config = replace(self.config, rerank=None if arg in ("off", "none") else arg)
        print(f"rerank: {self.config.rerank or 'off'} ({self.config.method}); choices: off, {', '.join(PRECISIONS)}")

    def set_top_k(self, arg: str) -> None:
        if arg:
            try:
                top_k = int(arg)
            except ValueError:
                raise RagError(f"expected a number, got '{arg}'") from None
            self.pipeline.retriever.config = replace(self.config, top_k=top_k)
        print(f"passages per answer: {self.config.top_k}")

    def toggle_passages(self, arg: str) -> None:
        self.show_passages = switch(arg, self.show_passages)
        print(f"passages: {'on' if self.show_passages else 'off'}")

    def show_passage(self, arg: str) -> None:
        if not self.context:
            raise RagError("no passages yet (ask a question first)")
        if not arg.isdigit() or not 1 <= int(arg) <= len(self.context):
            raise RagError(f"expected the number of a passage from 1 to {len(self.context)}")
        print(self.context[int(arg) - 1])

    def set_model(self, arg: str) -> None:
        if arg:
            self.pipeline.model = resolve_model(self.pipeline.client, arg)
        print(f"model: {self.pipeline.model}")

    def list_models(self, arg: str) -> None:
        for model_id in sorted(model["id"] for model in self.pipeline.client.models()):
            print(f"{'*' if model_id == self.pipeline.model else ' '} {model_id}")

    def set_param(self, arg: str) -> None:
        key, _, value = arg.partition(" ")
        if value.strip():
            self.pipeline.params[key] = parse_value(value.strip())
        elif key:
            self.pipeline.params.pop(key, None)
        params = self.pipeline.params
        print(f"parameters: {json.dumps(params, ensure_ascii=False) if params else '(none)'}")

    def toggle_reasoning(self, arg: str) -> None:
        self.show_reasoning = switch(arg, self.show_reasoning)
        print(f"reasoning: {'on' if self.show_reasoning else 'off'}")

    def toggle_usage(self, arg: str) -> None:
        self.show_usage = switch(arg, self.show_usage)
        print(f"usage: {'on' if self.show_usage else 'off'}")

    def stop(self, arg: str) -> None:
        self.running = False
