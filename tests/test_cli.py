import pytest

from scrydb_rag import RagError
from scrydb_rag.cli import build_parser, make_client, make_pipeline, set_platform_defaults
from scrydb_rag.datasets import resolve
from scrydb_rag.platforms import OllamaClient
from scrydb_rag.repl import Repl

OLLAMA = "http://localhost:11434/v1"
VLLM = "http://localhost:8000/v1"


@pytest.fixture
def environment(monkeypatch):
    """The environment without the variables of the API, which the shell might set."""
    for name in ("LLM_API_BASE", "LLM_API_KEY", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def parse(*argv):
    args = build_parser().parse_args(argv)
    set_platform_defaults(args)
    return args


def test_defaults(environment):
    args = parse("bench", "nfcorpus")
    assert (args.mode, args.rerank, args.top_k) == ("lexical", "none", 5)
    assert args.api_base == "https://chat.kiconnect.nrw/api/v1"
    assert args.model == "openai-gpt-oss-120b"
    assert args.embed_model == "qwen-qwen3-embedding-8b"
    assert args.judge == "openai-gpt-oss-120b"
    assert (args.split, args.workers, args.timeout, args.evaluate) == ("test", 3, 240, False)


def test_default_models_depend_on_the_platform(environment):
    args = parse("bench", "nfcorpus", "--api-base", "https://openrouter.ai/api/v1/")
    assert (args.model, args.judge, args.embed_model) == (
        "openai/gpt-oss-120b",
        "openai/gpt-oss-120b",
        "qwen/qwen3-embedding-8b",
    )
    environment.setenv("LLM_API_BASE", "https://openrouter.ai/api/v1")
    environment.setenv("LLM_MODEL", "nvidia/nemotron-3.5-lightning:free")
    args = parse("bench", "nfcorpus")
    assert args.api_base == "https://openrouter.ai/api/v1"
    # GPT-OSS 120B still judges, so that the answers of all models are judged alike
    assert (args.model, args.judge) == ("nvidia/nemotron-3.5-lightning:free", "openai/gpt-oss-120b")
    args = parse("bench", "nfcorpus", "-m", "mistral", "--judge", "qwen3.5", "--embed-model", "e5")
    assert (args.model, args.judge, args.embed_model) == ("mistral", "qwen3.5", "e5")


def test_ollama_needs_a_model_and_answers_one_request_at_a_time(environment, toy_index, client):
    args = parse("ask", "--api-base", OLLAMA, str(toy_index), "Anything?")
    assert (args.model, args.embed_model) == (None, "qwen3-embedding:8b")
    with pytest.raises(RagError, match="set LLM_MODEL or use -m"):
        make_pipeline(args, client, resolve(str(toy_index)))
    environment.setenv("LLM_MODEL", "qwen3:8b")
    args = parse("bench", "--api-base", OLLAMA, "nfcorpus")
    assert (args.model, args.judge, args.workers, args.timeout) == ("qwen3:8b", "qwen3:8b", 1, 900)
    args = parse("bench", "--api-base", OLLAMA, "--workers", "4", "--timeout", "60", "nfcorpus")
    assert (args.workers, args.timeout) == (4, 60)


def test_other_platforms_have_no_default_models(environment):
    environment.setenv("LLM_MODEL", "Qwen/Qwen3-8B")
    args = parse("bench", "--api-base", VLLM, "nfcorpus")
    assert (args.model, args.judge, args.embed_model, args.workers) == ("Qwen/Qwen3-8B", "Qwen/Qwen3-8B", None, 3)


def test_only_the_known_platforms_need_a_key(environment):
    with pytest.raises(RagError, match="LLM_API_KEY is not set"):
        make_client(parse("evaluate", "run"))
    client = make_client(parse("evaluate", "--api-base", OLLAMA, "run"))
    assert isinstance(client, OllamaClient)
    assert client.server == "http://localhost:11434"
    assert make_client(parse("evaluate", "--api-base", VLLM, "run")).api_base == VLLM


def test_repl_answers_with_sources(pipeline, capsys):
    repl = Repl(pipeline, "toy", stream=False, show_passages=True)
    repl.ask("Do statins cause breast cancer?")
    output = capsys.readouterr().out
    assert "[1] d1 (" in output  # the retrieved passages
    assert "Statins do not raise the risk of breast cancer [1]." in output
    assert "Sources:\n[1] d1  Statins and breast cancer" in output
    repl.run_command("show 1")
    assert capsys.readouterr().out.startswith("Statins and breast cancer\n\nStatin use")


def test_repl_commands_change_the_retrieval(pipeline, capsys):
    repl = Repl(pipeline, "toy", stream=False)
    repl.run_command("mode semantic")
    repl.run_command("precision int8")
    repl.run_command("rerank float")
    repl.run_command("k 3")
    config = pipeline.retriever.config
    assert (config.method, config.top_k) == ("cosine_int8-cosine_float", 3)
    repl.run_command("rerank off")
    assert pipeline.retriever.config.rerank is None
    repl.run_command("set reasoning_effort low")
    assert pipeline.params == {"reasoning_effort": "low"}
    capsys.readouterr()

    for command in ("k zero", "mode fuzzy", "show 1", "nonsense"):
        repl.run_command(command)
        assert "error:" in capsys.readouterr().err
    assert pipeline.retriever.config.mode == "semantic"
