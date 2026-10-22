from scrydb_rag.cli import build_parser
from scrydb_rag.repl import Repl


def test_defaults():
    args = build_parser().parse_args(["bench", "nfcorpus"])
    assert (args.mode, args.rerank, args.top_k) == ("lexical", "none", 5)
    assert args.model == "openai-gpt-oss-120b"
    assert args.embed_model == "qwen-qwen3-embedding-8b"
    assert args.judge == "openai-gpt-oss-120b"
    assert (args.split, args.workers, args.evaluate) == ("test", 3, False)


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
