# scrydb-rag

Retrieval-augmented generation with [scrydb](https://github.com/breuert/scrydb) as retrieval
component and the language models of an OpenAI-compatible API, e.g., of
[OpenRouter](https://openrouter.ai) or a local [Ollama](https://ollama.com/) server. The retrieval runs over the BEIR
indices of [scrydb-eval](https://huggingface.co/datasets/breuert/scrydb-eval), single SQLite
files with the documents, an FTS5 index, and Qwen3-Embedding-8B embeddings at three precisions.

```text
question ──► scrydb: BM25 (default) | vector search | RRF ──► top-k passages ──► LLM (e.g., GPT-OSS 120B) ──► answer with [n] citations
```

- **Interactive mode** (`repl`) and one-shot questions (`ask`) for arbitrary questions
- **Benchmark** (`bench`) over all queries of a test collection, resumable, with the retrieval
  evaluated against the qrels
- **Evaluation with DeepEval** (`evaluate`), judged by a model of the API as custom DeepEval model

## Setup

```bash
uv sync
echo "LLM_API_KEY=<your API key>" >> .env
uv run scrydb-rag download nfcorpus         # index (191 MB) and qrels from Hugging Face
```

`scrydb-rag datasets` lists the eight datasets and which are downloaded - more details given in the [setup documentation](./docs/setup.md).

## Usage

```bash
uv run scrydb-rag ask nfcorpus "Do statins cause breast cancer?"
uv run scrydb-rag ask nfcorpus --mode hybrid --precision float -k 10 --passages "How does fasting affect autophagy?"
uv run scrydb-rag repl scifact --mode semantic --rerank float

uv run scrydb-rag bench nfcorpus                     # all 323 test queries, BM25, k=5, GPT-OSS 120B
uv run scrydb-rag bench nfcorpus --limit 20 --evaluate
uv run scrydb-rag evaluate results/nfcorpus/bm25-k5-openai-gpt-oss-120b --judge qwen3.5
```

`scrydb-rag COMMAND --help` lists all options. Models can be given by a unique part of their
ID (e.g., `-m mistral`, `--judge gpt5.5`); `-p KEY=VALUE` passes request parameters, e.g.,
`-p reasoning_effort=low` - more details given in the [usage documentation](./docs/usage.md).
