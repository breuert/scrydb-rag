## Setup

```bash
uv sync
echo "LLM_API_KEY=<your API key>" >> .env
uv run scrydb-rag download nfcorpus         # index (191 MB) and qrels from Hugging Face
```

The `sqlite3` module of the Python has to support loading extensions, which uv's Python builds
and Homebrew's do (macOS's system Python does not, see scrydb's README). `scrydb-rag datasets`
lists the eight datasets and which are downloaded; they range from 191 MB (NFCorpus) to 11.5 GB
(Quora). Instead of a dataset name, `ask`, `repl`, and `bench` also take the path of a scrydb
index (`bench` then needs `--qrels`).

### Platforms

The models are requested from an API that is compatible with OpenAI's, which three variables set
(in the environment or in `.env`):

| Variable | |
| :-- | :-- |
| `LLM_API_BASE` | base URL of the API (or `--api-base`), by default the one of KI:connect.nrw |
| `LLM_API_KEY` | its API key, which local servers (e.g., of Ollama) do not need |
| `LLM_MODEL` | model that generates the answers (or `-m`), by default GPT-OSS 120B |

On the platforms with known model IDs, GPT-OSS 120B generates and judges the answers, and
Qwen3-Embedding-8B embeds questions for semantic search:

| Platform | `LLM_API_BASE` | GPT-OSS 120B | Qwen3-Embedding-8B |
| :-- | :-- | :-- | :-- |
| KI:connect.nrw | `https://chat.kiconnect.nrw/api/v1` (default) | `openai-gpt-oss-120b` | `qwen-qwen3-embedding-8b` |
| OpenRouter | `https://openrouter.ai/api/v1` | `openai/gpt-oss-120b` | `qwen/qwen3-embedding-8b` |
| Ollama | `http://localhost:11434/v1` | | `qwen3-embedding:8b` |

`LLM_MODEL` replaces GPT-OSS 120B only for generating the answers; GPT-OSS 120B stays the judge,
so that the answers of all models are judged alike (`--judge` changes it). Without GPT-OSS 120B
(on Ollama and other platforms), `LLM_MODEL` (or `-m`) is needed and also judges. On other
platforms, semantic search with questions that are not stored in the index needs `--embed-model`
with their ID of Qwen3-Embedding-8B.

- **Embeddings.** OpenRouter's Qwen3-Embedding-8B reproduces the stored query embeddings as well
  as KI:connect.nrw's (cosine similarity 0.9999 for samples of 20 NFCorpus and 20 SciFact
  queries), so the retrieval does not depend on the platform.
- **Request parameters** (`-p`) are passed on as given; both platforms take `reasoning_effort`.
  OpenRouter routes the requests for a model to one of its providers, which serve it differently
  (GPT-OSS 120B, e.g., at precisions from int4 to bf16):
  `-p 'provider={"only": ["deepinfra/bf16"]}'` pins one, and
  `--judge-param 'provider={"require_parameters": true}'` keeps the judge to those that support
  JSON schemas.
- **Rate limits.** KI:connect.nrw rejects a fourth concurrent request per key. OpenRouter does
  not limit paid models, but its free ones (IDs ending in `:free`) take 20 requests per minute and
  50 per day (1000 after buying $10 of credits), enough for questions, not for benchmarks.
- **Results** of different platforms are kept apart: `config.json` records the API of a run, and
  the slash of OpenRouter's model IDs becomes `__` in file names (e.g.,
  `results/nfcorpus/bm25-k5-openai__gpt-oss-120b`).

#### Ollama

A local [Ollama](https://ollama.com) server, recognized by its port 11434, needs no key and
serves the models pulled with `ollama pull`, one of which `LLM_MODEL` names:

```bash
LLM_API_BASE=http://localhost:11434/v1
LLM_MODEL=qwen3:8b
```

- **Context length.** Ollama runs models with a context of 4096 tokens on machines with less than
  24 GiB of VRAM. It cuts longer prompts without notice, keeping their end, and drops the
  beginning of a prompt when the reply fills the context, so that the instructions and the first
  passages are lost. Five NFCorpus passages take about 2,100 tokens, and reasoning can take
  thousands more, so scrydb-rag rejects such replies: `ask` and `repl` report them, and `bench`
  and `evaluate` stop (repeating the command resumes them). Set the context length to 16k or more
  with the slider in the settings of the Ollama app, or with
  `OLLAMA_CONTEXT_LENGTH=16384 ollama serve`; `ollama ps` shows it.
- **Embeddings.** `qwen3-embedding:8b` is quantized to 4 bits: its query embeddings have a cosine
  similarity of 0.97–0.99 to the stored ones, and 93% of the semantic top 10 agree with those of
  the stored embeddings (at least 80%, for the 40 queries above). The tags `8b-q8_0` and `8b-fp16`
  (8 and 15 GB) are quantized less (`--embed-model`).
- **Requests.** `--max-tokens` is sent as `max_tokens`, as Ollama ignores
  `max_completion_tokens`, and `-p reasoning_effort=none` turns thinking off. Ollama answers one
  request at a time unless `OLLAMA_NUM_PARALLEL` says otherwise, so `--workers` is 1 there, and
  local models may generate slowly (gemma4:12b took 1–3 minutes per answer with its reasoning on a
  laptop), so `--timeout` is 900 seconds.