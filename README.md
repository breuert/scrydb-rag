# scrydb-rag

Retrieval-augmented generation with [scrydb](https://github.com/breuert/scrydb) as retrieval
component and the language models of an OpenAI-compatible API: by default those of the
[KI:connect.nrw](https://chat.kiconnect.nrw) platform (THKI), or, e.g., of
[OpenRouter](https://openrouter.ai) or a local Ollama server. The retrieval runs over the BEIR
indices of [scrydb-eval](https://huggingface.co/datasets/breuert/scrydb-eval), single SQLite
files with the documents, an FTS5 index, and Qwen3-Embedding-8B embeddings at three precisions.

```text
question ──► scrydb: BM25 (default) | vector search | RRF ──► top-k passages ──► GPT-OSS 120B ──► answer with [n] citations
               ▲ query embedding: Qwen3-Embedding-8B (API), or the stored one of a benchmark query
```

- **Interactive mode** (`repl`) and one-shot questions (`ask`) for arbitrary questions
- **Benchmark** (`bench`) over all queries of a test collection, resumable, with the retrieval
  evaluated against the qrels
- **Evaluation with DeepEval** (`evaluate`), judged by a model of the API as custom DeepEval model

## Setup

```bash
uv sync
echo "LLM_API_KEY=<your KI:connect.nrw API key>" >> .env
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
`-p reasoning_effort=low`.

### Retrieval

The options are scrydb's search vocabulary; the default is BM25 over FTS5.

| Option | Values | Default |
| :-- | :-- | :-- |
| `--mode` | `lexical` (BM25), `semantic` (vector search), `hybrid` (reciprocal rank fusion) | `lexical` |
| `--precision` | `binary` (Hamming distance), `int8`, `float` (cosine similarity) | `binary` |
| `--rerank` | `none`, `binary`, `int8`, `float`: second-stage rerank of the candidates | `none` |
| `-k`, `--top-k` | passages per answer | 5 |
| `--rerank-depth`, `--candidate-limit` | candidates that are reranked, or fused per ranking | 1000 |

Runs are named like the run files of scrydb-eval: `bm25`, `bm25-cosine_float`, `hamming`,
`hamming-cosine_float`, `rrf-cosine_int8`, and so on.

- **Query embeddings.** The queries of a benchmark are searched with their stored embeddings, so
  the retrieval reproduces the published runs exactly: the nDCG@10 of BM25, BM25 >> cos_float,
  Hamming, Hamming >> cos_float, cos_float, and RRF(Hamming) on NFCorpus and SciFact, and of
  cos_float on ArguAna, match the results table of scrydb-eval. Other questions are embedded with
  Qwen3-Embedding-8B on the platform, with the instruction the stored query embeddings were
  computed with (`Instruct: <task of the dataset>\nQuery:<question>`; `--task` changes it).
  Without the instruction, the cosine similarity to the stored embedding of the same query drops
  from 0.9999 to 0.84–0.93 (for a sample of NFCorpus and SciFact queries).
- **Self-matches.** As in BEIR (and scrydb-eval), a document with the ID of the query is removed
  from its results (ArguAna draws its queries from the corpus); `--keep-self-matches` keeps it.
- **Semantic rerank.** scrydb reranks only the first-stage results it returns, so all
  `--rerank-depth` candidates are requested and cut to k after the rerank.
- **Single candidates.** sqlite-vec 0.1.9 finds nothing when a KNN query is restricted to a single
  rowid, so scrydb's rerank loses the only candidate of a first stage with one result (e.g., the
  NFCorpus queries "bagels", "lard", and "rickets" with BM25 >> cos_float). The pipeline then uses
  the first-stage result, which is what the rerank should return.

### Generation

GPT-OSS 120B (`-m` or `LLM_MODEL` selects another model) gets the passages numbered and cut after
`--max-passage-chars` (4000) characters, with instructions to answer only from them, to cite
them as `[n]`, and to say when they do not suffice. The request can also be a claim (SciFact), an
argument (ArguAna), or a topic, which the instructions cover; `-s` replaces them.

### Interactive mode

`scrydb-rag repl DATASET` answers each question from the passages retrieved for it, streams the
answer, and lists the cited passages. Commands:

| Command | |
| :-- | :-- |
| `/mode`, `/precision`, `/rerank`, `/k` | show or change the retrieval |
| `/passages on` | list the retrieved passages before each answer |
| `/show N` | print passage N of the last answer as the model saw it |
| `/model`, `/models`, `/set KEY VALUE` | change the model or the request parameters |
| `/reasoning on`, `/usage on` | show the reasoning of the model, or the token usage |

### Benchmark

`scrydb-rag bench DATASET` answers every query in the qrels of a split (`--split test` by default;
the indices also store the training queries). The results go to
`results/DATASET/METHOD-kN-MODEL/` (`-o` changes it):

| File | Content |
| :-- | :-- |
| `config.json` | settings of the run, including the API |
| `answers.jsonl` | per query: its text, the ranking (top max(k, 10) documents), the passages given to the model (`retrieval_context`), the answer, the reasoning, and the token usage |
| `run.trec` | the ranking as TREC run file |
| `retrieval.json` | nDCG@10, nDCG@k, RR@10, P@k, and R@k with the qrels (queries without results count as 0) |

Answers are generated by three concurrent requests (`--workers`), as KI:connect.nrw rejects a
fourth one per key with status 429 (other platforms may take more, Ollama one). Each answer is
saved when it is complete, so an interrupted run is resumed by repeating the command, which also
retries the queries that failed. A directory only takes runs with the same settings.

### Evaluation with DeepEval

`scrydb-rag evaluate RUN_DIR` (or `bench --evaluate`) evaluates the answers with three DeepEval
metrics that need no reference answers, which the test collections do not have:

- `answer_relevancy`: share of the statements of the answer that address the request
- `faithfulness`: share of the claims of the answer that the passages support
- `contextual_relevancy`: share of the statements of the passages that are relevant to the request

The judge is a model of the API (`--judge`, GPT-OSS 120B by default) wrapped as custom DeepEval
model (`JudgeModel` in `scrydb_rag/evaluation.py`). It asks for replies constrained to the JSON
schema each metric step expects, and falls back to DeepEval's JSON extraction for models that
reject schemas. The metrics run with `eval_mode="llm"`, so that no decision goes to another
service, and DeepEval's telemetry is turned off unless `DEEPEVAL_TELEMETRY_OPT_OUT` is set
otherwise.

The results go to `deepeval-JUDGE.jsonl` (score and reason per query and metric) and
`deepeval-JUDGE.json` (mean score and pass rate per metric, with `--threshold` 0.5) in the run
directory, so the answers of a run can be compared across judges. An interrupted evaluation is
resumed by repeating the command, which also retries the metrics that failed. Empty answers are
reported, not evaluated, as DeepEval would rate them perfectly relevant and faithful.

With k=5, an answer takes about 13 judge requests (3 for answer relevancy, 4 for faithfulness, 6
for contextual relevancy). With its default reasoning effort and three concurrent requests,
GPT-OSS on KI:connect.nrw took about 40 seconds per answer for the three metrics, which is about
3.5 hours for the 323 NFCorpus queries; `--judge-param reasoning_effort=low` about halves that.

## Development

```bash
uv run pytest
```

The tests run offline, with a toy index and a stand-in for the API client. `thki.py` stays a
standalone command-line client for KI:connect.nrw; the package reuses its `Client`, which works
with any OpenAI-compatible API.
