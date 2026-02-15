# tabrewind

tabrewind is a local-first tool for finding "lost tabs" and recent pages quickly.

Many browsers already store history in local SQLite databases, which makes retrieval reliable and easy to query. tabrewind builds on that with a search and indexing layer focused on Zen/Firefox first.

## what it does

- reads browser history from local SQLite data in read-only mode
- supports direct SQLite/FTS queries for exact and keyword search
- supports vector search with local embeddings
- supports hybrid search (BM25 + vector)
- exports deterministic outputs for repeatable indexing/search

## usage modes

- CLI workflow for direct local usage
- agent workflow (for example OpenCode/Codex-style agents) that can run queries and search flows

## tech

- Python (for now - may switch to Go/Rust/Zig later)
- SQLite
- sqlite-vec
- SQLite FTS5/BM25

## local model runtime

tabrewind uses `llama.cpp` for local model execution.

- use any model that is supported by your installed `llama.cpp` build
- download model files locally before running model-backed workflows
- no hosted model API is required for this project

Example model download via `llama.cpp`:

```bash
llama-cli \
  --hf-repo ggml-org/embeddinggemma-300M-qat-q4_0-GGUF \
  --hf-file embeddinggemma-300M-qat-Q4_0.gguf \
  -n 0 -p ""
```

## CLI workflow

Initialize a local config file once:

```bash
uv run main.py init
```

Inspect or update config values:

```bash
uv run main.py config show
uv run main.py config set llama.base_url http://127.0.0.1:8080
uv run main.py config set ingest.embedding_workers 4
uv run main.py config set ingest.embedding_batch_size 8
```

Start `llama-server` in embedding mode. If you plan to send concurrent embedding requests, set `--parallel` to match or exceed your configured worker count:

```bash
llama-server -m <path_to_model.gguf> \
  --embedding \
  --pooling mean \
  --parallel 4 \
  --port 8080
```

Run ingestion and print `vec[:10]` previews:

```bash
uv run main.py ingest --max-items 20
```

Run a quick local concurrency benchmark (20 sentences):

```bash
uv run main.py bench --sentences 20
```

## privacy

Privacy is a core goal of this repo: browser data stays local, processing is local-first, and the project is built to avoid shipping personal browsing data to third-party services.

Nothing here should require internet connections or API calls.
