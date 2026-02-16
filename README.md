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

Configure host filtering rules (top-to-bottom, last match wins):

```bash
# deny sensitive domains
uv run main.py domains add --deny "gmail.com"
uv run main.py domains add --deny "*.gmail.com"
uv run main.py domains add --deny "*.bank.*"

# allow a specific subdomain after a broad deny
uv run main.py domains add --allow "mail.google.com"

# inspect, reorder, and validate final decision
uv run main.py domains list
uv run main.py domains move --from-index 4 --to-index 2
uv run main.py domains check "https://mail.google.com/mail/u/0/#inbox"

# preview resolved allow/deny view for discovered hosts in recent history
uv run main.py domains preview --since-days 2 --browsers zen
```

Rules use a leading prefix in config:

- `-pattern` denies a host glob
- `+pattern` allows a host glob
- patterns may optionally include a path glob (for example `-mail.google.com/mail/*`)
- unmatched hosts are allowed by default
- rules are applied top-to-bottom, and the last matching rule wins

You can also set all rules at once via config:

```bash
uv run main.py config set ingest.domain_rules -- "-gmail.com,-*.gmail.com,-*.bank.*,+mail.google.com"
```

Common privacy rule recipes (order matters, last match wins):

```bash
# block personal inbox + broad banking surfaces
uv run main.py config set ingest.domain_rules -- "-gmail.com,-*.gmail.com,-*.bank.*,-*.paypal.com"

# allow one work-safe mailbox after a broad deny
uv run main.py config set ingest.domain_rules -- "-*.google.com,+mail.google.com"

# block only sensitive app routes, keep public pages
uv run main.py config set ingest.domain_rules -- "-app.example.com/settings/*,-app.example.com/billing/*"
```

Tip: run `uv run main.py domains preview --since-days 2 --browsers zen` to review the final resolved ALLOW/DENY view before ingesting.

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

Run schema-backed ingestion into `tabrewind.sqlite` (managed local DB path):

```bash
uv run main.py ingest-db --since-days 2 --browsers zen
```

Run the live vectorization health test (OpenAI-compatible `/v1/embeddings` on `llama-server`):

```bash
uv run tests/vectorization_concurrency.py
```

## privacy

Privacy is a core goal of this repo: browser data stays local, processing is local-first, and the project is built to avoid shipping personal browsing data to third-party services.

Nothing here should require internet connections or API calls.
