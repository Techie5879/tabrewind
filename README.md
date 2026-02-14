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

## privacy

Privacy is a core goal of this repo: browser data stays local, processing is local-first, and the project is built to avoid shipping personal browsing data to third-party services.

Nothing here should require internet connections or API calls.
