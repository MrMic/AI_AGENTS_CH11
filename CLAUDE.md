# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Chapter 11 companion project for a Manning "AI Agents" book. Judging by the dependency set
(`fastmcp`, `langchain-mcp-adapters`, `langgraph-supervisor` on top of the chapter 09 RAG
stack), the chapter covers **MCP tool servers driven by a LangGraph supervisor**.

**It is currently a scaffold — there is no working code yet.** Before assuming any
architecture, check what actually exists:

- `main_01_01.py` — 3-line stub; line 1 calls `load_dotenv()` with no import, so it does not run.
- `Untitled.ipynb` — a single empty cell.
- `src/ch11/__init__.py` — `uv init` stub (`main()` prints "Hello from ch11!"), wired as the
  `ch11` console script in `pyproject.toml`.

## Stale docs — do not trust as spec

Two markdown files were carried over from other chapters and describe code that is not here:

- `README.md` documents a **chapter 12** hotel-booking project: `main.py`, a `hotel_db/`
  SQLite directory, `pip install -r requirements.txt`, a PowerShell venv. None of it exists.
- `GUARDRAILS.md` documents the travel-only guardrail added to **chapter 09's**
  `main_09_01.py`. Useful as a pattern to port (pre-model hook + router short-circuit),
  not as a description of this repo.

Rewrite or delete them when real ch11 code lands rather than coding to match them.

## Running

Secrets come from `pass` via `secretspec` (see the gitignored `secretspec.toml`), injected
at launch:

```bash
secretspec run -P development --reason "ch11 notebook" -- uv run jupyter lab
```

`--reason` is mandatory (secretspec `require_reason` policy). Without it every command —
including `secretspec check` — fails with "Accessing secrets requires a reason", which reads
like a broken config.

Traps inherited from ch09, still applicable:

- `secretspec check` prints `✓ OPENAI_API_KEY` when the `pass` entry **exists but is empty**.
  Verify content, not existence:
  `pass show secretspec/ch11/development/OPENAI_API_KEY | head -1 | awk '{print length($0)}'`
  (valid key ≈164 chars, `sk-proj` prefix).
- Env is injected **once at lab launch**; kernels inherit it from the lab process. After
  changing a secret, relaunch `jupyter lab` — a kernel restart keeps the stale value.

`secretspec.toml` declares **only `OPENAI_API_KEY`**, but `.env_example` also lists
`LANGSMITH_*` and `ACCUWEATHER_API_KEY`. If chapter code needs those, add them to
`secretspec.toml` and `pass insert` them under `secretspec/ch11/development/` — do not fall
back to a `.env` file, which would split the secret source in two.

No test suite, no build. Lint/format: `uv run ruff check .` / `uv run ruff format .`.

## Dependencies

`pyproject.toml` + `uv.lock` are the single source of truth; the book's pinned
`requirements.txt` was deliberately deleted. Dependencies are declared as `>=` and resolved
**well above** the book's pins — notably `fastmcp>=4.0.4` where the book targets `2.10.5`,
and `langchain>=1.4.0` where the book targets `1.0.3`.

Expect book code to break on the fastmcp 2 → 4 API change. Pin down rather than debug
blind: `uv add 'fastmcp==2.10.5'`.
