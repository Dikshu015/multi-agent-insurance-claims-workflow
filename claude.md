# CLAUDE.md

Project context for Claude Code. Auto-loaded every session.

## Project Overview

Multi-agent insurance claims processor using **LangGraph** (orchestration)
**CrewAI** (fraud sub-crew). FastAPI backend, React+Vite frontend,
**Neon Postgres** persistence (not SQLite -- see docs/DECISIONS.md).
Two LLM providers (Groq, Gemini) and two country profiles (US, India)
switchable at runtime via `.env`.

## Common Commands

```bash
# Backend
source .venv/Scripts/activate          # Windows Git Bash
uv sync                                # install all dependencies
uvicorn api.main:app --port 8000       # start backend

# Frontend (separate terminal)
cd frontend && npm install && npm run dev   # http://localhost:3000

# Tests (no API key needed -- LLM is mocked in tests)
pytest tests/ -q
pytest tests/test_specific.py -q           # single file
pytest tests/ -q -k "test_name"            # keyword filter

# Data management
python scripts/seed_policies.py            # seed US + India test policies
python scripts/clean_data.py               # reset claims (keeps users)
python scripts/clean_data.py --all         # full reset including users
python scripts/generate_secret_key.py      # generate JWT secret into .env
```

## Architecture

### Pipeline (src/agents/graph.py)

LangGraph `StateGraph` with 7 agent nodes + 6 HITL checkpoint nodes.
Five execution paths:

- **Path A** (normal): intake → fraud_crew → damage → policy → settlement → evaluator → communication
- **Path B** (HITL): same as A but pauses at `hitl_checkpoint` when fraud score >= 0.45, amount above country threshold, or evaluation quality gate fails
- **Path C** (auto-reject): intake → fraud_crew → auto_reject → communication (fraud score >= 0.90)
- **Path D** (intake failure): intake → communication (invalid claim or missing docs)
- **Path E** (fast mode): intake → settlement → communication (amount < $500 with clean history)

Per-agent **confidence gates** can pause at any step via dedicated HITL
nodes that resume to the correct next agent.

HITL uses LangGraph `interrupt()` / `Command(resume=...)` with
**Postgres-backed checkpointer** (Neon free tier) for durable
pause/resume across Render cold starts.

### Configuration Hierarchy (src/config.py)

Precedence: runtime override > `.env` > country YAML
(`configs/countries/{code}.yaml`) > base YAML (`configs/base.yaml`).
Provider and country set in `.env` only; model IDs and tunables in YAML.

### Key Layers

| Layer | Location | Purpose |
|---|---|---|
| API | `api/` | FastAPI routes, SQLModel DB, JWT auth, role guards |
| Agents | `src/agents/` | Each agent takes `ClaimsState`, returns a dict update |
| State | `src/models/state.py` | `ClaimsState` TypedDict -- single object flowing through graph |
| HITL | `src/hitl/` | `queue.py` (Postgres ticket queue), `checkpoint.py` (trigger rules + priority scoring) |
| Memory | `src/memory/` | pgvector embeddings for similar-claim retrieval (lazy flush to Postgres) |
| LLM | `src/llm.py` | Provider factory -- ChatGroq or ChatGoogleGenerativeAI with token tracking |
| Guardrails | `src/guardrails/manager.py` | Per-claim caps on agent calls, tokens, cost |
| PII | `src/security/pii_masker.py` | Country-aware regex masking before LLM calls |
| Frontend | `frontend/` | React + Vite + Zustand + MUI; proxies `/api` to backend port 8000 |

### Data Storage

**Neon Postgres** (not SQLite) for all persistence:

- `api.db` equivalent → Postgres tables (users, claims, appeals)
- `hitl_queue.db` equivalent → Postgres HITL queue table
- `claims_checkpoints` → Postgres LangGraph checkpointer
- `memory` → pgvector table (lazy flush from in-process buffer)

See `docs/DECISIONS.md` for why Postgres over SQLite.

### Adding a New Country

Create `configs/countries/{code}.yaml` following `configs/countries/us.yaml`.
The config loader auto-discovers it via `glob("*.yaml")`.

## Environment

Key `.env` variables: `LLM_PROVIDER` (groq|gemini),
`GROQ_API_KEY`/`GOOGLE_API_KEY`, `COUNTRY` (us|india),
`DATABASE_URL` (Neon Postgres connection string), `API_SECRET_KEY`.
See `.env.example` for full list.

Dev accounts seeded on first startup:
`admin/admin123`, `reviewer1/review123`, `reviewer2/review123`, `claimant/claim123`.
