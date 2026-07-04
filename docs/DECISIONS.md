# Architectural Decisions

Deviations from the reference implementation
(github.com/genieincodebottle/aiml-companion/tree/main/projects/smart-claims-processor)
and the reasoning behind each one.

---

## Persistence Layer

**Decision**: Neon Postgres instead of SQLite + local ChromaDB files.

**Reference**: Uses `SqliteSaver` for LangGraph checkpoints and local
ChromaDB files for vector memory.

**Why we changed it**: Render's free web service has an ephemeral
filesystem -- local files are wiped on every restart/redeploy. SQLite
and local ChromaDB would silently lose all HITL checkpoint state and
agent memory on every cold start (which happens after 15 min idle on
free tier). Neon Postgres is free, permanent, and survives restarts.

**Scope**: LangGraph checkpointer targets Postgres. Agent memory
(ChromaDB-equivalent) uses pgvector on the same Neon DB. Core API
data (claims, users, audit logs) writes immediately to Postgres.

---

## Agent Memory: Lazy Flush

**Decision**: Agent memory (long-term episodic memory) is buffered
in-process and flushed to Postgres at natural checkpoints (end of
claim run, graceful shutdown) rather than on every micro-step.

**Reference**: No explicit lazy flush -- writes happen per-step.

**Why**: Reduces write frequency for the least critical data.
HITL checkpoints, audit logs, and core claim data still write
immediately -- lazy flush is scoped only to agent memory enrichment
where losing a few minutes of data on an unclean shutdown is
acceptable.

---

## Deployment

**Decision**: Backend on Render free web service, frontend on Vercel,
DB on Neon free tier.

**Reference**: No deployment config included.

**Known free-tier limitations**:

- Render free spins down after 15 min idle (30-60s cold start)
- No persistent disk on Render free (solved by Neon Postgres)
- Neon free tier has compute quota limits

**Upgrade path**: Render Starter ($7/mo) removes spin-down and adds
persistent disk. Dedicated Neon paid tier for higher connection limits.

---

## `src/models/claim.py` (not in reference)

**Decision**: Added a rich `Claim` Pydantic model with nested
`Claimant`, `PolicyInfo`, `IncidentDetails`, `Location` sub-models,
email/phone validation, `@model_validator` for country-aware `id_type`
validation, and a derived `currency` `@property`.

**Reference**: Uses a flat `ClaimInput` TypedDict with plain strings.
No input validation at the model layer.

**Why**: Input validation at the model boundary catches malformed
submissions before they reach any agent. Structured sub-models
(`Location` with lat/lng, `PolicyInfo` with typed dates) give agents
richer, more reliably shaped data to work with.

---

## `src/models/schemas.py` — `ClaimType`

**Decision**: Removed static `ClaimType` enum. `claim_type` in
`IntakeValidationOutput` is a plain `str` validated at runtime against
the active country YAML's `claim_types` list via `@field_validator`.

**Reference**: Static `ClaimType` enum with US-only values.

**Why**: A hardcoded enum requires a code change every time a new
country is added. Country YAMLs already define valid claim types --
validating against those at runtime means adding a new country is
purely a config change, zero code changes.

---

## `src/models/schemas.py` — Currency-neutral field names

**Decision**: Renamed `_usd` suffixed money fields to currency-neutral
names: `assessed_damage_amount`, `covered_amount`, `deductible`,
`settlement_amount`, `gross_damage`, `deductible_applied`,
`depreciation_applied`, `final_amount`.

**Reference**: All money fields suffixed with `_usd`.

**Why**: System supports both INR and USD. Currency is determined by
`claim.currency` derived from claimant country. `_usd` suffix is
misleading for Indian claims.

**Known risk**: Agent prompts in Phase 7 may reference original `_usd`
field names. Will fix prompt mismatches when building each agent.

---

## `src/models/state.py` — `masked_claim` type

**Decision**: `masked_claim: Optional[dict]` -- faithful to reference.

**We considered**: `Optional[Claim]` for typed field access.

**Why we reverted**: LangGraph's Postgres checkpointer serializes state
to JSON at every HITL pause and restores it on resume/cold restart.
A Pydantic BaseModel inside a TypedDict breaks this -- it isn't
directly JSON-serializable and comes back as a plain dict after
deserialization anyway, breaking any code expecting a Claim instance.

**How we preserve type safety at boundaries**: agents that need typed
access call `Claim(**state["masked_claim"])` to validate and convert
on demand, rather than storing a Pydantic object in the state itself.

---

## `src/models/schemas.py` — `default_factory=list` fixes

**Decision**: Added `default_factory=list` to all `list[str]` fields
where an empty list is a valid real-world outcome.

**Reference**: Several list fields in `FraudPatternOutput`,
`AnomalyDetectionOutput`, `SocialValidationOutput`,
`FraudAssessmentOutput` had no default, making them required fields.

**Why**: Forcing required list fields risks the LLM hallucinating
fake entries to satisfy a required field rather than honestly returning
empty. `default_factory=list` correctly expresses "empty is valid."

---

## `configs/base.yaml` — `app:` block

**Decision**: Added `app.name` and `app.version` to `base.yaml` and
a matching `get_app_config()` accessor in `config.py`.

**Reference**: No `app:` block in base YAML.

**Why**: FastAPI needs a display name for the docs page title and
`/health` endpoint response. Hardcoding it in Python would mean a
config value living outside the config layer.

---

## `src/config.py` — `deepcopy` fix

**Decision**: Added `copy.deepcopy()` in `get_security_config()` and
`get_guardrails_config()`.

**Reference**: Direct assignment `cfg = _load_raw()["security"]`
mutates the lru_cache object -- second call sees different data.

**Why**: Mutating a cached dict is a silent correctness bug. deepcopy
ensures every call gets an independent copy of the config dict.
