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

---

## `src/llm.py` — Runtime-configurable LLM factory

**Decision**: Extended `get_llm()` to support runtime overrides for `temperature`, `max_tokens`, and `streaming` while still defaulting to configuration values.

**Reference**: Configuration values are loaded entirely from YAML/environment and cannot be overridden per call.

**Why**: Different parts of the system have different inference requirements. Deterministic validation agents may require `temperature=0`, while summarization or explanation agents may benefit from higher temperatures. Allowing runtime overrides improves flexibility without sacrificing centralized configuration.

---

## `src/llm.py` — Builder abstraction

**Decision**: Introduced provider-specific builder functions (`_build_groq`, `_build_gemini`) registered through a `_BUILDERS` mapping.

**Reference**: Provider selection logic is less modular.

**Why**: Separating provider initialization behind builders isolates provider-specific SDK code. Adding a future provider (OpenAI, Anthropic, Ollama, etc.) requires only implementing a new builder and registering it in `_BUILDERS`, without modifying the factory logic.

---

## `src/llm.py` — Provider-independent token tracking

**Decision**: Implemented `_TokenTracker` as a LangChain callback to accumulate token usage and estimate inference cost.

**Reference**: Similar tracking exists but is tightly coupled to provider response metadata.

**Why**: LangChain exposes slightly different metadata depending on the provider. `_TokenTracker` normalizes these differences into a single internal representation while keeping provider-specific parsing isolated inside `_extract_usage()`.

---

## `src/llm.py` — Token pricing lookup

**Decision**: Pricing is resolved once during `_TokenTracker` initialization instead of performing repeated model lookups on every callback.

**Reference**: Pricing is looked up repeatedly during token accounting.

**Why**: Token callbacks execute after every LLM invocation. Resolving prices once avoids repeated dictionary lookups and keeps callback execution lightweight.

---

## `src/security/pii_masker.py` — Cached regex compilation

**Decision**: Added `@lru_cache(maxsize=1)` to `_build_patterns()`.

**Reference**: Regular expressions are recompiled every time masking functions are called.

**Why**: Regex compilation is relatively expensive compared to matching. Country-specific patterns rarely change during runtime, so compiling them once and reusing them significantly reduces overhead.

---

## `src/security/pii_masker.py` — Explicit cache invalidation

**Decision**: Added `clear_pattern_cache()`.

**Reference**: No explicit cache invalidation mechanism.

**Why**: If the active country configuration changes while the application is running, cached regex patterns would otherwise remain stale. Explicit cache invalidation allows configuration reloads without restarting the application.

---

## `src/security/pii_masker.py` — Reusable string masking helper

**Decision**: Introduced a dedicated `_mask_string()` helper used by all masking operations.

**Reference**: Regex substitution logic is duplicated across multiple code paths.

**Why**: Centralizing masking logic removes duplication, keeps placeholder generation consistent, and ensures future masking improvements only need to be implemented once.

---

## `src/security/pii_masker.py` — Improved type annotations

**Decision**: Added precise generic type hints such as `dict[str, Any]` and `re.Pattern[str]`.

**Reference**: Uses broader type annotations.

**Why**: More precise typing improves IDE autocomplete, static analysis, and long-term maintainability without changing runtime behavior.

---

## `src/security/pii_masker.py` — Country configuration error handling

**Decision**: Configuration loading failures are logged using `logger.exception()` before falling back to safe defaults.

**Reference**: Exceptions are silently ignored.

**Why**: Silent failures make configuration issues difficult to diagnose. Logging the complete traceback preserves production resilience while improving observability.

---

## `src/security/audit_log.py` — Immutable audit entry writes

**Decision**: `_write_entry()` creates a shallow copy of the provided entry before appending the computed hash.

**Reference**: The original implementation mutates the caller's dictionary by inserting the `"hash"` field.

**Why**: Audit logging should not unexpectedly modify objects owned by the caller. Copying preserves the immutability contract while still writing the hashed entry to disk.

---

## `src/security/audit_log.py` — Path-oriented file handling

**Decision**: Uses `Path.open()` instead of the built-in `open()` function.

**Reference**: Uses `open()` directly.

**Why**: The project already represents log locations as `pathlib.Path` objects. Using `Path.open()` keeps the implementation consistent with the surrounding API and avoids unnecessary conversions.

---

## `src/security/audit_log.py` — Direct JSON streaming

**Decision**: Uses `json.dump()` when writing audit entries instead of creating an intermediate string with `json.dumps()`.

**Reference**: Serializes to a string before writing.

**Why**: `json.dump()` is designed for file-backed serialization, avoids creating an intermediate string object, and more clearly communicates the intent of streaming JSON directly to disk.

---

## `src/security/audit_log.py` — Exception logging

**Decision**: Replaced formatted error logging with `logger.exception()` inside exception handlers.

**Reference**: Logs only the exception message.

**Why**: Audit log failures are operational issues that benefit from a complete traceback. `logger.exception()` records both the message and full stack trace, making production debugging significantly easier while preserving the original behavior of not interrupting the claim processing pipeline.

---

## `src/security/audit_log.py` — Removed unused parameter

**Decision**: Removed the unused `claim_id` parameter from `_get_log_path()`.

**Reference**: `_get_log_path(claim_id)` accepted `claim_id` but never used it.

**Why**: The audit log path depends only on the configured log directory and current date. Removing the unused parameter makes the function signature accurately describe its behavior and avoids misleading future contributors.

---

## `src/security/audit_log.py` — Deterministic hash serialization

**Decision**: Added compact deterministic JSON serialization (`sort_keys=True`, `separators=(",", ":")`) before computing the SHA-256 hash.

**Reference**: Uses `sort_keys=True` but relies on the default JSON formatting.

**Why**: Compact serialization removes insignificant whitespace from the serialized representation. This guarantees that logically identical audit entries always produce identical hashes regardless of formatting.

---

## `src/security/audit_log.py` — Stronger type annotations

**Decision**: Added explicit type annotations such as `dict[str, Any]` and `list[dict[str, Any]]` throughout the module.

**Reference**: Uses generic `dict` and `list` annotations.

**Why**: More precise typing improves IDE support, static analysis, and readability without changing runtime behavior.
