# Phase 6A — Chronicle Layer Specification

## Purpose
The chronicle layer records every learning or dream cycle in a deterministic, append-only log. The log enables reproducible historical analysis, audit trails, and aggregate statistics for the adaptive learning loop.

## Scope
- Module: `src/meta/chronicle.rs`
- Primary roles:
  - **Chronicle Writer**: serializes a `CycleRecord` and appends it to the on-disk history.
  - **Chronicle Reader**: loads the most recent `N` records for reporting and analytics.
- Persistent artifact: `data/meta/continuity_history.csv`, with an optional SQLite mirror for fast queries.

## Data Schema
```rust
pub struct CycleRecord {
    pub id: u64,
    pub timestamp: DateTime<Utc>,
    pub coherence: f32,
    pub entropy: f32,
    pub grad_energy: f32,
    pub loss: f32,
    pub val_accuracy: f32,
    pub dream_seed: Option<Uuid>,
    pub meta_score: f32,
}
```

### Serialization Rules
- CSV columns appear in the same order as the struct definition.
- `timestamp` is formatted as RFC 3339 with nanosecond precision.
- `dream_seed` serializes as a canonical hyphenated UUID string or an empty field when `None`.
- Floating-point values use `{:0.8}` formatting to guarantee deterministic precision.
- `id` increments monotonically starting from 0.

## Chronicle Writer Responsibilities
1. Load configuration from `[p6a]` in `config`:
   ```toml
   history_limit = 500
   log_path = "data/meta/continuity_history.csv"
   ```
2. Ensure the parent directory for `log_path` exists (create recursively when missing).
3. When appending:
   - Acquire an exclusive file lock during write to avoid interleaved records.
   - Write a header row when creating a fresh file.
   - Serialize `CycleRecord` to CSV using deterministic ordering and formatting.
   - Flush and fsync before releasing the lock.
4. Maintain an in-memory ring buffer capped to `history_limit` for fast access to recent cycles.
5. Emit a `tracing::info!` event with the serialized record after each append.

## Chronicle Reader Responsibilities
- Provide `load_recent(limit: usize) -> Vec<CycleRecord>` that reads at most `limit` newest entries from disk.
- Optionally hydrate from the SQLite mirror when enabled.
- Support aggregation helpers:
  - `mean(field)`
  - `stdev(field)`
  - `max(field)`
  - `min(field)`
- Guarantee that replaying the stored sequence yields identical floating-point aggregates within 1e-6 tolerance.

## Optional SQLite Mirror
- File path mirrors the CSV location with `.db` extension.
- Schema:
  ```sql
  CREATE TABLE IF NOT EXISTS cycle_history (
      id INTEGER PRIMARY KEY,
      timestamp TEXT NOT NULL,
      coherence REAL NOT NULL,
      entropy REAL NOT NULL,
      grad_energy REAL NOT NULL,
      loss REAL NOT NULL,
      val_accuracy REAL NOT NULL,
      dream_seed TEXT,
      meta_score REAL NOT NULL
  );
  ```
- Writer performs an UPSERT for each CSV append.
- Reader uses SQL window functions for aggregation when the mirror exists.

## Determinism & Validation
- Every append operation is deterministic given identical inputs and configuration.
- Replay test: load the full history, recompute aggregates, and ensure results match previously recorded summaries within tolerance.
- Long-haul durability: execute 1,000 cycle appends, restart the process halfway, and verify no gaps or duplicates.
- Detect and alert on out-of-order IDs or timestamps via `warn!` logs.

## Testing Strategy
- Unit tests for CSV serialization and deserialization round-tripping.
- Integration test simulating 1,000 cycles with random-but-seeded data to confirm append order and aggregate stability.
- Prop-test style fuzzing for optional dream seeds to ensure absence of panic on missing values.

## Operational Notes
- All logging runs on CPU; no GPU dependencies allowed.
- File paths are relative to the project root; ensure cross-platform compatibility.
- Document configuration options in `README` and expose metrics via the reporting stack in subsequent phases.

