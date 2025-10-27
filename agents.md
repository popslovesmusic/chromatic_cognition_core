# Agents Guide — Chromatic Cognition / Tiny Trainer

## Mission
Act as a co-developer focused on correctness, transparency, and modular growth.
Your task is to convert design intent into clean, verifiable Rust code without adding complexity.
Favor explicitness, determinism, and comments that explain reasoning.

## Boundaries
- Do **not** introduce frameworks larger than: `ndarray`, `rayon`, `serde`, `plotters`, or `candle`.
- Remain **CPU-only** until `GPU_ENABLED=true` appears in config.
- All code must compile with `cargo build --release` on Linux or Windows without external toolchains.
- Any stochastic or parallel behavior must log its seed and thread count.

## Conventions
- Follow `snake_case` for functions, `PascalCase` for structs/enums.
- Each public struct implements `Debug`, `Clone`, and `Serialize`.
- All functions that change tensors emit a log entry via `tracing` or plain JSON to `logs/`.

## Workflow
1. Read `spec.md` for current feature list.
2. Propose file diffs before major refactors.
3. After implementing, auto-generate:
   - a short `CHANGELOG` entry,
   - a single-page `report.md` summarizing new behavior and metrics.

## Key Roles
| Agent | Responsibility |
|--------|----------------|
| **Architect** | Maintain structure and dependencies. |
| **Coder** | Implement logic per spec. |
| **Tester** | Write minimal unit tests for every new function. |
| **DocBot** | Keep `README` and `api.md` synchronized. |

## Communication
When uncertain, prefer to ask “Is this in or out of scope for current spec?”  
Never assume hidden requirements.
