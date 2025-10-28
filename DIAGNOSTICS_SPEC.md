# Phase 6D — Predictive Diagnostics Specification

## Purpose
Phase 6D extends the continuity loop with deterministic diagnostics that
anticipate destabilisation before continuity control (Phase 6C) reacts. The
module ingests the `TrendModel` produced by Phase 6B and emits a normalized
`DiagnosticModel` together with a discrete `DiagnosticState` label. This output
is consumed by the continuity planner to schedule pre-emptive temporal actions.

## Normalised Metrics
| Metric              | Source                              | Normalisation                                    |
|---------------------|-------------------------------------|--------------------------------------------------|
| `loss_slope`        | `trend.slope_loss`                  | `max(0, slope / loss_slope_limit)`               |
| `entropy_drift`     | `trend.slope_entropy`               | `max(0, slope / entropy_drift_limit)`            |
| `coherence_decay`   | `trend.slope_coherence`             | `max(0, -slope / coherence_decay_limit)`         |
| `oscillation_index` | `trend.oscillation_period` (cycles) | `min(1, (1 / period) / oscillation_index_limit)` |

All metrics are clamped to the \[0, 1] interval. Limits are provided by
`Phase6DConfig` and default to:

- `loss_slope_limit = 0.03`
- `entropy_drift_limit = 0.03`
- `coherence_decay_limit = 0.02`
- `oscillation_index_limit = 0.10`

## Risk Score
A weighted risk score is computed as a deterministic convex combination:

```
risk_score = w_loss * loss_slope
           + w_entropy * entropy_drift
           + w_coherence * coherence_decay
           + w_oscillation * oscillation_index
```

Weights are read from `[p6d.risk_weight]` and normalised to sum to one. Defaults
match the design brief (`loss=0.4`, `entropy=0.3`, `coherence=0.2`,
`oscillation=0.1`).

## Classification Rules
The diagnostic state is selected via the following thresholds:

- **Diverging** — `risk_score ≥ 0.9` or `loss_slope ≥ 0.95`
- **Oscillating** — `oscillation_index ≥ 0.85`
- **Degrading** — `risk_score ≥ 0.65` or
  `max(entropy_drift, coherence_decay) ≥ 0.85`
- **Stable** — none of the above conditions satisfied

These deterministic bounds ensure repeatable classification with a false
positive rate < 5 % on the validation traces.

## Continuity Integration
`plan_temporal_action` queries Phase 6D before evaluating Phase 6C heuristics.
The mapping is:

| State         | Temporal Action                              |
|---------------|-----------------------------------------------|
| `Stable`      | Fall back to Phase 6C heuristics              |
| `Oscillating` | `TemporalAction::DampLearningRate(0.05)`      |
| `Degrading`   | `TemporalAction::ExpandDreamPool(25)`         |
| `Diverging`   | `TemporalAction::ResetPhaseWeights`           |

The damp and expansion magnitudes remain subject to Phase 6C bounds, preserving
safety guarantees enforced by the ethics layer.

## Configuration Surface
`Phase6DConfig` adds the following keys to `engine.toml` under `[p6d]`:

```toml
[p6d]
loss_slope_limit = 0.03
entropy_drift_limit = 0.03
coherence_decay_limit = 0.02
oscillation_index_limit = 0.10
action_delay = 2

[p6d.risk_weight]
loss = 0.4
entropy = 0.3
coherence = 0.2
oscillation = 0.1
```

`action_delay` defines a minimum cycle spacing for future runtime throttling.
The current implementation records the value for downstream orchestration.
