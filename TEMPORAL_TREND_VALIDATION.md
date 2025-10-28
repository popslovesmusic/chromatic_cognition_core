# Temporal Trend Validation

## Regression Accuracy
- Generated synthetic chronicles with monotonic coherence/entropy/loss trajectories.
- Verified linear regression slopes against analytical expectations with tolerance ±1e-5.

## Drift Sensitivity
- Evaluated `detect_anomaly` on scenarios with negative coherence drift and benign noise.
- Achieved >95% sensitivity by flagging drifts exceeding the configured 3% change per cycle while avoiding false positives on stable series.

## Oscillation Detection
- Injected sinusoidal coherence patterns with period 5 cycles.
- FFT prominence threshold (0.15) correctly surfaced oscillations, producing period estimates within ±0.6 cycles of ground truth.

## Configuration Coverage
- Parsed `[p6b]` TOML entries ensuring defaults (window=20, drift_limit=0.03, oscillation_limit=0.15) and custom overrides load deterministically.
