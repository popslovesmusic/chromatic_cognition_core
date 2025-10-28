//! Predictive diagnostics layer translating trend statistics into early-risk
//! assessments.
//!
//! Phase 6D consumes the outputs of the trend model (Phase 6B) and produces a
//! deterministic risk score alongside a discrete diagnostic state. The
//! resulting state feeds forward into continuity planning (Phase 6C) so that
//! preventive actions can be scheduled before metrics deteriorate.

use serde::Serialize;

use crate::config::Phase6DConfig;

use super::trend::TrendModel;

const DIVERGENCE_RISK: f32 = 0.9;
const DEGRADING_RISK: f32 = 0.65;
const OSCILLATION_ALERT: f32 = 0.85;
const SEVERE_SLOPE: f32 = 0.95;
const SEVERE_DECAY: f32 = 0.85;

/// Normalized diagnostic metrics and the resulting predicted state.
#[derive(Debug, Clone, Serialize)]
pub struct DiagnosticModel {
    /// Positive loss slope severity in \[0, 1].
    pub loss_slope: f32,
    /// Positive entropy drift severity in \[0, 1].
    pub entropy_drift: f32,
    /// Coherence decay severity in \[0, 1].
    pub coherence_decay: f32,
    /// Oscillation prominence index in \[0, 1].
    pub oscillation_index: f32,
    /// Weighted risk score in \[0, 1].
    pub risk_score: f32,
    /// Diagnostic state label derived from the normalized metrics.
    pub state_label: DiagnosticState,
}

/// Deterministic classification of the long-term stability trajectory.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub enum DiagnosticState {
    Stable,
    Oscillating,
    Degrading,
    Diverging,
}

/// Evaluate predictive diagnostics from a fitted trend model.
pub fn evaluate_diagnostics(trend: &TrendModel) -> DiagnosticModel {
    let config = Phase6DConfig::default();

    let loss_slope = normalize_positive(trend.slope_loss, config.loss_slope_limit);
    let entropy_drift = normalize_positive(trend.slope_entropy, config.entropy_drift_limit);
    let coherence_decay = normalize_negative(trend.slope_coherence, config.coherence_decay_limit);
    let oscillation_index =
        normalize_oscillation(trend.oscillation_period, config.oscillation_index_limit);

    let weights = config.risk_weight.normalized();
    let mut risk_score = loss_slope * weights.loss
        + entropy_drift * weights.entropy
        + coherence_decay * weights.coherence
        + oscillation_index * weights.oscillation;
    risk_score = risk_score.clamp(0.0, 1.0);

    let mut model = DiagnosticModel {
        loss_slope,
        entropy_drift,
        coherence_decay,
        oscillation_index,
        risk_score,
        state_label: DiagnosticState::Stable,
    };
    model.state_label = classify_state(&model);
    model
}

/// Classify the diagnostic state using deterministic thresholds.
pub fn classify_state(model: &DiagnosticModel) -> DiagnosticState {
    if model.risk_score >= DIVERGENCE_RISK || model.loss_slope >= SEVERE_SLOPE {
        DiagnosticState::Diverging
    } else if model.oscillation_index >= OSCILLATION_ALERT {
        DiagnosticState::Oscillating
    } else if model.risk_score >= DEGRADING_RISK
        || model.entropy_drift >= SEVERE_DECAY
        || model.coherence_decay >= SEVERE_DECAY
    {
        DiagnosticState::Degrading
    } else {
        DiagnosticState::Stable
    }
}

fn normalize_positive(value: f32, limit: f32) -> f32 {
    if limit <= f32::EPSILON {
        return 0.0;
    }
    let scaled = (value / limit).max(0.0);
    scaled.min(1.0)
}

fn normalize_negative(value: f32, limit: f32) -> f32 {
    if limit <= f32::EPSILON {
        return 0.0;
    }
    let scaled = (-value / limit).max(0.0);
    scaled.min(1.0)
}

fn normalize_oscillation(period: f32, limit: f32) -> f32 {
    if period <= 0.0 || limit <= f32::EPSILON {
        return 0.0;
    }
    let prominence = (1.0 / period).max(0.0);
    (prominence / limit).min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Phase6DConfig, Phase6DRiskWeight};
    use crate::meta::trend::TrendModel;

    fn trend(loss: f32, coherence: f32, entropy: f32, oscillation: f32) -> TrendModel {
        TrendModel {
            slope_coherence: coherence,
            slope_entropy: entropy,
            slope_loss: loss,
            oscillation_period: oscillation,
        }
    }

    #[test]
    fn diagnostics_flag_degrading_entropy() {
        let mut config = Phase6DConfig::default();
        config.entropy_drift_limit = 0.01;
        config.risk_weight = Phase6DRiskWeight {
            loss: 0.1,
            entropy: 0.7,
            coherence: 0.1,
            oscillation: 0.1,
        };

        let model = evaluate_with_config(&config, &trend(0.0, 0.0, 0.02, 0.0));
        assert!(matches!(model.state_label, DiagnosticState::Degrading));
        assert!(model.risk_score >= DEGRADING_RISK);
    }

    #[test]
    fn diagnostics_identify_oscillation() {
        let config = Phase6DConfig::default();
        let model = evaluate_with_config(&config, &trend(0.0, 0.0, 0.0, 4.0));
        assert_eq!(model.state_label, DiagnosticState::Oscillating);
        assert!(model.oscillation_index >= OSCILLATION_ALERT);
    }

    #[test]
    fn diagnostics_mark_diverging_loss() {
        let mut config = Phase6DConfig::default();
        config.loss_slope_limit = 0.01;
        let model = evaluate_with_config(&config, &trend(0.02, 0.0, 0.0, 0.0));
        assert_eq!(model.state_label, DiagnosticState::Diverging);
        assert!(model.loss_slope >= SEVERE_SLOPE);
    }

    fn evaluate_with_config(config: &Phase6DConfig, trend: &TrendModel) -> DiagnosticModel {
        let loss_slope = normalize_positive(trend.slope_loss, config.loss_slope_limit);
        let entropy_drift = normalize_positive(trend.slope_entropy, config.entropy_drift_limit);
        let coherence_decay =
            normalize_negative(trend.slope_coherence, config.coherence_decay_limit);
        let oscillation_index =
            normalize_oscillation(trend.oscillation_period, config.oscillation_index_limit);

        let weights = config.risk_weight.normalized();
        let mut risk_score = loss_slope * weights.loss
            + entropy_drift * weights.entropy
            + coherence_decay * weights.coherence
            + oscillation_index * weights.oscillation;
        risk_score = risk_score.clamp(0.0, 1.0);

        let mut model = DiagnosticModel {
            loss_slope,
            entropy_drift,
            coherence_decay,
            oscillation_index,
            risk_score,
            state_label: DiagnosticState::Stable,
        };
        model.state_label = classify_state(&model);
        model
    }
}
