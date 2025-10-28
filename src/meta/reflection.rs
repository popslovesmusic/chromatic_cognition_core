//! Adaptive reflection planning triggered by elevated dissonance.
//!
//! The planner converts the scalar dissonance score into a deterministic
//! sequence of mitigation steps sourced from configuration. Each step
//! includes a reversible directive to guarantee safe recovery.

use serde::Serialize;

use super::dissonance::Dissonance;
use crate::config::Phase5BConfig;

/// Actions supported by the reflection planner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum ReflectionAction {
    SeedFrom,
    DampLr,
    CoolTint,
    PauseAug,
}

impl ReflectionAction {
    fn apply_directive(&self) -> &'static str {
        match self {
            ReflectionAction::SeedFrom => {
                "Reseed predictors using the latest stable observation to reset drift."
            }
            ReflectionAction::DampLr => "Reduce learning rate by 20% for the next update window.",
            ReflectionAction::CoolTint => {
                "Blend a cool tint filter to dissipate excess chroma energy."
            }
            ReflectionAction::PauseAug => {
                "Pause augmentation pipeline for three cycles to stabilise inputs."
            }
        }
    }

    fn revert_directive(&self) -> &'static str {
        match self {
            ReflectionAction::SeedFrom => "Restore prior predictor seeds once metrics stabilise.",
            ReflectionAction::DampLr => {
                "Reinstate the baseline learning rate after evaluation window."
            }
            ReflectionAction::CoolTint => "Remove the tint filter and rebalance chroma weights.",
            ReflectionAction::PauseAug => "Resume augmentation pipeline with previous schedule.",
        }
    }
}

impl std::str::FromStr for ReflectionAction {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "SeedFrom" => Ok(ReflectionAction::SeedFrom),
            "DampLR" | "DampLr" => Ok(ReflectionAction::DampLr),
            "CoolTint" => Ok(ReflectionAction::CoolTint),
            "PauseAug" => Ok(ReflectionAction::PauseAug),
            other => Err(format!("unsupported reflection action '{}'", other)),
        }
    }
}

/// Single corrective step with matching revert instructions.
#[derive(Debug, Clone, Serialize)]
pub struct PlanStep {
    pub action: ReflectionAction,
    pub directive: String,
    pub revert: String,
}

impl PlanStep {
    fn new(action: ReflectionAction) -> Self {
        Self {
            action,
            directive: action.apply_directive().to_string(),
            revert: action.revert_directive().to_string(),
        }
    }
}

/// Reflection plan generated for a particular cycle.
#[derive(Debug, Clone, Serialize)]
pub struct Plan {
    pub step: usize,
    pub triggered: bool,
    pub score: f32,
    pub threshold: f32,
    pub rationale: String,
    pub steps: Vec<PlanStep>,
}

/// Build a deterministic reflection plan.
pub fn plan_reflection(step: usize, dissonance: &Dissonance, config: &Phase5BConfig) -> Plan {
    let triggered = dissonance.exceeds(config.dissonance_threshold);
    let mut steps = Vec::new();
    if triggered {
        steps = config
            .actions
            .iter()
            .map(|action| PlanStep::new(*action))
            .collect();
    }

    let rationale = if triggered {
        format!(
            "Dissonance {:.3} exceeded threshold {:.3}; executing {} corrective actions.",
            dissonance.score,
            config.dissonance_threshold,
            steps.len()
        )
    } else {
        format!(
            "Dissonance {:.3} below threshold {:.3}; maintaining current trajectory.",
            dissonance.score, config.dissonance_threshold
        )
    };

    Plan {
        step,
        triggered,
        score: dissonance.score,
        threshold: config.dissonance_threshold,
        rationale,
        steps,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::meta::dissonance::{Dissonance, DissonanceWeights};
    use crate::meta::predict::{FeatureForecast, PredictionSet};

    fn sample_plan(score: f32, threshold: f32) -> Plan {
        let config = Phase5BConfig {
            dissonance_threshold: threshold,
            weights: DissonanceWeights {
                coherence: 0.5,
                entropy: 0.3,
                energy: 0.2,
            },
            actions: vec![
                ReflectionAction::SeedFrom,
                ReflectionAction::DampLr,
                ReflectionAction::CoolTint,
                ReflectionAction::PauseAug,
            ],
        };

        let prediction = PredictionSet {
            horizon: 1,
            forecasts: vec![
                FeatureForecast {
                    feature: crate::meta::predict::Feature::Coherence,
                    values: vec![0.6],
                },
                FeatureForecast {
                    feature: crate::meta::predict::Feature::Entropy,
                    values: vec![0.4],
                },
                FeatureForecast {
                    feature: crate::meta::predict::Feature::GradEnergy,
                    values: vec![0.3],
                },
            ],
        };

        let observation = crate::meta::awareness::Observation {
            step: 42,
            coherence: 0.6,
            entropy: 0.4,
            grad_energy: 0.3,
            mean_rgb: [0.0; 3],
            variance: 0.0,
            mean_certainty: 0.0,
            low_freq_energy: 0.0,
            mid_freq_energy: 0.0,
            high_freq_energy: 0.0,
            mean_psd: 0.0,
            dominant_frequencies: [0; 3],
        };

        let mut dissonance = Dissonance::compare(&prediction, &observation, config.weights);
        dissonance.score = score;
        plan_reflection(42, &dissonance, &config)
    }

    #[test]
    fn plans_only_trigger_above_threshold() {
        let plan_low = sample_plan(0.2, 0.25);
        let plan_high = sample_plan(0.4, 0.25);

        assert!(!plan_low.triggered);
        assert!(plan_low.steps.is_empty());
        assert!(plan_high.triggered);
        assert_eq!(plan_high.steps.len(), 4);
    }

    #[test]
    fn plans_are_reversible() {
        let plan = sample_plan(0.4, 0.25);
        for step in &plan.steps {
            assert!(!step.directive.is_empty());
            assert!(!step.revert.is_empty());
            assert_ne!(step.directive, step.revert);
        }
    }
}
