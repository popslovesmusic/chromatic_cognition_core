//! Ethics guard enforcing bounded self-regulation actions.
//!
//! The guard is responsible for reviewing reflection actions before they are
//! applied to the training context. All requested changes are clamped to the
//! configured safety window and invalid directives are rejected with a
//! structured verdict.

use serde::Serialize;

use super::reflection::ReflectionAction;

/// Safety envelope for self-regulation controls.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct EthicsBounds {
    pub lr_damp_max: f32,
    pub cool_tint_max: f32,
    pub pause_aug_max_steps: usize,
    pub ethics_hue_jump_deg: f32,
}

impl EthicsBounds {
    pub fn sanitise(self) -> Self {
        Self {
            lr_damp_max: self.lr_damp_max.clamp(0.0, 1.0),
            cool_tint_max: self.cool_tint_max.clamp(0.0, 1.0),
            pause_aug_max_steps: self.pause_aug_max_steps,
            ethics_hue_jump_deg: self.ethics_hue_jump_deg.max(0.0),
        }
    }
}

/// Requested intervention derived from a reflection plan step.
#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(tag = "action", rename_all = "camelCase")]
pub enum InterventionSpec {
    SeedFrom,
    DampLr { factor: f32 },
    CoolTint { amount: f32, hue_shift_deg: f32 },
    PauseAug { steps: usize },
}

impl InterventionSpec {
    pub fn reflection_action(&self) -> ReflectionAction {
        match self {
            InterventionSpec::SeedFrom => ReflectionAction::SeedFrom,
            InterventionSpec::DampLr { .. } => ReflectionAction::DampLr,
            InterventionSpec::CoolTint { .. } => ReflectionAction::CoolTint,
            InterventionSpec::PauseAug { .. } => ReflectionAction::PauseAug,
        }
    }
}

/// Result of an ethics review for a requested intervention.
#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct EthicsVerdict {
    pub status: VerdictStatus,
    pub effect: Option<InterventionSpec>,
    pub notes: Vec<String>,
}

impl EthicsVerdict {
    pub fn approved(effect: InterventionSpec, status: VerdictStatus, notes: Vec<String>) -> Self {
        Self {
            status,
            effect: Some(effect),
            notes,
        }
    }

    pub fn rejected(notes: Vec<String>) -> Self {
        Self {
            status: VerdictStatus::Rejected,
            effect: None,
            notes,
        }
    }
}

/// Final decision returned by the [`EthicsGuard`].
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum VerdictStatus {
    Approved,
    Clipped,
    Rejected,
}

/// Ethics guard that validates and clips plan interventions.
#[derive(Debug, Clone, Serialize)]
pub struct EthicsGuard {
    bounds: EthicsBounds,
}

impl EthicsGuard {
    pub fn new(bounds: EthicsBounds) -> Self {
        Self {
            bounds: bounds.sanitise(),
        }
    }

    pub fn bounds(&self) -> EthicsBounds {
        self.bounds
    }

    /// Review a requested intervention and clamp it within bounds.
    pub fn review(&self, request: &InterventionSpec) -> EthicsVerdict {
        match request {
            InterventionSpec::SeedFrom => {
                EthicsVerdict::approved(request.clone(), VerdictStatus::Approved, vec![])
            }
            InterventionSpec::DampLr { factor } => {
                if *factor <= 0.0 {
                    return EthicsVerdict::rejected(vec![
                        "learning-rate damping factor must be positive".to_string(),
                    ]);
                }
                let clipped = factor.min(self.bounds.lr_damp_max);
                if clipped <= 0.0 {
                    return EthicsVerdict::rejected(vec![
                        "learning-rate damping disabled by configuration".to_string(),
                    ]);
                }
                let mut notes = Vec::new();
                let mut status = VerdictStatus::Approved;
                if (clipped - factor).abs() > f32::EPSILON {
                    status = VerdictStatus::Clipped;
                    notes.push(format!(
                        "learning-rate damping clipped from {:.3} to {:.3}",
                        factor, clipped
                    ));
                }
                EthicsVerdict::approved(InterventionSpec::DampLr { factor: clipped }, status, notes)
            }
            InterventionSpec::CoolTint {
                amount,
                hue_shift_deg,
            } => {
                if *amount <= 0.0 {
                    return EthicsVerdict::rejected(vec![
                        "cool tint amount must be positive".to_string()
                    ]);
                }
                let clipped_amount = amount.min(self.bounds.cool_tint_max);
                let clipped_hue = hue_shift_deg.min(self.bounds.ethics_hue_jump_deg);
                if clipped_amount <= 0.0 {
                    return EthicsVerdict::rejected(vec![
                        "cool tint adjustments disabled by configuration".to_string(),
                    ]);
                }
                let mut notes = Vec::new();
                let mut status = VerdictStatus::Approved;
                if (clipped_amount - amount).abs() > f32::EPSILON {
                    status = VerdictStatus::Clipped;
                    notes.push(format!(
                        "cool tint intensity clipped from {:.3} to {:.3}",
                        amount, clipped_amount
                    ));
                }
                if (clipped_hue - hue_shift_deg).abs() > f32::EPSILON {
                    status = VerdictStatus::Clipped;
                    notes.push(format!(
                        "hue shift clipped from {:.1}° to {:.1}°",
                        hue_shift_deg, clipped_hue
                    ));
                }
                EthicsVerdict::approved(
                    InterventionSpec::CoolTint {
                        amount: clipped_amount,
                        hue_shift_deg: clipped_hue,
                    },
                    status,
                    notes,
                )
            }
            InterventionSpec::PauseAug { steps } => {
                if *steps == 0 {
                    return EthicsVerdict::rejected(vec![
                        "pause augmentation steps must be positive".to_string(),
                    ]);
                }
                let clipped_steps = (*steps).min(self.bounds.pause_aug_max_steps);
                if clipped_steps == 0 {
                    return EthicsVerdict::rejected(vec![
                        "augmentation pauses disabled by configuration".to_string(),
                    ]);
                }
                let mut notes = Vec::new();
                let mut status = VerdictStatus::Approved;
                if clipped_steps != *steps {
                    status = VerdictStatus::Clipped;
                    notes.push(format!(
                        "augmentation pause clipped from {} to {} steps",
                        steps, clipped_steps
                    ));
                }
                EthicsVerdict::approved(
                    InterventionSpec::PauseAug {
                        steps: clipped_steps,
                    },
                    status,
                    notes,
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_guard() -> EthicsGuard {
        EthicsGuard::new(EthicsBounds {
            lr_damp_max: 0.5,
            cool_tint_max: 0.2,
            pause_aug_max_steps: 200,
            ethics_hue_jump_deg: 90.0,
        })
    }

    #[test]
    fn guard_clips_learning_rate() {
        let guard = default_guard();
        let verdict = guard.review(&InterventionSpec::DampLr { factor: 0.8 });
        assert_eq!(verdict.status, VerdictStatus::Clipped);
        assert_eq!(
            verdict.effect,
            Some(InterventionSpec::DampLr { factor: 0.5 })
        );
        assert_eq!(verdict.notes.len(), 1);
    }

    #[test]
    fn guard_rejects_negative_requests() {
        let guard = default_guard();
        let verdict = guard.review(&InterventionSpec::DampLr { factor: -0.1 });
        assert_eq!(verdict.status, VerdictStatus::Rejected);
        assert!(verdict.effect.is_none());
    }

    #[test]
    fn guard_clips_cool_tint_and_hue() {
        let guard = default_guard();
        let verdict = guard.review(&InterventionSpec::CoolTint {
            amount: 0.5,
            hue_shift_deg: 180.0,
        });
        assert_eq!(verdict.status, VerdictStatus::Clipped);
        assert_eq!(
            verdict.effect,
            Some(InterventionSpec::CoolTint {
                amount: 0.2,
                hue_shift_deg: 90.0,
            })
        );
        assert_eq!(verdict.notes.len(), 2);
    }

    #[test]
    fn guard_rejects_zero_pause() {
        let guard = default_guard();
        let verdict = guard.review(&InterventionSpec::PauseAug { steps: 0 });
        assert_eq!(verdict.status, VerdictStatus::Rejected);
    }
}
