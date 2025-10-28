//! Adapter that translates ethics-approved interventions into training control
//! updates with full rollback support.

use serde::Serialize;

use super::ethics::{EthicsGuard, InterventionSpec, VerdictStatus};
use super::log::{ControlStateSnapshot, MetaActionStatus, MetaLogEntry, MetaLogger};
use super::reflection::{Plan, ReflectionAction};

/// Deterministic defaults for reflection actions.
const DEFAULT_LR_DAMP_FACTOR: f32 = 0.2;
const DEFAULT_COOL_TINT_AMOUNT: f32 = 0.15;
const DEFAULT_COOL_TINT_HUE_SHIFT: f32 = 45.0;
const DEFAULT_PAUSE_AUG_STEPS: usize = 3;

/// Training controls affected by Phase 5C interventions.
#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct TrainingControls {
    pub learning_rate: f32,
    pub tint_cool_strength: f32,
    pub hue_shift_deg: f32,
    pub augmentation_pause_steps: usize,
    pub reseed_from_step: Option<usize>,
}

impl TrainingControls {
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            tint_cool_strength: 0.0,
            hue_shift_deg: 0.0,
            augmentation_pause_steps: 0,
            reseed_from_step: None,
        }
    }

    fn snapshot(&self) -> ControlStateSnapshot {
        ControlStateSnapshot {
            learning_rate: self.learning_rate,
            tint_cool_strength: self.tint_cool_strength,
            hue_shift_deg: self.hue_shift_deg,
            augmentation_pause_steps: self.augmentation_pause_steps,
            reseed_from_step: self.reseed_from_step,
        }
    }
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct MetaCycleReport {
    pub step: usize,
    pub triggered: bool,
    pub attempted: Vec<InterventionSpec>,
    pub applied: Vec<InterventionSpec>,
    pub violations: Vec<String>,
    pub rolled_back: bool,
}

impl MetaCycleReport {
    fn empty(step: usize, triggered: bool) -> Self {
        Self {
            step,
            triggered,
            attempted: Vec::new(),
            applied: Vec::new(),
            violations: Vec::new(),
            rolled_back: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
struct AppliedAction {
    effect: InterventionSpec,
    snapshot: ControlSnapshot,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
enum ControlSnapshot {
    SeedFrom {
        previous: Option<usize>,
    },
    DampLr {
        previous: f32,
    },
    CoolTint {
        previous_amount: f32,
        previous_hue: f32,
    },
    PauseAug {
        previous: usize,
    },
}

impl AppliedAction {
    fn rollback(&self, controls: &mut TrainingControls) {
        match (&self.effect, &self.snapshot) {
            (InterventionSpec::SeedFrom, ControlSnapshot::SeedFrom { previous }) => {
                controls.reseed_from_step = *previous;
            }
            (InterventionSpec::DampLr { .. }, ControlSnapshot::DampLr { previous }) => {
                controls.learning_rate = *previous;
            }
            (
                InterventionSpec::CoolTint { .. },
                ControlSnapshot::CoolTint {
                    previous_amount,
                    previous_hue,
                },
            ) => {
                controls.tint_cool_strength = *previous_amount;
                controls.hue_shift_deg = *previous_hue;
            }
            (InterventionSpec::PauseAug { .. }, ControlSnapshot::PauseAug { previous }) => {
                controls.augmentation_pause_steps = *previous;
            }
            _ => {
                debug_assert!(false, "snapshot/action mismatch in rollback");
            }
        }
    }
}

/// Meta adapter executing reflection plans under ethics review.
#[derive(Debug, Clone, Serialize)]
pub struct MetaAdapter<'a> {
    guard: &'a EthicsGuard,
}

impl<'a> MetaAdapter<'a> {
    pub fn new(guard: &'a EthicsGuard) -> Self {
        Self { guard }
    }

    pub fn execute_plan(
        &self,
        plan: &Plan,
        controls: &mut TrainingControls,
        logger: &mut MetaLogger,
    ) -> MetaCycleReport {
        if !plan.triggered {
            let report = MetaCycleReport::empty(plan.step, false);
            let _ = logger.record(MetaLogEntry {
                sequence: 0,
                step: plan.step,
                action: None,
                status: MetaActionStatus::Skipped,
                details: plan.rationale.clone(),
                timestamp_ms: MetaLogger::timestamp_now(),
                state: controls.snapshot(),
            });
            return report;
        }

        let mut report = MetaCycleReport::empty(plan.step, true);
        let mut applied_actions: Vec<AppliedAction> = Vec::new();
        let mut attempted_requests = Vec::new();
        let mut applied_effects = Vec::new();
        let mut violations = Vec::new();
        let mut rollback_needed = false;

        for step in &plan.steps {
            let request = self.request_from_action(step.action);
            attempted_requests.push(request.clone());

            let verdict = self.guard.review(&request);
            match verdict.status {
                VerdictStatus::Approved | VerdictStatus::Clipped => {
                    let effect = verdict
                        .effect
                        .clone()
                        .expect("approved verdict must contain effect");
                    let snapshot = apply_effect(plan.step, &effect, controls);
                    applied_effects.push(effect.clone());
                    applied_actions.push(AppliedAction {
                        effect: effect.clone(),
                        snapshot,
                    });

                    let status = match verdict.status {
                        VerdictStatus::Approved => MetaActionStatus::Applied,
                        VerdictStatus::Clipped => MetaActionStatus::Clipped,
                        VerdictStatus::Rejected => MetaActionStatus::Rejected,
                    };

                    let mut details = format!("Applied {:?}", effect.reflection_action());
                    if !verdict.notes.is_empty() {
                        details.push_str(&format!(": {}", verdict.notes.join("; ")));
                    }

                    let _ = logger.record(MetaLogEntry {
                        sequence: 0,
                        step: plan.step,
                        action: Some(effect.reflection_action()),
                        status,
                        details,
                        timestamp_ms: MetaLogger::timestamp_now(),
                        state: controls.snapshot(),
                    });
                }
                VerdictStatus::Rejected => {
                    rollback_needed = true;
                    let detail = if verdict.notes.is_empty() {
                        format!("{:?} rejected by ethics guard", request.reflection_action())
                    } else {
                        verdict.notes.join("; ")
                    };
                    let _ = logger.record(MetaLogEntry {
                        sequence: 0,
                        step: plan.step,
                        action: Some(request.reflection_action()),
                        status: MetaActionStatus::Rejected,
                        details: detail.clone(),
                        timestamp_ms: MetaLogger::timestamp_now(),
                        state: controls.snapshot(),
                    });
                    violations.push(detail);
                    break;
                }
            }
        }

        if rollback_needed {
            for action in applied_actions.iter().rev() {
                action.rollback(controls);
            }
            applied_effects.clear();
            let _ = logger.record(MetaLogEntry {
                sequence: 0,
                step: plan.step,
                action: None,
                status: MetaActionStatus::Rollback,
                details: format!("Rolled back {} interventions", applied_actions.len()),
                timestamp_ms: MetaLogger::timestamp_now(),
                state: controls.snapshot(),
            });
        }

        report.attempted = attempted_requests;
        report.applied = applied_effects;
        report.violations = violations;
        report.rolled_back = rollback_needed;
        report
    }

    fn request_from_action(&self, action: ReflectionAction) -> InterventionSpec {
        match action {
            ReflectionAction::SeedFrom => InterventionSpec::SeedFrom,
            ReflectionAction::DampLr => InterventionSpec::DampLr {
                factor: DEFAULT_LR_DAMP_FACTOR,
            },
            ReflectionAction::CoolTint => InterventionSpec::CoolTint {
                amount: DEFAULT_COOL_TINT_AMOUNT,
                hue_shift_deg: DEFAULT_COOL_TINT_HUE_SHIFT,
            },
            ReflectionAction::PauseAug => InterventionSpec::PauseAug {
                steps: DEFAULT_PAUSE_AUG_STEPS,
            },
        }
    }
}

fn apply_effect(
    step: usize,
    effect: &InterventionSpec,
    controls: &mut TrainingControls,
) -> ControlSnapshot {
    match effect {
        InterventionSpec::SeedFrom => {
            let previous = controls.reseed_from_step;
            controls.reseed_from_step = Some(step);
            ControlSnapshot::SeedFrom { previous }
        }
        InterventionSpec::DampLr { factor } => {
            let previous = controls.learning_rate;
            let damp_factor = 1.0 - factor.clamp(0.0, 1.0);
            controls.learning_rate = (previous * damp_factor).max(1e-6);
            ControlSnapshot::DampLr { previous }
        }
        InterventionSpec::CoolTint {
            amount,
            hue_shift_deg,
        } => {
            let snapshot = ControlSnapshot::CoolTint {
                previous_amount: controls.tint_cool_strength,
                previous_hue: controls.hue_shift_deg,
            };
            controls.tint_cool_strength = (*amount).clamp(0.0, 1.0);
            controls.hue_shift_deg = (*hue_shift_deg).clamp(0.0, 360.0);
            snapshot
        }
        InterventionSpec::PauseAug { steps } => {
            let previous = controls.augmentation_pause_steps;
            controls.augmentation_pause_steps = *steps;
            ControlSnapshot::PauseAug { previous }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::meta::reflection::{PlanStep, ReflectionAction};

    fn plan_with_actions(actions: Vec<ReflectionAction>) -> Plan {
        Plan {
            step: 42,
            triggered: true,
            score: 0.4,
            threshold: 0.25,
            rationale: "test".to_string(),
            steps: actions
                .into_iter()
                .map(|action| PlanStep {
                    action,
                    directive: String::new(),
                    revert: String::new(),
                })
                .collect(),
        }
    }

    fn guard_with_bounds(lr_max: f32, pause_max: usize) -> EthicsGuard {
        EthicsGuard::new(crate::meta::ethics::EthicsBounds {
            lr_damp_max: lr_max,
            cool_tint_max: 0.2,
            pause_aug_max_steps: pause_max,
            ethics_hue_jump_deg: 90.0,
        })
    }

    #[test]
    fn adapter_applies_actions() {
        let guard = guard_with_bounds(0.5, 10);
        let adapter = MetaAdapter::new(&guard);
        let mut controls = TrainingControls::new(0.1);
        let mut logger = MetaLogger::new(1);
        let plan = plan_with_actions(vec![
            ReflectionAction::SeedFrom,
            ReflectionAction::DampLr,
            ReflectionAction::CoolTint,
            ReflectionAction::PauseAug,
        ]);

        let report = adapter.execute_plan(&plan, &mut controls, &mut logger);

        assert!(report.triggered);
        assert!(!report.rolled_back);
        assert_eq!(report.applied.len(), 4);
        assert_eq!(controls.reseed_from_step, Some(plan.step));
        assert!(controls.learning_rate < 0.1);
        assert_eq!(controls.augmentation_pause_steps, DEFAULT_PAUSE_AUG_STEPS);
        assert_eq!(logger.entries().len(), 4);
        assert_eq!(logger.entries()[0].status, MetaActionStatus::Applied);
    }

    #[test]
    fn adapter_rolls_back_on_violation() {
        let guard = guard_with_bounds(0.5, 0);
        let adapter = MetaAdapter::new(&guard);
        let mut controls = TrainingControls::new(0.1);
        let mut logger = MetaLogger::new(1);
        let plan = plan_with_actions(vec![ReflectionAction::DampLr, ReflectionAction::PauseAug]);

        let report = adapter.execute_plan(&plan, &mut controls, &mut logger);

        assert!(report.triggered);
        assert!(report.rolled_back);
        assert!(report.applied.is_empty());
        assert_eq!(controls.learning_rate, 0.1);
        assert_eq!(controls.augmentation_pause_steps, 0);
        assert!(report
            .violations
            .iter()
            .any(|msg| msg.contains("augmentation")));
        assert_eq!(logger.entries().len(), 3); // apply, reject, rollback
        assert_eq!(logger.entries()[2].status, MetaActionStatus::Rollback);
    }
}
