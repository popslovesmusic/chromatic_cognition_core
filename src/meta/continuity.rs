//! Continuity control loop translating long-term trends into bounded training
//! adjustments.
//!
//! Phase 6C closes the loop opened by awareness (6A) and trend fitting (6B).
//! The planner maps trend slopes to deterministic actions while the executor
//! applies them within conservative configuration bounds.

use serde::Serialize;

use crate::config::Phase6CConfig;

use super::trend::TrendModel;

const LOSS_SLOPE_THRESHOLD: f32 = 0.005;
const COHERENCE_THRESHOLD: f32 = 0.005;
const ENTROPY_THRESHOLD: f32 = 0.005;

/// Temporal regulation actions emitted by the continuity planner.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub enum TemporalAction {
    AdjustLR(f32),
    ExpandDreamPool(usize),
    ContractDreamPool(usize),
    ResetPhaseWeights,
}

/// Minimal training context manipulated by the continuity executor.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct TrainCtx {
    pub cycle: usize,
    pub learning_rate: f32,
    pub dream_pool_size: usize,
    pub phase_weights: Vec<f32>,
    pub cooldown: usize,
    pub last_action_cycle: Option<usize>,
}

impl TrainCtx {
    /// Construct a new context with zeroed cooldown.
    pub fn new(learning_rate: f32, dream_pool_size: usize, phase_weights: Vec<f32>) -> Self {
        Self {
            cycle: 0,
            learning_rate,
            dream_pool_size,
            phase_weights,
            cooldown: 0,
            last_action_cycle: None,
        }
    }

    /// Advance the cycle counter and age any active cooldown.
    pub fn advance_cycle(&mut self) {
        self.cycle = self.cycle.saturating_add(1);
        if self.cooldown > 0 {
            self.cooldown -= 1;
        }
    }

    /// Determine whether continuity regulation may trigger this cycle.
    pub fn ready_for_temporal_action(&self) -> bool {
        self.cooldown == 0
    }
}

/// Determine whether the continuity controller should execute at the supplied cycle.
pub fn should_invoke_continuity(cycle: usize) -> bool {
    let config = Phase6CConfig::default();
    let interval = config.cycle_interval.max(1);
    cycle % interval == 0
}

/// Translate a trend model into a bounded temporal action.
pub fn plan_temporal_action(trend: &TrendModel) -> Option<TemporalAction> {
    let config = Phase6CConfig::default();

    if trend.oscillation_period > 0.0 {
        return Some(TemporalAction::ResetPhaseWeights);
    }

    if trend.slope_loss > LOSS_SLOPE_THRESHOLD {
        let magnitude = (trend.slope_loss).min(config.lr_adjust_max).max(0.0);
        if magnitude > 0.0 {
            return Some(TemporalAction::AdjustLR(-magnitude));
        }
    }

    if trend.slope_loss < -LOSS_SLOPE_THRESHOLD {
        let magnitude = (-trend.slope_loss).min(config.lr_adjust_max).max(0.0);
        if magnitude > 0.0 {
            return Some(TemporalAction::AdjustLR(magnitude));
        }
    }

    if trend.slope_coherence < -COHERENCE_THRESHOLD {
        let severity = (-trend.slope_coherence / COHERENCE_THRESHOLD).min(1.0);
        let change = (severity * config.dream_pool_expand_max as f32).round() as usize;
        if change > 0 {
            return Some(TemporalAction::ExpandDreamPool(
                change.min(config.dream_pool_expand_max),
            ));
        }
    }

    if trend.slope_entropy > ENTROPY_THRESHOLD {
        let severity = (trend.slope_entropy / ENTROPY_THRESHOLD).min(1.0);
        let change = (severity * config.dream_pool_expand_max as f32).round() as usize;
        if change > 0 {
            return Some(TemporalAction::ContractDreamPool(
                change.min(config.dream_pool_expand_max),
            ));
        }
    }

    None
}

/// Apply a temporal action to the mutable training context while honoring configuration limits.
pub fn apply_temporal_action(action: &TemporalAction, ctx: &mut TrainCtx) {
    let config = Phase6CConfig::default();

    match action {
        TemporalAction::AdjustLR(delta) => {
            let clipped = delta.clamp(-config.lr_adjust_max, config.lr_adjust_max);
            let factor = (1.0 + clipped).max(1e-6);
            ctx.learning_rate *= factor;
        }
        TemporalAction::ExpandDreamPool(amount) => {
            let allowed = (*amount).min(config.dream_pool_expand_max);
            ctx.dream_pool_size = ctx.dream_pool_size.saturating_add(allowed);
        }
        TemporalAction::ContractDreamPool(amount) => {
            let allowed = (*amount).min(config.dream_pool_expand_max);
            ctx.dream_pool_size = ctx.dream_pool_size.saturating_sub(allowed);
        }
        TemporalAction::ResetPhaseWeights => {
            if !ctx.phase_weights.is_empty() {
                let uniform = 1.0f32 / ctx.phase_weights.len() as f32;
                for weight in &mut ctx.phase_weights {
                    *weight = uniform;
                }
            }
        }
    }

    ctx.last_action_cycle = Some(ctx.cycle);
    ctx.cooldown = config.trend_anomaly_cooldown;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trend(loss: f32, coherence: f32, entropy: f32, oscillation: f32) -> TrendModel {
        TrendModel {
            slope_coherence: coherence,
            slope_entropy: entropy,
            slope_loss: loss,
            oscillation_period: oscillation,
        }
    }

    #[test]
    fn planner_reduces_learning_rate_on_loss_increase() {
        let action = plan_temporal_action(&trend(0.02, 0.0, 0.0, 0.0)).unwrap();
        match action {
            TemporalAction::AdjustLR(delta) => assert!(delta < 0.0),
            _ => panic!("expected AdjustLR action"),
        }
    }

    #[test]
    fn planner_expands_dream_pool_on_coherence_drop() {
        let action = plan_temporal_action(&trend(0.0, -0.02, 0.0, 0.0)).unwrap();
        assert!(matches!(action, TemporalAction::ExpandDreamPool(amount) if amount > 0));
    }

    #[test]
    fn planner_contracts_dream_pool_on_entropy_spike() {
        let action = plan_temporal_action(&trend(0.0, 0.0, 0.02, 0.0)).unwrap();
        assert!(matches!(action, TemporalAction::ContractDreamPool(amount) if amount > 0));
    }

    #[test]
    fn planner_resets_weights_on_oscillation_detection() {
        let action = plan_temporal_action(&trend(0.0, 0.0, 0.0, 4.0)).unwrap();
        assert_eq!(action, TemporalAction::ResetPhaseWeights);
    }

    #[test]
    fn apply_learning_rate_respects_bounds() {
        let mut ctx = TrainCtx::new(0.1, 32, vec![0.4, 0.3, 0.3]);
        ctx.cycle = 5;
        apply_temporal_action(&TemporalAction::AdjustLR(-0.5), &mut ctx);
        let max_delta = Phase6CConfig::default().lr_adjust_max;
        let expected = 0.1 * (1.0 - max_delta);
        assert!((ctx.learning_rate - expected).abs() < 1e-6);
        assert_eq!(
            ctx.cooldown,
            Phase6CConfig::default().trend_anomaly_cooldown
        );
        assert_eq!(ctx.last_action_cycle, Some(5));
    }

    #[test]
    fn apply_pool_changes_clip_to_bounds() {
        let mut ctx = TrainCtx::new(0.1, 10, vec![0.5, 0.5]);
        apply_temporal_action(&TemporalAction::ExpandDreamPool(500), &mut ctx);
        assert_eq!(
            ctx.dream_pool_size,
            10 + Phase6CConfig::default().dream_pool_expand_max
        );
        apply_temporal_action(&TemporalAction::ContractDreamPool(500), &mut ctx);
        assert_eq!(ctx.dream_pool_size, 10);
    }

    #[test]
    fn apply_reset_normalises_phase_weights() {
        let mut ctx = TrainCtx::new(0.1, 0, vec![0.2, 0.5, 0.3]);
        apply_temporal_action(&TemporalAction::ResetPhaseWeights, &mut ctx);
        for weight in ctx.phase_weights {
            assert!((weight - (1.0 / 3.0)).abs() < 1e-6);
        }
    }

    #[test]
    fn cooldown_decrements_on_advance_cycle() {
        let mut ctx = TrainCtx::new(0.1, 0, vec![]);
        ctx.cooldown = 2;
        ctx.advance_cycle();
        assert_eq!(ctx.cooldown, 1);
        ctx.advance_cycle();
        assert_eq!(ctx.cooldown, 0);
    }

    #[test]
    fn readiness_depends_on_cooldown() {
        let mut ctx = TrainCtx::new(0.1, 0, vec![]);
        assert!(ctx.ready_for_temporal_action());
        ctx.cooldown = 1;
        assert!(!ctx.ready_for_temporal_action());
    }

    #[test]
    fn should_invoke_respects_interval() {
        let config = Phase6CConfig::default();
        assert!(should_invoke_continuity(config.cycle_interval));
        assert!(!should_invoke_continuity(1));
    }
}
