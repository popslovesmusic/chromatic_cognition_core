use crate::solver::{Solver, SolverResult};
/// Native Rust implementation of chromatic field solver
///
/// This solver computes color-theory-informed metrics directly without
/// external dependencies. It provides:
///
/// - **Energy**: Total variation (smoothness) + saturation penalty
/// - **Coherence**: Color harmony based on complementary balance and hue consistency
/// - **Violation**: Gamut clipping, extreme saturation, local discontinuities
///
/// All metrics have analytical gradients for efficient training.
use crate::tensor::ChromaticTensor;
use anyhow::Result;
use serde_json::json;

/// Native Rust solver with color-space metrics
pub struct ChromaticNativeSolver {
    /// Weight for total variation term in energy
    pub lambda_tv: f32,

    /// Weight for saturation penalty term in energy
    pub lambda_sat: f32,

    /// Target saturation (0.5 = neutral, deviations penalized)
    pub target_saturation: f32,

    /// Threshold for local discontinuity detection (ΔE)
    pub discontinuity_threshold: f32,
}

impl Default for ChromaticNativeSolver {
    fn default() -> Self {
        Self {
            lambda_tv: 1.0,
            lambda_sat: 0.1,
            target_saturation: 0.5,
            discontinuity_threshold: 0.3,
        }
    }
}

impl ChromaticNativeSolver {
    /// Create a new native solver with default parameters
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a solver with custom parameters
    pub fn with_params(
        lambda_tv: f32,
        lambda_sat: f32,
        target_saturation: f32,
        discontinuity_threshold: f32,
    ) -> Self {
        Self {
            lambda_tv,
            lambda_sat,
            target_saturation,
            discontinuity_threshold,
        }
    }

    /// Compute total variation (spatial smoothness penalty)
    ///
    /// TV(F) = Σ ‖F[i,j,l] - F[i+1,j,l]‖ + ‖F[i,j,l] - F[i,j+1,l]‖
    fn compute_total_variation(&self, field: &ChromaticTensor) -> f32 {
        let mut tv = 0.0;

        for l in 0..field.layers() {
            for r in 0..field.rows() {
                for c in 0..field.cols() {
                    let curr = field.get_rgb(r, c, l);

                    // Right neighbor
                    if r + 1 < field.rows() {
                        let right = field.get_rgb(r + 1, c, l);
                        tv += rgb_distance(&curr, &right);
                    }

                    // Down neighbor
                    if c + 1 < field.cols() {
                        let down = field.get_rgb(r, c + 1, l);
                        tv += rgb_distance(&curr, &down);
                    }
                }
            }
        }

        tv
    }

    /// Compute saturation penalty (encourages target saturation)
    ///
    /// Sat_penalty = Σ (saturation[i,j,l] - target)²
    fn compute_saturation_penalty(&self, field: &ChromaticTensor) -> f32 {
        let mut penalty = 0.0;
        let total_cells = (field.rows() * field.cols() * field.layers()) as f32;

        for l in 0..field.layers() {
            for r in 0..field.rows() {
                for c in 0..field.cols() {
                    let rgb = field.get_rgb(r, c, l);
                    let saturation = rgb_saturation(&rgb);
                    let diff = saturation - self.target_saturation;
                    penalty += diff * diff;
                }
            }
        }

        penalty / total_cells
    }

    /// Compute color harmony score (0-1, higher is better)
    ///
    /// Measures:
    /// - Complementary balance (red-cyan, green-magenta, yellow-blue)
    /// - Hue consistency (lower std dev = more coherent)
    fn compute_color_harmony(&self, field: &ChromaticTensor) -> f32 {
        // 1. Complementary balance
        let mean_rgb = field.mean_rgb();
        let complementary_balance = 1.0
            - ((mean_rgb[0] - 0.5).abs() + (mean_rgb[1] - 0.5).abs() + (mean_rgb[2] - 0.5).abs())
                / 1.5; // Normalize to [0,1]

        // 2. Hue consistency
        let hue_std = self.compute_hue_std_dev(field);
        let hue_consistency = 1.0 - (hue_std / 180.0).min(1.0);

        // Combine metrics
        0.6 * complementary_balance + 0.4 * hue_consistency
    }

    /// Compute standard deviation of hue angles
    fn compute_hue_std_dev(&self, field: &ChromaticTensor) -> f32 {
        let mut hues = Vec::new();

        for l in 0..field.layers() {
            for r in 0..field.rows() {
                for c in 0..field.cols() {
                    let rgb = field.get_rgb(r, c, l);
                    let (_, _, hue) = rgb_to_hsv(&rgb);
                    hues.push(hue);
                }
            }
        }

        let mean_hue: f32 = hues.iter().sum::<f32>() / hues.len() as f32;
        let variance: f32 = hues
            .iter()
            .map(|&h| {
                let diff = angle_difference(h, mean_hue);
                diff * diff
            })
            .sum::<f32>()
            / hues.len() as f32;

        variance.sqrt()
    }

    /// Compute constraint violation score (0-1, lower is better)
    ///
    /// Measures:
    /// - Out-of-gamut pixels (RGB outside [0,1])
    /// - Extreme saturation (> 0.95)
    /// - Local discontinuities (sharp color jumps)
    fn compute_constraint_violation(&self, field: &ChromaticTensor) -> f32 {
        let total_cells = (field.rows() * field.cols() * field.layers()) as f32;
        let mut violation_count = 0.0;

        for l in 0..field.layers() {
            for r in 0..field.rows() {
                for c in 0..field.cols() {
                    let rgb = field.get_rgb(r, c, l);

                    // 1. Out-of-gamut check
                    if rgb.iter().any(|&v| v < 0.0 || v > 1.0) {
                        violation_count += 1.0;
                    }

                    // 2. Extreme saturation check
                    let saturation = rgb_saturation(&rgb);
                    if saturation > 0.95 {
                        violation_count += saturation - 0.95;
                    }

                    // 3. Local discontinuity check
                    if r + 1 < field.rows() {
                        let right = field.get_rgb(r + 1, c, l);
                        if rgb_distance(&rgb, &right) > self.discontinuity_threshold {
                            violation_count += 0.5;
                        }
                    }
                    if c + 1 < field.cols() {
                        let down = field.get_rgb(r, c + 1, l);
                        if rgb_distance(&rgb, &down) > self.discontinuity_threshold {
                            violation_count += 0.5;
                        }
                    }
                }
            }
        }

        (violation_count / total_cells).min(1.0)
    }

    /// Compute analytical gradients for all metrics
    ///
    /// Returns gradient with respect to RGB values, shape: [rows, cols, layers, 3]
    fn compute_gradients(&self, field: &ChromaticTensor) -> Vec<f32> {
        let total_size = field.rows() * field.cols() * field.layers() * 3;
        let mut grad = vec![0.0; total_size];

        // Gradient of total variation
        self.add_tv_gradient(field, &mut grad);

        // Gradient of saturation penalty
        self.add_saturation_gradient(field, &mut grad);

        grad
    }

    /// Add total variation gradient contribution
    fn add_tv_gradient(&self, field: &ChromaticTensor, grad: &mut [f32]) {
        for l in 0..field.layers() {
            for r in 0..field.rows() {
                for c in 0..field.cols() {
                    let idx = (l * field.rows() * field.cols() + r * field.cols() + c) * 3;
                    let curr = field.get_rgb(r, c, l);

                    // Current cell contribution
                    let mut grad_rgb = [0.0f32; 3];

                    // Right neighbor
                    if r + 1 < field.rows() {
                        let right = field.get_rgb(r + 1, c, l);
                        let dist = rgb_distance(&curr, &right);
                        if dist > 1e-6 {
                            for ch in 0..3 {
                                grad_rgb[ch] += self.lambda_tv * (curr[ch] - right[ch]) / dist;
                            }
                        }
                    }

                    // Down neighbor
                    if c + 1 < field.cols() {
                        let down = field.get_rgb(r, c + 1, l);
                        let dist = rgb_distance(&curr, &down);
                        if dist > 1e-6 {
                            for ch in 0..3 {
                                grad_rgb[ch] += self.lambda_tv * (curr[ch] - down[ch]) / dist;
                            }
                        }
                    }

                    // Left neighbor
                    if r > 0 {
                        let left = field.get_rgb(r - 1, c, l);
                        let dist = rgb_distance(&left, &curr);
                        if dist > 1e-6 {
                            for ch in 0..3 {
                                grad_rgb[ch] += self.lambda_tv * (curr[ch] - left[ch]) / dist;
                            }
                        }
                    }

                    // Up neighbor
                    if c > 0 {
                        let up = field.get_rgb(r, c - 1, l);
                        let dist = rgb_distance(&up, &curr);
                        if dist > 1e-6 {
                            for ch in 0..3 {
                                grad_rgb[ch] += self.lambda_tv * (curr[ch] - up[ch]) / dist;
                            }
                        }
                    }

                    // Add to gradient array
                    for ch in 0..3 {
                        grad[idx + ch] += grad_rgb[ch];
                    }
                }
            }
        }
    }

    /// Add saturation penalty gradient contribution
    fn add_saturation_gradient(&self, field: &ChromaticTensor, grad: &mut [f32]) {
        let total_cells = (field.rows() * field.cols() * field.layers()) as f32;
        let scale = 2.0 * self.lambda_sat / total_cells;

        for l in 0..field.layers() {
            for r in 0..field.rows() {
                for c in 0..field.cols() {
                    let idx = (l * field.rows() * field.cols() + r * field.cols() + c) * 3;
                    let rgb = field.get_rgb(r, c, l);

                    let saturation = rgb_saturation(&rgb);
                    let sat_diff = saturation - self.target_saturation;

                    // Gradient of saturation with respect to RGB
                    let grad_sat = rgb_saturation_gradient(&rgb);

                    for ch in 0..3 {
                        grad[idx + ch] += scale * sat_diff * grad_sat[ch];
                    }
                }
            }
        }
    }
}

impl Solver for ChromaticNativeSolver {
    fn evaluate(&mut self, field: &ChromaticTensor, with_grad: bool) -> Result<SolverResult> {
        // Compute metrics
        let tv = self.compute_total_variation(field);
        let sat_penalty = self.compute_saturation_penalty(field);
        let energy = (self.lambda_tv * tv + self.lambda_sat * sat_penalty) as f64;

        let coherence = self.compute_color_harmony(field) as f64;
        let violation = self.compute_constraint_violation(field) as f64;

        // Compute gradients if requested
        let grad = if with_grad {
            Some(self.compute_gradients(field))
        } else {
            None
        };

        Ok(SolverResult {
            energy,
            coherence,
            violation,
            grad,
            mask: None,
            meta: json!({
                "total_variation": tv,
                "saturation_penalty": sat_penalty,
                "lambda_tv": self.lambda_tv,
                "lambda_sat": self.lambda_sat,
            }),
        })
    }

    fn name(&self) -> &str {
        "ChromaticNativeSolver"
    }
}

// ============================================================================
// Color Space Utility Functions
// ============================================================================

/// Euclidean distance between two RGB colors
fn rgb_distance(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    let dr = a[0] - b[0];
    let dg = a[1] - b[1];
    let db = a[2] - b[2];
    (dr * dr + dg * dg + db * db).sqrt()
}

/// Compute saturation of an RGB color
///
/// Saturation = (max - min) / max (if max > 0, else 0)
fn rgb_saturation(rgb: &[f32; 3]) -> f32 {
    let max_val = rgb[0].max(rgb[1]).max(rgb[2]);
    let min_val = rgb[0].min(rgb[1]).min(rgb[2]);

    if max_val < 1e-6 {
        0.0
    } else {
        (max_val - min_val) / max_val
    }
}

/// Compute gradient of saturation with respect to RGB
fn rgb_saturation_gradient(rgb: &[f32; 3]) -> [f32; 3] {
    let max_val = rgb[0].max(rgb[1]).max(rgb[2]);
    let min_val = rgb[0].min(rgb[1]).min(rgb[2]);

    if max_val < 1e-6 {
        return [0.0, 0.0, 0.0];
    }

    let mut grad = [0.0f32; 3];
    let range = max_val - min_val;

    for ch in 0..3 {
        if (rgb[ch] - max_val).abs() < 1e-6 {
            // This channel is max
            grad[ch] = -range / (max_val * max_val);
        } else if (rgb[ch] - min_val).abs() < 1e-6 {
            // This channel is min
            grad[ch] = -1.0 / max_val;
        }
        // Else: middle channel, gradient is 0
    }

    grad
}

/// Convert RGB to HSV
///
/// Returns (hue [0-360], saturation [0-1], value [0-1])
fn rgb_to_hsv(rgb: &[f32; 3]) -> (f32, f32, f32) {
    let max_val = rgb[0].max(rgb[1]).max(rgb[2]);
    let min_val = rgb[0].min(rgb[1]).min(rgb[2]);
    let delta = max_val - min_val;

    // Value
    let v = max_val;

    // Saturation
    let s = if max_val < 1e-6 { 0.0 } else { delta / max_val };

    // Hue
    let h = if delta < 1e-6 {
        0.0 // Undefined, set to 0
    } else if (rgb[0] - max_val).abs() < 1e-6 {
        // Red is max
        60.0 * (((rgb[1] - rgb[2]) / delta) % 6.0)
    } else if (rgb[1] - max_val).abs() < 1e-6 {
        // Green is max
        60.0 * (((rgb[2] - rgb[0]) / delta) + 2.0)
    } else {
        // Blue is max
        60.0 * (((rgb[0] - rgb[1]) / delta) + 4.0)
    };

    let h_normalized = if h < 0.0 { h + 360.0 } else { h };

    (h_normalized, s, v)
}

/// Compute shortest angular difference between two hue angles (in degrees)
fn angle_difference(a: f32, b: f32) -> f32 {
    let diff = (a - b).abs();
    if diff > 180.0 {
        360.0 - diff
    } else {
        diff
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgb_distance() {
        let red = [1.0, 0.0, 0.0];
        let green = [0.0, 1.0, 0.0];
        let dist = rgb_distance(&red, &green);
        assert!((dist - 1.414).abs() < 0.01); // sqrt(2)
    }

    #[test]
    fn test_rgb_saturation() {
        let pure_red = [1.0, 0.0, 0.0];
        assert_eq!(rgb_saturation(&pure_red), 1.0);

        let gray = [0.5, 0.5, 0.5];
        assert_eq!(rgb_saturation(&gray), 0.0);

        let pink = [1.0, 0.5, 0.5];
        assert_eq!(rgb_saturation(&pink), 0.5);
    }

    #[test]
    fn test_rgb_to_hsv() {
        let red = [1.0, 0.0, 0.0];
        let (h, s, v) = rgb_to_hsv(&red);
        assert_eq!(h, 0.0);
        assert_eq!(s, 1.0);
        assert_eq!(v, 1.0);

        let green = [0.0, 1.0, 0.0];
        let (h, s, v) = rgb_to_hsv(&green);
        assert_eq!(h, 120.0);
        assert_eq!(s, 1.0);
        assert_eq!(v, 1.0);
    }

    #[test]
    fn test_angle_difference() {
        assert_eq!(angle_difference(10.0, 20.0), 10.0);
        assert_eq!(angle_difference(350.0, 10.0), 20.0);
        assert_eq!(angle_difference(0.0, 180.0), 180.0);
    }
}
