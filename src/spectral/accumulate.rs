use std::cmp::Ordering;

const FRACTIONAL_BITS: u32 = 48;
const SCALE_FACTOR: f64 = (1u128 << FRACTIONAL_BITS) as f64;

#[derive(Clone, Copy, Debug)]
struct AccumNode {
    sum_fixed: i128,
    compensation: f64,
}

impl AccumNode {
    fn from_value(value: f32) -> Self {
        if !value.is_finite() {
            return Self {
                sum_fixed: 0,
                compensation: 0.0,
            };
        }

        let scaled = (value as f64) * SCALE_FACTOR;
        let rounded = round_ties_even_to_i128(scaled);
        let quantized = rounded as f64 / SCALE_FACTOR;
        let residual = (value as f64) - quantized;

        Self {
            sum_fixed: rounded,
            compensation: residual,
        }
    }

    fn combine(self, other: Self) -> Self {
        let a = self.sum_fixed as f64 / SCALE_FACTOR + self.compensation;
        let b = other.sum_fixed as f64 / SCALE_FACTOR + other.compensation;

        let t = a + b;
        let mut compensation = match a.abs().partial_cmp(&b.abs()) {
            Some(Ordering::Greater) | Some(Ordering::Equal) => a - t + b,
            Some(Ordering::Less) => b - t + a,
            None => 0.0,
        };

        let scaled = t * SCALE_FACTOR;
        let rounded = round_ties_even_to_i128(scaled);
        let quantized = rounded as f64 / SCALE_FACTOR;
        compensation += t - quantized;

        Self {
            sum_fixed: rounded,
            compensation,
        }
    }

    fn finalize(self) -> f32 {
        let total = self.sum_fixed as f64 / SCALE_FACTOR + self.compensation;
        cast_f64_to_f32_rte(total)
    }
}

fn reduce_nodes(mut nodes: Vec<AccumNode>) -> AccumNode {
    debug_assert!(!nodes.is_empty());
    while nodes.len() > 1 {
        let mut next = Vec::with_capacity((nodes.len() + 1) / 2);
        let mut idx = 0;
        while idx < nodes.len() {
            if idx + 1 < nodes.len() {
                let combined = nodes[idx].combine(nodes[idx + 1]);
                next.push(combined);
                idx += 2;
            } else {
                next.push(nodes[idx]);
                idx += 1;
            }
        }
        nodes = next;
    }
    nodes[0]
}

fn round_ties_even_to_i128(value: f64) -> i128 {
    if !value.is_finite() {
        return 0;
    }

    let floor = value.floor();
    let frac = value - floor;
    if frac.abs() < 0.5 {
        floor as i128
    } else if frac.abs() > 0.5 {
        if value.is_sign_negative() {
            floor as i128 - 1
        } else {
            floor as i128 + 1
        }
    } else {
        let floor_int = floor as i128;
        if floor_int & 1 == 0 {
            floor_int
        } else if value.is_sign_negative() {
            floor_int - 1
        } else {
            floor_int + 1
        }
    }
}

fn next_up(value: f32) -> f32 {
    if value.is_nan() || value == f32::INFINITY {
        return value;
    }
    if value == -0.0 {
        return f32::from_bits(1);
    }
    if value >= 0.0 {
        f32::from_bits(value.to_bits() + 1)
    } else {
        f32::from_bits(value.to_bits() - 1)
    }
}

fn next_down(value: f32) -> f32 {
    if value.is_nan() || value == f32::NEG_INFINITY {
        return value;
    }
    if value == 0.0 {
        return -f32::from_bits(1);
    }
    if value > 0.0 {
        f32::from_bits(value.to_bits() - 1)
    } else {
        f32::from_bits(value.to_bits() + 1)
    }
}

fn cast_f64_to_f32_rte(value: f64) -> f32 {
    if !value.is_finite() {
        return value as f32;
    }

    let candidate = value as f32;
    let candidate_f64 = candidate as f64;
    if candidate_f64 == value {
        return candidate;
    }

    let (lower, upper) = if candidate_f64 < value {
        (candidate, next_up(candidate))
    } else {
        (next_down(candidate), candidate)
    };

    let lower_f64 = lower as f64;
    let upper_f64 = upper as f64;
    let dist_lower = (value - lower_f64).abs();
    let dist_upper = (value - upper_f64).abs();

    if dist_upper < dist_lower {
        upper
    } else if dist_lower < dist_upper {
        lower
    } else {
        // Exact tie: choose the value with an even mantissa (LSB = 0)
        if lower.to_bits() & 1 == 0 {
            lower
        } else if upper.to_bits() & 1 == 0 {
            upper
        } else {
            lower
        }
    }
}

/// Deterministically sum values using Q16.48 fixed-point accumulation and Neumaier compensation.
pub fn deterministic_sum(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let nodes: Vec<AccumNode> = values.iter().map(|&v| AccumNode::from_value(v)).collect();
    reduce_nodes(nodes).finalize()
}

/// Deterministic mean using the fixed-point accumulator.
pub fn deterministic_mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    deterministic_sum(values) / values.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_sum_matches_reordered() {
        let values = vec![0.1f32, 0.2, 0.3, 0.4, 0.5];
        let mut reversed = values.clone();
        reversed.reverse();
        let sum_original = deterministic_sum(&values);
        let sum_reversed = deterministic_sum(&reversed);
        assert!((sum_original - sum_reversed).abs() < 1e-9);
    }

    #[test]
    fn deterministic_mean_handles_empty() {
        assert_eq!(deterministic_mean(&[]), 0.0);
    }

    #[test]
    fn accumulator_handles_negative_values() {
        let values = vec![1.5f32, -0.25, 3.75, -1.0];
        let sum = deterministic_sum(&values);
        assert!((sum - 4.0).abs() < 1e-6);
    }
}
