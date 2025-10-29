//! Deterministic color space utilities for perceptual distance metrics.
//!
//! Provides fixed-parameter conversions between sRGB and CIELAB using the
//! CIE 1931 2° standard observer and D65 illuminant. The functions in this
//! module avoid platform color management differences by relying solely on
//! analytic transforms. They support the ΔE94 color difference metric used to
//! enforce round-trip chromatic reversibility.

const D65_WHITE_POINT: [f32; 3] = [0.95047, 1.0, 1.08883];
const EPSILON: f32 = 0.008856_452; // 216/24389
const KAPPA: f32 = 903.296_3; // 24389/27
const K1: f32 = 0.045; // Graphic arts weighting for ΔE94
const K2: f32 = 0.015;

/// Strict tolerance for round-trip chromatic comparisons (ΔE94 <= 1e-3).
pub const DELTA_E94_TOLERANCE: f32 = 1.0e-3;

/// Convert an sRGB triplet in [0, 1] to linear-light RGB.
fn srgb_to_linear(channel: f32) -> f32 {
    if channel <= 0.04045 {
        channel / 12.92
    } else {
        ((channel + 0.055) / 1.055).powf(2.4)
    }
}

/// Convert an sRGB color to XYZ using the D65 illuminant and CIE 1931 2° observer.
fn srgb_to_xyz(rgb: [f32; 3]) -> [f32; 3] {
    let r = srgb_to_linear(rgb[0]);
    let g = srgb_to_linear(rgb[1]);
    let b = srgb_to_linear(rgb[2]);

    let x = 0.412_456_4 * r + 0.357_576_1 * g + 0.180_437_5 * b;
    let y = 0.212_672_9 * r + 0.715_152_2 * g + 0.072_175_0 * b;
    let z = 0.019_333_9 * r + 0.119_192_0 * g + 0.950_304_1 * b;

    [x, y, z]
}

fn lab_f(t: f32) -> f32 {
    if t > EPSILON {
        t.powf(1.0 / 3.0)
    } else {
        (KAPPA * t + 16.0) / 116.0
    }
}

/// Convert an sRGB color in [0, 1] to CIELAB coordinates (L*, a*, b*).
pub fn srgb_to_lab(rgb: [f32; 3]) -> [f32; 3] {
    let xyz = srgb_to_xyz(rgb);
    let xr = xyz[0] / D65_WHITE_POINT[0];
    let yr = xyz[1] / D65_WHITE_POINT[1];
    let zr = xyz[2] / D65_WHITE_POINT[2];

    let fx = lab_f(xr);
    let fy = lab_f(yr);
    let fz = lab_f(zr);

    let l = 116.0 * fy - 16.0;
    let a = 500.0 * (fx - fy);
    let b = 200.0 * (fy - fz);

    [l, a, b]
}

/// Compute the CIE ΔE94 color difference between two sRGB colors in [0, 1].
///
/// Weighting factors kL, kC, and kH are fixed to 1.0. Application-specific
/// modifiers K1 and K2 follow the graphic arts standard (0.045, 0.015).
pub fn delta_e94(rgb_a: [f32; 3], rgb_b: [f32; 3]) -> f32 {
    let lab_a = srgb_to_lab(rgb_a);
    let lab_b = srgb_to_lab(rgb_b);

    let delta_l = lab_a[0] - lab_b[0];
    let c1 = (lab_a[1].powi(2) + lab_a[2].powi(2)).sqrt();
    let c2 = (lab_b[1].powi(2) + lab_b[2].powi(2)).sqrt();
    let delta_c = c1 - c2;

    let delta_a = lab_a[1] - lab_b[1];
    let delta_b = lab_a[2] - lab_b[2];
    let delta_h_sq = (delta_a * delta_a) + (delta_b * delta_b) - (delta_c * delta_c);
    let delta_h = delta_h_sq.max(0.0).sqrt();

    let s_l = 1.0;
    let s_c = 1.0 + K1 * c1;
    let s_h = 1.0 + K2 * c1;

    let term_l = delta_l / s_l;
    let term_c = delta_c / s_c;
    let term_h = delta_h / s_h;

    (term_l * term_l + term_c * term_c + term_h * term_h).sqrt()
}

#[cfg(test)]
mod tests {
    use super::{delta_e94, srgb_to_lab, DELTA_E94_TOLERANCE};

    fn approx_equal(a: f32, b: f32, eps: f32) {
        assert!((a - b).abs() <= eps, "{} !≈ {}", a, b);
    }

    #[test]
    fn srgb_to_lab_round_trip_reference_white() {
        let lab = srgb_to_lab([1.0, 1.0, 1.0]);
        approx_equal(lab[0], 100.0, 1e-3);
        approx_equal(lab[1], 0.0, 1e-3);
        approx_equal(lab[2], 0.0, 1e-3);
    }

    #[test]
    fn delta_e94_zero_for_identical_colors() {
        let diff = delta_e94([0.2, 0.4, 0.6], [0.2, 0.4, 0.6]);
        approx_equal(diff, 0.0, DELTA_E94_TOLERANCE);
    }

    #[test]
    fn delta_e94_matches_reference_pair() {
        // Pure red vs. pure green difference using the fixed ΔE94 parameters.
        let diff = delta_e94([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        approx_equal(diff, 73.430, 1e-3);
    }
}
