/// Demonstration of the chromatic field solver
///
/// This example shows how to:
/// 1. Create a chromatic tensor
/// 2. Evaluate it with the native solver
/// 3. Interpret energy, coherence, and violation metrics
/// 4. Compute gradients for optimization
use chromatic_cognition_core::{ChromaticNativeSolver, ChromaticTensor, Solver};

fn main() {
    println!("=== Chromatic Field Solver Demo ===\n");

    // Create a solver with default parameters
    let mut solver = ChromaticNativeSolver::new();
    println!("Solver: {}", solver.name());
    println!("Parameters:");
    println!("  lambda_tv (total variation): {}", solver.lambda_tv);
    println!("  lambda_sat (saturation penalty): {}", solver.lambda_sat);
    println!("  target_saturation: {}", solver.target_saturation);
    println!(
        "  discontinuity_threshold: {}\n",
        solver.discontinuity_threshold
    );

    // Example 1: Smooth random field
    println!("--- Example 1: Smooth Random Field ---");
    let smooth_field = ChromaticTensor::from_seed(42, 8, 8, 2);
    let result = solver.evaluate(&smooth_field, false).expect("eval failed");
    println!(
        "Energy: {:.4} (total variation + saturation penalty)",
        result.energy
    );
    println!(
        "Coherence: {:.4} (0-1, higher = more harmonious)",
        result.coherence
    );
    println!(
        "Violation: {:.4} (0-1, lower = fewer constraint violations)\n",
        result.violation
    );

    // Example 2: High contrast field (checkerboard pattern)
    println!("--- Example 2: High Contrast Field ---");
    let mut colors = ndarray::Array4::zeros((4, 4, 1, 3));
    for r in 0..4 {
        for c in 0..4 {
            let is_black_square = (r + c) % 2 == 0;
            if is_black_square {
                colors[[r, c, 0, 0]] = 0.0; // Black
                colors[[r, c, 0, 1]] = 0.0;
                colors[[r, c, 0, 2]] = 0.0;
            } else {
                colors[[r, c, 0, 0]] = 1.0; // White
                colors[[r, c, 0, 1]] = 1.0;
                colors[[r, c, 0, 2]] = 1.0;
            }
        }
    }
    let certainty = ndarray::Array3::ones((4, 4, 1));
    let checkerboard = ChromaticTensor::from_arrays(colors, certainty);

    let result = solver.evaluate(&checkerboard, false).expect("eval failed");
    println!(
        "Energy: {:.4} (high due to sharp transitions)",
        result.energy
    );
    println!(
        "Coherence: {:.4} (low due to extreme black/white)",
        result.coherence
    );
    println!(
        "Violation: {:.4} (low, colors in gamut)\n",
        result.violation
    );

    // Example 3: Pure colors (high saturation)
    println!("--- Example 3: Pure RGB Colors ---");
    let mut pure_colors = ndarray::Array4::zeros((3, 1, 1, 3));
    pure_colors[[0, 0, 0, 0]] = 1.0; // Pure red
    pure_colors[[1, 0, 0, 1]] = 1.0; // Pure green
    pure_colors[[2, 0, 0, 2]] = 1.0; // Pure blue
    let certainty = ndarray::Array3::ones((3, 1, 1));
    let pure_field = ChromaticTensor::from_arrays(pure_colors, certainty);

    let result = solver.evaluate(&pure_field, false).expect("eval failed");
    println!(
        "Energy: {:.4} (high saturation deviation from target)",
        result.energy
    );
    println!(
        "Coherence: {:.4} (complementary colors present)",
        result.coherence
    );
    println!(
        "Violation: {:.4} (high saturation flagged)\n",
        result.violation
    );

    // Example 4: Out-of-gamut colors (constraint violation)
    println!("--- Example 4: Out-of-Gamut Colors ---");
    let mut bad_colors = ndarray::Array4::zeros((2, 2, 1, 3));
    bad_colors[[0, 0, 0, 0]] = 1.5; // Out of gamut (> 1.0)
    bad_colors[[0, 1, 0, 1]] = -0.2; // Out of gamut (< 0.0)
    bad_colors[[1, 0, 0, 2]] = 0.5; // Valid
    bad_colors[[1, 1, 0, 0]] = 0.8; // Valid
    let certainty = ndarray::Array3::ones((2, 2, 1));
    let bad_field = ChromaticTensor::from_arrays(bad_colors, certainty);

    let result = solver.evaluate(&bad_field, false).expect("eval failed");
    println!("Energy: {:.4}", result.energy);
    println!("Coherence: {:.4}", result.coherence);
    println!(
        "Violation: {:.4} (HIGH - out-of-gamut colors detected)\n",
        result.violation
    );

    // Example 5: Gradient computation
    println!("--- Example 5: Gradient Computation ---");
    let test_field = ChromaticTensor::from_seed(123, 4, 4, 1);
    let result_with_grad = solver.evaluate(&test_field, true).expect("eval failed");

    if let Some(grad) = &result_with_grad.grad {
        println!("Gradient computed: {} values (4×4×1×3)", grad.len());
        println!("First 9 gradient components:");
        for i in 0..9.min(grad.len()) {
            println!("  grad[{}] = {:.4}", i, grad[i]);
        }
        println!("\nGradient can be used for optimization (gradient descent)");
    }

    println!("\n=== Interpretation Guide ===");
    println!("Energy: Total field \"cost\" (lower is smoother/better)");
    println!("  - Combines spatial smoothness (total variation)");
    println!("  - And deviation from target saturation");
    println!();
    println!("Coherence: Color harmony score (0-1, higher is better)");
    println!("  - Measures complementary balance (red-cyan, green-magenta, etc.)");
    println!("  - Measures hue consistency (similar hues = more coherent)");
    println!();
    println!("Violation: Constraint violation score (0-1, lower is better)");
    println!("  - Out-of-gamut colors (RGB outside [0,1])");
    println!("  - Extreme saturation (>0.95)");
    println!("  - Sharp discontinuities (large color jumps)");
    println!();
    println!("Gradients: Direction to adjust each RGB value to reduce energy");
    println!();
}
