//! Neural network components for chromatic cognition.
//!
//! This module provides the building blocks for creating neural networks
//! that operate in chromatic (color) space, using ChromaticTensors as
//! the fundamental computational unit.

pub mod gradient;
pub mod layer;
pub mod loss;
pub mod network;
pub mod optimizer;

pub use gradient::{backward_complement, backward_filter, backward_mix, backward_saturate};
pub use layer::{ChromaticLayer, ChromaticOp};
pub use loss::{cross_entropy_loss, mse_loss_with_gradients};
pub use network::ChromaticNetwork;
pub use optimizer::SGDOptimizer;
