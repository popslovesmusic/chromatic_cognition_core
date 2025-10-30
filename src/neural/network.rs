//! Chromatic neural network architecture.

use crate::checkpoint::{CheckpointError, Checkpointable};
use crate::neural::layer::{ChromaticLayer, ChromaticOp};
use crate::neural::loss::{accuracy, cross_entropy_loss};
use crate::neural::optimizer::{OptimizerStateSnapshot, SGDOptimizer};
use crate::tensor::ChromaticTensor;
use serde::{Deserialize, Serialize};

const NETWORK_CHECKPOINT_VERSION: u32 = 2;

#[derive(Clone, Serialize, Deserialize)]
struct LayerConfigSnapshot {
    operation: ChromaticOp,
    shape: (usize, usize, usize, usize),
}

#[derive(Clone, Serialize, Deserialize)]
struct NetworkConfigSnapshot {
    layer_configs: Vec<LayerConfigSnapshot>,
    num_classes: usize,
}

impl NetworkConfigSnapshot {
    fn from_layers(layers: &[ChromaticLayer], num_classes: usize) -> Self {
        let layer_configs = layers
            .iter()
            .map(|layer| LayerConfigSnapshot {
                operation: layer.operation,
                shape: layer.shape(),
            })
            .collect();

        Self {
            layer_configs,
            num_classes,
        }
    }

    fn validate_layers(&self, layers: &[ChromaticLayer]) -> Result<(), CheckpointError> {
        if self.layer_configs.len() != layers.len() {
            return Err(CheckpointError::InvalidFormat(format!(
                "Layer count mismatch: expected {}, found {}",
                self.layer_configs.len(),
                layers.len()
            )));
        }

        for (expected, layer) in self.layer_configs.iter().zip(layers.iter()) {
            if expected.operation != layer.operation {
                return Err(CheckpointError::InvalidFormat(format!(
                    "Layer operation mismatch: expected {:?}, found {:?}",
                    expected.operation, layer.operation
                )));
            }

            if expected.shape != layer.shape() {
                return Err(CheckpointError::InvalidFormat(format!(
                    "Layer shape mismatch: expected {:?}, found {:?}",
                    expected.shape,
                    layer.shape()
                )));
            }
        }

        if self.num_classes == 0 {
            return Err(CheckpointError::InvalidFormat(
                "Network configuration must specify at least one class".to_string(),
            ));
        }

        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
struct ChromaticNetworkCheckpoint {
    version: u32,
    layers: Vec<ChromaticLayer>,
    num_classes: usize,
    #[serde(default)]
    config: Option<NetworkConfigSnapshot>,
    #[serde(default)]
    optimizer_state: Option<OptimizerStateSnapshot>,
}

/// A chromatic neural network for classification.
///
/// Stacks multiple chromatic layers to learn complex color patterns.
#[derive(Serialize, Deserialize)]
pub struct ChromaticNetwork {
    layers: Vec<ChromaticLayer>,
    num_classes: usize,
    #[serde(default)]
    optimizer_state: Option<OptimizerStateSnapshot>,
}

impl ChromaticNetwork {
    /// Creates a new chromatic network.
    ///
    /// # Arguments
    ///
    /// * `layers` - Vector of chromatic layers
    /// * `num_classes` - Number of output classes
    pub fn new(layers: Vec<ChromaticLayer>, num_classes: usize) -> Self {
        Self {
            layers,
            num_classes,
            optimizer_state: None,
        }
    }

    /// Creates a simple 2-layer network for experiments.
    ///
    /// # Arguments
    ///
    /// * `input_size` - Size of input tensors (rows, cols, layers)
    /// * `num_classes` - Number of output classes
    /// * `seed` - Random seed
    pub fn simple(input_size: (usize, usize, usize), num_classes: usize, seed: u64) -> Self {
        let (rows, cols, layers) = input_size;

        let layer1 =
            ChromaticLayer::new(rows, cols, layers, ChromaticOp::Saturate, seed).with_param(1.2);
        let layer2 = ChromaticLayer::new(rows, cols, layers, ChromaticOp::Mix, seed + 1);

        Self::new(vec![layer1, layer2], num_classes)
    }

    /// Forward pass through the network.
    ///
    /// # Arguments
    ///
    /// * `input` - Input chromatic tensor
    ///
    /// # Returns
    ///
    /// Output chromatic tensor
    pub fn forward(&mut self, input: &ChromaticTensor) -> ChromaticTensor {
        let mut activation = input.clone();

        for layer in &mut self.layers {
            activation = layer.forward(&activation);
        }

        activation
    }

    /// Computes loss and gradients for a single sample.
    ///
    /// # Arguments
    ///
    /// * `input` - Input chromatic tensor
    /// * `label` - Target class label
    ///
    /// # Returns
    ///
    /// Tuple of (loss, accuracy for this sample)
    pub fn compute_loss(&mut self, input: &ChromaticTensor, label: usize) -> (f32, f32) {
        // Forward pass
        let output = self.forward(input);

        // Compute loss
        let (loss, _grad) = cross_entropy_loss(&output, label, self.num_classes);

        // Compute accuracy
        let acc = accuracy(&output, label, self.num_classes);

        (loss, acc)
    }

    /// Trains the network for one step.
    ///
    /// # Arguments
    ///
    /// * `input` - Input chromatic tensor
    /// * `label` - Target class label
    /// * `optimizer` - Optimizer to use for updates
    ///
    /// # Returns
    ///
    /// Tuple of (loss, accuracy)
    pub fn train_step(
        &mut self,
        input: &ChromaticTensor,
        label: usize,
        optimizer: &mut SGDOptimizer,
    ) -> (f32, f32) {
        // Forward pass
        let output = self.forward(input);

        // Compute loss and gradient
        let (loss, grad_output) = cross_entropy_loss(&output, label, self.num_classes);

        // Backward pass - collect gradients first
        let mut layer_gradients = Vec::new();
        let mut grad = grad_output;

        for layer in self.layers.iter().rev() {
            let (grad_input, grad_weights, grad_bias) = layer.backward(&grad);
            layer_gradients.push((grad_weights, grad_bias));
            grad = grad_input;
        }

        // Now update parameters
        layer_gradients.reverse();
        for (layer_idx, (grad_weights, grad_bias)) in layer_gradients.into_iter().enumerate() {
            optimizer.step(
                &format!("layer{}_weights", layer_idx),
                &mut self.layers[layer_idx].weights,
                &grad_weights,
            );
            optimizer.step(
                &format!("layer{}_bias", layer_idx),
                &mut self.layers[layer_idx].bias,
                &grad_bias,
            );
        }

        self.capture_optimizer_state_from_sgd(optimizer);

        // Compute accuracy
        let acc = accuracy(&output, label, self.num_classes);

        (loss, acc)
    }

    /// Returns the optimizer state captured during training, if any.
    pub fn optimizer_state(&self) -> Option<&OptimizerStateSnapshot> {
        self.optimizer_state.as_ref()
    }

    /// Takes ownership of the stored optimizer state.
    pub fn take_optimizer_state(&mut self) -> Option<OptimizerStateSnapshot> {
        self.optimizer_state.take()
    }

    /// Sets the optimizer state explicitly.
    pub fn set_optimizer_state(&mut self, state: Option<OptimizerStateSnapshot>) {
        self.optimizer_state = state;
    }

    /// Captures the state of an SGD optimizer.
    pub fn capture_optimizer_state_from_sgd(&mut self, optimizer: &SGDOptimizer) {
        self.optimizer_state = Some(OptimizerStateSnapshot::Sgd(optimizer.to_state()));
    }

    /// Captures the state of an Adam optimizer.
    pub fn capture_optimizer_state_from_adam(
        &mut self,
        optimizer: &crate::neural::optimizer::AdamOptimizer,
    ) {
        self.optimizer_state = Some(OptimizerStateSnapshot::Adam(optimizer.to_state()));
    }

    /// Applies the stored optimizer state to an SGD optimizer, if compatible.
    pub fn apply_optimizer_state_to_sgd(&self, optimizer: &mut SGDOptimizer) {
        if let Some(OptimizerStateSnapshot::Sgd(state)) = &self.optimizer_state {
            optimizer.apply_state(state.clone());
        }
    }

    /// Applies the stored optimizer state to an Adam optimizer, if compatible.
    pub fn apply_optimizer_state_to_adam(
        &self,
        optimizer: &mut crate::neural::optimizer::AdamOptimizer,
    ) {
        if let Some(OptimizerStateSnapshot::Adam(state)) = &self.optimizer_state {
            optimizer.apply_state(state.clone());
        }
    }

    /// Evaluates the network on a batch of samples.
    ///
    /// # Arguments
    ///
    /// * `inputs` - Batch of input tensors
    /// * `labels` - Batch of labels
    ///
    /// # Returns
    ///
    /// Tuple of (average loss, average accuracy)
    pub fn evaluate(&mut self, inputs: &[ChromaticTensor], labels: &[usize]) -> (f32, f32) {
        let mut total_loss = 0.0;
        let mut total_acc = 0.0;

        for (input, &label) in inputs.iter().zip(labels.iter()) {
            let (loss, acc) = self.compute_loss(input, label);
            total_loss += loss;
            total_acc += acc;
        }

        let n = inputs.len() as f32;
        (total_loss / n, total_acc / n)
    }
}

impl Checkpointable for ChromaticNetwork {
    fn save_checkpoint<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), CheckpointError> {
        let snapshot = ChromaticNetworkCheckpoint {
            version: NETWORK_CHECKPOINT_VERSION,
            layers: self.layers.clone(),
            num_classes: self.num_classes,
            config: Some(NetworkConfigSnapshot::from_layers(&self.layers, self.num_classes)),
            optimizer_state: self.optimizer_state.clone(),
        };

        Self::write_snapshot(&snapshot, path)
    }

    fn load_checkpoint<P: AsRef<std::path::Path>>(path: P) -> Result<Self, CheckpointError> {
        let snapshot: ChromaticNetworkCheckpoint = Self::read_snapshot(path)?;
        let mut snapshot = snapshot;
        if snapshot.version != NETWORK_CHECKPOINT_VERSION {
            return Err(CheckpointError::VersionMismatch {
                expected: NETWORK_CHECKPOINT_VERSION,
                found: snapshot.version,
            });
        }

        let config = snapshot
            .config
            .take()
            .unwrap_or_else(|| NetworkConfigSnapshot::from_layers(&snapshot.layers, snapshot.num_classes));
        config.validate_layers(&snapshot.layers)?;

        Ok(Self {
            layers: snapshot.layers,
            num_classes: snapshot.num_classes,
            optimizer_state: snapshot.optimizer_state,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_creation() {
        let net = ChromaticNetwork::simple((16, 16, 4), 3, 42);
        assert_eq!(net.layers.len(), 2);
        assert_eq!(net.num_classes, 3);
        assert!(net.optimizer_state().is_none());
    }

    #[test]
    fn test_forward_pass() {
        let mut net = ChromaticNetwork::simple((8, 8, 2), 3, 42);
        let input = ChromaticTensor::from_seed(100, 8, 8, 2);

        let output = net.forward(&input);
        assert_eq!(output.shape(), input.shape());
    }

    #[test]
    fn test_train_step() {
        let mut net = ChromaticNetwork::simple((8, 8, 2), 3, 42);
        let mut optimizer = SGDOptimizer::new(0.01, 0.0, 0.0);

        let input = ChromaticTensor::from_seed(100, 8, 8, 2);
        let label = 0;

        let (loss1, _acc1) = net.train_step(&input, label, &mut optimizer);
        let (loss2, _acc2) = net.train_step(&input, label, &mut optimizer);

        // Loss should decrease (or at least change) after training
        assert_ne!(loss1, loss2);
        assert!(net.optimizer_state().is_some());
    }

    #[test]
    fn test_checkpoint_roundtrip_preserves_optimizer_state() {
        use std::fs;

        let mut net = ChromaticNetwork::simple((4, 4, 2), 2, 7);
        let mut optimizer = SGDOptimizer::new(0.1, 0.9, 0.0001);
        let input = ChromaticTensor::from_seed(10, 4, 4, 2);
        let label = 1;
        let _ = net.train_step(&input, label, &mut optimizer);

        let mut path = std::env::temp_dir();
        path.push(format!(
            "chromatic_network_checkpoint_{}.bin",
            uuid::Uuid::new_v4()
        ));

        net.save_checkpoint(&path).expect("save checkpoint");
        let restored = ChromaticNetwork::load_checkpoint(&path).expect("load checkpoint");
        fs::remove_file(&path).ok();

        assert!(matches!(
            restored.optimizer_state(),
            Some(OptimizerStateSnapshot::Sgd(_))
        ));
        assert_eq!(restored.layers.len(), net.layers.len());
    }

    #[test]
    fn test_config_validation_detects_mismatch() {
        let layers = vec![
            ChromaticLayer::new(2, 2, 1, ChromaticOp::Mix, 1),
            ChromaticLayer::new(2, 2, 1, ChromaticOp::Filter, 2),
        ];

        let mut config = NetworkConfigSnapshot::from_layers(&layers, 2);
        config.layer_configs[0].operation = ChromaticOp::Saturate;

        let err = config.validate_layers(&layers).expect_err("mismatch");
        match err {
            CheckpointError::InvalidFormat(msg) => {
                assert!(msg.contains("Layer operation mismatch"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
