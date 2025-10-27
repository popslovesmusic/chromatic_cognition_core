//! Training example for chromatic neural network on color classification.
//!
//! This example trains a chromatic neural network to classify patterns
//! into primary colors (red, green, blue).

use chromatic_cognition_core::data::{generate_primary_color_dataset, shuffle_dataset, split_dataset};
use chromatic_cognition_core::neural::{ChromaticNetwork, SGDOptimizer};
use chromatic_cognition_core::tensor::gradient::GradientLayer;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎨 Chromatic Neural Network - Color Classification");
    println!("==================================================\n");

    // Hyperparameters
    let samples_per_class = 50;
    let tensor_size = (16, 16, 4); // Small for fast training
    let num_classes = 3; // Red, Green, Blue
    let epochs = 20;
    let learning_rate = 0.05;
    let momentum = 0.9;
    let weight_decay = 0.0001;

    println!("Configuration:");
    println!("  Samples per class: {}", samples_per_class);
    println!("  Tensor size: {:?}", tensor_size);
    println!("  Epochs: {}", epochs);
    println!("  Learning rate: {}", learning_rate);
    println!("  Momentum: {}", momentum);
    println!("  Weight decay: {}", weight_decay);
    println!();

    // Generate dataset
    println!("📊 Generating dataset...");
    let mut dataset = generate_primary_color_dataset(
        samples_per_class,
        tensor_size.0,
        tensor_size.1,
        tensor_size.2,
        42,
    );

    // Shuffle and split
    shuffle_dataset(&mut dataset, 123);
    let (train_data, val_data) = split_dataset(dataset, 0.8);

    println!("  Training samples: {}", train_data.len());
    println!("  Validation samples: {}", val_data.len());
    println!();

    // Create network
    println!("🧠 Creating chromatic neural network...");
    let mut network = ChromaticNetwork::simple(tensor_size, num_classes, 777);
    let mut optimizer = SGDOptimizer::new(learning_rate, momentum, weight_decay);
    println!("  Network created with {} layers", 2);
    println!();

    // Training loop
    println!("🎓 Starting training...\n");

    for epoch in 0..epochs {
        let mut epoch_loss = 0.0;
        let mut epoch_acc = 0.0;

        // Train on each sample
        for pattern in &train_data {
            let (loss, acc) = network.train_step(&pattern.tensor, pattern.label, &mut optimizer);
            epoch_loss += loss;
            epoch_acc += acc;
        }

        // Average metrics
        let train_loss = epoch_loss / train_data.len() as f32;
        let train_acc = epoch_acc / train_data.len() as f32;

        // Validation
        let val_inputs: Vec<_> = val_data.iter().map(|p| p.tensor.clone()).collect();
        let val_labels: Vec<_> = val_data.iter().map(|p| p.label).collect();
        let (val_loss, val_acc) = network.evaluate(&val_inputs, &val_labels);

        println!(
            "Epoch {:2}/{} | Train Loss: {:.4} Acc: {:.2}% | Val Loss: {:.4} Acc: {:.2}%",
            epoch + 1,
            epochs,
            train_loss,
            train_acc * 100.0,
            val_loss,
            val_acc * 100.0
        );
    }

    println!("\n✅ Training complete!");
    println!();

    // Final evaluation
    println!("📈 Final Evaluation:");
    let val_inputs: Vec<_> = val_data.iter().map(|p| p.tensor.clone()).collect();
    let val_labels: Vec<_> = val_data.iter().map(|p| p.label).collect();
    let (final_loss, final_acc) = network.evaluate(&val_inputs, &val_labels);

    println!("  Validation Loss: {:.4}", final_loss);
    println!("  Validation Accuracy: {:.2}%", final_acc * 100.0);
    println!();

    // Per-class accuracy
    println!("📊 Per-Class Performance:");
    let class_names = ["Red", "Green", "Blue"];
    for class in 0..num_classes {
        let class_samples: Vec<_> = val_data
            .iter()
            .filter(|p| p.label == class)
            .collect();

        if !class_samples.is_empty() {
            let class_inputs: Vec<_> = class_samples.iter().map(|p| p.tensor.clone()).collect();
            let class_labels: Vec<_> = class_samples.iter().map(|p| p.label).collect();
            let (_loss, acc) = network.evaluate(&class_inputs, &class_labels);

            println!("  {}: {:.2}%", class_names[class], acc * 100.0);
        }
    }
    println!();

    // Visualize some predictions
    println!("🖼️  Visualizing sample predictions...");
    std::fs::create_dir_all("out/predictions")?;

    for (i, pattern) in val_data.iter().take(9).enumerate() {
        let output = network.forward(&pattern.tensor);
        let gradient = GradientLayer::from_tensor(&output);

        let filename = format!("out/predictions/sample_{}_label_{}.png", i, pattern.label);
        gradient.to_png(PathBuf::from(&filename))?;

        let stats = output.statistics();
        let predicted_class = if stats.mean_rgb[0] > stats.mean_rgb[1] && stats.mean_rgb[0] > stats.mean_rgb[2] {
            0
        } else if stats.mean_rgb[1] > stats.mean_rgb[2] {
            1
        } else {
            2
        };

        let correct = if predicted_class == pattern.label { "✓" } else { "✗" };
        println!("  {} Sample {}: True={}, Pred={}", correct, i, class_names[pattern.label], class_names[predicted_class]);
    }
    println!();

    println!("✨ Done! Check out/predictions/ for visualizations.");

    Ok(())
}
