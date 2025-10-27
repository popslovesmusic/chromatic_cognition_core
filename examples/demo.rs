use std::path::PathBuf;

use chromatic_cognition_core::config::ConfigError;
use chromatic_cognition_core::logging;
use chromatic_cognition_core::{
    ChromaticTensor, EngineConfig, GradientLayer, complement, filter, mix, mse_loss, saturate,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = load_config()?;
    println!(
        "Loaded config: rows={} cols={} layers={} seed={}",
        config.rows, config.cols, config.layers, config.seed
    );

    let primary = ChromaticTensor::from_seed(config.seed, config.rows, config.cols, config.layers);
    let secondary = ChromaticTensor::from_seed(
        config.seed ^ 0xABCD_EF01,
        config.rows,
        config.cols,
        config.layers,
    );

    let mixed = mix(&primary, &secondary);
    let filtered = filter(&mixed, &secondary);
    let complemented = complement(&filtered);
    let saturated = saturate(&complemented, 1.25);

    let gradient = GradientLayer::from_tensor(&saturated);
    gradient.to_png(PathBuf::from("out/frame_0001.png"))?;

    let metrics = mse_loss(&saturated, &primary);
    logging::log_training_iteration(0, &metrics)?;

    println!("Demo complete. Loss {:.6}", metrics.loss);
    Ok(())
}

fn load_config() -> Result<EngineConfig, ConfigError> {
    EngineConfig::load_from_file("config/engine.toml").or_else(|err| {
        eprintln!("Falling back to default config: {err}");
        Ok(EngineConfig::default())
    })
}
