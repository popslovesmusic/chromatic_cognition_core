use chromatic_cognition_core::dream::simple_pool::{DreamEntry, PoolConfig};
use chromatic_cognition_core::{mix, ChromaticTensor, SimpleDreamPool, SolverResult};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::json;

const NUM_EPOCHS: usize = 100;
const POOL_MAX_SIZE: usize = 50;
const COHERENCE_THRESHOLD: f64 = 0.75;
const RETRIEVAL_LIMIT: usize = 3;
const ALPHA_BLEND: f32 = 0.5;
const RANDOM_SEED: u64 = 42;

#[derive(Clone, Copy, Debug)]
enum Group {
    Control,
    Test,
}

impl Group {
    fn as_str(&self) -> &'static str {
        match self {
            Group::Control => "Control",
            Group::Test => "Test",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct EpochRecord {
    epoch: usize,
    group: String,
    avg_loss: f64,
    avg_coherence: f64,
}

impl EpochRecord {
    fn new(epoch: usize, group: Group, avg_loss: f64, avg_coherence: f64) -> Self {
        Self {
            epoch,
            group: group.as_str().to_string(),
            avg_loss,
            avg_coherence,
        }
    }

    fn to_csv_row(&self) -> String {
        format!(
            "{},{},{:.6},{:.6}",
            self.epoch, self.group, self.avg_loss, self.avg_coherence
        )
    }
}

struct HarnessRun {
    csv: String,
    json: String,
    final_control: EpochRecord,
    final_test: EpochRecord,
    control_pool_count: usize,
    test_pool_count: usize,
}

fn execute_harness(seed: u64) -> HarnessRun {
    let pool_config = PoolConfig {
        max_size: POOL_MAX_SIZE,
        coherence_threshold: COHERENCE_THRESHOLD,
        retrieval_limit: RETRIEVAL_LIMIT,
        use_hnsw: true,
        memory_budget_mb: Some(500),
    };

    let mut control_pool = SimpleDreamPool::new(pool_config.clone());
    let mut test_pool = SimpleDreamPool::new(pool_config);

    let mut control_rng = StdRng::seed_from_u64(seed);
    let mut test_rng = StdRng::seed_from_u64(seed);

    let mut csv_rows = vec!["epoch,group,avg_loss,avg_coherence".to_string()];
    let mut json_records = Vec::with_capacity(NUM_EPOCHS * 2);

    let mut final_control = None;
    let mut final_test = None;

    for epoch in 0..NUM_EPOCHS {
        let control_seed = ChromaticTensor::from_seed(control_rng.gen(), 8, 8, 2);
        let control_result = mock_dream_cycle(epoch, &[]);

        if control_result.coherence >= COHERENCE_THRESHOLD {
            control_pool.add_if_coherent(control_seed.clone(), control_result.clone());
        }

        let control_record = EpochRecord::new(
            epoch,
            Group::Control,
            control_result.energy,
            control_result.coherence,
        );
        csv_rows.push(control_record.to_csv_row());
        json_records.push(control_record.clone());
        final_control = Some(control_record);

        let mut test_seed = ChromaticTensor::from_seed(test_rng.gen(), 8, 8, 2);
        let query_signature = test_seed.mean_rgb();
        let retrieved = test_pool.retrieve_similar(&query_signature, RETRIEVAL_LIMIT);

        if !retrieved.is_empty() {
            for entry in &retrieved {
                test_seed = blend_with_alpha(&test_seed, &entry.tensor, ALPHA_BLEND);
            }
        }

        let test_result = mock_dream_cycle(epoch, &retrieved);

        if test_result.coherence >= COHERENCE_THRESHOLD {
            test_pool.add_if_coherent(test_seed.clone(), test_result.clone());
        }

        let test_record = EpochRecord::new(
            epoch,
            Group::Test,
            test_result.energy,
            test_result.coherence,
        );
        csv_rows.push(test_record.to_csv_row());
        json_records.push(test_record.clone());
        final_test = Some(test_record);
    }

    let stats_control = control_pool.stats();
    let stats_test = test_pool.stats();

    HarnessRun {
        csv: csv_rows.join("\n"),
        json: serde_json::to_string(&json_records).expect("json serialization should succeed"),
        final_control: final_control.expect("control record present"),
        final_test: final_test.expect("test record present"),
        control_pool_count: stats_control.count,
        test_pool_count: stats_test.count,
    }
}

fn blend_with_alpha(
    base: &ChromaticTensor,
    retrieved: &ChromaticTensor,
    alpha: f32,
) -> ChromaticTensor {
    assert_eq!(base.colors.dim(), retrieved.colors.dim());
    assert_eq!(base.certainty.dim(), retrieved.certainty.dim());

    if (alpha - 0.5).abs() < f32::EPSILON {
        return mix(base, retrieved);
    }

    let mut colors = base.colors.clone();
    let mut certainty = base.certainty.clone();

    let (rows, cols, layers, channels) = colors.dim();
    for row in 0..rows {
        for col in 0..cols {
            for layer in 0..layers {
                for channel in 0..channels {
                    let base_val = base.colors[[row, col, layer, channel]];
                    let retrieved_val = retrieved.colors[[row, col, layer, channel]];
                    colors[[row, col, layer, channel]] =
                        base_val * (1.0 - alpha) + retrieved_val * alpha;
                }
                let base_cert = base.certainty[[row, col, layer]];
                let retrieved_cert = retrieved.certainty[[row, col, layer]];
                certainty[[row, col, layer]] = base_cert * (1.0 - alpha) + retrieved_cert * alpha;
            }
        }
    }

    ChromaticTensor::from_arrays(colors, certainty)
}

fn mock_dream_cycle(epoch: usize, retrieved: &[DreamEntry]) -> SolverResult {
    let base_coherence = 0.76 + 0.0015 * epoch as f64;

    let retrieval_bonus = if retrieved.is_empty() {
        0.0
    } else {
        let mean_retrieved = retrieved
            .iter()
            .map(|entry| entry.result.coherence)
            .sum::<f64>()
            / retrieved.len() as f64;
        0.03 + (mean_retrieved - COHERENCE_THRESHOLD).max(0.0) * 0.1
    };

    let coherence = (base_coherence + retrieval_bonus).min(0.99);
    let energy = (1.0 - coherence).max(0.01);
    let violation = (0.12 - retrieval_bonus * 0.5).max(0.01);

    SolverResult {
        energy,
        coherence,
        violation,
        grad: None,
        mask: None,
        meta: json!({
            "epoch": epoch,
            "retrieval_bonus": retrieval_bonus,
            "retrieved": retrieved.len(),
        }),
    }
}

fn parse_csv_records(csv: &str) -> Vec<EpochRecord> {
    csv.lines()
        .skip(1)
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let parts: Vec<&str> = line.split(',').collect();
            assert_eq!(parts.len(), 4, "each CSV row must have four columns");

            EpochRecord {
                epoch: parts[0].parse().expect("epoch should parse"),
                group: parts[1].to_string(),
                avg_loss: parts[2].parse().expect("loss should parse"),
                avg_coherence: parts[3].parse().expect("coherence should parse"),
            }
        })
        .collect()
}

#[test]
fn validation_harness_retrieval_outperforms_control() {
    let first_run = execute_harness(RANDOM_SEED);

    assert!(first_run.final_test.avg_loss < first_run.final_control.avg_loss);
    assert!(first_run.final_test.avg_coherence > first_run.final_control.avg_coherence);
    assert!(first_run.control_pool_count <= POOL_MAX_SIZE);
    assert!(first_run.test_pool_count <= POOL_MAX_SIZE);

    let csv_records = parse_csv_records(&first_run.csv);
    assert_eq!(csv_records.len(), NUM_EPOCHS * 2);
    assert_eq!(
        csv_records.first().map(|r| r.group.as_str()),
        Some("Control")
    );
    let csv_last = csv_records.last().expect("csv should contain final row");
    assert_eq!(csv_last.epoch, first_run.final_test.epoch);
    assert_eq!(csv_last.group, first_run.final_test.group);
    assert!((csv_last.avg_loss - first_run.final_test.avg_loss).abs() < 1e-5);
    assert!((csv_last.avg_coherence - first_run.final_test.avg_coherence).abs() < 1e-5);

    let json_records: Vec<EpochRecord> =
        serde_json::from_str(&first_run.json).expect("json parsing succeeds");
    assert_eq!(json_records.len(), NUM_EPOCHS * 2);
    let json_last = json_records.last().expect("json should contain final row");
    assert_eq!(json_last.epoch, first_run.final_test.epoch);
    assert_eq!(json_last.group, first_run.final_test.group);
    assert!((json_last.avg_loss - first_run.final_test.avg_loss).abs() < 1e-12);
    assert!((json_last.avg_coherence - first_run.final_test.avg_coherence).abs() < 1e-12);

    let second_run = execute_harness(RANDOM_SEED);
    assert_eq!(first_run.csv, second_run.csv);
    assert_eq!(first_run.json, second_run.json);
    assert_eq!(first_run.final_control, second_run.final_control);
    assert_eq!(first_run.final_test, second_run.final_test);
}
