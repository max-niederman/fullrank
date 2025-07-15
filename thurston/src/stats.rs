use nalgebra::DMatrix;
use statrs::function::erf::erf;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct SampleStats {
    /// Matrix of samples, with each column being a sample.
    pub(crate) samples: DMatrix<f32>,
}

#[wasm_bindgen]
impl SampleStats {
    #[wasm_bindgen]
    /// Compute statistics on the entropy of a comparison between two items.
    pub fn comparison_entropy(&self) -> ComparisonEntropyStats {
        let mut sum_entropy = 0.0;
        let mut max_entropy = f64::NEG_INFINITY;
        let mut max_entropy_matchup = (0, 0);

        for i in 0..self.samples.nrows() {
            for j in 0..self.samples.nrows() {
                if i == j {
                    continue;
                }

                let mut sum_matchup_entropy = 0.0;
                let mut sum_p_i_wins = 0.0;
                for sample in self.samples.column_iter() {
                    let s_i = sample[i];
                    let s_j = sample[j];
                    let p_i_wins = standard_normal_cdf((s_i - s_j) as f64);
                    let entropy =
                        -p_i_wins * p_i_wins.ln() - (1.0 - p_i_wins) * (1.0 - p_i_wins).ln();
                    sum_p_i_wins += p_i_wins;
                    sum_matchup_entropy += entropy;
                }
                let avg_matchup_entropy = sum_matchup_entropy / self.samples.ncols() as f64;
                let avg_p_i_wins = sum_p_i_wins / self.samples.ncols() as f64;
                if avg_matchup_entropy > max_entropy {
                    max_entropy = avg_matchup_entropy;
                    max_entropy_matchup = if avg_p_i_wins > 0.5 { (i, j) } else { (j, i) };
                }
                sum_entropy += avg_matchup_entropy;
            }
        }
        let avg_entropy = sum_entropy / (self.samples.ncols() * (self.samples.ncols() - 1)) as f64;

        ComparisonEntropyStats {
            expected_entropy: avg_entropy,
            max_entropy_winner: max_entropy_matchup.0,
            max_entropy_loser: max_entropy_matchup.1,
            max_entropy,
        }
    }
}

#[wasm_bindgen]
pub struct ComparisonEntropyStats {
    /// Expected entropy of a randomly chosen matchup of distinct items.
    pub expected_entropy: f64,
    /// The winner from the matchup with the highest entropy.
    pub max_entropy_winner: usize,
    /// The loser from the matchup with the highest entropy.
    pub max_entropy_loser: usize,
    /// The entropy of the matchup with the highest entropy.
    pub max_entropy: f64,
}

fn standard_normal_cdf(x: f64) -> f64 {
    (1.0 + erf(x / std::f64::consts::SQRT_2)) / 2.0
}
