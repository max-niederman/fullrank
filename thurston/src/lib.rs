mod dist;
mod trunc_mvn;
mod utils;
mod stats;

use dist::*;

use nalgebra::DMatrix;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Comparison {
    pub winner: usize,
    pub loser: usize,
}

#[wasm_bindgen]
impl Comparison {
    #[wasm_bindgen(constructor)]
    pub fn new(winner: usize, loser: usize) -> Self {
        Self { winner, loser }
    }
}

#[wasm_bindgen]
/// Given a normal prior and a set of comparisons, return the posterior distribution under the Thurstonian model.
/// This distribution is of closed skew normal form.
pub fn model_comparisons(
    prior: NormalDistribution,
    comparisons: Vec<Comparison>,
    probit_scale: f32,
) -> ClosedSkewNormalDistribution {
    let m = comparisons.len();
    let n = prior.mean.len();

    assert!(
        comparisons
            .iter()
            .map(|c| std::cmp::max(c.winner, c.loser))
            .max()
            .unwrap()
            < n,
        "comparison indices must be less than the number of items"
    );

    let tilt_matrix = {
        let mut tilt_matrix = DMatrix::zeros(n, n);
        for (i, c) in comparisons.iter().enumerate() {
            tilt_matrix[(i, c.winner)] = 1.0;
            tilt_matrix[(i, c.loser)] = -1.0;
        }
        tilt_matrix
    };

    ClosedSkewNormalDistribution {
        mean: prior.mean.clone(),
        covariance: prior.covariance,
        tilt_matrix,
        latent_mean: -prior.mean,
        latent_covariance: probit_scale * DMatrix::identity(m, m),
    }
}

pub use dist::{ClosedSkewNormalDistribution, NormalDistribution};
pub use stats::SampleStats;
