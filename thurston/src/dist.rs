use nalgebra::{DMatrix, DVector};
use rand::Rng;
use rand_distr::StandardNormal;
use wasm_bindgen::prelude::*;

use crate::{stats::SampleStats, trunc_mvn};

#[wasm_bindgen]
pub struct NormalDistribution {
    pub(crate) mean: DVector<f32>,
    pub(crate) covariance: DMatrix<f32>,
}

#[wasm_bindgen]
impl NormalDistribution {
    #[wasm_bindgen(constructor)]
    pub fn new(mean: Box<[f32]>, covariance: Box<[f32]>) -> Self {
        let dim = mean.len();
        assert_eq!(dim * dim, covariance.len());
        Self {
            mean: DVector::from_vec(mean.to_vec()),
            covariance: DMatrix::from_vec(dim, dim, covariance.to_vec()),
        }
    }

    #[wasm_bindgen]
    pub fn standard(dimension: usize) -> Self {
        Self {
            mean: DVector::zeros(dimension),
            covariance: DMatrix::identity(dimension, dimension),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn dimension(&self) -> usize {
        self.mean.len()
    }

    #[wasm_bindgen(getter)]
    pub fn mean(&self) -> Box<[f32]> {
        self.mean.as_slice().to_vec().into_boxed_slice()
    }

    #[wasm_bindgen(getter)]
    pub fn covariance(&self) -> Box<[f32]> {
        self.covariance.as_slice().to_vec().into_boxed_slice()
    }
}

#[wasm_bindgen]
pub struct ClosedSkewNormalDistribution {
    pub(crate) mean: DVector<f32>,
    pub(crate) covariance: DMatrix<f32>,
    pub(crate) tilt_matrix: DMatrix<f32>,
    pub(crate) latent_mean: DVector<f32>,
    pub(crate) latent_covariance: DMatrix<f32>,
}

#[wasm_bindgen]
impl ClosedSkewNormalDistribution {
    #[wasm_bindgen(constructor)]
    pub fn new(
        mean: Box<[f32]>,
        covariance: Box<[f32]>,
        tilt_matrix: Box<[f32]>,
        latent_mean: Box<[f32]>,
        latent_covariance: Box<[f32]>,
    ) -> Self {
        let dim = mean.len();
        let latent_dim = latent_mean.len();

        assert_eq!(dim * dim, covariance.len());
        assert_eq!(latent_dim * latent_dim, latent_covariance.len());
        assert_eq!(dim * latent_dim, tilt_matrix.len());

        Self {
            mean: DVector::from_vec(mean.to_vec()),
            covariance: DMatrix::from_vec(dim, dim, covariance.to_vec()),
            tilt_matrix: DMatrix::from_vec(dim, latent_dim, tilt_matrix.to_vec()),
            latent_mean: DVector::from_vec(latent_mean.to_vec()),
            latent_covariance: DMatrix::from_vec(
                latent_dim,
                latent_dim,
                latent_covariance.to_vec(),
            ),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn dimension(&self) -> usize {
        self.mean.len()
    }

    #[wasm_bindgen(getter)]
    pub fn latent_dimension(&self) -> usize {
        self.latent_mean.len()
    }

    #[wasm_bindgen(getter)]
    pub fn mean(&self) -> Box<[f32]> {
        self.mean.as_slice().to_vec().into_boxed_slice()
    }

    #[wasm_bindgen(getter)]
    pub fn covariance(&self) -> Box<[f32]> {
        self.covariance.as_slice().to_vec().into_boxed_slice()
    }

    #[wasm_bindgen(getter)]
    pub fn tilt_matrix(&self) -> Box<[f32]> {
        self.tilt_matrix.as_slice().to_vec().into_boxed_slice()
    }

    #[wasm_bindgen(getter)]
    pub fn latent_mean(&self) -> Box<[f32]> {
        self.latent_mean.as_slice().to_vec().into_boxed_slice()
    }

    #[wasm_bindgen(getter)]
    pub fn latent_covariance(&self) -> Box<[f32]> {
        self.latent_covariance
            .as_slice()
            .to_vec()
            .into_boxed_slice()
    }

    pub fn affine_transform(&self, matrix: Box<[f32]>, offset: Box<[f32]>) -> Self {
        let dim = self.dimension();
        let latent_dim = self.latent_dimension();

        assert_eq!(matrix.len(), dim * latent_dim);
        assert_eq!(offset.len(), dim);

        let matrix = DMatrix::from_vec(dim, latent_dim, matrix.to_vec());
        let b = DVector::from_vec(offset.to_vec());

        let mean = &matrix * &self.mean;
        let covariance = &matrix * &self.covariance * matrix.transpose();
        let tilt_matrix = &matrix * &self.tilt_matrix;
        let latent_mean = &matrix * &self.latent_mean + &b;
        let latent_covariance = &matrix * &self.latent_covariance * matrix.transpose();

        Self {
            mean,
            covariance,
            tilt_matrix,
            latent_mean,
            latent_covariance,
        }
    }

    #[wasm_bindgen]
    pub fn sample_stats(&self, n: usize) -> SampleStats {
        let samples = self.sample_n(n, &mut rand::thread_rng());
        SampleStats { samples }
    }

    /// Sample `n` values column-wise from the distribution.
    ///
    /// See section 2.1 of doi:10.1111/j.1467-9469.2006.00503.x for details on the implementation.
    pub(crate) fn sample_n(&self, num_samples: usize, rng: &mut impl Rng) -> DMatrix<f32> {
        let dim = self.dimension();
        let latent_dim = self.latent_dimension();

        // Compute necessary SUN parameters
        let Delta = &self.covariance * self.tilt_matrix.transpose();
        let Gamma = DMatrix::identity(latent_dim, latent_dim)
            + &self.tilt_matrix * &self.covariance * self.tilt_matrix.transpose();
        let Gamma_inv = Gamma
            .clone()
            .try_inverse()
            .expect("Gamma matrix is not invertible");

        // Decompose equation (13). We choose Psi = I for convenience.
        // This way the Cholesky decomposition is handled by [`MultivariateNormal`].
        let b0 = &Delta * &Gamma_inv;
        let b1_squared = DMatrix::identity(dim, dim) - &Delta * &Gamma_inv * Delta.transpose();
        let b1 = b1_squared.cholesky().unwrap().l();

        // Sample V_0 using truncated normal (need to convert to/from ndarray)
        let v0 = trunc_mvn::sample_n(
            &(-&self.tilt_matrix * &self.mean).map(|x| x as f64),
            &DVector::from_element(latent_dim, f64::INFINITY),
            &Gamma_inv.map(|x| x as f64),
            num_samples,
        )
        .map(|x| x as f32);

        // Sample V_1 from standard normal
        let v1 = DMatrix::from_distribution(dim, num_samples, &StandardNormal, rng);

        // Compute the final samples
        &self.mean * DMatrix::from_element(1, num_samples, 1.) + &b0 * v0 + &b1 * v1
    }
}
