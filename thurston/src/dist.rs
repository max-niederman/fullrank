use ndarray::{Array, Array1, Array2};
use ndarray_linalg::{Cholesky, Inverse, UPLO};
use ndarray_rand::{rand, rand_distr, RandomExt};
use truncnorm::distributions::MultivariateTruncatedNormal;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct NormalDistribution {
    pub(crate) mean: Array1<f32>,
    pub(crate) covariance: Array2<f32>,
}

#[wasm_bindgen]
impl NormalDistribution {
    #[wasm_bindgen(constructor)]
    pub fn new(mean: Box<[f32]>, covariance: Box<[f32]>) -> Self {
        assert_eq!(mean.len() * mean.len(), covariance.len());
        Self {
            mean: Array1::from(mean.to_vec()),
            covariance: Array2::from_shape_vec((mean.len(), mean.len()), covariance.to_vec())
                .unwrap(),
        }
    }

    #[wasm_bindgen]
    pub fn standard(dimension: usize) -> Self {
        Self {
            mean: Array1::zeros(dimension),
            covariance: Array2::eye(dimension),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn dimension(&self) -> usize {
        self.mean.len()
    }

    #[wasm_bindgen(getter)]
    pub fn mean(&self) -> Box<[f32]> {
        self.mean.as_slice().unwrap().to_vec().into_boxed_slice()
    }

    #[wasm_bindgen(getter)]
    pub fn covariance(&self) -> Box<[f32]> {
        self.covariance
            .as_slice()
            .unwrap()
            .to_vec()
            .into_boxed_slice()
    }
}

#[wasm_bindgen]
pub struct ClosedSkewNormalDistribution {
    pub(crate) mean: Array1<f32>,
    pub(crate) covariance: Array2<f32>,
    pub(crate) tilt_matrix: Array2<f32>,
    pub(crate) latent_mean: Array1<f32>,
    pub(crate) latent_covariance: Array2<f32>,
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
        assert_eq!(mean.len() * mean.len(), covariance.len());
        assert_eq!(
            latent_mean.len() * latent_mean.len(),
            latent_covariance.len()
        );
        assert_eq!(latent_mean.len() * mean.len(), tilt_matrix.len());

        Self {
            mean: Array1::from(mean.to_vec()),
            covariance: Array2::from_shape_vec((mean.len(), mean.len()), covariance.to_vec())
                .unwrap(),
            tilt_matrix: Array2::from_shape_vec((mean.len(), mean.len()), tilt_matrix.to_vec())
                .unwrap(),
            latent_mean: Array1::from(latent_mean.to_vec()),
            latent_covariance: Array2::from_shape_vec(
                (latent_mean.len(), latent_mean.len()),
                latent_covariance.to_vec(),
            )
            .unwrap(),
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
        self.mean.as_slice().unwrap().to_vec().into_boxed_slice()
    }

    #[wasm_bindgen(getter)]
    pub fn covariance(&self) -> Box<[f32]> {
        self.covariance
            .as_slice()
            .unwrap()
            .to_vec()
            .into_boxed_slice()
    }

    #[wasm_bindgen(getter)]
    pub fn tilt_matrix(&self) -> Box<[f32]> {
        self.tilt_matrix
            .as_slice()
            .unwrap()
            .to_vec()
            .into_boxed_slice()
    }

    #[wasm_bindgen(getter)]
    pub fn latent_mean(&self) -> Box<[f32]> {
        self.latent_mean
            .as_slice()
            .unwrap()
            .to_vec()
            .into_boxed_slice()
    }

    #[wasm_bindgen(getter)]
    pub fn latent_covariance(&self) -> Box<[f32]> {
        self.latent_covariance
            .as_slice()
            .unwrap()
            .to_vec()
            .into_boxed_slice()
    }

    pub fn affine_transform(&self, matrix: Box<[f32]>, offset: Box<[f32]>) -> Self {
        assert_eq!(matrix.len(), self.dimension() * self.latent_dimension());
        assert_eq!(offset.len(), self.dimension());

        let matrix = Array2::<f32>::from_shape_vec(
            (self.dimension(), self.latent_dimension()),
            matrix.to_vec(),
        )
        .unwrap();
        let matrix_tr = matrix.t();
        let b = Array1::<f32>::from_vec(offset.to_vec());

        let mean = matrix.dot(&self.mean);
        let covariance = matrix.dot(&self.covariance).dot(&matrix_tr);
        let tilt_matrix = matrix.dot(&self.tilt_matrix);
        let latent_mean = matrix.dot(&self.latent_mean) + &b;
        let latent_covariance = matrix.dot(&self.latent_covariance).dot(&matrix_tr);

        Self {
            mean,
            covariance,
            tilt_matrix,
            latent_mean,
            latent_covariance,
        }
    }

    #[wasm_bindgen]
    pub fn sample(&self, n: usize) -> Box<[f32]> {
        self.sample_convolutional(n)
            .as_slice()
            .unwrap()
            .to_vec()
            .into_boxed_slice()
    }

    /// Sample `n` values column-wise from the distribution.
    ///
    /// See section 2.1 of doi:10.1111/j.1467-9469.2006.00503.x for details on the implementation.
    pub(crate) fn sample_convolutional(&self, n: usize) -> Array2<f32> {
        // Compute necessary SUN parameters
        let Delta = self.covariance * self.tilt_matrix.t();
        let Gamma = Array2::<f32>::eye(self.latent_dimension())
            + self.tilt_matrix * self.covariance * self.tilt_matrix.t();
        let Gamma_inv = Gamma.inv().unwrap();

        // Decompose equation (13). We choose Psi = I for convenience of V_2 sampling.
        let B0 = Delta * Gamma_inv;
        let B1 = (Array2::eye(self.latent_dimension()) - Delta * Gamma_inv * Delta.t())
            .cholesky(UPLO::Lower)
            .unwrap();

        // Sample V_0 and V_1
        let mut rng = rand::thread_rng();
        let V0_dist = MultivariateTruncatedNormal::new(
            Array1::zeros(self.latent_dimension()),
            Gamma.map(|&x| x as f64),
            Array1::zeros(self.latent_dimension()),
            Array1::from_elem(self.latent_dimension(), f64::INFINITY),
            32,
        );
        let V0 = V0_dist.sample_n(n, &mut rng).map(|&x| x as f32);
        let V1 = Array::random_using(
            (self.latent_dimension(), n),
            rand_distr::Normal::new(0.0, 1.0).unwrap(),
            &mut rng,
        );

        self.mean.into_shape((self.dimension(), 1)) + B0 * V0 + B1 * V1
    }
}
