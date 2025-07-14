use argmin::core::Jacobian as ArgminJacobian;
use argmin::core::Operator as ArgminOperator;
use argmin::core::{Error, Executor};
use argmin::solver::gaussnewton::GaussNewton;
use nalgebra::{DMatrix, DVector};
use ordered_float::OrderedFloat;
use rand::prelude::*;
use rand_distr::Normal;
use statrs::function::erf::{erfc, erfc_inv};

#[derive(Clone)]
struct PsiProblem {
    l: DVector<f64>,
    u: DVector<f64>,
    l_mat: DMatrix<f64>,
}

impl ArgminOperator for PsiProblem {
    type Param = DVector<f64>;
    type Output = DVector<f64>;

    fn apply(&self, y: &Self::Param) -> Result<Self::Output, Error> {
        let d = self.l.len();
        let (x, mu) = self.split_y(y, d);
        let (grad, _) = self.grad_psi(&x, &mu, false);
        Ok(grad)
    }
}

impl ArgminJacobian for PsiProblem {
    type Param = DVector<f64>;
    type Jacobian = DMatrix<f64>;

    fn jacobian(&self, y: &Self::Param) -> Result<Self::Jacobian, Error> {
        let d = self.l.len();
        let (x, mu) = self.split_y(y, d);
        let (_, jacob) = self.grad_psi(&x, &mu, true);
        Ok(jacob.unwrap())
    }
}

impl PsiProblem {
    fn split_y(&self, y: &DVector<f64>, d: usize) -> (DVector<f64>, DVector<f64>) {
        let mut x = DVector::zeros(d - 1);
        let mut mu = DVector::zeros(d - 1);
        x.copy_from(&y.rows(0, d - 1));
        mu.copy_from(&y.rows(d - 1, d - 1));
        (x, mu)
    }

    fn psy(&self, x: &DVector<f64>, mu: &DVector<f64>) -> f64 {
        let d = self.l.len();
        let mut full_x = DVector::zeros(d);
        let mut full_mu = DVector::zeros(d);
        full_x.rows_mut(0, d - 1).copy_from(x);
        full_mu.rows_mut(0, d - 1).copy_from(mu);
        let c = &self.l_mat * &full_x;
        let lt = &self.l - &full_mu - &c;
        let ut = &self.u - &full_mu - &c;
        let sum_terms = (0..d)
            .map(|i| ln_n_pr(lt[i], ut[i]) + 0.5 * full_mu[i] * full_mu[i])
            .sum::<f64>();
        sum_terms - full_x.dot(&full_mu)
    }

    fn grad_psi(
        &self,
        x: &DVector<f64>,
        mu: &DVector<f64>,
        compute_jacob: bool,
    ) -> (DVector<f64>, Option<DMatrix<f64>>) {
        let d = self.l.len();
        let mut full_x = DVector::zeros(d);
        let mut full_mu = DVector::zeros(d);
        full_x.rows_mut(0, d - 1).copy_from(x);
        full_mu.rows_mut(0, d - 1).copy_from(mu);
        let mut c = DVector::zeros(d);
        c.rows_mut(1, d - 1)
            .copy_from(&(&self.l_mat.rows(1, d - 1) * &full_x));
        let lt = &self.l - &full_mu - &c;
        let ut = &self.u - &full_mu - &c;
        let w_vec = (0..d).map(|i| ln_n_pr(lt[i], ut[i])).collect::<Vec<f64>>();
        let w = DVector::from_vec(w_vec);
        let pl_vec = (0..d)
            .map(|i| (-0.5 * lt[i] * lt[i] - w[i]).exp() / (2.0 * std::f64::consts::PI).sqrt())
            .collect::<Vec<f64>>();
        let pl = DVector::from_vec(pl_vec);
        let pu_vec = (0..d)
            .map(|i| (-0.5 * ut[i] * ut[i] - w[i]).exp() / (2.0 * std::f64::consts::PI).sqrt())
            .collect::<Vec<f64>>();
        let pu = DVector::from_vec(pu_vec);
        let p = &pl - &pu;
        let dfdx = -full_mu.rows(0, d - 1) + self.l_mat.columns(0, d - 1).transpose() * &p;
        let dfdm = full_mu - full_x + &p;
        let mut grad = DVector::zeros(2 * (d - 1));
        grad.rows_mut(0, d - 1).copy_from(&dfdx);
        grad.rows_mut(d - 1, d - 1).copy_from(&dfdm.rows(0, d - 1));
        if compute_jacob {
            let mut lt2 = lt.clone();
            for v in lt2.iter_mut() {
                if v.is_infinite() {
                    *v = 0.0;
                }
            }
            let mut ut2 = ut.clone();
            for v in ut2.iter_mut() {
                if v.is_infinite() {
                    *v = 0.0;
                }
            }
            let dp = -p.component_mul(&p) + lt2.component_mul(&pl) - ut2.component_mul(&pu);
            let dl = DMatrix::from_diagonal(&dp) * &self.l_mat;
            let mx = -DMatrix::identity(d, d) + &dl;
            let xx = self.l_mat.transpose() * dl;
            let mx_slice = mx.view((0, 0), (d - 1, d - 1));
            let xx_slice = xx.view((0, 0), (d - 1, d - 1));
            let mut jacob = DMatrix::zeros(2 * (d - 1), 2 * (d - 1));
            for i in 0..2 * (d - 1) {
                for j in 0..2 * (d - 1) {
                    jacob[(i, j)] = if i < d - 1 {
                        if j < d - 1 {
                            xx_slice[(i, j)]
                        } else {
                            mx_slice[(i, j - (d - 1))]
                        }
                    } else {
                        if j < d - 1 {
                            mx_slice[(i - (d - 1), j)]
                        } else {
                            if i == j {
                                1.0 + dp[i - (d - 1)]
                            } else {
                                0.0
                            }
                        }
                    };
                }
            }
            (grad, Some(jacob))
        } else {
            (grad, None)
        }
    }
}

fn ln_n_pr(a: f64, b: f64) -> f64 {
    let mut p = 0.0;
    if a > 0.0 {
        let pa = ln_phi(a);
        let pb = ln_phi(b);
        p = pa + (1.0 - (pb - pa).exp()).ln();
    } else if b < 0.0 {
        let pa = ln_phi(-a);
        let pb = ln_phi(-b);
        p = pb + (1.0 - (pa - pb).exp()).ln();
    } else {
        let pa = erfc(-a / 2.0f64.sqrt()) / 2.0;
        let pb = erfc(b / 2.0f64.sqrt()) / 2.0;
        p = (1.0 - pa - pb).ln();
    }
    p
}

fn ln_phi(x: f64) -> f64 {
    -0.5 * x * x - 2.0f64.ln() + erfcx(x / 2.0f64.sqrt()).ln()
}

fn erfcx(x: f64) -> f64 {
    (x * x).exp() * erfc(x)
}

fn trandn(l: f64, u: f64) -> f64 {
    let a = 0.66;
    if l > a {
        ntail(l, u)
    } else if u < -a {
        -ntail(-u, -l)
    } else {
        tn(l, u)
    }
}

fn tn(l: f64, u: f64) -> f64 {
    let tol = 2.0;
    if (u - l).abs() > tol {
        trnd(l, u)
    } else {
        let pl = erfc(l / 2.0f64.sqrt()) / 2.0;
        let pu = erfc(u / 2.0f64.sqrt()) / 2.0;
        2.0f64.sqrt() * erfc_inv(2.0 * (pl - (pl - pu) * rand::thread_rng().gen::<f64>()))
    }
}

fn trnd(l: f64, u: f64) -> f64 {
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut rng = rand::thread_rng();
    let mut x = normal.sample(&mut rng);
    while x < l || x > u {
        x = normal.sample(&mut rng);
    }
    x
}

fn ntail(l: f64, u: f64) -> f64 {
    let c = l * l / 2.0;
    let f = (c - u * u / 2.0).exp_m1();
    let mut rng = rand::thread_rng();
    let mut x = c - (1.0 + rng.gen::<f64>() * f).ln();
    while rng.gen::<f64>().powi(2) * x > c {
        x = c - (1.0 + rng.gen::<f64>() * f).ln();
    }
    (2.0 * x).sqrt()
}

struct CholPermResult {
    l_mat: DMatrix<f64>,
    l: DVector<f64>,
    u: DVector<f64>,
    perm: Vec<usize>,
}

fn chol_perm(mut sig: DMatrix<f64>, mut l: DVector<f64>, mut u: DVector<f64>) -> CholPermResult {
    let d = l.len();
    let mut perm = (0..d).collect::<Vec<usize>>();
    let mut l_mat = DMatrix::<f64>::zeros(d, d);
    let mut z = DVector::zeros(d);
    for j in 0..d {
        let mut pr = DVector::from_element(d, f64::INFINITY);
        for i in j..d {
            let s = sig[(i, i)] - (0..j).map(|k| l_mat[(i, k)].powi(2)).sum::<f64>();
            let s_sqrt = if s < 0.0 {
                f64::EPSILON.sqrt()
            } else {
                s.sqrt()
            };
            let tl = (l[i] - (0..j).map(|k| l_mat[(i, k)] * z[k]).sum::<f64>()) / s_sqrt;
            let tu = (u[i] - (0..j).map(|k| l_mat[(i, k)] * z[k]).sum::<f64>()) / s_sqrt;
            pr[i] = ln_n_pr(tl, tu);
        }
        let k = (j..d).min_by_key(|&i| OrderedFloat(pr[i])).unwrap();
        sig.swap_rows(j, k);
        sig.swap_columns(j, k);
        l_mat.swap_rows(j, k);
        l.swap_rows(j, k);
        u.swap_rows(j, k);
        perm.swap(j, k);
        let s = sig[(j, j)] - (0..j).map(|k| l_mat[(j, k)].powi(2)).sum::<f64>();
        let s = if s < 0.0 { f64::EPSILON } else { s };
        l_mat[(j, j)] = s.sqrt();
        for i in (j + 1)..d {
            l_mat[(i, j)] = (sig[(i, j)]
                - (0..j).map(|k| l_mat[(i, k)] * l_mat[(j, k)]).sum::<f64>())
                / l_mat[(j, j)];
        }
        let tl = (l[j] - (0..j).map(|k| l_mat[(j, k)] * z[k]).sum::<f64>()) / l_mat[(j, j)];
        let tu = (u[j] - (0..j).map(|k| l_mat[(j, k)] * z[k]).sum::<f64>()) / l_mat[(j, j)];
        let w = ln_n_pr(tl, tu);
        z[j] = ((-0.5 * tl * tl - w).exp() - (-0.5 * tu * tu - w).exp())
            / (2.0 * std::f64::consts::PI).sqrt();
    }
    CholPermResult { l_mat, l, u, perm }
}

fn mvnrnd(
    n: usize,
    l_mat: &DMatrix<f64>,
    l: &DVector<f64>,
    u: &DVector<f64>,
    mu: &DVector<f64>,
) -> (DVector<f64>, DMatrix<f64>) {
    let d = l.len();
    let mut z = DMatrix::zeros(d, n);
    let mut p = DVector::zeros(n);
    let mut full_mu = DVector::zeros(d);
    full_mu.rows_mut(0, d - 1).copy_from(mu);
    for k in 0..d {
        for i in 0..n {
            let col = (0..k).fold(0.0, |acc, j| acc + l_mat[(k, j)] * z[(j, i)]);
            let tl = l[k] - full_mu[k] - col;
            let tu = u[k] - full_mu[k] - col;
            z[(k, i)] = full_mu[k] + trandn(tl, tu);
            p[i] += ln_n_pr(tl, tu) + 0.5 * full_mu[k].powi(2) - full_mu[k] * z[(k, i)];
        }
    }
    (p, z)
}

/// Sample from a truncated multivariate normal distribution with zero mean, the given lower and upper bounds, and covariance matrix.
/// Infinite bounds are accepted.
pub fn sample_n(
    lower_bound: &DVector<f64>,
    upper: &DVector<f64>,
    cov: &DMatrix<f64>,
    n: usize,
) -> DMatrix<f64> {
    let d = lower_bound.len();
    if upper.len() != d
        || cov.nrows() != d
        || cov.ncols() != d
        || lower_bound.iter().zip(upper.iter()).any(|(&l, &u)| l > u)
    {
        panic!("Dimensions mismatch or l > u");
    }
    let CholPermResult {
        l_mat: l_full,
        mut l,
        mut u,
        perm,
    } = chol_perm(cov.clone(), lower_bound.clone(), upper.clone());
    let diag = l_full.diagonal();
    if diag.iter().any(|&d| d < f64::EPSILON) {
        eprintln!("Warning: Covariance matrix is singular!");
    }
    let mut l_mat = l_full.clone();
    for i in 0..d {
        l_mat.row_mut(i).scale_mut(1.0 / diag[i]);
    }
    l.component_div_assign(&diag);
    u.component_div_assign(&diag);
    l_mat = l_mat - DMatrix::identity(d, d);
    let problem = PsiProblem {
        l_mat: l_mat.clone(),
        l: l.clone(),
        u: u.clone(),
    };
    let init_param = DVector::zeros(2 * (d - 1));
    let solver = GaussNewton::new();
    let res = Executor::new(problem.clone(), solver)
        .configure(|state| state.param(init_param).max_iters(100))
        .run()
        .unwrap();
    let soln = res.state.best_param.unwrap();
    let (x, mu) = problem.split_y(&soln, d);
    let psistar = problem.psy(&x, &mu);
    let mut rv_vec: Vec<DVector<f64>> = Vec::with_capacity(n);
    let mut accept = 0;
    let mut iter = 0;
    while accept < n {
        let current_n = n - accept;
        let (logpr, z) = mvnrnd(current_n, &l_mat, &l, &u, &mu);
        let mut rng = rand::thread_rng();
        for i in 0..current_n {
            if -rng.gen::<f64>().ln() > psistar - logpr[i] {
                rv_vec.push(DVector::from(z.column(i)));
            }
        }
        accept = rv_vec.len();
        iter += 1;
        if iter == 1000 {
            eprintln!("Warning: Acceptance prob smaller than 0.001");
        } else if iter > 10000 {
            for _ in 0..(n - accept) {
                let (_, approx_z) = mvnrnd(1, &l_mat, &l, &u, &mu);
                rv_vec.push(DVector::from(approx_z.column(0)));
            }
            eprintln!("Warning: Sample is only approximately distributed.");
            break;
        }
    }
    let mut rv = DMatrix::from_columns(&rv_vec);
    rv = l_full * rv;
    let mut ordered_rv = DMatrix::zeros(d, n);
    for (i, &p) in perm.iter().enumerate() {
        ordered_rv.row_mut(p).copy_from(&rv.row(i));
    }
    ordered_rv
}
