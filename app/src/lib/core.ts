export type Item = { text: string };
export type Comparison = { winner: number; loser: number };

export type NormalDistribution = {
  mean: NDArray;
  covariance: NDArray;
};

export class ClosedSkewNormalDistribution {
  mu: NDArray;
  Sigma: NDArray;
  D: NDArray;
  nu: NDArray;
  Delta: NDArray;

  constructor(
    mu: NDArray,
    Sigma: NDArray,
    D: NDArray,
    nu: NDArray,
    Delta: NDArray
  ) {
    this.mu = mu;
    this.Sigma = Sigma;
    this.D = D;
    this.nu = nu;
    this.Delta = Delta;
  }

  affineTransform(A: NDArray, b: NDArray): ClosedSkewNormalDistribution {
    const mu_A = A.multiply(this.mu).add(b);
    const Sigma_A = A.multiply(this.Sigma).multiply(A.transpose());
    const D_A = this.D.multiply(this.Sigma)
      .multiply(A.transpose())
      .multiply(Sigma_A.inv());
    const nu_A = this.nu;
    const Delta_A = this.Delta.add(
      this.D.multiply(this.Sigma).multiply(this.D.transpose())
    ).subtract(
      D_A.multiply(A).multiply(this.Sigma).multiply(this.D.transpose())
    );

    return new ClosedSkewNormalDistribution(mu_A, Sigma_A, D_A, nu_A, Delta_A);
  }
}

export function modelComparisons(
  priorMean: NDArray,
  priorCovariance: NDArray,
  probitScale: number,
  comparisons: Comparison[]
): ClosedSkewNormalDistribution {
  const m = comparisons.length;
  const n = priorMean.length;

  const mu = priorMean;
  const Sigma = priorCovariance;
  const nu = mu.scale(-1);
  const Delta = eye(m).scale(1 / probitScale);

  const D = zeros(m, n);
  comparisons.forEach((comparison, i) => {
    D.set(i, comparison.winner, 1);
    D.set(i, comparison.loser, -1);
  });

  return new ClosedSkewNormalDistribution(mu, Sigma, D, nu, Delta);
}

export function comparisonProbitDistributionFromModel(
  model: ClosedSkewNormalDistribution
): ClosedSkewNormalDistribution {
  const m = model.D.shape[0];
  const n = model.D.shape[1];

  const A = zeros(n * n, n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      A.set(i * n + j, i, 1);
      A.set(i * n + j, j, -1);
    }
  }

  const b = zeros(n * n);

  return model.affineTransform(A, b);
}
