use thurston::{model_comparisons, ClosedSkewNormalDistribution, Comparison, NormalDistribution};

#[test]
fn test_model_comparisons() {
    let prior = NormalDistribution::new(Box::new([0.0, 0.0]), Box::new([1.0, 0.0, 0.0, 1.0]));

    let comparisons = vec![Comparison::new(0, 1), Comparison::new(0, 1)];

    let probit_scale = 1.0f32;

    let posterior: ClosedSkewNormalDistribution =
        model_comparisons(prior, comparisons, probit_scale);

    assert_eq!(posterior.dimension(), 2);
    assert_eq!(posterior.latent_dimension(), 2);
    assert_eq!(*posterior.mean(), [0.0, 0.0]);
    assert_eq!(*posterior.covariance(), [1.0, 0.0, 0.0, 1.0]);
    assert_eq!(*posterior.tilt_matrix(), [1.0, 1.0, -1.0, -1.0]);
    assert_eq!(*posterior.latent_mean(), [0.0, 0.0]);
    assert_eq!(*posterior.latent_covariance(), [1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn test_sampling() {
    let prior = NormalDistribution::new(Box::new([0.0, 0.0]), Box::new([1.0, 0.0, 0.0, 1.0]));

    let comparisons = vec![Comparison::new(0, 1), Comparison::new(0, 1)];

    let probit_scale = 1.0f32;

    let posterior = model_comparisons(prior, comparisons, probit_scale);

    let stats = posterior.sample_stats(100);

    let entropy = stats.comparison_entropy();

    // With two identical comparisons favoring 0 over 1, the expected entropy should be less than the maximum possible (ln(2) ≈ 0.693)
    assert!(entropy.expected_entropy < 0.693);
    assert!(entropy.expected_entropy > 0.0);

    assert_eq!(entropy.max_entropy_winner, 0);
    assert_eq!(entropy.max_entropy_loser, 1);
    assert!(entropy.max_entropy > 0.0);
}
