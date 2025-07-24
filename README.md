# Fullrank

Fullrank is an interactive CLI tool for Bayesian inference of list rankings based on noisy comparisons.
It takes a list of items,
then efficiently prompts the user to compare pairs of items until the user decides that the posterior distribution is sufficiently low entropy.
It can then sample from the resulting posterior distribution and compute various statistics.

## Installation and Usage

```bash
# install via pip
pip install git+https://github.com/max-niederman/fullrank.git

# interactively compare items
fullrank compare items.txt > comparisons.json
# infer a ranking distribution and compute statistics
fullrank stats < comparisons.json
# compute raw samples in JSONL format for your own processing
fullrank raw-sample 10000 samples.jsonl < comparisons.json
```

## Background

Deterministic sorting algorithms rank lists by comparing pairs of items.
If an item is greater than another,
it is moved higher in the list.
However,
sometimes it is uncertain which item is greater.
For example:

- The best chess player wins only 60% of their non-draw games against the second-best player.
- Only three-quarters of human raters prefer one LLM completion over another.
- A person prefers one food over another, but only on 80% of days.

Estimating rankings in the presence of this uncertainty is called **noisy sorting**.
A common approach is to model comparisons between items as depending on a latent numerical value ("skill" or "rating") for each item.
For example, the commonly used [Bradley–Terry model](https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model) assumes that

```math
p(i > j) = \sigma (s_i - s_j)
```

where $s_i$ denotes the latent skill of item $i$ and $\sigma$ is the logistic function.

## Motivation

Gwern Branwen's [Resorter](https://gwern.net/resorter) is a CLI tool for
manual noisy sorting of lists based on the [Bradley–Terry model](https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model).
However, its frequentist approach limits it in a few ways:

- It has to use imperfect heuristics to guess which comparison is most informative.
- There's no principled way to quantify how accurate the resulting [maximum-likelihood ranking](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation) is, or tell when to stop comparing items.
- It can't answer questions like "What is the probability that item $i$ is truly the best item?"

As a project to learn more about Bayesian inference,
I decided to build a Bayesian version of Resorter.

## Thurstonian Model

The Bradley–Terry model is quite nice for maximum-likelihood estimation,
but I was unable to get it to work well in a Bayesian setting.
Given a normal prior on the skills $\mathbf{s} \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)$, the posterior density is

```math
p(\mathbf{s}|w > l)
=
\phi (\mathbf{s}; \boldsymbol{\mu}, \Sigma)
\prod_i^m \sigma (\mathbf{s}_{w_i} - \mathbf{s}_{l_i})
\left[\int_{\mathbb{R}^n} \phi (\mathbf{s}; \boldsymbol{\mu}, \Sigma) \prod_i^m \sigma (\mathbf{s}_{w_i} - \mathbf{s}_{l_i}) d \mathbf{s}\right]^{-1}
```

where $\phi$ denotes the normal density, $m$ is the number of comparisons, and $w_i$ and $l_i$ are the winning and losing items in the $i$th comparison.
It appears some researchers have designed efficient sampling procedures for this posterior,
but frankly they are beyond me.

Instead, I used a probability model very similar to Bradley–Terry, but using a probit link instead of a logit link.
That is, under the Thurstonian model,

```math
p(i > j) = \Phi (\mathbf{s}_i - \mathbf{s}_j)
```

where $\Phi$ denotes the cumulative distribution function of the standard normal distribution.

I'll now derive the posterior density in the Thurstonian model.
For convenience, I'll represent the observed comparisons as a matrix $D \in \mathbb{R}^{m \times n}$ mapping score vectors to probits for each comparison.
That is, $D_{i j} = 1$ if item $j$ wins the $i$th comparison, $D_{i j} = -1$ if item $j$ loses the $i$th comparison, and $D_{i j} = 0$ otherwise.

```math
\begin{align}
p(\mathbf{s}|D) &= p(\mathbf{s}) \frac{p(D|\mathbf{s})}{p(D)} \\\\
&= \phi (\mathbf{s}; \boldsymbol{\mu}, \Sigma) \frac{\Pr[D \mathbf{s} < \mathbf{z}]}{\Pr[D \mathbf{t} < \mathbf{z}]}
\end{align}
```

where $\mathbf{t} \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)$ and $\mathbf{z} \sim \mathcal{N}(0, I_m)$.

It turns out that the normalization constant can be represented quite nicely
using the multivariate normal CDF $\Phi_m$:

```math
\begin{align}
\Pr[D \mathbf{t} < \mathbf{z}] &= \Pr[D \mathbf{t} < \mathbf{z}] \\\\
&= \Pr[D (\mathbf{t} - \boldsymbol{\mu}) + D \boldsymbol{\mu} < \mathbf{z}] \\\\
&= \Pr[D \boldsymbol{\mu} < \mathbf{z} - D (\mathbf{t} - \boldsymbol{\mu})]
\end{align}
```

And since $D(\mathbf{t} - \boldsymbol{\mu}) \sim \mathcal{N}(0, D \Sigma D^T)$, we have

```math
\begin{align}
\mathbf{z} - D (\mathbf{t} - \boldsymbol{\mu}) &\sim \mathcal{N}(0, I_m + D \Sigma D^T) \\\\
\Pr[D \mathbf{t} < \mathbf{z}] &= \Phi_m (D \boldsymbol{\mu}; I_m + D \Sigma D^T)
\end{align}
```

Likewise, $\Pr[D \mathbf{s} < \mathbf{z}] = \Phi (D \mathbf{s})$.
Therefore,

```math
p(\mathbf{s}|D) = \phi (\mathbf{s}; \boldsymbol{\mu}, \Sigma) \Phi_m (D \mathbf{s}) \left[\Phi_m (D \boldsymbol{\mu}; I_m + D \Sigma D^T)\right]^{-1}
```

This is called a unified skew-normal (SUN) distribution,
and it is the posterior of most probit models.
Using the convention of [1], we can write

```math
\mathbf{s}|D \sim \text{SUN}_{n,m}(\boldsymbol{\mu}, \Sigma, \Sigma^T D, D \boldsymbol{\mu}, I_m + D \Sigma D^T)
```

## Efficient Sampling

[1] also gives us a convolutional representation of the posterior.
If

```math
\begin{align}
\mathbf{u} &\sim \mathcal{N}(0, \Sigma - \Sigma^T D (I_m + D \Sigma D^T)^{-1} D^T \Sigma), \text{ and} \\\\
\mathbf{v} &\sim \mathcal{N}_{-D \boldsymbol{\mu}}(0, I_m + D \Sigma D^T)
\end{align}
```

where $\mathcal{N}_{\boldsymbol{\tau}}$ denotes the normal distribution truncated below $\boldsymbol{\tau}$, then

```math
\boldsymbol{\mu} + \mathbf{u} + \Sigma^T D (I_m + D \Sigma D^T)^{-1} \mathbf{v} \sim \text{SUN}_{n,m}(\boldsymbol{\mu}, \Sigma, \Sigma^T D, D \boldsymbol{\mu}, I_m + D \Sigma D^T)
```

Fullrank exploits this fact to efficiently sample from the posterior
using samples of $\mathbf{u}$ and $\mathbf{v}$.

## Optimal Comparison Choice

Ideally, Fullrank should always present the user with the most informative comparison.
That is, the comparison whose probit has maximal entropy.

Unified skew-normal distributions are closed under full-rank linear transformations,
so each comparison probit is distributed according to a one-dimensional SUN distribution.
At least in the case of a standard normal prior, each comparison has identical first and second moments,
so intuitively the entropy should be controlled by the skewness.

Fullrank currently assumes that the entropy is decreasing in the $L_2$ norm of the skewness parameter $\Delta$,
which seems to work well in practice.
However, I haven't been able to prove that this works,
and it definitely fails for certain non-scalar choices of prior covariance
(though these are currently not supported anyway).
If you have any better ideas for choosing comparisons,
please let me know!

## References

[1] R. B. Arellano-Valle and A. Azzalini, "Some properties of the unified skew-normal distribution," *Statistical Papers*, vol. 63, pp. 461–487, 2022. [doi:10.1007/s00362-021-01235-2](https://doi.org/10.1007/s00362-021-01235-2)
