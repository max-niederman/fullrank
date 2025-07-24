#set text(size: 13pt, font: "Garamond Premier Pro")

Fullrank is an interactive CLI tool for Bayesian inference of list rankings based on noisy comparisons.
It takes a list of items,
then efficiently prompts the user to compare pairs of items until the user decides that the posterior distribution is sufficiently low entropy.
It can then sample from the resulting posterior distribution and compute various statistics.

= Background

Deterministic sorting algorithms rank lists by comparing pairs of items.
If an item is greater than another,
it is moved higher in the list.
However,
sometimes it is uncertain which item is greater.
For example:

- The best chess player wins only 60% of their non-draw games against the second-best player.
- Only three-quarters of human raters prefer one LLM completion over another.
- A person prefers one food over another, but only on 80% of days.

Estimating rankings in the presence of this uncertainty is called *noisy sorting*.
A common approach is to model comparisons between items as depending on a latent numerical value ("skill" or "rating") for each item.
For example, the commonly used #link("https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model")[Bradley--Terry model] assumes that
$ p(i > j) = sigma (s_i - s_j) $
where $s_i$ denotes the latent skill of item $i$ and $sigma$ is the logistic function.

= Motivation

Gwern Branwen's #link("https://gwern.net/resorter")[Resorter] is a CLI tool for
manual noisy sorting of lists based on the #link("https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model")[Bradley--Terry model].
However, its frequentist approach limits it in a few ways:

- It has to use imperfect heuristics to guess which comparison is most informative.
- There's no principled way to quantify how accurate the resulting #link("https://en.wikipedia.org/wiki/Maximum_likelihood_estimation")[maximum-likelihood ranking] is, or tell when to stop comparing items.
- It can't answer questions like "What is the probability that item $i$ is truly the best item?"

As a project to learn more about Bayesian inference,
I decided to build a Bayesian version of Resorter.

= Thurstonian Model

The Bradley--Terry model is quite nice for maximum-likelihood estimation,
but I was unable to get it to work well in a Bayesian setting.
Given a normal prior on the skills $s ~ N(mu, Sigma)$, the posterior density is

$
  p(bold(s)|w > l)
  =
  phi (bold(s); mu, Sigma)
  product_i^m sigma (bold(s)_w_i - bold(s)_l_i)
  [integral_(RR^n) phi (bold(s); mu, Sigma) product_i^m sigma (bold(s)_w_i - bold(s)_l_i) d bold(s)]^(-1)
$

where $phi$ denotes the normal density, $m$ is the number of comparisons, and $w_i$ and $l_i$ are the winning and losing items in the $i$th comparison.
It appears some researchers have designed efficient sampling procedures for this posterior,
but frankly they are beyond me.

Instead, I used a probability model very similar to Bradley--Terry, but using a probit link instead of a logit link.
That is, under the Thurstonian model,

$ p(i > j) = Phi (bold(s)_i - bold(s)_j) $

where $Phi$ denotes the cumulative distribution function of the standard normal distribution.

I'll now derive the posterior density in the Thurstonian model.
For convenience, I'll represent the observed comparisons as a matrix $D in RR^(m times n)$ mapping score vectors to probits for each comparison.
That is, $D_(i j) = 1$ if item $j$ wins the $i$th comparison, $D_(i j) = -1$ if item $j$ loses the $i$th comparison, and $D_(i j) = 0$ otherwise.

$
  p(bold(s)|D) & = p(bold(s)) p(D|bold(s)) / p(D)                                                     \
               & = phi (bold(s); bold(mu), Sigma) (Pr[D bold(s) < bold(z)]) / Pr[D bold(t) < bold(z)]
$
where $bold(t) ~ cal(N)(bold(mu), Sigma)$ and $bold(z) ~ cal(N)(0, I_m)$.

It turns out that the normalization constant can be represented quite nicely
using the multivariate normal CDF $Phi_m$:

$
  Pr[D bold(t) < bold(z)] & = Pr[D bold(t) < bold(z)]                               \
                          & = Pr[D (bold(t) - bold(mu)) + D bold(mu) < bold(z)]     \
                          & = Pr[D bold(mu) < bold(z) - D (bold(t) - bold(mu))] "."
$

And since $D(bold(t) - bold(mu)) ~ cal(N)(0, D Sigma D^T)$, we have

$
  bold(z) - D (bold(t) - bold(mu)) & ~ cal(N)(0, I_m + D Sigma D^T)              \
           Pr[D bold(t) < bold(z)] & = Phi_m (D bold(mu); I_m + D Sigma D^T) "."
$

Likewise, $Pr[D bold(s) < bold(z)] = Phi (D bold(s))$.
Therefore,

$
  p(bold(s)|D) & = phi (bold(s); bold(mu), Sigma) Phi_m (D bold(s)) [Phi_m (D bold(mu); I_m + D Sigma D^T)]^(-1) "."
$

This is called a unified skew-normal (SUN) distribution,
and it is the posterior of most probit models.
Using the convention of @sun-props, we can write

$
  bold(s)|D & ~ "SUN"_(n,m)(bold(mu), Sigma, Sigma^T D, D bold(mu), I_m + D Sigma D^T) "."
$

= Efficient Sampling

@sun-props also gives us a convolutional representation of the posterior.
If
$
  bold(u) & ~ cal(N)(0, Sigma - Sigma^T D (I_m + D Sigma D^T)^(-1) D^T Sigma)", and" \
  bold(v) & ~ cal(N)_(-D bold(mu))(0, I_m + D Sigma D^T) ","
$
where $cal(N)_(bold(tau))$ denotes the normal distribution truncated below $bold(tau)$, then
$
  bold(mu) + bold(u) + Sigma^T D (I_m + D Sigma D^T)^(-1) bold(v) & ~ "SUN"_(n,m)(bold(mu), Sigma, Sigma^T D, D bold(mu), I_m + D Sigma D^T) "."
$

Fullrank exploits this fact to efficiently sample from the posterior
using samples of $bold(u)$ and $bold(v)$.

= Optimal Comparison Choice

#let inc = math.class("relation", "⧡")

Ideally, Fullrank should always present the user with the most informative comparison.
That is, the comparison whose probit has maximal entropy.

Unified skew-normal distributions are closed under full-rank linear transformations,
so each comparison probit is distributed according to a one-dimensional SUN distribution.
At least in the case of a standard normal prior, each comparison has identical first and second moments,
so intuitively the entropy should be controlled by the skewness.

Fullrank currently assumes that the entropy is decreasing in the $L_2$ norm of the skewness parameter $Delta$,
which seems to work well in practice.
However, I haven't been able to prove that this works,
and it definitely fails for certain non-scalar choices of prior covariance
(though these are currently not supported anyway).
If you have any better ideas for choosing comparisons,
please let me know!

= Using the Posterior

Fullrank can be used either as a CLI tool or as a Python library to get raw samples from the posterior distribution.
It also implements a few useful statistics:

```
$ fullrank compare fruits.txt | fullrank stats --entropy
Finished inferring posterior.
Sampling... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:01
Mean:  [-6.76037714e-01 -1.67944768e+00  1.61404920e-03  1.58269652e+00
  7.65822584e-01]
Ranking Deciles:
[[0 1 1 1 1 1 1 1 1 1]
 [0 0 0 0 0 0 0 0 1 2]
 [0 0 2 2 2 2 2 2 2 4]
 [0 2 3 4 4 4 4 4 4 4]
 [0 3 3 3 3 3 3 3 3 4]]
Ranking Probabilities (rows are items, columns are rankings):
[[6.0780e-02 9.3543e-01 3.7400e-03 0.0000e+00 5.0000e-05]
 [7.8972e-01 6.2380e-02 1.4145e-01 1.7000e-04 6.2800e-03]
 [1.4292e-01 2.1300e-03 7.3222e-01 5.4500e-03 1.1728e-01]
 [6.4300e-03 6.0000e-05 1.1810e-01 1.2262e-01 7.5279e-01]
 [1.5000e-04 0.0000e+00 4.4900e-03 8.7176e-01 1.2360e-01]]
Entropy:  -4.45792309625278
```

You can see the source code on GitHub #link("https://github.com/max-niederman/fullrank")[here].

#bibliography("bibliography.bib")