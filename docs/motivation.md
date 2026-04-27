# IXPLORE: Motivation

## Why a 2D bounded latent space?

IXPLORE is designed to produce an interpretable visual layout of users and items in a shared bounded latent space. Every modeling choice — dimensionality, prior, inference scheme, output API — follows from this commitment.

Given a bounded 2D latent space and the desire for a per-user posterior, the natural inference substrate is a discrete grid covering the canvas. The grid serves two roles simultaneously:

1. **Posterior**: the posterior $p(x_n \mid y_n, \theta)$ is evaluated exactly at every grid cell, with no approximation beyond the discretization itself.
2. **Inference**: missing-answer imputation integrates the predictive distribution over the user's posterior on the grid — a closed-form operation.

## Why logit + BCE?

Within the bounded-grid model, the choice of likelihood is a choice of how individual answers contribute to a user's log-posterior. Under the BCE (Bernoulli cross-entropy) likelihood:

$$\log p(y_n \mid x, \theta) = \sum_k m_{nk}\big[y_{nk} \log \sigma(\beta_k^\top x + \alpha_k) + (1-y_{nk})\log(1-\sigma(\beta_k^\top x + \alpha_k))\big]$$

the log-likelihood at each grid cell decomposes linearly in $y_{nk}$ and $1-y_{nk}$. This linearity is what enables the vectorized implementation: the full $(N, G)$ log-likelihood matrix factors into two BLAS matmuls,

$$L = \tilde Y \cdot \mathrm{LP}^\top + \tilde Z \cdot \mathrm{LN}^\top,$$

where $\tilde Y, \tilde Z$ are mask-zeroed answer matrices and $\mathrm{LP}, \mathrm{LN}$ cache the per-grid-point log-predictions. This collapses the dominant cost from a Python loop over users into two matrix products, yielding ~100× speedup at typical sizes without changing the model.

The BCE likelihood also unifies the algorithm's two halves under a common probabilistic story. Both steps reference the same predictive function $p_k(x) = \sigma(\beta_k^\top x + \alpha_k)$ and the same per-observation Bernoulli loss. The algorithm alternates between:

- **User step (E-step-like)**: compute the full posterior $p(x_n \mid y_n, \theta)$ on the grid, summarized as the posterior mean for the embedding.
- **Item step (M-step)**: refit each item's logistic regression — either at the posterior mean (default, `point_estimates=True`, hard-EM-style plug-in) or weighted by the full grid posterior (`point_estimates=False`, the standard EM M-step that maximizes the expected complete-data log-likelihood).

With `point_estimates=False`, this is textbook EM on the marginal likelihood $\sum_n \log p(y_n \mid \theta)$, with the standard monotone-improvement guarantee. The default plug-in mode (`point_estimates=True`) is a faster approximation that uses the posterior mean as a point summary — empirically convergent, marginally less principled, but several times faster per iteration. Both modes optimize the same predictive model; they differ only in how the user posterior is summarized for the item update.

## Why Mean and not MAP?

We summarize each user's posterior by its mean rather than its mode (MAP). On a discrete grid, the MAP estimator would force every user's reported coordinate onto a grid point, producing visible lattice artifacts in the visualization (users aligning along regular rows and columns at the grid resolution). The posterior mean lives in the continuous interior of the bounded latent space and varies smoothly with the data and grid resolution, which preserves the visual fidelity of the deliverable. The mean has a small inward bias for users whose posterior piles against a boundary, but this bias is smooth and small, while MAP's lattice artifacts would be visually obvious at any practical grid resolution.

## Relation to existing work

The dominant approaches to ideal-point and item-response estimation — Poole-Rosenthal NOMINATE (Poole and Rosenthal 1985, 1997) and the Bayesian methodology of Clinton, Jackman, and Rivers (2004) — both fit user positions and item parameters jointly under utility-function-based generative models. Both are well-established but, as Potthoff (2018) documents in detail, both suffer from nonidentifiability beyond one dimension and from arbitrary modeling constraints (e.g., NOMINATE forces ideal points onto the unit circle in 2D). IXPLORE shares neither commitment: it makes no utility-function assumption and its 2D bounded latent space is identified up to rotation and reflection.

Two more recent works are closer in spirit and worth comparing directly.

**Potthoff (2018) — PCA + per-item logistic regression.** This is the closest prior work in modeling stance. Potthoff advocates a two-stage decomposition: principal-components analysis of the (mean-adjusted) reaction matrix yields user scores; per-item logistic regression on those scores yields item parameters. IXPLORE shares the substantive commitments of this approach — separable user/item estimation, logit link, orthogonal axes, no utility function — and its PCA initialization is in fact Potthoff's full method as a starting point. Where IXPLORE departs is the user-side estimator: Potthoff's PCA gives a single point per user with no uncertainty (a weakness he explicitly acknowledges in §8); IXPLORE replaces it with a Bayesian posterior on a bounded grid, then iterates between the user and item updates rather than running them once. This addresses Potthoff's main self-identified gap (uncertainty quantification) while preserving his identifiability and simplicity arguments.

**Imai, Lo, and Olmsted (2016) — closed-form EM via Albert-Chib augmentation.** Imai et al. develop fast (variational) EM algorithms for the standard Bayesian ideal-point models, exploiting probit + data augmentation to get closed-form Gaussian conditional posteriors that scale to millions of users. They are far from IXPLORE in both modeling stance (probit + utility function, unbounded latent space) and deliverable (point estimates + bootstrap SEs at scale, no visualization layer), but close in the specific sense that they too solve the closed-form-EM problem more cleanly than NOMINATE/CJR.

| | Potthoff (2018) | IXPLORE | Imai et al. (2016) |
|---|---|---|---|
| Item model | Logit, per-item logistic regression | Logit, per-item logistic regression | Probit + augmentation |
| User model | PCA (one-shot) | Bayesian posterior on grid (iterated) | Closed-form Gaussian via augmentation |
| Latent space | Unbounded $\mathbb{R}^K$, $K$ chosen by eigenvalue test | Bounded $[-L, L]^2$, $K=2$ fixed | Unbounded $\mathbb{R}^K$, $K$ variable |
| Identifiability | Orthogonality (clean) | 2D rotation only (clean) | Sign restrictions; problematic for $K>1$ |
| Utility function | No | No | Yes |
| Uncertainty | None (point estimate only) | Full posterior on grid | Bootstrap SEs |
| Iterative refinement | No | Yes (alternating EM-style) | Yes (textbook EM) |
| Missing data | Iterative imputation (§4.5) | Bayesian marginalization | Latent variable in EM |
| New-user inference | Implicit, ad-hoc | First-class API (`embed_user`) | Refit |
| Visualization | Acknowledged but not central | Primary deliverable | Acknowledged but not central |
| Scale regime | Hundreds–thousands | Thousands–tens of thousands | Millions |

The genealogy is therefore: IXPLORE inherits Potthoff's modeling stance (logit, no utility function, separable user/item estimation, orthogonality) and adds the Bayesian posterior + iteration that Potthoff identifies as missing — specialized to 2D for visualization-first analysis based on Imai's unbounded ideal-point estimation for massive populations.