# IXPLORE Documentation

**Iterative Probabilistic Logistic Regression Embedding**

This document walks through IXPLORE's algorithm in the order it executes — initialization, posterior computation, iterative refinement, and inference — and then provides the full API reference. For the high-level pitch and installation, see the [README](../README.md). For the design rationale behind the modeling choices (bounded 2D space, grid inference, BCE likelihood, posterior mean over MAP), see [motivation.md](motivation.md).

---

## Stage 1: Initialization (PCA + Logistic Regression)

### Purpose

Before any iterative refinement, IXPLORE needs an initial layout of users in 2D. A good initialization speeds up convergence and avoids poor local optima. PCA provides a data-driven starting point by projecting the high-dimensional response matrix onto its two principal components.

### How it works

1. **Iterative PCA imputation**: Missing values (NaN) in the reaction matrix are filled via iterative PCA reconstruction (Grung & Manne 1998, implemented in `ixplore.utils.iterative_pca_impute`). This is only used for initialization and does not affect the final model.

2. **PCA projection**: The imputed matrix is reduced to 2 dimensions using a truncated SVD (`ixplore.optimization.pca_decompose`), capturing the two directions of greatest variance in user responses.

3. **Normalization**: The resulting coordinates are centered on the midpoint between min and max and scaled to fit inside the configured `limits` (default `[-1, 1]`) via `ixplore.utils.normalize_embedding`.

4. **Logistic regression fit**: For each item, a logistic regression is fit with a Newton solver (`ixplore.optimization.fit_logistic_newton`) using the 2D user coordinates as features and the normalized (soft) responses as targets. This yields three parameters per item: two coefficients (`beta1`, `beta2`) defining the orientation of the decision boundary, and an intercept (`alpha`) defining its offset.

After this step, the model already has a meaningful (though approximate) embedding and set of decision boundaries.

### Code example

For a runnable end-to-end version of the snippet below, see the [Quick Start in the README](../README.md#quick-start).

```python
import pandas as pd
from ixplore import IXPLORE

# Load user-item reaction matrix
reactions = pd.read_csv('data/likert_reactions.csv', index_col=0)

# Initialize with PCA (default)
model = IXPLORE(reactions, pca_initialization=True, random_state=17)

# Inspect initial embedding and item parameters
embedding = model.get_embedding()    # DataFrame (N x 2) with columns ['x', 'y']
parameters = model.get_parameters()  # DataFrame (K x 3) with columns ['beta1', 'beta2', 'alpha']
```

Alternatively, you can initialize with random positions instead of PCA:

```python
model = IXPLORE(reactions, pca_initialization=False, random_state=17)
```


## Stage 2: Posterior Computation

### Purpose

A single point estimate for a user's position discards information about how confident the model is. IXPLORE computes a posterior distribution over the 2D space for each user, which captures:

- **Localization uncertainty**: Users with many answers have tight, concentrated posteriors. Users with few answers have broad, diffuse posteriors.
- **Multimodality**: If a user's answers are consistent with multiple regions, the posterior reflects this ambiguity.
- **Prior influence**: The Gaussian prior pulls estimates toward the center, acting as regularization especially when data is sparse.

### How it works

IXPLORE evaluates the posterior on a discrete grid (default: 100 x 100 = 10,000 points) covering the latent space using Bayes' rule:

**P(position | answers) proportional to P(answers | position) * P(position)**

The components are:

1. **Prior P(position)**: A multivariate Gaussian (default: zero mean, unit covariance) evaluated at each grid point and normalized. This encodes the belief that users are likely near the center of the space before observing any data.

2. **Likelihood P(answers | position)**: For each grid point, the model computes how well that position explains the user's observed answers. For a single item with response value `y` and predicted probability `p(x)`, the (L1) log-likelihood contribution is `log(1 - |y - p(x)|)`. The total log-likelihood is the sum across all answered items.

3. **Posterior**: The element-wise sum of log-prior and log-likelihood is exponentiated and normalized to sum to 1. The result is a probability distribution over the 10,000 grid points.

The **posterior mean** is extracted from the posterior distribution (via `ixplore.posteriors.posterior_means`) and used as the user's point embedding.

### Code example

```python
import pandas as pd
from ixplore import IXPLORE
from ixplore.visualization import plot_posterior, plot_overview

reactions = pd.read_csv('data/likert_reactions.csv', index_col=0)
users = pd.read_csv('data/synthetic_users.csv', index_col=0)

# Fit the model
model = IXPLORE(reactions, pca_initialization=True)

# Compute and visualize posteriors
model.fit_posteriors()

# Visualize the posterior of a specific user overlaid on the embedding
fig, (ax1, ax2) = plot_overview(model, question='Q15', user='1', colors=users.color, figsize=(7, 2.5))

# You can also compute a posterior for a new user with partial answers
new_user = pd.Series({'Q15': 0, 'Q1': 1}, name='new_user')
fig, ax = plot_posterior(model, new_user)

# Access raw posterior values (array of shape (10000,))
posterior = model.compute_posteriors(new_user)

# Or get all user posteriors as a DataFrame
posteriors = model.get_posteriors()
```

### Tuning the prior

The prior strength controls the trade-off between data and regularization. A tighter prior (smaller covariance) pulls positions toward the center more strongly; a wider prior lets the data dominate:

```python
# Wider prior: less regularization, posteriors follow data more closely
model = IXPLORE(reactions, prior_variance=1.0)

# Tighter prior: stronger regularization, useful for sparse data
model = IXPLORE(reactions, prior_variance=0.1)

# Re-apply a different prior without recomputing likelihoods (requires fit_posteriors first)
model.fit_posteriors()
model.apply_prior(prior_variance=0.3)
```

### Notebook

See [notebooks/demo.ipynb](../notebooks/demo.ipynb) for posterior visualizations showing how uncertainty decreases as more answers are provided and how the prior influences the posterior shape.

---

## Stage 3: Iterative Optimization

### Purpose

Neither the user positions nor the item boundaries are known in advance. IXPLORE works with iterative alternation: given current item models, update user positions; given current user positions, update item models. Each iteration improves the mutual consistency between embeddings and models.

### How it works

Each iteration consists of two steps:

1. **Fit posteriors**: For every user, compute the posterior distribution over the grid using the current item models (logistic regression parameters). Extract the posterior mean as the new user position.

2. **Fit models**: For each item, fit a new logistic regression using the updated user positions as features and the (soft) responses as targets. By default (`point_estimates=True`), the fit uses each user's posterior-mean position. When `point_estimates=False`, the fit instead uses the full grid posteriors as sample weights — propagating user-position uncertainty into the item fit. Update the prediction grid accordingly.

The convergence of this process is measured by decreasing MAE (mean absolute error) and increasing accuracy on the training data, tracked in `model.fit_logger`.

### Sparse responses: `scale_weights`

By default, every observed answer contributes one log-likelihood term to its user's posterior. Users who answered few questions therefore have flatter likelihoods than users who answered many, and the prior dominates their embedding more strongly.

When `scale_weights=True`, each user's per-item weight row is rescaled so it sums to `K` (the full item count), independent of how many items they observed. This rescales the *effective* sample size to `K` for every user, so users with sparse responses are pulled by their data with the same total weight as users with dense responses. Useful when response sparsity varies substantially across users and you don't want sparse users to be dominated by the prior.

`scale_weights` is a constructor flag stored on the model; the same setting applies to training (`iterate`, `fit_posteriors`) and to new-user inference (`compute_posteriors`, `embed`, etc.) so embeddings are computed under a consistent likelihood. With no missing values and uniform weights it is a no-op.

### Code example

```python
import pandas as pd
from ixplore import IXPLORE

reactions = pd.read_csv('data/likert_reactions.csv', index_col=0)

# Initialize model
model = IXPLORE(reactions, pca_initialization=True, random_state=17)

# Run 5 iterations of refinement
model.iterate(n_iterations=5)

# Or propagate uncertainty into the item fits via the grid posteriors
model.iterate(n_iterations=5, point_estimates=False)

# Or equalize effective sample size across users with varying response sparsity
# (set on the constructor, then iterate normally)
model = IXPLORE(reactions, scale_weights=True, random_state=17)
model.iterate(n_iterations=5)

# Check fit quality
metrics = model.evaluate()
print(f"MAE: {metrics['mae']:.4f}, Accuracy: {metrics['accuracy']:.4f}, "
      f"Boundary: {metrics['boundary']:.4f}, Spread: {metrics['spread']:.4f}")

# You can also run the steps manually for fine-grained control
model.fit_posteriors()  # Update user positions from posteriors (posterior mean)
model.fit_models()      # Refit item logistic regressions on the updated embedding
```

After convergence, you can optionally apply a linear transformation (rotation, scaling, shear) to align the embedding axes with interpretable dimensions:

```python
from ixplore.utils import transformation_matrix

M = transformation_matrix(rotate=55, scale=(1.1, 1))
model.transform_embedding(M)
```

### Notebook

See [notebooks/demo.ipynb](../notebooks/demo.ipynb) for a step-by-step walkthrough of the iterative process, including plots at each stage showing how the embedding and decision boundaries evolve.

---

## Stage 4: Inference & Imputation

### Purpose

In practice, users rarely answer every question. IXPLORE ignores missing values during model training. Afterwards, it can predict unobserved responses for both existing and new users. Here, the user's position in the latent space determines their expected response to any item.

### How it works

**During training**: Missing values (NaN) are simply skipped. When computing a user's posterior, only their observed answers contribute to the likelihood. When fitting logistic regression for an item, only users who answered that item are included. No heuristic imputation is needed beyond the initial PCA step.

**For imputation**: Given a (possibly partial) set of answers, IXPLORE:

1. Computes the posterior distribution P(position | observed answers) over the grid.
2. For each item, computes the predicted probability by marginalizing over the posterior:
   **P(answer_k | observed) = sum over all positions of P(answer_k | position) * P(position | observed)**
3. Fills in missing values with these predicted probabilities.

This is Bayesian prediction: it accounts for uncertainty in the user's position rather than relying on a single point estimate.

**For new users**: The same mechanism works for users not in the training set. Provide any subset of answers (even a single one), and the model computes a posterior and predicts all remaining responses.

### Code example

```python
import pandas as pd
from ixplore import IXPLORE

reactions = pd.read_csv('data/likert_reactions.csv', index_col=0)

# Fit the model
model = IXPLORE(reactions, pca_initialization=True)

# --- Embed a new user with partial answers ---
new_user_answers = pd.Series({'Q1': 0, 'Q15': 1, 'Q30': 0}, name='new_user')

# Get their position in the latent space
position = model.embed(new_user_answers)
print(f"New user position: x={position[0]:.3f}, y={position[1]:.3f}")

# --- Impute missing answers, keeping observed values ---
imputed = model.impute_answers(new_user_answers)
print(imputed.head())
# Q1, Q15, Q30 retain their original values; all others are filled with predictions

# --- Draw synthetic answer samples from the posterior ---
samples = model.sample_answers(new_user_answers, method='posterior', num_samples=100)
# shape (100, K): probabilities drawn from posterior-sampled latent positions
```

### Notebook

See [notebooks/demo.ipynb](../notebooks/demo.ipynb) for examples of embedding new users with varying numbers of answers and visualizing how the posterior (and thus predictions) become more precise as more answers are provided.

---

## Algorithm Summary

```
Input: Reaction matrix R (N users x K items), possibly with missing values

[INITIALIZATION]
  Option A: Iterative-PCA-imputed R -> initial 2D positions (then normalize)
  Option B: Random uniform positions in limits^2
  Option C: Load pretrained positions
  Optional: apply a 2x2 transformation matrix to the initial embedding
  -> Fit K logistic regressions on initial positions (Newton solver)

[ITERATIVE REFINEMENT] (repeat for n iterations)
  1. For each user:
       - Compute posterior P(x | answers) on sampling_resolution^2 grid
       - Extract posterior mean as new position
  2. For each item:
       - Fit logistic regression on the updated embedding (or, if
         point_estimates=False, weight the fit by the grid posteriors)
       - Update prediction grid

[INFERENCE]
  - New user embedding:  model.embed(answers)            -> posterior mean(s)
  - Posterior on grid:   model.compute_posteriors(answers)     -> (G,) or (B, G)
  - Answer prediction:   model.predict(positions, items=...)   -> (N, K) probs
  - Imputation:          model.impute_answers(...)             -> filled series/frame
  - Synthetic samples:   model.sample_answers(..., method=...) -> (num_samples, K)
```

---

## API Reference

### Constructor parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reactions` | pd.DataFrame | required | User-item reaction matrix (users as index, items as columns) |
| `prior_variance` | float | 1.0 | Diagonal entry of the Gaussian prior covariance matrix |
| `sampling_resolution` | int | 100 | Grid resolution per axis for posterior computation |
| `limits` | tuple | (-1, 1) | (min, max) limits applied to both axes of the square latent space |
| `pretrained_models` | pd.DataFrame | None | Optional pretrained item parameters |
| `pretrained_embedding` | pd.DataFrame | None | Optional pretrained user embedding |
| `pca_initialization` | bool | True | Initialize embeddings with PCA (ignored if pretrained embedding is given) |
| `random_state` | int | 0 | Random seed for reproducibility |
| `transformation` | np.ndarray | identity(2) | 2×2 linear transformation applied to the initial embedding |
| `kernel` | callable | None | Feature transform `(N, 2) -> (N, D)` for polynomial / RFF features |
| `use_point_estimates` | bool | True | Default for `fit_models`: fit each item on posterior-mean embeddings (True) or on the grid weighted by full posteriors (False) |
| `scale_weights` | bool | False | Rescale each user's per-item weight row to sum to `K`, equalizing effective sample size across users with varying response sparsity |

### Key methods

| Method | Description |
|--------|-------------|
| `iterate(n_iterations)` | Alternate posterior fitting and model updating for n iterations |
| `fit_posteriors()` | Compute posterior distributions for all users and update the embedding |
| `fit_models(point_estimates=True)` | Fit each item on posterior-mean embeddings (default) or on the grid weighted by full posteriors (`point_estimates=False`) |
| `apply_prior(prior_variance)` | Re-apply a Gaussian prior using cached log-likelihoods (cheap re-evaluation, no full likelihood pass) |
| `transform_embedding(M)` | Apply a 2×2 linear transformation to the embedding and refit models |
| `predict(positions, items=None)` | Predict P(Y=1) for items at given 2D positions |
| `compute_posteriors(answers, weights=None)` | Grid posteriors for one user (Series → `(G,)`) or a batch (DataFrame → `(B, G)`) |
| `embed(answers, weights=None)` | Embed one user (Series → `(2,)`) or a batch (DataFrame → `(B, 2)`) |
| `predict_answers(answers, weights=None)` | Predict P(Y=1) for every item — branches on `use_point_estimates` (point prediction at embedded position vs. marginalization over the posterior) |
| `impute_answers(answers, weights=None)` | Fill missing entries via `predict_answers`; observed answers are kept intact |
| `sample_answers(answers, method)` | Draw synthetic answer vectors (`"rasch"`, `"posterior"`, `"random"`) |
| `get_embedding()` | Return current user embeddings as a DataFrame |
| `get_parameters()` | Return item model parameters as a DataFrame |
| `get_posteriors()` | Return current grid posteriors as a DataFrame |
| `evaluate()` | Return a dict with `mae`, `accuracy`, `boundary`, and `spread` on training data |
