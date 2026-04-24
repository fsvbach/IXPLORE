# IXPLORE Documentation

**Iterative Probabilistic Logistic Regression Embedding**

## Overview

IXPLORE is a Python package that jointly embeds users and questionnaire items in a shared 2D latent space. It is designed for user-item reaction matrices commonly found in political questionnaires, where each row represents a user and each column represents an item (question). Responses can be binary (agree/disagree) or Likert-scale values, which are automatically normalized to the [0, 1] range.

The core idea is simple: a user's position in 2D space should predict their responses to all items. Each item defines a logistic regression decision boundary in this space, separating regions of agreement from disagreement. The model iteratively refines both the user positions and the item boundaries until they are mutually consistent.

IXPLORE produces:
- **User embeddings** (N x 2): a 2D coordinate for each user representing their latent preferences
- **Item parameters** (K x 3): logistic regression coefficients (beta1, beta2, intercept) defining each item's decision boundary
- **Posterior distributions**: full probability distributions over the 2D space quantifying uncertainty about each user's position

This enables interpretable visualization of preference landscapes, principled uncertainty quantification, and missing value imputation grounded in the learned geometry.

---

## Feature 1: Baseline Embedding (PCA + Logistic Regression)

### Purpose

Before any iterative refinement, IXPLORE needs an initial layout of users in 2D. A good initialization speeds up convergence and avoids poor local optima. PCA provides a data-driven starting point by projecting the high-dimensional response matrix onto its two principal components.

### How it works

1. **Mean imputation**: Missing values (NaN) in the reaction matrix are filled with column means using `sklearn.impute.SimpleImputer`. This is only used for initialization and does not affect the final model.

2. **PCA projection**: The imputed matrix is reduced to 2 dimensions using `sklearn.decomposition.PCA`, capturing the two directions of greatest variance in user responses.

3. **Normalization**: The resulting coordinates are centered (subtract the midpoint between min and max) and scaled to fit within the [-1, 1] bounds of the latent space.

4. **Logistic regression fit**: For each item, a `sklearn.linear_model.LogisticRegression` model is fit using the 2D user coordinates as features and the (stochastically binarized) responses as targets. This yields three parameters per item: two coefficients (beta1, beta2) defining the orientation of the decision boundary, and an intercept (alpha) defining its offset.

After this step, the model already has a meaningful (though approximate) embedding and set of decision boundaries.

### Code example

```python
import pandas as pd
from ixplore import IXPLORE

# Load user-item reaction matrix
reactions = pd.read_csv('data/likert_reactions.csv', index_col=0)

# Initialize with PCA (default)
model = IXPLORE(reactions, pca_initialization=True, random_state=17)

# Inspect initial embedding and item parameters
embedding = model.get_embedding()        # DataFrame (N x 2) with columns ['x', 'y']
parameters = model.get_item_parameters() # DataFrame (K x 3) with columns ['beta1', 'beta2', 'alpha']

print(embedding.head())
print(parameters.head())
```

Alternatively, you can initialize with random positions instead of PCA:

```python
model = IXPLORE(reactions, pca_initialization=False, random_state=17)
```

### Notebook

See [notebooks/demo.ipynb](../notebooks/demo.ipynb) for a side-by-side comparison of random vs. PCA initialization, including visualizations of the initial embeddings and decision boundaries.

---

## Feature 2: Posterior Distributions for Uncertainty Quantification

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

2. **Likelihood P(answers | position)**: For each grid point, the model computes how well that position explains the user's observed answers. For a single item with response value `y` and predicted probability `p(x)`, the likelihood contribution is `1 - |y - p(x)|`. The total likelihood is the product across all answered items.

3. **Posterior**: The element-wise product of prior and likelihood, normalized to sum to 1. The result is a probability distribution over the 10,000 grid points.

The **MAP estimate** (mean a posteriori) is extracted as the grid point with the expectation value of the posterior distribution and used as the user's point embedding.

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
posterior = model.posterior_X(new_user)
```

### Tuning the prior

The prior strength controls the trade-off between data and regularization. A tighter prior (smaller covariance) pulls positions toward the center more strongly; a wider prior lets the data dominate:

```python
# Wider prior: less regularization, posteriors follow data more closely
model = IXPLORE(reactions, prior_variance=1.0)

# Tighter prior: stronger regularization, useful for sparse data
model = IXPLORE(reactions, prior_variance=0.1)
```

### Notebook

See [notebooks/demo.ipynb](../notebooks/demo.ipynb) for posterior visualizations showing how uncertainty decreases as more answers are provided and how the prior influences the posterior shape.

---

## Feature 3: Iterative Refinement

### Purpose

Neither the user positions nor the item boundaries are known in advance. IXPLORE works with iterative alternation: given current item models, update user positions; given current user positions, update item models. Each iteration improves the mutual consistency between embeddings and models.

### How it works

Each iteration consists of three steps:

1. **Fit posteriors**: For every user, compute the posterior distribution over the grid using the current item models (logistic regression parameters). Extract the MAP estimate as the new user position.

2. **Normalize embedding**: Center the embedding (subtract midpoint) and scale it to fit within [-1, 1] bounds. This prevents drift and ensures the grid covers all users.

3. **Fit models**: For each item, fit a new logistic regression using the updated user positions as features and the binarized responses as targets. Update the likelihood grid accordingly.

The convergence of this process is measured by decreasing MAE (mean absolute error) and increasing accuracy on the training data.

### Code example

```python
import pandas as pd
from ixplore import IXPLORE

reactions = pd.read_csv('data/likert_reactions.csv', index_col=0)

# Initialize model
model = IXPLORE(reactions, pca_initialization=True, random_state=17)

# Run 5 iterations of refinement
model.iterate(n_iterations=5)

# Check fit quality
metrics = model.evaluate()
print(f"MAE: {metrics['mae']:.4f}, Accuracy: {metrics['accuracy']:.4f}, "
      f"Boundary: {metrics['boundary']:.4f}, Spread: {metrics['spread']:.4f}")

# You can also run the steps manually for fine-grained control
model.fit_posteriors()       # Step 1: update user positions from posteriors
model.normalize_embedding()  # Step 2: center and scale
model.fit_models()           # Step 3: refit item logistic regressions
```

After convergence, you can optionally apply a linear transformation (rotation, scaling, shear) to align the embedding axes with interpretable dimensions:

```python
from ixplore.utils import transformation_matrix

M = transformation_matrix(rotation=55, scale=(1.1, 1))
model.transform_embedding(M)
```

### Notebook

See [notebooks/demo.ipynb](../notebooks/demo.ipynb) for a step-by-step walkthrough of the iterative process, including plots at each stage showing how the embedding and decision boundaries evolve.

---

## Feature 4: Missing Value Imputation

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
position = model.embed_new_user(new_user_answers)
print(f"New user position: x={position[0]:.3f}, y={position[1]:.3f}")

# --- Impute missing answers, keeping observed values ---
imputed = model.impute_remaining_answers(new_user_answers)
print(imputed.head())
# Q1, Q15, Q30 retain their original values; all others are filled with predictions
```

### Notebook

See [notebooks/demo.ipynb](../notebooks/demo.ipynb) for examples of embedding new users with varying numbers of answers and visualizing how the posterior (and thus predictions) become more precise as more answers are provided.

---

## Algorithm Summary

```
Input: Reaction matrix R (N users x K items), possibly with missing values

[INITIALIZATION]
  Option A: PCA on mean-imputed R -> initial 2D positions
  Option B: Random uniform positions in [-1, 1]^2
  Option C: Load pretrained positions
  -> Fit K logistic regressions on initial positions

[ITERATIVE REFINEMENT] (repeat for n iterations)
  1. For each user:
       - Compute posterior P(x | answers) on 100x100 grid
       - Extract MAP estimate as new position
  2. Center and scale all positions to [-1, 1]^2
  3. For each item:
       - Fit logistic regression on updated positions
       - Update likelihood grid

[INFERENCE]
  - New user embedding: posterior -> MAP estimate
  - Answer prediction: marginalize likelihood over posterior
  - Imputation: fill missing with predicted probabilities
```
