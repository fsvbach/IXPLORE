# IXPLORE

**Iterative Probabilistic Logistic Regression Embedding:** A Python package for embedding users and questionnaire items in a shared 2D latent space. 

![IXPLORE overview](https://raw.githubusercontent.com/fsvbach/IXPLORE/refs/heads/main/figures/overview.png)

## Applicable data

IXPLORE is designed for user-item reaction matrices commonly found in political questionnaires. It's well suited for binary data (e.g., agree/disagree) and Likert-scale responses, where each question measures different preferences dimensions. Use IXPLORE when you want a compact, interpretable 2D visualization of the latent political landscape or when you need interpretable imputation of missing responses based on users' latent positions.

## Features

IXPLORE jointly learns a posterior distribution for each user and a logistic regression model for each questionnaire item. It visualizes the political landscpae in a two dimensional space. In inference, it can impute missing values and generate answers based on any latent position.

- **User Embedding**: Compute posterior distributions for users based on their reactions
- **Item Models**: Define decision boundaries for each questions with logistic regression models
- **Iterative Refinement**: Jointly optimize user embeddings and item models through iterative updates
- **Flexible Initialization**: Initialize embeddings via PCA, random values, or load pretrained embeddings
- **Missing Data Handling**: Robust to missing values in the user-item reaction matrix
- **Answer Imputation**: Predict answers based on positions in latent space
- **New User Embedding**: Embed new users in the latent space with uncertainty quantification
- **Visualization Tools**: Built-in plotting functions for embeddings, posteriors, and item decision boundaries

## Installation

```bash
pip install ixplore
```

Or install from source:

```bash
git clone https://github.com/fsvbach/ixplore.git
cd ixplore
pip install -e .
```

### Quick Start

```python
import pandas as pd
from ixplore import IXPLORE

# Load reaction data (users × items matrix, values in {0, 1} or Likert-scale)
reactions = pd.read_csv('../data/likert_reactions.csv', index_col=0)

# Initialize and fit the model
model = IXPLORE(reactions, pca_initialization=True)

# Refine with a few iterations of joint optimization
model.iterate(n_iterations=1)

# Get user embeddings and item parameters
embedding  = model.get_embedding()    # User positions (N × 2) with columns ['x', 'y']
parameters = model.get_parameters()   # Item parameters (K × 3): ['beta1', 'beta2', 'alpha']

# Embed a new user based on their answers
new_user_answers = pd.Series({'Q1': 0.8, 'Q2': 0.2, 'Q3': 0.6}, name='new_user')
position = model.embed_user(new_user_answers)

# Impute all answers for a user
predicted = model.impute_remaining_answers(new_user_answers)
```

### Custom Configuration

```python
import numpy as np

model = IXPLORE(
    reactions,
    prior_variance=1.0,                         # Prior regularization
    sampling_resolution=100,                    # Grid resolution for posteriors
    limits=(-1, 1),                             # Bounds for both axes (square space)
    pca_initialization=True,                    # Initialize with PCA
    random_state=17                             # For reproducibility
)
```

### Loading Pretrained Models

```python
# Load pretrained embedding and model parameters
pretrained_embedding = pd.read_csv('../data/pretrained_embedding_likert.csv', index_col=0)
pretrained_models = pd.read_csv('../data/pretrained_models_likert.csv', index_col=0)

model = IXPLORE(
    reactions,
    pretrained_embedding=pretrained_embedding,
    pretrained_models=pretrained_models
)
```

### Visualization

```python
from ixplore.visualization import plot_overview

# Load user metadata (e.g., colors for plotting)
users = pd.read_csv('../data/synthetic_users.csv', index_col=0)

# Plot user embeddings
_ = plot_overview(model, question='Q12', user='1', colors=users.color)
```

### IXPLORE Class

#### Constructor Parameters

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

#### Key Methods

| Method | Description |
|--------|-------------|
| `iterate(n_iterations, use_posteriors=False)` | Alternate posterior fitting and model updating for n iterations |
| `fit_posteriors()` | Compute posterior distributions for all users and update the embedding |
| `fit_models(use_posteriors=False)` | Fit logistic regression models for all items (optionally weighted by posteriors) |
| `apply_prior(prior_variance)` | Re-apply a Gaussian prior using cached log-likelihoods |
| `transform_embedding(M)` | Apply a 2×2 linear transformation to the embedding and refit models |
| `predict(positions, items=None)` | Predict P(Y=1) for items at given 2D positions |
| `compute_posterior_X(answers)` | Grid posterior for a given answer vector |
| `embed_user(answers)` | Embed a (new or existing) user given their answers |
| `impute_remaining_answers(answers)` | Impute missing answers for a user via Bayesian marginalization |
| `sample_answers(answers, method)` | Draw synthetic answer vectors (`"rasch"`, `"posterior"`, `"random"`) |
| `get_embedding()` | Return current user embeddings as a DataFrame |
| `get_parameters()` | Return item model parameters as a DataFrame |
| `get_posteriors()` | Return current grid posteriors as a DataFrame |
| `evaluate()` | Return a dict with `mae`, `accuracy`, `boundary`, and `spread` on training data |

## Dependencies

- numpy
- pandas
- scikit-learn
- scipy
- matplotlib

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use IXPLORE in your research, please cite:

```bibtex
@software{bachmann2026ixplore,
  author       = {Bachmann, Fynn},
  title        = {IXPLORE: Iterative Probabilistic Logistic Regression Embedding},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/fsvbach/ixplore}
}
```

Or in text format:

> Bachmann, F. (2026). IXPLORE: Iterative Probabilistic Logistic Regression Embedding. GitHub. https://github.com/fsvbach/ixplore
