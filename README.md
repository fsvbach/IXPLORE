# IXPLORE

**Iterative Positioning via Logistic Regression Embedding:** A Python package for embedding users and items of political questionnaires in a shared 2D latent space. 

IXPLORE jointly learns a posterior distribution for each user and a logistic regression model for each question. This can be used to visualize the political landscpae in a two dimensional, interpretable space. In inference, it can be used for missing value imputation and answer prediction.

## Features

- **User Embedding**: Compute posterior distributions over user positions using observed reactions
- **Item Models**: Define decision boundaries in the latent space for each item by learning logistic regression models
- **Iterative Refinement**: Jointly optimize user embeddings and item models through iterative updates
- **Flexible Initialization**: Initialize embeddings via PCA, random values, or load pretrained embeddings
- **Missing Data Handling**: Robust to missing values in the user-item reaction matrix
- **Answer Imputation**: Predict missing answers based on learned user positions
- **New User Embedding**: Embed new users based on their responses and obtain optimal positions with uncertainty quantification
- **Visualization Tools**: Built-in plotting functions for embeddings, posteriors, and item decision boundaries

## Installation

```bash
pip install ixplore
```

Or install from source:

```bash
git clone https://github.com/fybach/ixplore.git
cd ixplore
pip install -e .
```

## Quick Start

```python
import pandas as pd
from ixplore import IXPLORE

# Load reaction data (users × items matrix, values in {0, 1})
reactions = pd.read_csv('../data/binary_reactions.csv', index_col=0)

# Initialize and fit the model
model = IXPLORE(reactions, pca_initialization=True)

# Get user embeddings
embedding  = model.get_embedding()        # User positions (N × 2)
parameters = model.get_item_parameters()  # Item parameters (K × 3)

# Embed a new user based on their answers
new_user_answers = pd.Series({'Q1': 0.8, 'Q2': 0.2, 'Q3': 0.6}, name='new_user')
position = model.embed_new_user(new_user_answers)

# Predict all answers for a user
predicted = model.predict_all_answers(new_user_answers)
```

## Usage

### Custom Configuration

```python
import numpy as np

model = IXPLORE(
    reactions,
    prior_mean=np.array([0, 0]),                # Prior center
    prior_cov=np.array([[0.1, 0], [0, 0.1]]),   # Prior covariance
    sampling_resolution=200,                    # Grid resolution for posteriors
    xlimits=(-1, 1),                            # X-axis bounds
    ylimits=(-1, 1),                            # Y-axis bounds
    pca_initialization=True,                    # Initialize with PCA
    random_state=17                             # For reproducibility
)
```

### Loading Pretrained Models

```python
# Load pretrained embedding and model parameters
model = IXPLORE(
    reactions,
    pretrained_embedding='../data/synthetic_embedding.csv',
)
```

### Visualization

```python
from ixplore.visualization import plot_overview

# Load user metadata (e.g., colors for plotting)
users = pd.read_csv('../data/synthetic_users.csv', index_col=0)

# Plot user embeddings
plot_overview(model, n='0', q='Q12', colors=users.color, figsize=(7,2.5))
```

## API Reference

### IXPLORE Class

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reactions` | pd.DataFrame | required | User-item reaction matrix (users as index, items as columns) |
| `prior_mean` | np.ndarray | [0, 0] | Mean of the prior distribution |
| `prior_cov` | np.ndarray | [[0.1, 0], [0, 0.1]] | Covariance of the prior distribution |
| `sampling_resolution` | int | 200 | Grid resolution for posterior computation |
| `xlimits` | tuple | (-1, 1) | X-axis limits of the latent space |
| `ylimits` | tuple | (-1, 1) | Y-axis limits of the latent space |
| `pca_initialization` | bool | True | Initialize embeddings with PCA |
| `random_state` | int | 0 | Random seed for reproducibility |

#### Key Methods

| Method | Description |
|--------|-------------|
| `iterate(n_iterations)` | Run n iterations of posterior fitting and model updating |
| `fit_posteriors()` | Compute posterior distributions for all users |
| `fit_models()` | Fit logistic regression models for all items |
| `get_embedding()` | Return current user embeddings as DataFrame |
| `get_item_parameters()` | Return item model parameters |
| `embed_new_user(answers)` | Embed a new user given their answers |
| `predict_all_answers(answers)` | Predict answers for all items given partial answers |
| `impute_remaining_answers(answers)` | Impute missing answers for a user |
| `evaluate()` | Return MAE and accuracy on training data |

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
  title        = {IXPLORE: Iterative Positioning via Logistic Regression Embedding},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/fybach/ixplore}
}
```

Or in text format:

> Bachmann, F. (2026). IXPLORE: Iterative Positioning via Logistic Regression Embedding. GitHub. https://github.com/fybach/ixplore
