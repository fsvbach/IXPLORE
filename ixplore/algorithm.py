from __future__ import annotations

from typing import Callable, Literal

from scipy.special import expit
import pandas as pd
import numpy as np

from .logger import logger, FitLogger
from .optimization import fit_logistic_newton, pca_decompose
from .likelihood import l1_log_likelihood
from .prior import set_gaussian_prior
from .posteriors import posterior_means
from . import utils


class IXPLORE:
    embedding: np.ndarray
    item_parameters: np.ndarray
    predictions_X: np.ndarray

    def __init__(
        self,
        reactions: pd.DataFrame,
        prior_variance: float = 1.0,
        sampling_resolution: int = 100,
        limits: tuple[float, float] = (-1, 1),
        pretrained_models: pd.DataFrame = None,
        pretrained_embedding: pd.DataFrame = None,
        pca_initialization: bool = True,
        random_state: int = 0,
        transformation: np.ndarray = np.identity(2),
        kernel: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        """Initialize the IXPLORE model.

        Parameters
        ----------
        reactions: pd.DataFrame
            The user-item reaction matrix with users as index and items as columns. Can contain missing values (NaN).
        prior_variance: float
            The diagonal entry of the Gaussian prior covariance matrix. Default is 1.0.
        sampling_resolution: int
            The resolution of the grid over the 2D space.
        limits: tuple
            The (min, max) limits applied to both axes of the square 2D space.
        pretrained_models: pd.DataFrame
            The pretrained model parameters. If provided, the model parameters will be loaded from this DataFrame.
        pretrained_embedding: pd.DataFrame
            The pretrained user embeddings. If provided, the user embeddings will be loaded from this DataFrame.
        pca_initialization: bool
            Whether to initialize the embedding with PCA. If False, the embedding will be initialized with random values.
        random_state: int
            The random state for reproducibility.
        transformation: np.ndarray
            A 2x2 transformation matrix to apply to the embedding. Default is the identity matrix.
        kernel: callable, optional
            A feature transform function mapping (N, 2) arrays to (N, D) arrays. If None, the identity
            function is used (no feature engineering). Can be used for e.g. polynomial or RFF features.
        """
        self.kernel = kernel if kernel is not None else lambda X: X
        self.log_likelihood_fn = l1_log_likelihood
        self.scaled_likelihood = False
        self.impute_fn = utils.iterative_pca_impute
        self.get_point_estimates = posterior_means

        ### Store data as numpy arrays
        self.reactions = utils.scale_reactions(reactions.values)
        self.users = reactions.index.astype(str)
        self.items = reactions.columns.astype(str)
        self.number_of_users = len(self.users) # N
        self.number_of_items = len(self.items) # K
        logger.info(f"Number of users for model: {self.number_of_users}")
        logger.info(f"Number of items: {self.number_of_items}")
        logger.info(f"Number of missing values: {np.isnan(self.reactions).sum()} ({np.isnan(self.reactions).mean()*100:.2f}%)")

        ### Create grid
        self.sampling_resolution = sampling_resolution
        self.limits = limits
        self.X = utils.create_meshgrid(self.limits, self.limits, self.sampling_resolution)
        self.X_transformed = self.kernel(self.X)
        self.n_features = self.X_transformed.shape[1]
        self.parameter_names = [f'beta{i+1}' for i in range(self.n_features)] + ['alpha']
        logger.info(f"Grid created with resolution {self.sampling_resolution}x{self.sampling_resolution}, total {self.X.shape[0]} points")
        logger.info(f"Feature dimensions: {self.n_features} (from 2D input)")

        ### Set prior
        self.prior_variance = prior_variance
        self.log_prior_X = set_gaussian_prior(self.X, prior_variance, log=True)
        logger.info(f"Gaussian prior set with covariance diagonal entry {self.prior_variance}")

        ### Initialize other variables
        self.log_likelihoods_X: np.ndarray | None = None
        self.generator = np.random.Generator(np.random.PCG64(seed=random_state))
        self.fit_logger = FitLogger()
        logger.info(f"Random state set to {random_state}")

        ### Initialize embedding and models
        self.transformation = transformation
        if pretrained_embedding is not None:
            self.load_embedding(pretrained_embedding)
            logger.info("Used pretrained embedding.")
        elif pca_initialization:
            self.initialize_with_pca()
            self.embedding = self.embedding @ transformation.T
            logger.info("Initialized embedding with PCA.")
        else:
            self.embedding = self.generator.uniform(limits[0], limits[1], size=(self.number_of_users, 2))
            self.embedding = self.embedding @ transformation.T
            logger.info("Initialized embedding with random values.")

        if pretrained_models is not None:
            self.load_models(pretrained_models)
            logger.info("Used pretrained model parameters.")
        else:
            self.fit_models()
            logger.info("Fitted model parameters from embedding.")

        self._log_metrics(prefix="Initial — ")

    def __str__(self) -> str:
        return 'IXPLORE'

    # ------------------------------------------------------------------
    # Loaders & embedding initialization
    # ------------------------------------------------------------------
    def load_embedding(self, embedding: pd.DataFrame) -> None:
        """Load pretrained user embeddings from a DataFrame."""
        assert embedding.index.astype(str).equals(self.users), "User indices in the pretrained embedding do not match the user indices in the data."
        assert embedding.columns.tolist() == ['x', 'y'], "Columns in the pretrained embedding must be ['x', 'y']."
        self.embedding = embedding.values
        logger.debug(f"Pretrained embedding was given.")

    def load_models(self, item_parameters: pd.DataFrame) -> None:
        """Load pretrained model parameters."""
        assert item_parameters.index.astype(str).equals(self.items), "Items in the pretrained model parameters do not match the items in the data."
        assert item_parameters.columns.tolist() == self.parameter_names, "Columns in the pretrained model parameters do not match the expected columns for the XPLORE model."
        self.item_parameters = item_parameters.values
        self.predictions_X = self.predict(self.X)
        logger.debug(f"Pretrained model parameters were given.")

    def initialize_with_pca(self) -> None:
        """Initialize user embeddings via PCA on the reaction matrix."""
        X_imputed = self.impute_fn(self.reactions)
        self.embedding = pca_decompose(X_imputed, n_components=2)
        self.embedding = utils.normalize_embedding(self.embedding, limits=self.limits)
        logger.debug(f"Initialized embedding with PCA.")

    def transform_embedding(self, transformation: np.ndarray) -> None:
        """Apply a 2x2 linear transformation to the embedding and refit models."""
        assert transformation.shape == (2,2), "Transformation matrix must be of shape (2,2)."
        self.embedding = utils.normalize_embedding(self.embedding @ transformation.T, limits=self.limits)
        self.fit_models(use_posteriors=False)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def iterate(self, n_iterations: int = 10, use_posteriors: bool = False, parallelize: bool = False) -> None:
        """Alternate fit_posteriors and fit_models for n_iterations."""
        for i in range(n_iterations):
            logger.info(f"Iteration {i+1}/{n_iterations}")
            self.fit_posteriors(parallelize=parallelize)
            self.fit_models(use_posteriors=use_posteriors)
            self._log_metrics(prefix=f"Iter {i+1}/{n_iterations} — ")

    def fit_posteriors(self, parallelize: bool = False) -> None:
        """Compute log-likelihoods on the grid for every user and update the embedding."""
        ### TODO: parallelize this
        rows = []
        for n in self.users:
            i = self.users.get_loc(n)
            user = self.reactions[i, :]
            mask = ~np.isnan(user)
            answer_values = user[mask]
            answer_index = np.where(mask)[0]
            rows.append(self._compute_log_likelihood_X(answer_values, answer_index))
        self.log_likelihoods_X = np.array(rows)
        self.embedding = self.get_point_estimates(self._posteriors_X(), self.X)

    def apply_prior(self, prior_variance: float) -> None:
        """Re-apply a Gaussian prior with the given variance using cached log-likelihoods.

        Recomputes `self.log_prior_X` and `self.embedding` without redoing the
        expensive likelihood pass. Requires `fit_posteriors` to have been called.
        """
        assert self.log_likelihoods_X is not None, "fit_posteriors must be called before apply_prior."
        self.prior_variance = prior_variance
        self.log_prior_X = set_gaussian_prior(self.X, prior_variance, log=True)
        self.embedding = self.get_point_estimates(self._posteriors_X(), self.X)

    def fit_models(self, use_posteriors: bool = False) -> None:
        """Fit logistic regression models for each item.

        Parameters
        ----------
        use_posteriors : bool
            If True and posteriors are available, uses aggregated posteriors
            (per-grid-point weights and effective soft labels) for uncertainty-aware
            fitting. If False or posteriors are not available, falls back to fitting
            on point embeddings with soft labels.
        """
        if use_posteriors and self.log_likelihoods_X is not None:
            item_parameters = self._fit_models_posterior()
        else:
            item_parameters = self._fit_models_embedding()
        self.item_parameters = np.vstack(item_parameters)
        self.predictions_X = self.predict(self.X)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(
        self,
        positions: np.ndarray,
        items: list[str] | pd.Index | None = None,
    ) -> np.ndarray:
        """Predict P(Y=1) for each item at each 2D position, shape (N, len(items))."""
        if items is None:
            items = self.items
        index = self.items.get_indexer(items)
        if not len(positions):
            return np.array([])
        features = utils.add_ones(self.kernel(positions.reshape(-1, 2)))
        return expit(features @ self.item_parameters[index, :].T)

    def compute_posterior_X(self, answers: pd.Series, weights: pd.Series | None = None) -> np.ndarray:
        """Compute the posterior over the grid for a user's answers, shape (G,)."""
        answer_values = answers.dropna().values
        if weights is not None:
            weights = weights.dropna().values
        answer_index = self.items.get_indexer(answers.dropna().index)
        logger.debug("Answer values: %s, Answer indices: %s", answer_values, answer_index)
        return self._compute_posterior_X(answer_values, answer_index, weights)

    def embed_user(self, answers: pd.Series) -> np.ndarray:
        """Embed a single user from their answers, returning (x, y)."""
        return self.get_point_estimates(self.compute_posterior_X(answers)[None, :], self.X)[0]

    def impute_remaining_answers(self, answers: pd.Series) -> pd.Series:
        """Fill missing answers via Bayesian marginalization over the grid.

        Computes P(Y=1 | observed answers) for every item and fills only the
        missing entries; observed answers are kept intact.
        """
        P_X_Yi  = self.compute_posterior_X(answers).reshape(-1,1)
        P_Yn1_X = self.predictions_X
        P_XYn1_Yi = P_Yn1_X * P_X_Yi                                        # (G, K)
        P_Yn1_Yi  = P_XYn1_Yi.sum(axis=0)                                   # (K,)
        marginal_predictions = pd.Series(P_Yn1_Yi, name=answers.name, index=self.items)
        answers = pd.Series(index=self.items, dtype=float).fillna(answers)
        return answers.fillna(marginal_predictions)

    def sample_answers(
        self,
        answers: pd.Series,
        method: Literal["rasch", "posterior", "random"] = "posterior",
        num_samples: int = 1000,
        num_options: int = 5,
        variance: float = 0.1,
    ) -> np.ndarray:
        """Draw synthetic answer vectors for a user, shape (num_samples, K).

        Parameters
        ----------
        method: "rasch" | "posterior" | "random"
            Sampling strategy. "rasch" uses the Rasch model on imputed answers;
            "posterior" samples grid points from the posterior and predicts;
            "random" returns uniform noise.
        num_options: int
            Number of answer options (only for "rasch").
        variance: float
            Variance of the Rasch normal distributions (only for "rasch").
        """
        if method == 'rasch':
            mean_answer = self.impute_remaining_answers(answers)
            probs, answer_options = utils.compute_rasch_values(mean_answer, num_options, variance=variance)
            K, Q = probs.shape
            # Repeat probs for k samples: shape (k, K, Q)
            log_probs = np.log(probs)[None, :, :]          # (1, K, Q)
            gumbel_noise = -np.log(-np.log(self.generator.random((num_samples, K, Q))))  # (k, K, Q)
            samples = np.argmax(log_probs + gumbel_noise, axis=1)  # shape: (k, Q)
            samples = answer_options[samples]
        elif method == 'posterior':
            answer_values = answers.dropna().values
            answer_index = self.items.get_indexer(answers.dropna().index)
            posterior = self._compute_posterior_X(answer_values, answer_index)
            samples = self.generator.choice(len(posterior), size=num_samples, p=posterior)
            samples = self.X[samples]
            samples = self.predict(samples)
        else:
            samples = self.generator.random((num_samples, self.number_of_items))
        return samples

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self, boundary_threshold: float = 0.05) -> dict[str, float]:
        """Evaluate model fit on training data.

        Returns
        -------
        dict with keys:
            - 'mae': mean absolute error of the predictions.
            - 'accuracy': fraction of rounded predictions matching rounded reactions.
            - 'boundary': fraction of users within `boundary_threshold` of the grid edge.
            - 'spread': sqrt(std_x^2 + std_y^2) of the embedding.
        """
        predictions = pd.DataFrame(self.predict(self.embedding),
                                   index=self.users,
                                   columns=self.items)
        fit_accuracy = 1 - np.abs(self.reactions.round() - predictions.round()).mean().mean()
        fit_mae = np.mean(np.abs(self.reactions - predictions))
        lo, hi = self.limits
        margin = (hi - lo) * boundary_threshold
        near_border = (
            (self.embedding[:, 0] - lo < margin) |
            (hi - self.embedding[:, 0] < margin) |
            (self.embedding[:, 1] - lo < margin) |
            (hi - self.embedding[:, 1] < margin)
        )
        boundary_fraction = near_border.mean()
        spread = float(np.sqrt(self.embedding[:, 0].std()**2 + self.embedding[:, 1].std()**2))
        return {
            'mae': float(fit_mae),
            'accuracy': float(fit_accuracy),
            'boundary': float(boundary_fraction),
            'spread': spread,
        }

    # ------------------------------------------------------------------
    # Accessors (DataFrame views)
    # ------------------------------------------------------------------
    def get_embedding(self) -> pd.DataFrame:
        """Return the current embedding as a DataFrame indexed by users."""
        return pd.DataFrame(self.embedding.round(3), index=self.users, columns=['x','y'])

    def get_parameters(self) -> pd.DataFrame:
        """Return the item parameters as a DataFrame indexed by items."""
        return pd.DataFrame(self.item_parameters.round(3), index=self.items, columns=self.parameter_names)

    def get_posteriors(self) -> pd.DataFrame:
        """Return the current posteriors on the grid as a DataFrame indexed by users."""
        return pd.DataFrame(self._posteriors_X(), index=self.users)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _log_metrics(self, prefix: str = "") -> None:
        """Evaluate, record to fit_logger, and emit an INFO line."""
        metrics = self.evaluate()
        self.fit_logger.log(metrics)
        body = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        logger.info(f"{prefix}{body}")

    def _fit_models_embedding(self) -> list[np.ndarray]:
        """Fit logistic regression on point embeddings with soft labels."""
        item_parameters = []
        for k in range(self.number_of_items):
            mask = ~np.isnan(self.reactions[:, k])
            train_data = self.kernel(self.embedding[mask])  # (N_k, D)
            train_labels = self.reactions[mask, k]                     # (N_k,)
            params = fit_logistic_newton(train_data, train_labels, regularization=1e-8)
            item_parameters.append(params)
        return item_parameters

    def _fit_models_posterior(self) -> list[np.ndarray]:
        """Fit logistic regression using aggregated posteriors as sample weights."""
        posteriors = self._posteriors_X()
        item_parameters = []
        for k in range(self.number_of_items):
            mask = ~np.isnan(self.reactions[:, k])
            posteriors_k = posteriors[mask]               # (N_k, G)
            labels_k = self.reactions[mask, k]            # (N_k,)
            W_g = posteriors_k.sum(axis=0)                # (G,)
            A_g = posteriors_k.T @ labels_k               # (G,)
            y_eff = np.divide(A_g, W_g, out=np.zeros_like(A_g), where=W_g > 0)
            params = fit_logistic_newton(self.X_transformed, y_eff, sample_weight=W_g, regularization=1e-8)
            item_parameters.append(params)
        return item_parameters

    def _compute_log_likelihood_X(self, answer_values: np.ndarray, answer_index: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
        """Grid log-likelihood for a single user's answers, shape (G,)."""
        mask = ~np.isnan(answer_values.astype(float))
        answer_values = answer_values[mask]
        answer_index = answer_index[mask]
        if weights is None:
            weights = np.ones_like(answer_index, dtype=float)
        assert answer_values.shape == weights.shape, "Length of weights must match length of answers."
        predictions = self.predictions_X[:, answer_index]
        if self.scaled_likelihood and weights.sum() > 0:
            weights *= self.number_of_items / weights.sum()
        return self.log_likelihood_fn(answer_values.reshape(-1), predictions, weights=weights)

    def _compute_posterior_X(self, answer_values: np.ndarray, answer_index: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
        """Single-user posterior on the grid from numpy answer arrays, shape (G,)."""
        log_likelihood = self._compute_log_likelihood_X(answer_values, answer_index, weights)
        log_posterior = log_likelihood + self.log_prior_X
        log_posterior -= log_posterior.max()
        posterior = np.exp(log_posterior)
        return posterior / posterior.sum()

    def _posteriors_X(self) -> np.ndarray:
        """All-user posteriors from cached log-likelihoods and current prior, shape (N, G)."""
        assert self.log_likelihoods_X is not None, "log_likelihoods_X must be computed before deriving posteriors."
        log_posteriors = self.log_likelihoods_X + self.log_prior_X
        log_posteriors -= log_posteriors.max(axis=1, keepdims=True)
        posteriors = np.exp(log_posteriors)
        return posteriors / posteriors.sum(axis=1, keepdims=True)
