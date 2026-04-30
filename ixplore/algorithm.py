from __future__ import annotations

import logging
from typing import Callable, Literal

from scipy.special import expit, log_expit
import pandas as pd
import numpy as np

from .logger import logger, FitLogger
from .optimization import fit_logistic_newton, pca_decompose, posterior_means
from .metrics import compute_accuracy, compute_mae, compute_spread
from .prior import set_gaussian_prior
from . import utils


class IXPLORE:
    embedding: np.ndarray
    item_parameters: np.ndarray
    _log_p1: np.ndarray
    _log_p0: np.ndarray

    def __init__(
        self,
        reactions: pd.DataFrame,
        weights: pd.DataFrame | None = None,
        prior_variance: float = 1.0,
        sampling_resolution: int = 200,
        limits: tuple[float, float] = (-1, 1),
        pretrained_models: pd.DataFrame = None,
        pretrained_embedding: pd.DataFrame = None,
        pca_initialization: bool = True,
        random_state: int = 0,
        transformation: np.ndarray = np.identity(2),
        kernel: Callable[[np.ndarray], np.ndarray] | None = None,
        use_point_estimates: bool = True,
        scale_weights: bool = False,
        model_regularization: float = 1e-8,
    ) -> None:
        """Initialize the IXPLORE model.

        Parameters
        ----------
        reactions: pd.DataFrame
            The user-item reaction matrix with users as index and items as columns. Can contain missing values (NaN).
        weights: pd.DataFrame, optional
            The weights applied to the log-likelihood at fit time. Same shape and indices as `reactions`.
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
            A 2x2 transformation matrix applied to the embedding. Default is the identity matrix.
        kernel: callable, optional
            A feature transform function mapping (N, 2) arrays to (N, D) arrays. If None, the identity function is used (no feature engineering). Can be used for e.g. polynomial or RFF features.
        use_point_estimates: bool
            Default for `fit_models`: if True, fit each item on the users' posterior-mean point
            estimates; if False, fit on the grid using the full posteriors as sample weights.
        scale_weights: bool
            If True, rescale per-user weights so each user's row sums to K (the
            full item count), independent of how many items they observed. Default False.
        model_regularization: float
            L2 regularization strength on the per-item logistic regression coefficients.
            Larger values shrink the coefficients toward zero.
        """
        self.kernel = kernel if kernel is not None else lambda X: X
        self.use_point_estimates = use_point_estimates
        self.scale_weights = scale_weights
        self.model_regularization = model_regularization
        self.get_point_estimates = posterior_means

        ### Store data as numpy arrays
        self.reactions = utils.scale_reactions(reactions.values).astype(np.float32)
        self.users = reactions.index.astype(str)
        self.items = reactions.columns.astype(str)
        self.number_of_users = len(self.users) # N
        self.number_of_items = len(self.items) # K

        ### Store weights
        self.weights = None
        if weights is not None:
            assert weights.index.equals(reactions.index), "weights.index must match reactions.index."
            assert weights.columns.equals(reactions.columns), "weights.columns must match reactions.columns."
            self.weights = weights.values.astype(np.float32)
            logger.debug(f"Per-(user, item) weights provided with shape {self.weights.shape}.")

        ### Create grid
        self.sampling_resolution = sampling_resolution
        self.limits = limits
        self.X = utils.create_meshgrid(self.limits, self.limits, self.sampling_resolution).astype(np.float32)
        self.X_transformed = self.kernel(self.X).astype(np.float32)
        self.n_features = self.X_transformed.shape[1]
        self.parameter_names = [f'beta{i+1}' for i in range(self.n_features)] + ['alpha']
        logger.debug(f"Grid {self.X.shape[0]} points, {self.n_features} features.")

        ### Set prior
        self.prior_variance = prior_variance
        self.log_prior = set_gaussian_prior(self.X, prior_variance, log=True).astype(np.float32)

        ### Initialize other variables
        self.log_likelihoods: np.ndarray | None = None
        self.generator = np.random.Generator(np.random.PCG64(seed=random_state))
        self.fit_logger = FitLogger()
        logger.debug(f"Random state set to {random_state}")

        ### Initialize embedding and models
        self.transformation = transformation
        if pretrained_embedding is not None:
            self.load_embedding(pretrained_embedding)
        elif pca_initialization:
            self.initialize_with_pca()
            self.embedding = self.embedding @ transformation.T
        else:
            self.embedding = self.generator.uniform(limits[0], limits[1], size=(self.number_of_users, 2))
            self.embedding = self.embedding @ transformation.T
            logger.debug("Initialized embedding with random values.")

        if pretrained_models is not None:
            self.load_models(pretrained_models)
        else:
            self.fit_models()

        nan_mask = np.isnan(self.reactions)
        logger.info(
            f"IXPLORE initialized: {self.number_of_users} users × {self.number_of_items} items "
            f"({nan_mask.sum()} missing, {nan_mask.mean()*100:.2f}%), "
            f"grid {self.sampling_resolution}×{self.sampling_resolution}, prior var={self.prior_variance}."
        )
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
        self._recompute_log_predictions()
        logger.debug(f"Pretrained model parameters were given.")

    def initialize_with_pca(self) -> None:
        """Initialize user embeddings via PCA on the reaction matrix."""
        X_imputed = utils.iterative_pca_impute(self.reactions)
        self.embedding = pca_decompose(X_imputed, n_components=2)
        self.embedding = utils.normalize_embedding(self.embedding, limits=self.limits)
        logger.debug(f"Initialized embedding with PCA.")

    def transform_embedding(self, transformation: np.ndarray) -> None:
        """Apply a 2x2 linear transformation to the embedding and refit models."""
        assert transformation.shape == (2,2), "Transformation matrix must be of shape (2,2)."
        self.embedding = utils.normalize_embedding(self.embedding @ transformation.T, limits=self.limits)
        self.fit_models()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def iterate(self, n_iterations: int = 10) -> None:
        """Alternate fit_posteriors and fit_models for n_iterations."""
        for i in range(n_iterations):
            self.fit_posteriors()
            self.fit_models()
            self._log_metrics(prefix=f"Iter {i+1}/{n_iterations} — ", level=logging.DEBUG)
        self._log_metrics(prefix=f"Fit complete after {n_iterations} iterations — ")

    def fit_posteriors(self) -> None:
        """Compute log-likelihoods on the grid for every user and update the embedding."""
        Y = np.nan_to_num(self.reactions, nan=0.0)               # (N, K)
        M = (~np.isnan(self.reactions)).astype(Y.dtype)          # (N, K)
        self.log_likelihoods = self._compute_log_likelihoods(Y, M, weights=self.weights)
        self.embedding = self.get_point_estimates(self._posteriors(), self.X)

    def apply_prior(self, prior_variance: float) -> None:
        """Re-evaluate posteriors and embeddings under a different prior."""
        self.prior_variance = prior_variance
        self.log_prior = set_gaussian_prior(self.X, prior_variance, log=True).astype(np.float32)
        self.embedding = self.get_point_estimates(self._posteriors(), self.X)

    def fit_models(self) -> None:
        """Fit logistic regression models for each item."""
        if not self.use_point_estimates and self.log_likelihoods is not None:
            item_parameters = self._fit_models_posterior()
        else:
            item_parameters = self._fit_models_embedding()
        self.item_parameters = np.vstack(item_parameters)
        self._recompute_log_predictions()

    def predict(
        self,
        positions: np.ndarray,
        items: list[str] | pd.Index | None = None,
        logits: bool = False,
    ) -> np.ndarray:
        """Predict P(Y=1) for each item at each 2D position, shape (N, len(items)).

        If `logits=True`, return the raw logit z = beta . phi(x) + alpha instead of sigma(z).
        """
        if items is None:
            items = self.items
        index = self.items.get_indexer(items)
        if not len(positions):
            return np.array([])
        features = utils.add_ones(self.kernel(positions.reshape(-1, 2)))
        Z = features @ self.item_parameters[index, :].T
        return Z if logits else expit(Z)
    
    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def compute_posteriors(
        self,
        answers: pd.Series | pd.DataFrame,
        weights: pd.Series | pd.DataFrame | None = None,
    ) -> np.ndarray:
        """Compute grid posteriors for one user (Series) or a batch (DataFrame).

        Returns shape (G,) for a Series input, (B, G) for a DataFrame input.
        """
        Y, M, W, single_user = self._align_answers(answers, weights)
        log_lik = self._compute_log_likelihoods(Y, M, W)
        post = self._posteriors(log_lik)
        return post[0] if single_user else post

    def embed(
        self,
        answers: pd.Series | pd.DataFrame,
        weights: pd.Series | pd.DataFrame | None = None,
    ) -> np.ndarray:
        """Embed one user (Series → (2,)) or a batch (DataFrame → (B, 2))."""
        post = self.compute_posteriors(answers, weights)
        if post.ndim == 1:
            return self.get_point_estimates(post[None, :], self.X)[0]
        return self.get_point_estimates(post, self.X)

    def predict_answers(
        self,
        answers: pd.Series | pd.DataFrame,
        weights: pd.Series | pd.DataFrame | None = None,
    ) -> pd.Series | pd.DataFrame:
        """Predict P(Y=1) for every item.

        If `self.use_point_estimates`, predict at each user's embedded position.
        Otherwise, marginalize the per-grid item probabilities over the posterior.
        """
        if self.use_point_estimates:
            position = np.atleast_2d(self.embed(answers, weights))
            predictions = self.predict(position)                       # (B, K)
        else:
            posteriors = np.atleast_2d(self.compute_posteriors(answers, weights))  # (B, G)
            predictions = posteriors @ np.exp(self._log_p1)                  # (B, K)
        if isinstance(answers, pd.Series):
            return pd.Series(predictions[0], name=answers.name, index=self.items)
        return pd.DataFrame(predictions, index=answers.index, columns=self.items)

    def impute_answers(
        self,
        answers: pd.Series | pd.DataFrame,
        weights: pd.Series | pd.DataFrame | None = None,
    ) -> pd.Series | pd.DataFrame:
        """Fill missing answers via `predict_answers`; observed answers are kept intact."""
        predictions = self.predict_answers(answers, weights)
        return answers.combine_first(predictions)

    def sample_answers(
        self,
        answers: pd.Series,
        weights: pd.Series | None = None,
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
            mean_answer = self.impute_answers(answers, weights)
            probs, answer_options = utils.compute_rasch_values(mean_answer, num_options, variance=variance)
            K, Q = probs.shape
            log_probs = np.log(probs)[None, :, :]          # (1, K, Q)
            gumbel_noise = -np.log(-np.log(self.generator.random((num_samples, K, Q))))  # (k, K, Q)
            samples = np.argmax(log_probs + gumbel_noise, axis=1)  # shape: (k, Q)
            samples = answer_options[samples]
        elif method == 'posterior':
            posterior = self.compute_posteriors(answers, weights)
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
        """Evaluate model fit on training data, returning {mae, accuracy, boundary, spread}."""
        predictions = self.predict(self.embedding)
        observed = ~np.isnan(self.reactions)
        true_vals = self.reactions[observed]
        pred_vals = predictions[observed]
        return {
            'mae': compute_mae(true_vals, pred_vals),
            'accuracy': compute_accuracy(true_vals, pred_vals),
            **compute_spread(self.embedding, self.limits, boundary_threshold),
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
        return pd.DataFrame(self._posteriors(), index=self.users)

    def get_reactions(self, user: str) -> tuple[pd.Series, pd.Series | None]:
        """Return a user's observed answers and their per-item weights (or None if unweighted)."""
        i = self.users.get_loc(user)
        answers = pd.Series(self.reactions[i, :], index=self.items, name=user).dropna()
        if self.weights is None:
            return answers, None
        weights = pd.Series(self.weights[i, :], index=self.items, name=user).loc[answers.index]
        return answers, weights

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _log_metrics(self, prefix: str = "", level: int = logging.INFO) -> None:
        """Compute and log evaluation metrics with an optional prefix."""
        metrics = self.evaluate()
        self.fit_logger.log(metrics)
        body = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        logger.log(level, f"{prefix}{body}")

    def _fit_models_embedding(self) -> list[np.ndarray]:
        """Fit logistic regression on point embeddings with soft labels."""
        item_parameters = []
        for k in range(self.number_of_items):
            mask = ~np.isnan(self.reactions[:, k])
            train_data = self.kernel(self.embedding[mask])  # (N_k, D)
            train_labels = self.reactions[mask, k]                     # (N_k,)
            params = fit_logistic_newton(train_data, train_labels, regularization=self.model_regularization)
            item_parameters.append(params)
        return item_parameters

    def _fit_models_posterior(self) -> list[np.ndarray]:
        """Fit logistic regression using aggregated posteriors as sample weights."""
        posteriors = self._posteriors()
        item_parameters = []
        for k in range(self.number_of_items):
            mask = ~np.isnan(self.reactions[:, k])
            posteriors_k = posteriors[mask]               # (N_k, G)
            labels_k = self.reactions[mask, k]            # (N_k,)
            W_g = posteriors_k.sum(axis=0)                # (G,)
            A_g = posteriors_k.T @ labels_k               # (G,)
            y_eff = np.divide(A_g, W_g, out=np.zeros_like(A_g), where=W_g > 0)
            params = fit_logistic_newton(self.X_transformed, y_eff, sample_weight=W_g, regularization=self.model_regularization)
            item_parameters.append(params)
        return item_parameters

    def _align_answers(
        self,
        answers: pd.Series | pd.DataFrame,
        weights: pd.Series | pd.DataFrame | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, bool]:
        """Normalize Series/DataFrame inputs into aligned (B, K) arrays.

        Returns (Y, M, W, single_user). Y is the answer matrix (zeros where unobserved),
        M is the observation mask, W is the weight matrix or None, and single_user is
        True if the input was a Series (so callers can squeeze the leading axis).
        """
        single_user = isinstance(answers, pd.Series)
        if single_user:
            answers = answers.to_frame().T
            if weights is not None:
                weights = weights.to_frame().T
        values = answers.reindex(columns=self.items).values.astype(np.float32)
        finite = ~np.isnan(values)
        if finite.any():
            assert ((values[finite] >= 0) & (values[finite] <= 1)).all(), "answers must lie in [0, 1]."
        Y = np.where(finite, values, 0.0).astype(np.float32)
        M = finite.astype(np.float32)
        W = None
        if weights is not None:
            W = weights.reindex(columns=self.items).fillna(1.0).values.astype(np.float32)
        return Y, M, W, single_user

    def _compute_log_likelihoods(
        self,
        Y: np.ndarray,
        M: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """Vectorized BCE log-likelihood for an (N, K) input, shape (N, G).

        log L_n(g) = sum_k m_nk * w_nk * [ y_nk log p_k(g) + (1 - y_nk) log(1 - p_k(g)) ]
        """
        W = M
        if weights is not None:
            assert not np.isnan(weights).any(), "NaN in weights array is not allowed."
            assert (weights >= 0).all(), "Negative weights are not allowed."
            W = M * weights
        if self.scale_weights:
            n_eff = W.sum(axis=1, keepdims=True)                 # (N, 1)
            scale = np.where(n_eff > 0, self.number_of_items / np.maximum(n_eff, 1), 0.0)
            W = W * scale
        return (W * Y) @ self._log_p1.T + (W * (1.0 - Y)) @ self._log_p0.T

    def _recompute_log_predictions(self) -> None:
        """Recompute the per-grid-cell log-prediction tables from item_parameters."""
        Z = self.predict(self.X, logits=True)            # (G, K) logits
        self._log_p1 = log_expit(Z).astype(np.float32)    # (G, K)  log p_k(g)
        self._log_p0 = log_expit(-Z).astype(np.float32) # (G, K)  log (1 - p_k(g))

    def _posteriors(self, log_lik: np.ndarray | None = None) -> np.ndarray:
        """Softmax-normalize (log_lik + log_prior) along the grid axis, shape (B, G)."""
        log_lik = self.log_likelihoods if log_lik is None else log_lik
        assert log_lik is not None, "fit_posteriors must be called first, or pass log_lik."
        log_post = (log_lik + self.log_prior).astype(np.float64)
        log_post -= log_post.max(axis=1, keepdims=True)
        post = np.exp(log_post)
        return post / post.sum(axis=1, keepdims=True)
