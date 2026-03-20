from __future__ import annotations

from typing import Callable, Literal

from scipy.special import expit
import pandas as pd
import numpy as np

from .logger import logger, FitLogger
from .optimization import fit_logistic_newton, pca_decompose
from .likelihood import l1_log_likelihood
from .prior import set_gaussian_prior
from . import utils

class IXPLORE:
    def __init__(
        self,
        reactions: pd.DataFrame,
        prior_variance: float = 1.0,
        sampling_resolution: int = 100,
        xlimits: tuple[float, float] = (-1, 1),
        ylimits: tuple[float, float] = (-1, 1),
        pretrained_models: pd.DataFrame = None,
        pretrained_embedding: pd.DataFrame = None,
        pca_initialization: bool = True,
        random_state: int = 0,
        transformation: np.ndarray = np.identity(2),
        kernel: Callable[[np.ndarray], np.ndarray] | None = None,
        lr_regularization: float = 1e-6,
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
        xlimits: tuple
            The limits of the x-axis of the 2D space.
        ylimits: tuple
            The limits of the y-axis of the 2D space.
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
        lr_regularization: float
            L2 regularization strength for logistic regression. Default is 1e-6.
        """

        ### Feature transform and regularization
        self.kernel = kernel if kernel is not None else lambda X: X
        self.lr_regularization = lr_regularization
        self.log_likelihood_fn = l1_log_likelihood

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
        self.limits = (xlimits[0], xlimits[1], ylimits[0], ylimits[1])
        self.X = utils.create_meshgrid(self.limits, self.sampling_resolution)
        self.X_transformed = self.kernel(self.X)
        self.n_features = self.X_transformed.shape[1]
        self.parameters = [f'beta{i+1}' for i in range(self.n_features)] + ['alpha']
        logger.info(f"Grid created with resolution {self.sampling_resolution}x{self.sampling_resolution}, total {self.X.shape[0]} points")
        logger.info(f"Feature dimensions: {self.n_features} (from 2D input)")

        ### Set prior
        self.prior_variance = prior_variance
        self.log_prior_X = set_gaussian_prior(self.X, prior_variance, log=True)
        logger.info(f"Gaussian prior set with covariance diagonal entry {self.prior_variance}")

        ### Initialize other variables
        self.item_parameters: np.ndarray | None = None
        self.likelihood_X: np.ndarray | None = None
        self.posteriors: np.ndarray | None = None
        self.embedding: np.ndarray | None = None
        self.generator = np.random.Generator(np.random.PCG64(seed=random_state))
        self.fit_logger = FitLogger()
        logger.info(f"Random state set to {random_state}")

        ### Initialize embedding and models
        self.transformation = transformation
        if pretrained_embedding is not None:
            self.load_embedding(pretrained_embedding)
            logger.info("Used pretrained embedding.")
        elif pca_initialization:
            self.initialize_with_PCA()
            self.embedding = utils.normalize_embedding(self.embedding @ transformation.T)
            logger.info("Initialized embedding with PCA.")
        else:
            self.embedding = self.generator.uniform(-1, 1,(self.number_of_users, 2))
            self.embedding = utils.normalize_embedding(self.embedding @ transformation.T)
            logger.info("Initialized embedding with random values.")

        if pretrained_models is not None:
            self.load_models(pretrained_models)
            logger.info("Used pretrained model parameters.")
        else:
            self.fit_models()
            logger.info("Fitted model parameters from embedding.")

        mae, acc, boundary = self.evaluate()
        self.fit_logger.log(mae, acc, boundary)
        logger.info(f"Initial MAE: {mae:.4f}, Initial accuracy: {acc:.4f}, Boundary fraction: {boundary:.4f}")

    def __str__(self) -> str:
        return 'IXPLORE'

    def load_embedding(self, embedding: pd.DataFrame) -> None:
        """Load pretrained user embeddings from a DataFrame."""
        assert embedding.index.astype(str).equals(self.users), "User indices in the pretrained embedding do not match the user indices in the data."
        assert embedding.columns.tolist() == ['x', 'y'], "Columns in the pretrained embedding must be ['x', 'y']."
        self.embedding = embedding.values
        logger.debug(f"Pretrained embedding was given.")

    def load_models(self, parameters: pd.DataFrame) -> None:
        """Load pretrained model parameters."""
        assert parameters.index.astype(str).equals(self.items), "Items in the pretrained model parameters do not match the items in the data."
        assert parameters.columns.tolist() == self.parameters, "Columns in the pretrained model parameters do not match the expected columns for the XPLORE model."
        self.item_parameters = parameters.values
        self.likelihood_X = self.predict(self.X)
        logger.debug(f"Pretrained model parameters were given.")

    def initialize_with_PCA(self) -> None:
        """Initialize user embeddings using PCA on the reaction data and center them."""
        X_imputed = utils.mean_impute(self.reactions)
        self.embedding = pca_decompose(X_imputed, n_components=2)
        self.embedding = utils.normalize_embedding(self.embedding)
        logger.debug(f"Initialized embedding with PCA.")

    def get_embedding(self) -> pd.DataFrame:
        """Get the current user embeddings."""
        return pd.DataFrame(self.embedding, index=self.users, columns=['x','y'])

    def iterate(self, n_iterations: int = 10) -> None:
        """Perform a number of iterations of fitting posteriors and models."""
        for i in range(n_iterations):
            logger.info(f"Iteration {i+1}/{n_iterations}")
            self.fit_posteriors()
            self.fit_models()
            mae, acc, boundary = self.evaluate()
            self.fit_logger.log(mae, acc, boundary)
            logger.info(f"Fit MAE: {mae:.4f}, Fit accuracy: {acc:.4f}, Boundary fraction: {boundary:.4f}")

    def fit_posteriors(self, parallelize: bool = False) -> None:
        """Compute posteriors on X-grid for every user in train set (self.reactions)."""
        ### TODO: parallelize this
        posteriors = []
        for n in self.users:
            i = self.users.get_loc(n)
            user = self.reactions[i, :]
            mask = ~np.isnan(user)
            answers_values = user[mask]
            answers_indices = np.where(mask)[0]
            posteriors.append(self._posterior_X(answers_values, answers_indices))
        self.posteriors = np.array(posteriors)
        self.embedding = self.posteriors2coordinates(self.posteriors)

    def get_posteriors(self) -> pd.DataFrame:
        """Get the current posteriors on X-grid for every user in train set (self.reactions)."""
        return pd.DataFrame(self.posteriors, index=self.users)

    def transform_embedding(self, transformation: np.ndarray) -> None:
        """Apply a linear transformation to the current embedding."""
        assert transformation.shape == (2,2), "Transformation matrix must be of shape (2,2)."
        self.embedding = utils.normalize_embedding(self.embedding @ transformation.T)
        self.fit_models(use_posteriors=False)

    def fit_models(self, use_posteriors: bool = True) -> None:
        """Fit logistic regression models for each item.

        Parameters
        ----------
        use_posteriors : bool
            If True and posteriors are available, uses aggregated posteriors
            (per-grid-point weights and effective soft labels) for uncertainty-aware
            fitting. If False or posteriors are not available, falls back to fitting
            on point embeddings with soft labels.
        """
        if use_posteriors and self.posteriors is not None:
            item_parameters = self._fit_models_posterior()
        else:
            item_parameters = self._fit_models_embedding()
        self.item_parameters = np.vstack(item_parameters)
        self.likelihood_X = self.predict(self.X)

    def _fit_models_embedding(self) -> None:
        """Fit logistic regression on point embeddings with soft labels."""
        item_parameters = []
        for k in range(self.number_of_items):
            mask = ~np.isnan(self.reactions[:, k])
            train_data = self.kernel(self.embedding[mask])  # (N_k, D)
            train_labels = self.reactions[mask, k]                     # (N_k,)
            params = fit_logistic_newton(train_data, train_labels, regularization=self.lr_regularization)
            item_parameters.append(params)
        return item_parameters

    def _fit_models_posterior(self) -> None:
        """Fit logistic regression using aggregated posteriors."""
        item_parameters = []
        for k in range(self.number_of_items):
            mask = ~np.isnan(self.reactions[:, k])
            posteriors_k = self.posteriors[mask]          # (N_k, G)
            labels_k = self.reactions[mask, k]            # (N_k,)
            W_g = posteriors_k.sum(axis=0)                # (G,)
            A_g = posteriors_k.T @ labels_k               # (G,)
            y_eff = np.divide(A_g, W_g, out=np.zeros_like(A_g), where=W_g > 0)
            params = fit_logistic_newton(self.X_transformed, y_eff, sample_weight=W_g, regularization=self.lr_regularization)
            item_parameters.append(params)
        return item_parameters

    def get_item_parameters(self) -> pd.DataFrame:
        """Get the current item parameters."""
        return pd.DataFrame(self.item_parameters, index=self.items, columns=self.parameters)

    def _posterior_X(self, answer_values: np.ndarray, answer_index: np.ndarray) -> np.ndarray:
        """Compute posterior distribution over X based on the given answers in numpy format.

        Parameters
        ----------
        answer_values: np.array
            The given answers as numpy array.
        answer_index: np.array
            The numpy indices of the given answers.

        Returns
        -------
        np.array
            The normalized posterior distribution of shape (sampling_resolution*sampling_resolution,)
        """
        mask = ~np.isnan(answer_values.astype(float))
        answer_values = answer_values[mask]
        answer_index = answer_index[mask]
        assert self.likelihood_X is not None, "Likelihoods must be computed before computing posterior."
        likelihood = self.likelihood_X[:, answer_index]
        # Work in log-space to avoid underflow when many items are multiplied
        log_likelihood = self.log_likelihood_fn(answer_values.reshape(-1), likelihood)
        log_posterior = log_likelihood + self.log_prior_X
        log_posterior -= log_posterior.max()  # shift for numerical stability
        posterior = np.exp(log_posterior)
        return posterior / posterior.sum()

    def posterior_X(self, answers: pd.Series) -> np.ndarray:
        """Compute the posterior distribution over X based on the given answers in pandas format.

        Parameters
        ----------
        answers: pd.Series
            The given answers with index as item names and values as answers.

        Returns
        -------
        np.array
            The normalized posterior distribution of shape (sampling_resolution*sampling_resolution,)
        """
        answer_values = answers.dropna().values
        answer_indices = self.items.get_indexer(answers.dropna().index)
        logger.debug("Answer values: %s, Answer indices: %s", answer_values, answer_indices)
        return self._posterior_X(answer_values, answer_indices)

    def sample_pseudo_answers(
        self,
        answers: pd.Series,
        method: Literal["rasch", "posterior", "random"] = "posterior",
        num_samples: int = 1000,
        num_options: int = 5,
        variance: float = 0.1,
    ) -> np.ndarray:   
        """Sample pseudo answers for a user with given answers.

        Parameters
        ----------
        answers: pd.Series
            The given answers with index as item names and values as answers.
        method: string
            The method to draw samples from the posterior distribution. Options are 'rasch', 'posterior', or 'random'.
        num_samples: int
            The number of samples to draw
        num_options: int
            The number of possible answer options (only for 'rasch' method)
        variance: float
            The variance of the normal distributions (only for 'rasch' method)

        Returns
        -------
        np.array
            The sampled answers of shape (num_samples, number_of_items)
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
            answers_values = answers.dropna().values
            answers_indices = self.items.get_indexer(answers.dropna().index)
            posterior = self._posterior_X(answers_values, answers_indices)
            samples = self.generator.choice(len(posterior), size=num_samples, p=posterior)
            samples = self.X[samples]
            samples = self.predict(samples)
        else:
            samples = self.generator.random((num_samples, self.number_of_items))
        return samples

    def predict(
        self,
        params: np.ndarray,
        items: list[str] | pd.Index | None = None,
    ) -> np.ndarray:
        """Compute predictions for given 2D positions. Applies the kernel transform internally.

        Parameters
        ----------
        params: np.array
            The 2D positions to predict of shape (N, 2) where N is the number of positions.
        items: list, optional
            The items to predict. If None, predict all items. Default is None.

        Returns
        -------
        np.array
            The predicted probabilities of shape (N, len(items)).
        """
        if items is None:
            items = self.items
        index = self.items.get_indexer(items)
        if not len(params):
            return np.array([])
        assert self.item_parameters is not None, "Item parameters must be fitted before predicting."
        params = utils.add_ones(self.kernel(params.reshape(-1, 2)))
        return_value = expit(params@self.item_parameters[index,:].T)
        return return_value

    def posteriors2coordinates(self, posteriors: np.ndarray) -> np.ndarray:
        """Convert posteriors on X-grid to coordinates in 2D space.

        Parameters
        ----------
        posteriors: np.array
            The posteriors on the X-grid of shape (number_of_users, sampling_resolution*sampling_resolution).

        Returns
        -------
        np.array
            The predicted coordinates in 2D space of shape (number_of_users, 2).
        """
        p = posteriors.reshape(-1, self.sampling_resolution * self.sampling_resolution)
        return p @ self.X

    def embed_new_user(self, answers: pd.Series) -> np.ndarray:
        """Embed a single user with given answers.

        Parameters
        ----------
        answers: pd.Series
            The given answers with index as item names and values as answers.

        Returns
        -------
        np.array
            The predicted coordinates in 2D space as (x, y).
        """
        return self.posteriors2coordinates(self.posterior_X(answers))[0]

    def impute_remaining_answers(self, answers: pd.Series) -> pd.Series:
        """Impute answers to all items for a user with given answers.

        Uses Bayesian marginalisation to predict P(Y=1 | observed answers) for
        all items, then fills in only the missing values while keeping observed
        answers intact.

        Parameters
        ----------
        answers: pd.Series
            The given answers with index as item names and values as answers.

        Returns
        -------
        pd.Series
            The imputed answers with index as item names and values as answers.
        """
        P_X_Yi  = self.posterior_X(answers).reshape(-1,1)
        P_Yn1_X = self.likelihood_X
        P_XYn1_Yi = P_Yn1_X * P_X_Yi                                        # (grid_size*grid_size, K)
        P_Yn1_Yi  = P_XYn1_Yi.sum(axis=0)                                   # (K,)
        predictions = pd.Series(P_Yn1_Yi, name=answers.name, index=self.items)
        answers = pd.Series(index=self.items, dtype=float).fillna(answers)
        return answers.fillna(predictions)

    def evaluate(self) -> tuple[float, float, float]:
        """Evaluate model fit on training data using MAE, accuracy, and boundary fraction.

        Returns
        -------
        float
            Mean absolute error of the model predictions.
        float
            Accuracy of the model predictions.
        float
            Fraction of predictions near the decision boundary (within 0.05 of 0.5).
        """
        assert self.embedding is not None, "Embedding must be initialized before evaluating."
        predictions = pd.DataFrame(self.predict(self.embedding),
                                   index=self.users,
                                   columns=self.items)
        fit_accuracy = 1 - np.abs(self.reactions.round() - predictions.round()).mean().mean()
        fit_mae = np.mean(np.abs(self.reactions - predictions))
        xmin, xmax, ymin, ymax = self.limits
        near_border = (
            (self.embedding[:, 0] - xmin < 0.05) |
            (xmax - self.embedding[:, 0] < 0.05) |
            (self.embedding[:, 1] - ymin < 0.05) |
            (ymax - self.embedding[:, 1] < 0.05)
        )
        boundary_fraction = near_border.mean()
        return fit_mae, fit_accuracy, boundary_fraction
