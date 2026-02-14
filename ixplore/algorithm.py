from __future__ import annotations

from typing import Literal

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from scipy.stats import multivariate_normal
from scipy.special import expit
import pandas as pd
import numpy as np

from .logger import logger
from .utils import extract_parameters, binarize, add_ones, create_meshgrid, compute_rasch_values


class IXPLORE:
    def __init__(
        self,
        reactions: pd.DataFrame,
        prior_mean: np.ndarray = np.array([0, 0]),
        prior_cov: np.ndarray = np.array([[.1, 0], [0, .1]]),
        sampling_resolution: int = 200,
        xlimits: tuple[float, float] = (-1, 1),
        ylimits: tuple[float, float] = (-1, 1),
        pretrained_models: str = '',
        pretrained_embedding: str = '',
        pca_initialization: bool = True,
        random_state: int = 0,
        transformation: np.ndarray = np.identity(2),
    ) -> None:
        """Initialize the iXpLoRE model.

        Parameters
        ----------
        reactions: pd.DataFrame
            The user-item reaction matrix with users as index and items as columns. Can contain missing values (NaN).
        prior_mean: np.ndarray
            The mean of the prior distribution over the 2D space.
        prior_cov: np.ndarray
            The covariance of the prior distribution over the 2D space. Must be a 2x2 matrix.
        sampling_resolution: int
            The resolution of the grid over the 2D space.
        xlimits: tuple
            The limits of the x-axis of the 2D space.
        ylimits: tuple
            The limits of the y-axis of the 2D space.
        pretrained_models: str 
            The path to the pretrained models. If provided, the model parameters will be loaded from this file.
        pretrained_embedding: str
            The path to the pretrained embedding. If provided, the user embeddings will be loaded from this file.
        pca_initialization: bool
            Whether to initialize the embedding with PCA. If False, the embedding will be initialized with random values.
        random_state: int
            The random state for reproducibility. 
        transformation: np.ndarray
            A 2x2 transformation matrix to apply to the embedding. Default is the identity matrix
        """

        ### Store data as numpy arrays
        self.reactions = reactions.values
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
        self.X = create_meshgrid(self.limits, self.sampling_resolution) 
        logger.info(f"Grid created with resolution {self.sampling_resolution}x{self.sampling_resolution}, total {self.X.shape[0]} points")

        ### Set prior
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        self.set_prior(prior_mean, prior_cov) 
        logger.info(f"Prior set with mean {self.prior_mean} and covariance {self.prior_cov.flatten()}")

        ### Initialize other variables
        self.models: dict[str, LogisticRegression] = {}
        self.item_parameters: np.ndarray | None = None
        self.likelihood_X: np.ndarray | None = None
        self.posteriors: np.ndarray | None = None
        self.embedding: np.ndarray | None = None
        self.generator = np.random.Generator(np.random.PCG64(seed=random_state))
        logger.info(f"Random state set to {random_state}")

        ### Initialize embedding and models
        self.transformation = transformation
        if pretrained_embedding:
            self.load_embedding(pretrained_embedding)
            logger.info("Used pretrained embedding.")
        elif pca_initialization:
            self.initialize_with_PCA()
            self.embedding = self.embedding @ transformation.T
            self.normalize_embedding()
            logger.info("Initialized embedding with PCA.")
        else:
            self.embedding = self.generator.uniform(-1, 1,(self.number_of_users, 2))    # (N, 2)
            self.embedding = self.embedding @ transformation.T                          # (N, 2) @ (2, 2) -> (N, 2)
            self.normalize_embedding()
            logger.info("Initialized embedding with random values.")

        if pretrained_models:
            self.load_models(pretrained_models)
            logger.info("Used pretrained model parameters.")
        else:
            self.fit_models()
            logger.info("Fitted model parameters from embedding.")

    def __str__(self) -> str:
        return 'iXpLoRE'

    def set_prior(self, prior_mean: np.ndarray, prior_cov: np.ndarray) -> None:
        """Set the prior distribution over the 2D space."""
        self.prior = multivariate_normal(prior_mean, prior_cov)                 # scipy obj
        self.prior_X = self.prior.pdf(self.X) / self.prior.pdf(self.X).sum()    # (grid_size*grid_size,)

    def load_embedding(self, path: str) -> None:
        """Load pretrained user embeddings from a CSV file."""
        dataframe = pd.read_csv(path, index_col=0)
        self.embedding = dataframe.values
        assert dataframe.index.astype(str).equals(self.users), "User indices in the pretrained embedding do not match the user indices in the data."
        assert dataframe.columns.tolist() == ['x', 'y'], "Columns in the pretrained embedding must be ['x', 'y']."
        logger.info(f"Pretrained embedding loaded from {path}")

    def load_models(self, path: str) -> None:
        """Load pretrained model parameters from a CSV file."""
        self.item_parameters = pd.read_csv(path, index_col=0).rename(index=str)
        assert self.item_parameters.shape[0] == self.number_of_items, "Number of items in the pretrained model parameters does not match the number of items in the data."
        assert self.item_parameters.shape[1] == 3, "Number of columns in the pretrained model parameters does not match the expected number of columns for the XPLORE model."
        self.likelihood_X = self.predict(self.X)
        logger.info(f"Pretrained model parameters loaded from {path}")

    def initialize_with_PCA(self) -> None:
        """Initialize user embeddings using PCA on the reaction data and center them."""
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(self.reactions)
        self.embedding = PCA(n_components=2).fit_transform(X_imputed)
        self.normalize_embedding()

    def get_embedding(self) -> pd.DataFrame:
        """Get the current user embeddings."""
        return pd.DataFrame(self.embedding, index=self.users, columns=['x','y'])

    def normalize_embedding(self, scaling: float = 1.05) -> None:
        """Center and scale the embedding to fit within the defined limits."""
        assert self.embedding is not None, "Embedding must be initialized before normalizing."
        centroid = (self.embedding.max(axis=0) + self.embedding.min(axis=0))/2
        self.embedding -= centroid
        max_extent = np.abs(self.embedding).max(axis=0)*scaling
        self.embedding /= max_extent

    def iterate(self, n_iterations: int = 10) -> None:
        """Perform a number of iterations of fitting posteriors and models."""
        for i in range(n_iterations):
            logger.info(f"Iteration {i+1}/{n_iterations}")
            self.fit_posteriors()
            self.normalize_embedding()
            self.fit_models()
            mae, acc = self.evaluate()
            logger.info(f"Fit MAE: {mae:.4f}, Fit accuracy: {acc:.4f}")

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
        self.update_embedding()

    def get_posteriors(self) -> pd.DataFrame:
        """Get the current posteriors on X-grid for every user in train set (self.reactions)."""
        return pd.DataFrame(self.posteriors, index=self.users)

    def update_embedding(self) -> None:
        """Update user embeddings based on current posteriors."""
        assert self.posteriors is not None, "Posteriors must be computed before updating embedding."
        self.embedding = self.posteriors2coordinates(self.posteriors)

    def transform_embedding(self, transformation: np.ndarray) -> None:
        """Apply a linear transformation to the current embedding."""
        assert transformation.shape == (2,2), "Transformation matrix must be of shape (2,2)."
        self.embedding = self.embedding @ transformation.T
        self.normalize_embedding()
        self.fit_models()

    def fit_models(self) -> None:
        """Fit logistic regression models for each item based on current embeddings."""
        assert self.embedding is not None, "Embedding must be initialized before fitting models."
        ### TODO: parallelize this
        models: dict[str, LogisticRegression] = {}
        for item in self.items:
            model = LogisticRegression(random_state=0)
            i = self.items.get_loc(item)
            mask = ~np.isnan(self.reactions[:, i])
            train_data   = self.embedding[mask]
            train_labels = self.reactions[mask, i]
            train_labels = binarize(train_labels, self.generator)
            assert len(np.unique(train_labels)) == 2, f"No model fitted for Feature {item}."
            model.fit(train_data, train_labels)
            models[item] = model
        self.models = models
        self.update_likelihoods()

    def update_likelihoods(self) -> None:
        """Update likelihoods on X-grid based on current models."""
        self.item_parameters = np.vstack([extract_parameters(model) for model in self.models.values()])
        self.likelihood_X = self.predict(self.X)

    def get_likelihoods(self) -> pd.DataFrame:
        """Get the current likelihoods on X-grid for every item."""
        return pd.DataFrame(self.likelihood_X, columns=self.items)

    def get_item_parameters(self) -> pd.DataFrame:
        """Get the current item parameters."""
        return pd.DataFrame(self.item_parameters, index=self.items, columns=['beta1', 'beta2', 'alpha'])

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
        posterior = np.prod(1 - np.abs(answer_values.reshape(-1) - likelihood), axis=1)
        posterior = posterior * self.prior_X
        return posterior/ posterior.sum()

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
            probs, answer_options = compute_rasch_values(mean_answer, num_options, variance=variance)
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
        """Compute predictions for given positions outside the X-grid.

        Parameters
        ----------
        params: np.array
            The positions to predict of shape (N, 2) where N is the number of positions to predict.
        items: list, optional
            The items to predict. If None, predict all items. Default is None.

        Returns
        -------
        np.array
            The predicted probabilities of shape (N, len(queries)).
        """
        if items is None:
            items = self.items
        index = self.items.get_indexer(items)
        if not len(params):
            return np.array([])
        assert self.item_parameters is not None, "Item parameters must be fitted before predicting."
        params = add_ones(params.reshape(-1,2))
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
        maxidxs = np.argmax(posteriors.reshape(-1,self.sampling_resolution*self.sampling_resolution), axis=1)
        return self.X[maxidxs]

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

    def predict_all_answers(self, answers: pd.Series) -> pd.Series:
        """Predict answers to all items for a single user with given answers.

        Parameters
        ----------
        answers: pd.Series
            The given answers with index as item names and values as answers.

        Returns
        -------
        pd.Series
            The predicted answers with index as item names and values as answers.
        """
        P_X_Yi  = self.posterior_X(answers).reshape(-1,1) 
        P_Yn1_X = self.likelihood_X
        P_XYn1_Yi = P_Yn1_X * P_X_Yi                                        # (grid_size*grid_size, K)
        P_Yn1_Yi  = P_XYn1_Yi.sum(axis=0)                                   # (K,)
        return pd.Series(P_Yn1_Yi, name=answers.name, index=self.items)

    def impute_remaining_answers(self, answers: pd.Series) -> pd.Series:
        """Impute answers to all items for a user with given answers.

        Parameters
        ----------
        answers: pd.Series
            The given answers with index as item names and values as answers.

        Returns
        -------
        pd.Series
            The imputed answers with index as item names and values as answers.
        """
        answers = pd.Series(index=self.items, dtype=float).fillna(answers)
        return answers.fillna(self.predict_all_answers(answers))

    def evaluate(self) -> tuple[float, float]:
        """Evaluate model fit on training data using MAE and accuracy.

        Returns
        -------
        float
            Mean absolute error of the model predictions.
        float
            Accuracy of the model predictions.
        """
        assert self.embedding is not None, "Embedding must be initialized before evaluating."
        predictions = pd.DataFrame(self.predict(self.embedding), 
                                   index=self.users, 
                                   columns=self.items)
        fit_accuracy = 1 - np.abs(self.reactions.round() - predictions.round()).mean().mean()
        fit_mae = np.mean(np.abs(self.reactions - predictions))
        return fit_mae, fit_accuracy
