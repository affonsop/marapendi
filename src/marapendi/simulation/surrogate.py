"""
Neural-network surrogate for expensive ``model_fn`` callables.

``SurrogateModel`` wraps a :class:`~marapendi.simulation.estimation.ParameterEstimation`
instance (the *base model*) and learns the mapping from the normalised
parameter vector ``theta`` to the model output ``y = model_fn(params)``.

Workflow
--------
1. ``generate_training_data`` samples the unknown-parameter space (Latin
   hypercube) and evaluates the physics-based ``model_fn`` for each sample.
2. ``fit`` trains a multi-layer-perceptron regressor (with input/output
   standardisation) on the resulting ``(theta, y)`` pairs.
3. ``predict`` / ``__call__`` / ``get_model_fn`` expose the trained surrogate
   as a drop-in replacement for ``model_fn``, including support for slicing
   the output vector down to a subset of cases (as required by
   :class:`~marapendi.estimation.cross_validation.CrossValidation`).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import qmc

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None

__all__ = ['SurrogateModel']


class SurrogateModel:
    """Multi-layer-perceptron surrogate for a ``model_fn(params) -> np.ndarray`` callable.

    Parameters
    ----------
    base_model : ParameterEstimation
        Provides ``unknown_parameters``, ``params`` (nominal values for fixed
        parameters), and the normalisation helpers ``p_to_theta`` /
        ``params_from_theta`` used to sample and convert parameter vectors.
    case_sizes : dict
        Ordered mapping ``case -> n_outputs`` giving the number of entries
        each case contributes to ``model_fn``'s output vector, in the same
        order as the case list used to generate training data (e.g.
        ``{case: 2 * len(exp_current[case]) for case in full_case_list}``
        for a model that returns ``[V_cell..., HFR...]`` per case).
    hidden_layer_sizes : tuple[int, ...]
        Hidden-layer sizes forwarded to ``sklearn.neural_network.MLPRegressor``.
    **mlp_kwargs
        Additional keyword arguments forwarded to ``MLPRegressor``.
    """

    def __init__(
        self,
        base_model,
        case_sizes: dict,
        hidden_layer_sizes: tuple = (128, 128, 64),
        random_state: int = 0,
        max_iter: int = 2000,
        early_stopping: bool = True,
        **mlp_kwargs,
    ):
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler

        self.base_model = base_model
        self.case_sizes = dict(case_sizes)

        offsets = {}
        offset = 0
        for case, size in self.case_sizes.items():
            offsets[case] = (offset, offset + size)
            offset += size
        self.offsets = offsets
        self.n_outputs = offset
        self.n_params = len(base_model.unknown_parameters)

        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            random_state=random_state,
            max_iter=max_iter,
            early_stopping=early_stopping,
            **mlp_kwargs,
        )
        self.is_fitted = False

        self.theta_ = None
        self.Y_ = None
        self.theta_test_ = None
        self.Y_test_ = None

    # ------------------------------------------------------------------
    # Sampling / training-data generation
    # ------------------------------------------------------------------

    def sample_theta(self, n_samples: int, seed: int | None = None) -> np.ndarray:
        """Latin-hypercube samples in normalised ``[0, 1]^n_params`` space."""
        sampler = qmc.LatinHypercube(d=self.n_params, seed=seed)
        return sampler.random(n_samples)

    def generate_training_data(
        self,
        model_fn=None,
        n_samples: int = 500,
        seed: int | None = None,
        theta: np.ndarray | None = None,
        print_progress: bool = False,
        append: bool = False,
    ):
        """Sample the parameter space and evaluate ``model_fn`` for each sample.

        Parameters
        ----------
        model_fn : callable, optional
            Defaults to ``self.base_model.model_fn``.
        n_samples : int
            Number of Latin-hypercube samples (ignored if ``theta`` given).
        theta : np.ndarray, shape (n_samples, n_params), optional
            Explicit normalised parameter samples to evaluate.
        print_progress : bool
            Print progress every ~5%.
        append : bool
            If ``True``, append to any previously generated training data
            instead of overwriting it.

        Returns
        -------
        theta, Y : np.ndarray
            Valid (finite-output) samples and their model outputs.
        """
        model_fn = model_fn or self.base_model.model_fn
        if theta is None:
            theta = self.sample_theta(n_samples, seed=seed)
        n_samples = len(theta)

        Y = np.full((n_samples, self.n_outputs), np.nan)
        report_every = max(1, n_samples // 20)
        for i, th in enumerate(theta):
            params = self.base_model.params_from_theta(th)
            try:
                y = np.asarray(model_fn(params), dtype=float)
                if y.shape == (self.n_outputs,) and np.all(np.isfinite(y)):
                    Y[i] = y
            except Exception:
                pass
            if print_progress and ((i + 1) % report_every == 0 or i + 1 == n_samples):
                print(f'  sample {i + 1}/{n_samples}')

        valid = np.all(np.isfinite(Y), axis=1)
        theta, Y = theta[valid], Y[valid]
        if print_progress:
            print(f'  -> {valid.sum()}/{n_samples} valid samples')

        if append and self.theta_ is not None:
            theta = np.concatenate([self.theta_, theta])
            Y = np.concatenate([self.Y_, Y])

        self.theta_, self.Y_ = theta, Y
        return theta, Y

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, theta=None, Y=None, test_size: float = 0.2, random_state: int = 0):
        """Fit the MLP on ``(theta, Y)``, holding out ``test_size`` for validation."""
        from sklearn.model_selection import train_test_split

        theta = theta if theta is not None else self.theta_
        Y = Y if Y is not None else self.Y_
        if theta is None or len(theta) == 0:
            raise ValueError('No training data available; call generate_training_data first.')

        theta_train, theta_test, Y_train, Y_test = train_test_split(
            theta, Y, test_size=test_size, random_state=random_state
        )

        Xs = self.x_scaler.fit_transform(theta_train)
        Ys = self.y_scaler.fit_transform(Y_train)
        self.model.fit(Xs, Ys)
        self.is_fitted = True

        self.theta_test_, self.Y_test_ = theta_test, Y_test
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_theta(self, theta: np.ndarray) -> np.ndarray:
        """Predict ``Y`` for normalised parameter samples ``theta``."""
        if not self.is_fitted:
            raise RuntimeError('SurrogateModel is not fitted yet.')
        theta = np.atleast_2d(theta)
        Xs = self.x_scaler.transform(theta)
        Ys = self.model.predict(Xs)
        return self.y_scaler.inverse_transform(np.atleast_2d(Ys))

    def predict(self, params: dict) -> np.ndarray:
        """Predict the full output vector ``y`` for a parameter dict ``params``."""
        theta = self.base_model.p_to_theta(
            np.array([params[up.key] for up in self.base_model.unknown_parameters])
        )
        return self.predict_theta(theta)[0]

    def __call__(self, params: dict, case_list=None) -> np.ndarray:
        y_full = self.predict(params)
        if case_list is None:
            return y_full
        return np.concatenate([y_full[slice(*self.offsets[case])] for case in case_list])

    def get_model_fn(self, case_list=None):
        """Return a ``model_fn(params) -> np.ndarray`` callable for ``case_list``.

        ``case_list`` defaults to all cases (the full output vector).
        """
        def model_fn(params):
            return self(params, case_list=case_list)
        return model_fn

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, theta_test=None, Y_test=None):
        """Evaluate the surrogate on held-out data.

        Returns
        -------
        Y_pred, Y_test, rmse : np.ndarray
            Predictions, ground truth, and per-output RMSE.
        """
        theta_test = theta_test if theta_test is not None else self.theta_test_
        Y_test = Y_test if Y_test is not None else self.Y_test_
        if theta_test is None:
            raise ValueError('No held-out data available; call fit first.')

        Y_pred = self.predict_theta(theta_test)
        rmse = np.sqrt(np.mean((Y_pred - Y_test) ** 2, axis=0))
        return Y_pred, Y_test, rmse

    def plot_validation(self, theta_test=None, Y_test=None, figsize=(5, 5), max_points=2000):
        """Scatter plot of predicted vs. true outputs on held-out data."""
        import matplotlib.pyplot as plt

        Y_pred, Y_test, rmse = self.validate(theta_test, Y_test)

        y_true = Y_test.ravel()
        y_pred = Y_pred.ravel()
        if len(y_true) > max_points:
            idx = np.random.default_rng(0).choice(len(y_true), max_points, replace=False)
            y_true, y_pred = y_true[idx], y_pred[idx]

        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(y_true, y_pred, s=4, alpha=0.3)
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, 'k--', lw=1)
        ax.set_xlabel('True output')
        ax.set_ylabel('Predicted output')
        ax.set_title(f'Surrogate validation (overall RMSE = {np.sqrt(np.mean(rmse**2)):.3e})')
        fig.tight_layout()
        return fig, ax, rmse

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath: str):
        """Save the fitted MLP, scalers, and metadata (not ``base_model``)."""
        if joblib is None:
            raise ImportError('joblib is required to save/load SurrogateModel.')
        joblib.dump({
            'model': self.model,
            'x_scaler': self.x_scaler,
            'y_scaler': self.y_scaler,
            'case_sizes': self.case_sizes,
            'is_fitted': self.is_fitted,
            'theta_test_': self.theta_test_,
            'Y_test_': self.Y_test_,
        }, filepath)

    @classmethod
    def load(cls, filepath: str, base_model, **kwargs):
        """Load a surrogate previously saved with :meth:`save`.

        ``base_model`` must be supplied again (it is not pickled).
        """
        if joblib is None:
            raise ImportError('joblib is required to save/load SurrogateModel.')
        state = joblib.load(filepath)

        surrogate = cls(base_model, state['case_sizes'], **kwargs)
        surrogate.model = state['model']
        surrogate.x_scaler = state['x_scaler']
        surrogate.y_scaler = state['y_scaler']
        surrogate.is_fitted = state['is_fitted']
        surrogate.theta_test_ = state['theta_test_']
        surrogate.Y_test_ = state['Y_test_']
        return surrogate
