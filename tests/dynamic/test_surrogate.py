"""Tests for SurrogateModel."""
import numpy as np
import pytest

from marapendi.dynamic.simulation.estimation import ParameterEstimation, UnknownParameter
from marapendi.dynamic.simulation.surrogate import SurrogateModel


@pytest.fixture
def linear_params():
    return [
        UnknownParameter('a', r'$a$', 'm', lower=0., upper=10., log_scale=False),
        UnknownParameter('b', r'$b$', 'K', lower=1., upper=100., log_scale=False),
    ]


def _linear_model(params):
    """Two cases, each with 2 outputs: [a + b, a - b]."""
    a, b = params['a'], params['b']
    return np.array([a + b, a - b, 2 * a + b, 2 * a - b])


@pytest.fixture
def base_model(linear_params):
    return ParameterEstimation(_linear_model, {'a': 5., 'b': 50.}, linear_params)


@pytest.fixture
def case_sizes():
    return {'case1': 2, 'case2': 2}


class TestSurrogateModelSetup:
    def test_offsets_and_n_outputs(self, base_model, case_sizes):
        surrogate = SurrogateModel(base_model, case_sizes)
        assert surrogate.offsets == {'case1': (0, 2), 'case2': (2, 4)}
        assert surrogate.n_outputs == 4
        assert surrogate.n_params == 2
        assert surrogate.is_fitted is False


class TestSampling:
    def test_sample_theta_shape_and_bounds(self, base_model, case_sizes):
        surrogate = SurrogateModel(base_model, case_sizes)
        theta = surrogate.sample_theta(20, seed=0)
        assert theta.shape == (20, 2)
        assert np.all(theta >= 0.) and np.all(theta <= 1.)


class TestTrainingDataGeneration:
    def test_generate_training_data(self, base_model, case_sizes):
        surrogate = SurrogateModel(base_model, case_sizes)
        theta, Y = surrogate.generate_training_data(n_samples=30, seed=0)
        assert theta.shape[1] == 2
        assert Y.shape == (theta.shape[0], 4)
        assert np.all(np.isfinite(Y))

    def test_invalid_outputs_are_dropped(self, base_model, case_sizes):
        def flaky_model(params):
            if params['a'] > 8.:
                return np.array([np.nan, np.nan, np.nan, np.nan])
            return _linear_model(params)

        surrogate = SurrogateModel(base_model, case_sizes)
        theta, Y = surrogate.generate_training_data(model_fn=flaky_model, n_samples=50, seed=0)
        assert len(theta) <= 50
        assert np.all(np.isfinite(Y))

    def test_append(self, base_model, case_sizes):
        surrogate = SurrogateModel(base_model, case_sizes)
        theta1, Y1 = surrogate.generate_training_data(n_samples=10, seed=0)
        theta2, Y2 = surrogate.generate_training_data(n_samples=10, seed=1, append=True)
        assert len(theta2) == len(theta1) + 10


class TestFitPredict:
    def test_fit_and_predict(self, base_model, case_sizes):
        surrogate = SurrogateModel(base_model, case_sizes, hidden_layer_sizes=(32, 32), random_state=0)
        surrogate.generate_training_data(n_samples=200, seed=0)
        surrogate.fit(test_size=0.2, random_state=0)
        assert surrogate.is_fitted is True

        y = surrogate.predict({'a': 5., 'b': 50.})
        y_true = _linear_model({'a': 5., 'b': 50.})
        np.testing.assert_allclose(y, y_true, atol=2.0)

    def test_predict_before_fit_raises(self, base_model, case_sizes):
        surrogate = SurrogateModel(base_model, case_sizes)
        with pytest.raises(RuntimeError):
            surrogate.predict_theta(np.array([0.5, 0.5]))

    def test_call_with_case_list_slices_output(self, base_model, case_sizes):
        surrogate = SurrogateModel(base_model, case_sizes, hidden_layer_sizes=(32, 32), random_state=0)
        surrogate.generate_training_data(n_samples=200, seed=0)
        surrogate.fit(test_size=0.2, random_state=0)

        params = {'a': 5., 'b': 50.}
        y_full = surrogate(params)
        y_case1 = surrogate(params, case_list=['case1'])
        y_case2_case1 = surrogate(params, case_list=['case2', 'case1'])

        np.testing.assert_allclose(y_case1, y_full[:2])
        np.testing.assert_allclose(y_case2_case1, np.concatenate([y_full[2:], y_full[:2]]))

    def test_get_model_fn(self, base_model, case_sizes):
        surrogate = SurrogateModel(base_model, case_sizes, hidden_layer_sizes=(32, 32), random_state=0)
        surrogate.generate_training_data(n_samples=200, seed=0)
        surrogate.fit(test_size=0.2, random_state=0)

        model_fn = surrogate.get_model_fn(case_list=['case1'])
        params = {'a': 5., 'b': 50.}
        np.testing.assert_allclose(model_fn(params), surrogate(params, case_list=['case1']))


class TestValidate:
    def test_validate_returns_rmse(self, base_model, case_sizes):
        surrogate = SurrogateModel(base_model, case_sizes, hidden_layer_sizes=(32, 32), random_state=0)
        surrogate.generate_training_data(n_samples=200, seed=0)
        surrogate.fit(test_size=0.2, random_state=0)

        Y_pred, Y_test, rmse = surrogate.validate()
        assert Y_pred.shape == Y_test.shape
        assert rmse.shape == (4,)
        assert np.all(rmse >= 0.)

    def test_validate_before_fit_raises(self, base_model, case_sizes):
        surrogate = SurrogateModel(base_model, case_sizes)
        with pytest.raises(ValueError):
            surrogate.validate()


class TestPersistence:
    def test_save_and_load_roundtrip(self, base_model, case_sizes, tmp_path):
        surrogate = SurrogateModel(base_model, case_sizes, hidden_layer_sizes=(32, 32), random_state=0)
        surrogate.generate_training_data(n_samples=200, seed=0)
        surrogate.fit(test_size=0.2, random_state=0)

        filepath = tmp_path / 'surrogate.joblib'
        surrogate.save(str(filepath))

        loaded = SurrogateModel.load(str(filepath), base_model)
        assert loaded.is_fitted is True
        assert loaded.case_sizes == surrogate.case_sizes

        params = {'a': 5., 'b': 50.}
        np.testing.assert_allclose(loaded.predict(params), surrogate.predict(params))
