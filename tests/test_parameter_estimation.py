"""Tests for parameter estimation (estimation/estimation.py).

Uses a simple synthetic model to verify the SteadyStateModel class
without requiring external data files.
"""
import numpy as np
import pytest
import marapendi as mrpd


# ---------------------------------------------------------------------------
# Synthetic model: y = a * x^2 + b * x over x in [0, 1]
# ---------------------------------------------------------------------------

_X = np.linspace(0., 1., 20)
_TRUE_PARAMS = {'a': 3.0, 'b': 1.5}


def _quadratic(params):
    return params['a'] * _X ** 2 + params['b'] * _X


_Y_EXP = _quadratic(_TRUE_PARAMS)


@pytest.fixture
def estimator():
    return mrpd.SteadyStateModel(_quadratic, {'a': 2.0, 'b': 1.0})


class TestSteadyStateModelBasics:
    def test_solve_returns_correct_shape(self, estimator):
        _, _, y = estimator.solve(0)
        assert y.shape == _Y_EXP.shape

    def test_residuals_at_true_params_near_zero(self):
        model = mrpd.SteadyStateModel(_quadratic, _TRUE_PARAMS)
        res = model.residuals(_Y_EXP, t=0)
        assert np.allclose(res, 0., atol=1e-12)

    def test_residuals_nonzero_at_wrong_params(self, estimator):
        res = estimator.residuals(_Y_EXP, t=0)
        assert not np.allclose(res, 0.)

    def test_set_params(self, estimator):
        estimator.set_params({'a': 5.0, 'b': 2.0})
        assert estimator.p['a'] == 5.0


class TestParameterEstimation:
    def test_estimate_recovers_true_params(self, estimator):
        estimator.set_unknown_params([
            ('a', (0., 10.), True, 'a'),
            ('b', (0., 5.), True, 'b'),
        ])
        sol, p_est = estimator.estimate(
            _Y_EXP, t=0, print_iterations=False,
            popsize=20, ftol=1e-10, penalty_threshold=0,
        )
        assert np.isclose(p_est[0], _TRUE_PARAMS['a'], atol=0.2)
        assert np.isclose(p_est[1], _TRUE_PARAMS['b'], atol=0.2)

    def test_single_parameter_estimate(self):
        model = mrpd.SteadyStateModel(_quadratic, {'a': 3.0, 'b': 1.0})
        model.set_unknown_params([('b', (0., 5.), True, 'b')])
        _, p_est = model.estimate(
            _Y_EXP, t=0, print_iterations=False,
            popsize=15, ftol=1e-10, penalty_threshold=0,
        )
        assert np.isclose(p_est[0], _TRUE_PARAMS['b'], atol=0.1)
