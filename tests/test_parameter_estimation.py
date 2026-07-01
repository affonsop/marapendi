"""Tests for the parameter estimation / calibration API.

Uses a simple synthetic model (quadratic) to exercise BaseModelCalibration
without requiring external data files or a real FuelCell.
"""
import numpy as np
import pytest

from marapendi.estimation.base_calibration import BaseModelCalibration
from marapendi.estimation.parameters import Parameter, UnknownParameter


# ---------------------------------------------------------------------------
# Synthetic calibration subclass: y = a * x^2 + b * x over x in [0, 1]
# ---------------------------------------------------------------------------

_X = np.linspace(0.0, 1.0, 20)
_TRUE_PARAMS = {'a': 3.0, 'b': 1.5}
_Y_EXP = _TRUE_PARAMS['a'] * _X ** 2 + _TRUE_PARAMS['b'] * _X

_KNOWN = [Parameter(value=0.0, key='offset')]

_UNKNOWN = [
    UnknownParameter(value=2.0, initial_guess=2.0, lower_bound=0.0, upper_bound=10.0,
                     key='a', is_linear=True),
    UnknownParameter(value=1.0, initial_guess=1.0, lower_bound=0.0, upper_bound=5.0,
                     key='b', is_linear=True),
]


class QuadraticCalibration(BaseModelCalibration):
    def compute_y_sim(self, p_values, case_list=None):
        p = dict(zip(self.p_i_name, p_values))
        return p['a'] * _X ** 2 + p['b'] * _X

    def compute_residuals(self, p_values, case_list=None):
        return _Y_EXP - self.compute_y_sim(p_values, case_list)


@pytest.fixture
def calibration():
    cal = QuadraticCalibration(known_parameters=_KNOWN, unknown_parameters=_UNKNOWN)
    cal.full_case_list = ['case0']
    return cal


# ---------------------------------------------------------------------------
# Parameter normalisation
# ---------------------------------------------------------------------------

class TestNormalisation:
    def test_p_to_theta_linear_midpoint(self, calibration):
        p = np.array([5.0, 2.5])   # midpoint of [0,10] and [0,5]
        theta = calibration.p_to_theta(p)
        assert np.allclose(theta, [0.5, 0.5])

    def test_theta_to_p_roundtrip(self, calibration):
        p_orig = np.array([3.0, 1.5])
        assert np.allclose(calibration.theta_to_p(calibration.p_to_theta(p_orig)), p_orig)

    def test_p_to_theta_log_scale(self):
        unknown_log = [
            UnknownParameter(value=1e-3, initial_guess=1e-3, lower_bound=1e-4, upper_bound=1e-2,
                             key='k', is_linear=False),
        ]
        cal = QuadraticCalibration(known_parameters=[], unknown_parameters=unknown_log)
        p = np.array([1e-3])   # geometric midpoint of [1e-4, 1e-2]
        theta = cal.p_to_theta(p)
        assert np.isclose(theta[0], 0.5, atol=1e-6)

    def test_theta_zero_maps_to_lower_bound(self, calibration):
        p = calibration.theta_to_p(np.array([0.0, 0.0]))
        assert np.allclose(p, [0.0, 0.0])

    def test_theta_one_maps_to_upper_bound(self, calibration):
        p = calibration.theta_to_p(np.array([1.0, 1.0]))
        assert np.allclose(p, [10.0, 5.0])


# ---------------------------------------------------------------------------
# Unknown parameter setup
# ---------------------------------------------------------------------------

class TestSetUnknownParams:
    def test_bounds_arrays(self, calibration):
        assert np.allclose(calibration.p_i_min, [0.0, 0.0])
        assert np.allclose(calibration.p_i_max, [10.0, 5.0])

    def test_n_unknown(self, calibration):
        assert calibration.n_unkown_p == 2

    def test_initial_guess_loaded_into_params(self, calibration):
        assert calibration.params['a'] == 2.0
        assert calibration.params['b'] == 1.0

    def test_subset_reduces_unknown_count(self, calibration):
        calibration.subset_of_unknown_parameters(keys=['a'])
        assert calibration.n_unkown_p == 1
        assert calibration.p_i_name == ['a']

    def test_reset_restores_full_unknown_list(self, calibration):
        calibration.subset_of_unknown_parameters(keys=['a'])
        calibration.reset_unknown_parameters()
        assert calibration.n_unkown_p == 2


# ---------------------------------------------------------------------------
# compute_y_sim / compute_residuals
# ---------------------------------------------------------------------------

class TestSyntheticModel:
    def test_residuals_at_true_params_near_zero(self, calibration):
        p_true = np.array([_TRUE_PARAMS['a'], _TRUE_PARAMS['b']])
        res = calibration.compute_residuals(p_true)
        assert np.allclose(res, 0.0, atol=1e-12)

    def test_residuals_nonzero_at_wrong_params(self, calibration):
        p_wrong = np.array([2.0, 1.0])
        res = calibration.compute_residuals(p_wrong)
        assert not np.allclose(res, 0.0)


# ---------------------------------------------------------------------------
# k-fold splitting
# ---------------------------------------------------------------------------

class TestKFoldSplit:
    def test_k_folds_covers_all_cases(self, calibration):
        calibration.full_case_list = ['A', 'B', 'C', 'D']
        calibration.set_k_folds(k=2)
        all_cases = [c for fold in calibration.k_folds for c in fold]
        assert sorted(all_cases) == ['A', 'B', 'C', 'D']

    def test_k_folds_count(self, calibration):
        calibration.full_case_list = list(range(6))
        calibration.set_k_folds(k=3)
        assert len(calibration.k_folds) == 3

    def test_leave_one_out_folds(self, calibration):
        cases = ['A', 'B', 'C']
        calibration.full_case_list = cases
        calibration.set_k_folds(k=3)
        for fold in calibration.k_folds:
            assert len(fold) == 1
