"""Tests for ParameterEstimation and UnknownParameter."""
import numpy as np
import pytest

from marapendi.simulation.estimation import ParameterEstimation, UnknownParameter


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def linear_params():
    return [
        UnknownParameter('a', r'$a$', 'm', lower=0., upper=10., log_scale=False),
        UnknownParameter('b', r'$b$', 'K', lower=1., upper=100., log_scale=False),
    ]


@pytest.fixture
def log_params():
    return [
        UnknownParameter('k', r'$k$', 's⁻¹', lower=1e-4, upper=1e-1, log_scale=True),
    ]


def _identity_model(params):
    """Returns [a, b] directly — exact solution is just y_exp."""
    return np.array([params['a'], params['b']])


def _scalar_model(params):
    return np.array([params['k']])


# ---------------------------------------------------------------------------
# UnknownParameter
# ---------------------------------------------------------------------------

class TestUnknownParameter:
    def test_fields(self):
        up = UnknownParameter('x', r'$x$', 'm', lower=1., upper=5., log_scale=True)
        assert up.key == 'x'
        assert up.label == r'$x$'
        assert up.units == 'm'
        assert up.lower == 1.
        assert up.upper == 5.
        assert up.log_scale is True

    def test_default_log_scale_is_false(self):
        up = UnknownParameter('x', 'x', '-', 0., 1.)
        assert up.log_scale is False


# ---------------------------------------------------------------------------
# Normalisation: p_to_theta / theta_to_p
# ---------------------------------------------------------------------------

class TestNormalisation:
    def test_linear_bounds_map_to_zero_one(self, linear_params):
        est = ParameterEstimation(_identity_model, {'a': 5., 'b': 50.}, linear_params)
        p_min = np.array([0., 1.])
        p_max = np.array([10., 100.])
        np.testing.assert_allclose(est.p_to_theta(p_min), [0., 0.])
        np.testing.assert_allclose(est.p_to_theta(p_max), [1., 1.])

    def test_log_bounds_map_to_zero_one(self, log_params):
        est = ParameterEstimation(_scalar_model, {'k': 1e-3}, log_params)
        np.testing.assert_allclose(est.p_to_theta(np.array([1e-4])), [0.])
        np.testing.assert_allclose(est.p_to_theta(np.array([1e-1])), [1.])

    def test_linear_roundtrip(self, linear_params):
        est = ParameterEstimation(_identity_model, {'a': 5., 'b': 50.}, linear_params)
        p_orig = np.array([3.7, 42.1])
        np.testing.assert_allclose(est.theta_to_p(est.p_to_theta(p_orig)), p_orig, rtol=1e-12)

    def test_log_roundtrip(self, log_params):
        est = ParameterEstimation(_scalar_model, {'k': 1e-3}, log_params)
        p_orig = np.array([5e-3])
        np.testing.assert_allclose(est.theta_to_p(est.p_to_theta(p_orig)), p_orig, rtol=1e-12)

    def test_linear_midpoint(self, linear_params):
        est = ParameterEstimation(_identity_model, {'a': 5., 'b': 50.}, linear_params)
        p_mid = np.array([5., 50.5])
        theta_mid = est.p_to_theta(p_mid)
        np.testing.assert_allclose(theta_mid, [0.5, 0.5], atol=1e-10)

    def test_log_midpoint_is_geometric_mean(self, log_params):
        est = ParameterEstimation(_scalar_model, {'k': 1e-3}, log_params)
        p_geo = np.array([np.sqrt(1e-4 * 1e-1)])  # geometric mean
        np.testing.assert_allclose(est.p_to_theta(p_geo), [0.5], atol=1e-10)


# ---------------------------------------------------------------------------
# params_from_theta / nominal_theta
# ---------------------------------------------------------------------------

class TestParamsFromTheta:
    def test_overlays_only_unknowns(self, linear_params):
        nominal = {'a': 5., 'b': 50., 'fixed': 99.}
        est = ParameterEstimation(_identity_model, nominal, linear_params)
        px = est.params_from_theta(np.array([0., 0.]))   # a=0, b=1
        assert px['a'] == pytest.approx(0.)
        assert px['b'] == pytest.approx(1.)
        assert px['fixed'] == 99.    # fixed param untouched

    def test_nominal_theta_uses_self_params(self, linear_params):
        # a ∈ [0, 10]:  theta_a = (2.5 - 0) / (10 - 0) = 0.25
        # b ∈ [1, 100]: theta_b = (10 - 1) / (100 - 1) ≈ 0.0909
        nominal = {'a': 2.5, 'b': 10.}
        est = ParameterEstimation(_identity_model, nominal, linear_params)
        theta = est.nominal_theta()
        np.testing.assert_allclose(theta[0], 0.25, rtol=1e-12)
        np.testing.assert_allclose(theta[1], 9. / 99., rtol=1e-12)

    def test_nominal_theta_roundtrip(self, log_params):
        nominal = {'k': 3e-3}
        est = ParameterEstimation(_scalar_model, nominal, log_params)
        theta = est.nominal_theta()
        p_back = est.theta_to_p(theta)
        np.testing.assert_allclose(p_back, [3e-3], rtol=1e-12)


# ---------------------------------------------------------------------------
# evaluate / residuals
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_evaluate_at_nominal(self, linear_params):
        nominal = {'a': 3., 'b': 7.}
        est = ParameterEstimation(_identity_model, nominal, linear_params)
        y = est.evaluate()
        np.testing.assert_allclose(y, [3., 7.])

    def test_evaluate_at_custom_params(self, linear_params):
        est = ParameterEstimation(_identity_model, {'a': 0., 'b': 0.}, linear_params)
        y = est.evaluate({'a': 1.5, 'b': 9.})
        np.testing.assert_allclose(y, [1.5, 9.])

    def test_residuals_correct(self, linear_params):
        est = ParameterEstimation(_identity_model, {'a': 2., 'b': 4.}, linear_params)
        res = est.residuals(np.array([3., 5.]))
        np.testing.assert_allclose(res, [1., 1.])

    def test_residuals_nan_safe(self, linear_params):
        """NaN entries in either y_exp or y_model are excluded from residuals."""
        def model_with_nan(p):
            return np.array([p['a'], np.nan])

        est = ParameterEstimation(model_with_nan, {'a': 2., 'b': 0.}, linear_params)
        res = est.residuals(np.array([5., 7.]))
        assert res.shape == (1,)   # only the finite entry
        assert res[0] == pytest.approx(3.)

    def test_residuals_nan_in_y_exp(self, linear_params):
        est = ParameterEstimation(_identity_model, {'a': 2., 'b': 4.}, linear_params)
        res = est.residuals(np.array([np.nan, 6.]))
        assert res.shape == (1,)
        assert res[0] == pytest.approx(2.)


# ---------------------------------------------------------------------------
# estimate — DE
# ---------------------------------------------------------------------------

class TestEstimateDE:
    def test_recovers_exact_solution(self, linear_params):
        """DE must recover the exact target on a trivial identity model."""
        true_params = {'a': 4.2, 'b': 73.}
        nominal    = {'a': 5.,  'b': 50.}
        y_exp = np.array([true_params['a'], true_params['b']])

        est = ParameterEstimation(_identity_model, nominal, linear_params)
        _, p_hat = est.estimate(y_exp, popsize=5, maxiter=200, seed=0)

        assert p_hat['a'] == pytest.approx(true_params['a'], rel=1e-4)
        assert p_hat['b'] == pytest.approx(true_params['b'], rel=1e-4)

    def test_returns_full_params_dict(self, linear_params):
        nominal = {'a': 5., 'b': 50., 'extra': 42.}
        est = ParameterEstimation(_identity_model, nominal, linear_params)
        _, p_hat = est.estimate(np.array([1., 2.]), popsize=3, maxiter=5)
        assert 'extra' in p_hat   # fixed params must be preserved

    def test_sol_has_fun_attribute(self, linear_params):
        est = ParameterEstimation(_identity_model, {'a': 5., 'b': 50.}, linear_params)
        sol, _ = est.estimate(np.array([1., 2.]), popsize=3, maxiter=5)
        assert hasattr(sol, 'fun')
        assert np.isfinite(sol.fun)

    def test_objective_nan_safe(self, linear_params):
        """If model returns NaN for some theta, DE should not crash."""
        call_count = {'n': 0}

        def flaky_model(p):
            call_count['n'] += 1
            if call_count['n'] % 3 == 0:
                return np.array([np.nan, np.nan])
            return _identity_model(p)

        est = ParameterEstimation(flaky_model, {'a': 5., 'b': 50.}, linear_params)
        sol, _ = est.estimate(np.array([3., 7.]), popsize=3, maxiter=10)
        assert np.isfinite(sol.fun)


# ---------------------------------------------------------------------------
# estimate — gradient methods with n_restarts
# ---------------------------------------------------------------------------

class TestEstimateLBFGSB:
    def test_single_restart_starts_from_nominal(self, linear_params):
        """n_restarts=1 with no initial_guess must give a deterministic result
        starting from self.params (the nominal values)."""
        true_params = {'a': 4.2, 'b': 73.}
        nominal     = {'a': 4.2, 'b': 73.}   # nominal IS the solution
        y_exp = np.array([true_params['a'], true_params['b']])

        est = ParameterEstimation(_identity_model, nominal, linear_params)
        _, p_hat = est.estimate(y_exp, method='L-BFGS-B', n_restarts=1)
        assert p_hat['a'] == pytest.approx(true_params['a'], rel=1e-6)
        assert p_hat['b'] == pytest.approx(true_params['b'], rel=1e-6)

    def test_single_restart_is_deterministic(self, linear_params):
        """Two calls with n_restarts=1 must return the same result."""
        nominal = {'a': 5., 'b': 50.}
        est = ParameterEstimation(_identity_model, nominal, linear_params)
        y_exp = np.array([3., 7.])

        _, p1 = est.estimate(y_exp, method='L-BFGS-B', n_restarts=1)
        _, p2 = est.estimate(y_exp, method='L-BFGS-B', n_restarts=1)
        assert p1['a'] == pytest.approx(p2['a'])
        assert p1['b'] == pytest.approx(p2['b'])

    def test_initial_guess_overrides_nominal(self, linear_params):
        nominal = {'a': 5., 'b': 50.}
        est = ParameterEstimation(_identity_model, nominal, linear_params)
        y_exp = np.array([3., 7.])

        # Start from the true solution — should converge in zero steps
        _, p_hat = est.estimate(
            y_exp, method='L-BFGS-B', n_restarts=1,
            initial_guess=np.array([3., 7.]),
        )
        assert p_hat['a'] == pytest.approx(3., rel=1e-6)
        assert p_hat['b'] == pytest.approx(7., rel=1e-6)

    def test_multi_restart_finds_solution(self, linear_params):
        """With enough restarts the global minimum is reached."""
        true_params = {'a': 1.1, 'b': 88.}
        nominal     = {'a': 9.,  'b': 2.}   # far from truth
        y_exp = np.array([true_params['a'], true_params['b']])

        est = ParameterEstimation(_identity_model, nominal, linear_params)
        _, p_hat = est.estimate(
            y_exp, method='L-BFGS-B', n_restarts=8, ftol=1e-14, seed=42,
        )
        assert p_hat['a'] == pytest.approx(true_params['a'], rel=1e-4)
        assert p_hat['b'] == pytest.approx(true_params['b'], rel=1e-4)

    def test_best_restart_is_returned(self, linear_params):
        """sol.fun must be the minimum across all restarts."""
        est = ParameterEstimation(_identity_model, {'a': 5., 'b': 50.}, linear_params)
        y_exp = np.array([2., 6.])
        sol, _ = est.estimate(y_exp, method='L-BFGS-B', n_restarts=5)
        assert sol.fun >= 0.
        assert sol.fun == pytest.approx(0., abs=1e-10)


# ---------------------------------------------------------------------------
# Local sensitivity
# ---------------------------------------------------------------------------

class TestLocalSensitivity:
    def test_shape(self, linear_params):
        est = ParameterEstimation(_identity_model, {'a': 5., 'b': 50.}, linear_params)
        S = est.calculate_local_sensitivity(n_samples=5)
        assert S.shape == (2, 2)   # (n_unknown, n_outputs)

    def test_diagonal_dominance(self, linear_params):
        """Output i is sensitive to param i and insensitive to param j≠i."""
        est = ParameterEstimation(_identity_model, {'a': 5., 'b': 50.}, linear_params)
        S = est.calculate_local_sensitivity(n_samples=7)
        # |S[0,0]| > |S[0,1]|  (a drives output 0)
        assert abs(S[0, 0]) > abs(S[0, 1])
        # |S[1,1]| > |S[1,0]|  (b drives output 1)
        assert abs(S[1, 1]) > abs(S[1, 0])

    def test_stores_result_on_self(self, linear_params):
        est = ParameterEstimation(_identity_model, {'a': 5., 'b': 50.}, linear_params)
        S = est.calculate_local_sensitivity()
        assert est.S is S
