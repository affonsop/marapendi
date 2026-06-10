"""
Parameter estimation and sensitivity analysis for callable forward models.

Classes
-------
UnknownParameter
    Dataclass describing a single parameter to be estimated.
ParameterEstimation
    Wraps a callable model and a list of UnknownParameter objects.
    Provides normalisation helpers, differential-evolution estimation,
    local/global sensitivity analysis, identifiability ranking, and plots.

Normalisation convention (Goshtasbi et al. 2020)
-------------------------------------------------
* Linear:  θ = (p − p_min) / (p_max − p_min)
* Log:     θ = (ln p − ln p_min) / (ln p_max − ln p_min)

Both transforms map p ∈ [p_min, p_max] → θ ∈ [0, 1].

References
----------
Goshtasbi, A. et al. J. Electrochem. Soc. 167, 024518 (2020).
Goshtasbi, A. et al. J. Electrochem. Soc. 167, 044504 (2020).
Lund, B. F. & Foss, B. A. Automatica 44, 278–281 (2008).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp as _solve_ivp
from scipy.optimize import differential_evolution, minimize
from scipy.stats import qmc
from scipy.linalg import qr, eigvals

__all__ = ['UnknownParameter', 'ParameterEstimation']

_EPS = 1e-300   # avoids log(0)


# ---------------------------------------------------------------------------
# UnknownParameter
# ---------------------------------------------------------------------------

@dataclass
class UnknownParameter:
    """Descriptor for a single parameter to be estimated.

    Parameters
    ----------
    key : str
        Key in the parameter dict passed to the model callable.
    label : str
        Display label, e.g. a LaTeX string used in plots.
    units : str
        SI unit string for display (e.g. ``'Ω·m²'``).
    lower : float
        Lower bound in physical units.
    upper : float
        Upper bound in physical units.
    log_scale : bool
        If ``True``, normalisation is logarithmic — appropriate for parameters
        that span orders of magnitude.  If ``False`` (default), linear.
    """
    key: str
    label: str
    units: str
    lower: float
    upper: float
    log_scale: bool = False


# ---------------------------------------------------------------------------
# ParameterEstimation
# ---------------------------------------------------------------------------

class ParameterEstimation:
    """Parameter estimation and sensitivity analysis for a callable model.

    Parameters
    ----------
    model_fn : callable
        Forward model ``model_fn(params: dict) -> np.ndarray``.  Must accept a
        full parameter dict and return a 1-D output vector.  NaN and Inf values
        in the output are handled gracefully in the objective function.
    params : dict
        Nominal parameter values for *all* parameters (fixed and unknown).
        Serves as the base dict; unknown parameters are overlaid before each
        model evaluation.
    unknown_parameters : list[UnknownParameter]
        Parameters to be estimated, with their bounds and scale.

    Examples
    --------
    ::

        est = ParameterEstimation(
            model_fn=my_model,
            params=nominal_params,
            unknown_parameters=[
                UnknownParameter('r_elec', r'$r_{elec}$', 'Ω·m²',
                                 lower=1e-5, upper=1e-3, log_scale=True),
            ],
        )
        sol, p_hat = est.estimate(y_exp)
    """

    def __init__(
        self,
        model_fn: Callable,
        params: dict,
        unknown_parameters: list[UnknownParameter],
    ):
        self.model_fn           = model_fn
        self.params             = params.copy()
        self.unknown_parameters = list(unknown_parameters)

        self._p_min = np.array([up.lower      for up in unknown_parameters], dtype=float)
        self._p_max = np.array([up.upper      for up in unknown_parameters], dtype=float)
        self._log   = np.array([up.log_scale  for up in unknown_parameters], dtype=bool)

        # Results populated by sensitivity / estimation methods
        self.S            = None
        self.S_med        = None
        self.S_std        = None
        self.S_med_i      = None
        self.S_std_i      = None
        self.norm_s_i     = None
        self.cosPhi_med_ij = None
        self.S_n          = None
        self.n_valid      = None

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------

    def p_to_theta(self, p_values: np.ndarray) -> np.ndarray:
        """Convert physical parameter values to normalised [0, 1] space."""
        p = np.asarray(p_values, dtype=float)
        return np.where(
            self._log,
            (np.log(p + _EPS) - np.log(self._p_min + _EPS))
            / (np.log(self._p_max + _EPS) - np.log(self._p_min + _EPS)),
            (p - self._p_min) / (self._p_max - self._p_min),
        )

    def theta_to_p(self, theta: np.ndarray) -> np.ndarray:
        """Convert normalised [0, 1] values to physical parameter values."""
        t = np.asarray(theta, dtype=float)
        return np.where(
            self._log,
            self._p_min * np.exp(
                (np.log(self._p_max + _EPS) - np.log(self._p_min + _EPS)) * t
            ),
            self._p_min + (self._p_max - self._p_min) * t,
        )

    def params_from_theta(self, theta: np.ndarray) -> dict:
        """Return a full parameter dict with unknown parameters set from *theta*."""
        px = self.params.copy()
        for up, v in zip(self.unknown_parameters, self.theta_to_p(theta)):
            px[up.key] = float(v)
        return px

    def nominal_theta(self) -> np.ndarray:
        """Return normalised coordinates of the nominal (``self.params``) values."""
        return self.p_to_theta(
            np.array([self.params[up.key] for up in self.unknown_parameters])
        )

    # ------------------------------------------------------------------
    # Model evaluation
    # ------------------------------------------------------------------

    def evaluate(self, params: dict | None = None) -> np.ndarray:
        """Evaluate the model at *params* (or at the nominal values)."""
        return np.asarray(self.model_fn(params if params is not None else self.params))

    def residuals(self, y_exp: np.ndarray, params: dict | None = None) -> np.ndarray:
        """Return ``y_exp − model(params)`` over finite-valued entries."""
        y_exp   = np.asarray(y_exp)
        y_model = self.evaluate(params)
        valid   = np.isfinite(y_model) & np.isfinite(y_exp)
        return y_exp[valid] - y_model[valid]

    # ------------------------------------------------------------------
    # Parameter estimation
    # ------------------------------------------------------------------

    def estimate(
        self,
        y_exp: np.ndarray,
        *,
        method: str = 'differential_evolution',
        penalty_threshold: float = 1e-2,
        popsize: int = 10,
        workers: int = 1,
        rtol: float = 0.,
        atol: float = 0.,
        ftol: float = 0.,
        maxiter: int = 200,
        print_iterations: bool = False,
        initial_guess: np.ndarray | None = None,
        seed: int = 2,
    ):
        """Estimate unknown parameters by minimising the mean-squared residual.

        Parameters
        ----------
        y_exp : array_like
            Measured (or synthetic) output vector.  NaN/Inf entries are ignored.
        method : str
            ``'differential_evolution'`` (default, global) or any
            ``scipy.optimize.minimize`` method for local refinement.
        penalty_threshold : float
            Residuals exceeding this value carry an additional quadratic
            penalty — see Goshtasbi et al. (2020).  Set to ``0`` to disable.
        popsize, workers, rtol, atol, maxiter, seed :
            Passed to ``scipy.optimize.differential_evolution``.
        print_iterations : bool
            Print objective value at each callback.
        initial_guess : np.ndarray, optional
            Physical initial values for gradient-based methods.

        Returns
        -------
        sol : OptimizeResult
        params_estimated : dict
            Full parameter dict with estimated values merged in.
        """
        y_exp = np.asarray(y_exp, dtype=float)

        def _objective(theta):
            px = self.params_from_theta(theta)
            try:
                y_model = np.asarray(self.model_fn(px), dtype=float)
            except Exception:
                return 1e10
            valid = np.isfinite(y_model) & np.isfinite(y_exp)
            if not np.any(valid):
                return 1e10
            res = y_exp[valid] - y_model[valid]
            obj = np.dot(res, res) / len(res)
            if penalty_threshold > 0:
                pen = np.where(np.abs(res) > penalty_threshold,
                               10. * (np.abs(res) - penalty_threshold), 0.)
                obj += np.dot(pen, pen)
            return float(obj)

        bounds = [(0., 1.)] * len(self.unknown_parameters)

        if method == 'differential_evolution':
            sol = differential_evolution(
                _objective, bounds=bounds,
                popsize=popsize, workers=workers,
                mutation=(0., 1.), recombination=0.5,
                seed=seed, init='latinhypercube',
                tol=rtol, atol=atol, maxiter=maxiter,
                polish=True, disp=print_iterations,
                callback=(lambda xk, convergence: print(f"  convergence={convergence:.4g}"))
                if print_iterations else None,
            )
        else:
            if initial_guess is not None:
                x0 = self.p_to_theta(np.asarray(initial_guess, dtype=float))
            else:
                x0 = self.nominal_theta()
            sol = minimize(
                _objective, x0=x0, method=method, bounds=bounds,
                options={'ftol': ftol, 'maxiter': maxiter},
            )

        return sol, self.params_from_theta(sol.x)

    # ------------------------------------------------------------------
    # Local sensitivity (single parameter point, sweep over full range)
    # ------------------------------------------------------------------

    def calculate_local_sensitivity(
        self,
        n_samples: int = 7,
        params: dict | None = None,
        measures: Callable | None = None,
    ) -> np.ndarray:
        """Normalised local sensitivity by sweeping each parameter (Eq. 7, Goshtasbi 2020).

        For each unknown parameter, sweeps θ_i uniformly from 0 to 1 in
        ``n_samples`` steps while holding all other parameters fixed at their
        nominal values.  The normalised sensitivity is

            S_i = mean(dy/dθ_i) / mean(|y|)

        Parameters
        ----------
        n_samples : int
            Number of θ-samples per parameter (default 7).
        params : dict, optional
            Base parameter dict (defaults to ``self.params``).
        measures : callable, optional
            ``measures(params_dict) -> ndarray``; defaults to ``self.model_fn``.

        Returns
        -------
        S : ndarray, shape (n_unknown, n_outputs)
            Normalised sensitivities.
        """
        p_base = (params or self.params).copy()
        fn     = measures if measures is not None else self.model_fn
        theta_nom = self.p_to_theta(
            np.array([p_base[up.key] for up in self.unknown_parameters])
        )
        theta_grid = np.linspace(0., 1., n_samples)

        y_all = []
        for i in range(len(self.unknown_parameters)):
            y_row = []
            for ti in theta_grid:
                theta = theta_nom.copy()
                theta[i] = ti
                y_row.append(np.asarray(fn(self.params_from_theta(theta)), dtype=float))
            y_all.append(y_row)

        y_arr    = np.array(y_all)                        # (n_unknown, n_samples, n_out)
        dy       = np.diff(y_arr, axis=1)                 # (n_unknown, n_samples-1, n_out)
        dydtheta = dy / np.diff(theta_grid)[:, np.newaxis] # broadcast over outputs
        y_mean   = np.mean(y_arr, axis=1)                 # (n_unknown, n_out)
        S = np.mean(dydtheta, axis=1) / (np.abs(y_mean) + 1e-12)
        self.S = S
        return S

    def _local_sensitivity_at(
        self,
        params: dict,
        eps_theta: float = 1e-6,
        measures: Callable | None = None,
    ) -> np.ndarray:
        """One-sided finite-difference sensitivity at a given parameter point.

        Returns
        -------
        S : ndarray, shape (n_unknown, n_outputs)
        """
        fn = measures if measures is not None else self.model_fn
        theta_base = self.p_to_theta(
            np.array([params[up.key] for up in self.unknown_parameters])
        )
        y_base = np.asarray(fn(self.params_from_theta(theta_base)), dtype=float)

        S = []
        for i in range(len(self.unknown_parameters)):
            t_pert    = theta_base.copy()
            t_pert[i] = min(1., theta_base[i] + eps_theta)
            y_pert    = np.asarray(fn(self.params_from_theta(t_pert)), dtype=float)
            dt        = t_pert[i] - theta_base[i] + _EPS
            S.append((y_pert - y_base) / dt / (np.abs(y_base) + 1e-12))
        return np.array(S)   # (n_unknown, n_outputs)

    # ------------------------------------------------------------------
    # Global sensitivity (Sobol sampling)
    # ------------------------------------------------------------------

    def compute_global_sensitivity(
        self,
        m: int = 8,
        y_exp: np.ndarray | None = None,
        rmse_limit: float = 0.3,
        check_samples: bool = False,
        measures: Callable | None = None,
        filename_to_save: str | None = None,
        print_progress: bool = False,
    ):
        """Global sensitivity via Sobol sampling (Goshtasbi 2020, Eqs 8–10).

        Generates 2^m Sobol samples in normalised parameter space,
        evaluates local sensitivities at each valid sample, then computes
        median/std sensitivity norms and pairwise collinearity indices.

        Parameters
        ----------
        m : int
            Sobol exponent; total samples = 2**m.
        y_exp : array_like, optional
            Needed only when ``check_samples=True``.
        rmse_limit : float
            Reject samples whose RMSE exceeds this value.
        check_samples : bool
            If True, evaluate the model and reject samples with poor fit.
        measures : callable, optional
        filename_to_save : str, optional
            If given, save results to a .npz file.
        print_progress : bool

        Returns
        -------
        cosPhi_med_ij, norm_s_i, S_med, S_std,
        S_med_i, S_std_i, S_n, n_valid, p_valid
        """
        n_unk = len(self.unknown_parameters)
        fn    = measures if measures is not None else self.model_fn

        sampler       = qmc.Sobol(d=n_unk, scramble=False)
        theta_samples = sampler.random_base2(m)
        n_total       = 2 ** m

        S_n, p_valid = [], []
        for n, theta in enumerate(theta_samples):
            if print_progress and n % max(1, n_total // 20) == 0:
                print(f"  sample {n}/{n_total} ({len(S_n)} valid)", end='\r')
            px = self.params_from_theta(theta)

            if check_samples and y_exp is not None:
                try:
                    y_m  = np.asarray(fn(px), dtype=float)
                    valid = np.isfinite(y_m) & np.isfinite(np.asarray(y_exp))
                    res  = np.asarray(y_exp)[valid] - y_m[valid]
                    ok   = float(np.sqrt(np.dot(res, res) / max(len(res), 1))) < rmse_limit
                except Exception:
                    ok = False
            else:
                ok = True

            if ok:
                try:
                    s = self._local_sensitivity_at(px, measures=fn)
                    if np.all(np.isfinite(s)):
                        S_n.append(s)
                        p_valid.append(px)
                except Exception:
                    pass

        if print_progress:
            print(f"\n  → {len(S_n)}/{n_total} valid samples")

        S_n = np.array(S_n)                              # (n_valid, n_unk, n_out)
        norm_s_i    = np.linalg.norm(S_n, axis=-1)       # (n_valid, n_unk)
        dot_product = np.einsum('nij,nkj->nik', S_n, S_n)
        norm_prod   = np.einsum('ni,nj->nij', norm_s_i, norm_s_i)
        cosPhi_n    = np.tril(np.abs(dot_product / np.maximum(norm_prod, 1e-12)))

        self.norm_s_i      = norm_s_i
        self.cosPhi_med_ij = np.median(cosPhi_n, axis=0)
        self.S_med_i       = np.median(norm_s_i, axis=0)
        self.S_std_i       = np.std(norm_s_i, axis=0)
        self.S_med         = np.median(S_n, axis=0)
        self.S_std         = np.std(S_n, axis=0)
        self.S_n           = S_n
        self.n_valid        = len(S_n)

        if filename_to_save:
            np.savez(filename_to_save,
                     cosPhi_med_ij=self.cosPhi_med_ij,
                     norm_s_i=self.norm_s_i,
                     S_med=self.S_med, S_std=self.S_std,
                     S_med_i=self.S_med_i, S_std_i=self.S_std_i,
                     S_n=self.S_n, n_valid=self.n_valid)

        return (self.cosPhi_med_ij, self.norm_s_i,
                self.S_med, self.S_std,
                self.S_med_i, self.S_std_i,
                self.S_n, self.n_valid, p_valid)

    def load_global_sensitivity_results(self, filename: str):
        """Load saved global sensitivity results from a .npz file."""
        d = np.load(filename)
        self.norm_s_i       = d['norm_s_i']
        self.cosPhi_med_ij  = d['cosPhi_med_ij']
        self.S_med_i        = d['S_med_i']
        self.S_std_i        = d['S_std_i']
        self.S_med          = d['S_med']
        self.S_std          = d['S_std']
        self.S_n            = d['S_n']
        self.n_valid        = int(d['n_valid'])

    # ------------------------------------------------------------------
    # Identifiability ranking
    # ------------------------------------------------------------------

    def get_smallest_hessian_eigenvalues(self):
        """Incremental Hessian eigenvalue analysis (Lund & Foss 2008).

        Uses QR decomposition with column pivoting to order parameters by
        importance, then adds one parameter at a time and records the
        smallest eigenvalue of the incremental Hessian H = S·Sᵀ.

        Returns
        -------
        P : ndarray
            Parameter indices in order of decreasing identifiability.
        min_eigvals : list[float]
        n_params : int
        """
        if self.S_med is None:
            raise RuntimeError("Run compute_global_sensitivity first.")
        S = self.S_med.copy().T     # (n_outputs, n_unknown)
        _, _, P = qr(S, mode='economic', pivoting=True)

        min_eigvals = []
        for i in range(len(self.unknown_parameters)):
            sel = np.sort(P[:i + 1])
            H   = self.S_med[sel] @ self.S_med[sel].T
            min_eigvals.append(float(np.min(np.abs(eigvals(H)))))
        return P, min_eigvals, len(self.unknown_parameters)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot_global_sensitivity(
        self,
        fig=None, ax=None,
        color: str = 'C0',
        xlabel_angle: float = 45,
        figsize=(5, 3),
        parameter_order=None,
    ):
        """Plot median normalised sensitivity norms with sample scatter."""
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        n   = len(self.unknown_parameters)
        xi  = np.arange(n) if parameter_order is None else np.asarray(parameter_order)
        lbl = [self.unknown_parameters[j].label for j in xi]

        for pos, pidx in enumerate(xi):
            ax.plot(pos * np.ones(self.norm_s_i.shape[0]),
                    self.norm_s_i[:, pidx], '.' + color, alpha=0.2, ms=3)
        ax.semilogy(np.arange(n), self.S_med_i[xi], '-s' + color)
        ax.set_xticks(np.arange(n), labels=lbl)
        ax.set_xlim(-0.5, n - 0.5)
        if xlabel_angle > 0:
            plt.setp(ax.get_xticklabels(), rotation=xlabel_angle, ha='right')
        ax.set_ylabel('Median normalised sensitivity')
        if fig is not None:
            fig.tight_layout()
        return fig, ax

    def plot_colinearity_map(
        self,
        fig=None, ax=None,
        cmap: str = 'viridis',
        xlabel_angle: float = 45,
        figsize=(4, 4),
        write_text: bool = True,
    ):
        """Plot the pairwise median collinearity index matrix."""
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        n   = len(self.unknown_parameters)
        xi  = np.arange(n)
        lbl = [up.label for up in self.unknown_parameters]

        im = ax.pcolormesh(xi, xi, self.cosPhi_med_ij, vmin=0, vmax=1,
                           cmap=plt.colormaps[cmap])
        if write_text:
            for i in xi:
                for j in xi:
                    ax.text(j, i, f'{self.cosPhi_med_ij[i, j]:.2f}',
                            ha='center', va='center', color='w', fontsize=7)
        fig.colorbar(im, ax=ax)
        ax.set_xticks(xi, labels=lbl)
        ax.set_yticks(xi, labels=lbl)
        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
        if xlabel_angle > 0:
            plt.setp(ax.get_xticklabels(), rotation=xlabel_angle,
                     ha='center', va='bottom', rotation_mode='default')
        ax.set_aspect('equal')
        if fig is not None:
            fig.tight_layout()
        return fig, ax

    def plot_parameter_ranking(self, fig=None, ax=None):
        """Plot Hessian eigenvalues vs. parameter ranking order."""
        n = len(self.unknown_parameters)
        P, eigvals_list, _ = self.get_smallest_hessian_eigenvalues()
        if ax is None:
            fig, ax = plt.subplots(figsize=(max(6, n * 0.6), 3))
        lbl = [self.unknown_parameters[i].label for i in P]
        ax.semilogy(1 + np.arange(n), np.abs(eigvals_list), '-s')
        ax.set_xticks(1 + np.arange(n), labels=lbl, rotation=45, ha='right')
        ax2 = ax.twiny()
        ax.set_xlim(0.5, n + 0.5)
        ax2.set_xticks(ax.get_xticks())
        ax2.set_xlim(0.5, n + 0.5)
        ax2.set_xlabel('Number of selected parameters')
        ax.set_xlabel('Ranked parameters')
        ax.set_ylabel('Smallest Hessian\neigenvalue')
        ax.grid(True, alpha=0.3)
        if fig is not None:
            fig.tight_layout()
        return P, fig, ax


# ---------------------------------------------------------------------------
# Legacy classes kept for backward compatibility with older notebooks.
# New code should use ParameterEstimation + UnknownParameter instead.
# ---------------------------------------------------------------------------

from scipy.integrate import solve_ivp as _solve_ivp  # noqa: E402


class DynamicModel:  # noqa: D101 — legacy, see ParameterEstimation
    def __init__(self, f, h, x0=[], p=None, u=None, ode_solution_method='BDF'):
        self.f = f; self.h = h; self.p = p; self.x0 = x0
        self.u = u; self.ode_solution_method = ode_solution_method

    def set_conditions(self, u):  self.u = u
    def set_params(self, p):      self.p = p
    def set_initial_conditions(self, x0): self.x0 = x0

    def set_unknown_params(self, p_list):
        self.unknown_p_list  = p_list
        self.p_i_min         = np.array([p[1][0] for p in p_list])
        self.p_i_max         = np.array([p[1][1] for p in p_list])
        self.p_i_isLinear    = np.array([p[2] for p in p_list])
        self.p_i_name        = [p[0] for p in p_list]
        self.p_i_label       = [p[3] for p in p_list]
        if len(p_list[0]) > 4:
            self.p_i_guess = [p[4] for p in p_list]
            self.p.update(dict(zip(self.p_i_name, self.p_i_guess)))

    def solve(self, time, u=None, x0=[], parameters_dict=None,
              rtol=1e-4, atol=1e-5, vectorized=False, sparsity=None):
        parameters_dict = parameters_dict if parameters_dict else self.p
        u   = u  if u   else self.u
        x0  = x0 if len(x0) > 0 else self.x0
        sol = _solve_ivp(self.f, (time[0], time[-1]), y0=x0, t_eval=time,
                         args=(u, parameters_dict), method=self.ode_solution_method,
                         rtol=rtol, atol=atol, vectorized=vectorized, jac_sparsity=sparsity)
        y = self.h(sol.t, sol.y, u, parameters_dict)
        self.t, self.x, self.y = sol.t, sol.y, y
        return sol.t, sol.y, y

    def p_to_theta(self, unknown_p_values):
        log_mask = ~self.p_i_isLinear
        t = (unknown_p_values - self.p_i_min) / (self.p_i_max - self.p_i_min)
        t[log_mask] = ((np.log(unknown_p_values[log_mask]) - np.log(self.p_i_min[log_mask]))
                       / (np.log(self.p_i_max[log_mask]) - np.log(self.p_i_min[log_mask])))
        return t

    def theta_to_p(self, theta_k):
        return np.where(self.p_i_isLinear,
                        self.p_i_min + (self.p_i_max - self.p_i_min) * theta_k,
                        self.p_i_min * np.exp(
                            (np.log(self.p_i_max + 1e-20) - np.log(self.p_i_min + 1e-20)) * theta_k))

    def residuals(self, y_exp, t, u=None, x0=None, p=None):
        _, _, y = self.solve(t, u, x0, p)
        return y_exp - y

    def estimate(self, y_exp=None, t=0, u=None, x0=None, p=None, residuals=None,
                 print_iterations=False, popsize=10, workers=1,
                 rtol=0, atol=0, ftol=0, penalty_threshold=1e-2,
                 vectorized=False, method='differential_evolution',
                 initial_guess=None, maxiter=120):
        if not p:
            p = self.p if self.p else {}
        if not residuals:
            residuals = lambda px: self.residuals(y_exp, t, u, x0, px)

        def f(x):
            px  = p.copy()
            px.update({pi[0]: v for pi, v in zip(self.unknown_p_list, self.theta_to_p(x))})
            res = residuals(px)
            pen = (np.where(np.abs(res) > penalty_threshold,
                            10 * (res - penalty_threshold), 0)
                   if penalty_threshold > 0 else 0)
            return np.dot(res, res) / len(res) + (np.dot(pen, pen) if penalty_threshold > 0 else 0)

        bounds = [(0, 1)] * len(self.unknown_p_list)
        if method == 'differential_evolution':
            sol = differential_evolution(f, bounds=bounds, disp=True,
                                         popsize=popsize, polish=True, workers=workers,
                                         mutation=(0, 1), recombination=0.5, vectorized=vectorized,
                                         seed=2, init='latinhypercube', tol=rtol, atol=atol,
                                         maxiter=maxiter)
        else:
            sol = minimize(f,
                           x0=[0]*len(self.unknown_p_list) if not initial_guess
                           else self.p_to_theta(initial_guess),
                           method=method, bounds=bounds,
                           options={'ftol': ftol, 'maxiter': maxiter})
        return sol, self.theta_to_p(sol.x)

    # (sensitivity / plot methods omitted from legacy class for brevity)


class SteadyStateModel(DynamicModel):
    """Legacy steady-state model (wraps a callable h(params) -> y)."""

    def __init__(self, h, p=None):
        self.h = h
        self.p = p if p is not None else {}
        self.f = None; self.x0 = []; self.u = None
        self.ode_solution_method = None

    def solve(self, time=None, u=None, x0=None, parameters_dict=None):
        px = parameters_dict if parameters_dict else self.p
        y  = self.h(px)
        self.y = y
        return None, None, y
