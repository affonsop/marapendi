"""
Base class for model calibration using sensitivity analysis and cross-validation.

:class:`BaseModelCalibration` handles parameter normalisation, global sensitivity
analysis (Sobol + local finite-differences, Goshtasbi et al. 2020), Hessian-based
parameter ranking, and k-fold cross-validation with optional model-complexity sweeps.
Concrete subclasses implement :meth:`compute_y_sim` and :meth:`compute_residuals`.
"""
from dataclasses import dataclass
import time
import random
import os

import numpy as np
import pandas as pd
from scipy.stats import qmc
from scipy.linalg import qr, eigvals
from scipy.optimize import differential_evolution, minimize

from .parameters import Parameter, UnknownParameter


@dataclass
class BaseModelCalibration:

    known_parameters: list
    unknown_parameters: list

    def __post_init__(self):
        self.params = {}
        self.reset_unknown_parameters()
        self.full_case_list = []

    def set_params(self, params):
        """Replace the full parameter dict with *params*."""
        self.params = params

    def set_known_params(self, known_p_list):
        """Merge fixed-value parameters from *known_p_list* into ``self.params``."""
        self.params.update({p.key: p.value for p in known_p_list})

    def set_unknown_params(self, unknown_p_list):
        """Rebuild all derived arrays and mappings from a list of UnknownParameter objects."""
        self.unknown_p_list = unknown_p_list
        self.p_i_min = np.array([p.lower_bound for p in unknown_p_list])
        self.p_i_max = np.array([p.upper_bound for p in unknown_p_list])
        self.p_i_is_linear = np.array([p.is_linear for p in unknown_p_list])
        self.p_i_name = [p.key for p in unknown_p_list]
        self.p_i_symbol = [p.symbol for p in unknown_p_list]
        self.p_i_index = {key: i for i, key in enumerate(self.p_i_name)}
        self.p_initial_guess = np.array([p.initial_guess for p in unknown_p_list])
        self.params.update({p.key: p.initial_guess for p in unknown_p_list})
        self.n_unkown_p = len(self.unknown_p_list)

    def subset_of_unknown_parameters(self, indices=None, keys=None):
        """Promote a subset of unknown parameters to known, keeping only the selected ones unknown.

        Parameters
        ----------
        indices : list of int, optional
            Positions in ``self.unknown_parameters`` to keep as unknown.
        keys : list of str, optional
            Alternative to *indices* — parameter keys to keep as unknown.
        """
        if indices is None:
            indices = [self.p_i_index[key] for key in keys]
        known_list, unknown_list = [], []
        for idx in range(len(self.unknown_parameters)):
            if idx in indices:
                unknown_list.append(self.unknown_parameters[idx])
            else:
                known_list.append(self.unknown_parameters[idx])
        self.set_known_params(self.known_parameters + known_list)
        self.set_unknown_params(unknown_list)

    def reset_unknown_parameters(self):
        """Restore all parameters to their original known/unknown split."""
        self.set_known_params(self.known_parameters)
        self.set_unknown_params(self.unknown_parameters)

    def p_to_theta(self, unknown_p_values):
        """Map physical parameter values to the normalised [0, 1] space (eqs. 5–6, Goshtasbi et al. 2020).

        Parameters with ``is_linear=True`` use linear scaling; those with
        ``is_linear=False`` use log-scale normalisation.
        """
        log_mask = ~self.p_i_is_linear
        theta_i_k = (unknown_p_values - self.p_i_min) / (self.p_i_max - self.p_i_min)
        theta_i_k[log_mask] = (
            (np.log(unknown_p_values[log_mask]) - np.log(self.p_i_min[log_mask]))
            / (np.log(self.p_i_max[log_mask]) - np.log(self.p_i_min[log_mask]))
        )
        return theta_i_k

    def theta_to_p(self, theta_k):
        """Map normalised [0, 1] parameters to physical values (eqs. 5–6, Goshtasbi et al. 2020)."""
        p_i_k = np.where(
            self.p_i_is_linear,
            self.p_i_min + (self.p_i_max - self.p_i_min) * theta_k,
            self.p_i_min * np.exp(
                (np.log(self.p_i_max + 1e-20) - np.log(self.p_i_min + 1e-20)) * theta_k
            ),
        )
        return p_i_k

    def compute_y_sim(self, params, case_list=[]):
        """Return the concatenated simulated output vector for *case_list*. Override in subclasses."""
        pass

    def compute_residuals(self, params, case_list=[]):
        """Return element-wise residuals (y_exp − y_sim) for *case_list*. Override in subclasses."""
        pass

    def calculate_local_sensitivity_neighborhood(self, unknown_p_values, eps_p=0):
        """Local sensitivity by finite differences in the normalised parameter space (eq. 7, Goshtasbi et al. 2020)."""
        y, dtheta = [], []
        theta_ref = self.p_to_theta(unknown_p_values)
        eps_matrix = np.eye(len(unknown_p_values)) * eps_p
        for i, p_i in enumerate(self.unknown_p_list):
            y_i = []
            for k in [0, 1]:
                p_i_k = self.theta_to_p(np.minimum(1, theta_ref + k * eps_matrix[i, :]))
                y_i_k = self.compute_y_sim(p_i_k)
                y_i.append(y_i_k)
            y.append(y_i)
            dtheta.append(eps_p)

        y = np.array(y)
        dtheta = np.array(dtheta)
        dy = np.diff(y, axis=1)
        dydtheta = dy / dtheta[:, np.newaxis]
        S = 1 / (1e-20 + np.mean(y, axis=1)) * np.mean(dydtheta, axis=1)

        self.S = S
        return S

    def compute_global_sensitivity(self, params=None, m=8, check_samples=False,
                                   rmse_limit=0.3, print_px=False, filename_to_save=None):
        """Global sensitivity via Sobol sampling and local finite-difference sensitivities (eqs. 8–10, Goshtasbi et al. 2020).

        Uses 2**m Sobol samples. If check_samples=True, rejects parameter sets whose RMSE exceeds rmse_limit.
        Results are stored on self (norm_s_i, cosPhi_med_ij, S_med_i, …) and optionally saved to filename_to_save.npz.
        """
        n_unknown_p = len(self.unknown_p_list)
        sampler = qmc.Sobol(d=n_unknown_p, scramble=False)
        theta_samples = sampler.random_base2(m)

        params = params if params else self.params

        S_n = []
        for n in range(2**m):
            p_i_n = self.theta_to_p(theta_samples[n])
            if check_samples:
                res = self.compute_residuals(p_i_n)
                res = res[~np.isnan(res)]
                isValid = np.sqrt(np.dot(res, res) / len(res)) < rmse_limit
            else:
                isValid = True

            if isValid:
                s_n = self.calculate_local_sensitivity_neighborhood(p_i_n, eps_p=1e-6)
                S_n.append(s_n)

        n_valid = len(S_n)
        S_n = np.array(S_n)

        norm_s_i = np.linalg.norm(S_n, axis=-1)

        dot_product = np.einsum('nij,nkj->nik', S_n, S_n)
        norm_product = np.einsum('ni,nj->nij', norm_s_i, norm_s_i)
        norm_product = np.maximum(norm_product, 1e-12)
        cosPhi_n = np.abs(dot_product / norm_product)
        cosPhi_n = np.tril(cosPhi_n)

        self.norm_s_i = norm_s_i
        self.cosPhi_med_ij = np.median(cosPhi_n, axis=0)
        self.S_med_i = np.median(norm_s_i, axis=0)
        self.S_std_i = np.std(norm_s_i, axis=0)
        self.S_med = np.median(S_n, axis=0)
        self.S_std = np.std(S_n, axis=0)
        self.S_n = S_n
        self.n_valid = n_valid

        if filename_to_save:
            np.savez(
                filename_to_save,
                cosPhi_med_ij=self.cosPhi_med_ij,
                norm_s_i=self.norm_s_i,
                S_med=self.S_med,
                S_std=self.S_std,
                S_med_i=self.S_med_i,
                S_std_i=self.S_std_i,
                S_n=self.S_n,
                n_valid=self.n_valid,
            )

    def load_global_sensitivity_results(self, filename):
        """Load sensitivity results previously saved by compute_global_sensitivity."""
        npzfile = np.load(filename)
        self.norm_s_i = npzfile['norm_s_i']
        self.cosPhi_med_ij = npzfile['cosPhi_med_ij']
        self.S_med_i = npzfile['S_med_i']
        self.S_std_i = npzfile['S_std_i']
        self.S_med = npzfile['S_med']
        self.S_std = npzfile['S_std']
        self.S_n = npzfile['S_n']
        self.n_valid = npzfile['n_valid']

    def get_smallest_hessian_eigenvalues(self):
        """Rank parameters by identifiability via QR-pivoted Hessian eigenvalues (Goshtasbi et al. 2020; Lund & Foss 2008).

        Stores self.P (parameter ranking by QR pivot) and self.min_eigvals (smallest eigenvalue per subset size).
        Small eigenvalues indicate collinearity or poor identifiability.
        """
        S = self.S_med.copy().transpose()
        Q, R, P = qr(S, mode='economic', pivoting=True)

        min_eigvals = []
        indices = np.arange(self.n_unkown_p)
        for i in range(self.n_unkown_p):
            selected_indices = indices[np.isin(indices, P[:i + 1])]
            H = np.matmul(
                self.S_med[selected_indices, :],
                self.S_med[selected_indices, :].transpose(),
            )
            min_eigvals.append(np.min(eigvals(H)))

        self.P = P
        self.min_eigvals = min_eigvals

    def estimate(self, params=None, case_list=None,
                 popsize=10, workers=1,
                 rtol=0, atol=0, ftol=0,
                 method='differential_evolution', initial_guess=None, maxiter=120):
        """Minimise mean squared residuals over case_list. Defaults to differential_evolution; pass method='Nelder-Mead' etc. for scipy.optimize.minimize."""
        params = self.params if params is None else params
        case_list = self.full_case_list if case_list is None else case_list

        def f(x):
            res = self.compute_residuals(self.theta_to_p(x), case_list=case_list)
            n_measures = sum(1 - np.isnan(res))
            res = np.nan_to_num(res, nan=0)
            return np.dot(res, res) / n_measures

        if method == 'differential_evolution':
            sol = differential_evolution(
                f,
                bounds=tuple([0, 1] for _ in self.unknown_p_list),
                disp=True,
                callback=None,
                popsize=popsize, polish=True,
                workers=workers, mutation=(0, 1), recombination=0.5, vectorized=False,
                seed=2, init='latinhypercube', tol=rtol, atol=atol, maxiter=maxiter,
            )
        else:
            sol = minimize(
                f,
                x0=[0 for _ in self.unknown_p_list] if not initial_guess else self.p_to_theta(initial_guess),
                method=method,
                bounds=tuple([0, 1] for _ in self.unknown_p_list),
                options={'ftol': ftol, 'maxiter': maxiter},
            )

        return sol, self.theta_to_p(sol.x)

    def run_cross_validation(self, testing_cases, training_cases=None, estimate_kwargs=None):
        """Estimate on training_cases (all cases minus testing_cases by default) and return a results DataFrame."""
        if estimate_kwargs is None:
            estimate_kwargs = dict(popsize=10, ftol=1e-5, rtol=0.1, maxiter=120)

        if training_cases is None:
            training_cases = [case for case in self.full_case_list if case not in testing_cases]

        start_time = time.time()
        optimization_result, estimated_parameters = self.estimate(
            case_list=training_cases,
            **estimate_kwargs,
        )
        elapsed_time = time.time() - start_time

        result_dict = {
            "n_params": [int(self.n_unkown_p)],
            "computation_time": elapsed_time,
            "objective_value": (
                optimization_result.fun
                if hasattr(optimization_result, "fun")
                else optimization_result
            ),
        }
        result_dict.update(dict(zip(self.p_i_name, estimated_parameters)))
        return pd.DataFrame(result_dict)

    def set_k_folds(self, k=1):
        """Partition ``self.full_case_list`` into *k* folds and store them in ``self.k_folds``.

        When ``k == len(cases)`` the order is preserved (leave-one-out); otherwise
        the list is shuffled randomly before splitting.
        """
        cases = list(self.full_case_list)
        if k != len(cases):
            random.shuffle(cases)
        self.k_folds = [cases[i * len(cases) // k:(i + 1) * len(cases) // k] for i in range(k)]

    def run_k_fold_cross_validation(self, k_folds=None, estimate_kwargs=None,
                                    filename=None, output_dir='.'):
        """Run cross-validation for each fold and return a concatenated results DataFrame."""
        if k_folds is None:
            k_folds = self.k_folds

        if filename is not None:
            filepath = os.path.join(output_dir, f"k_fold_results_{filename}.csv")

        k_fold_results_df = pd.DataFrame()

        for fold_id, testing_cases in enumerate(k_folds):
            result_df = self.run_cross_validation(
                testing_cases=testing_cases,
                training_cases=None,
                estimate_kwargs=estimate_kwargs,
            )
            result_df.insert(0, 'fold_id', int(fold_id))
            k_fold_results_df = pd.concat([k_fold_results_df, result_df])
        k_fold_results_df['n_params'] = k_fold_results_df['n_params'].astype('Int64')
        k_fold_results_df['fold_id'] = k_fold_results_df['fold_id'].astype('Int64')
        if filename is not None:
            k_fold_results_df.to_csv(filepath, index=False, mode='w')
        return k_fold_results_df

    def run_k_fold_cross_validation_vs_complexity(
        self,
        n_params_list,
        force_restart=True,
        k_folds=None,
        estimate_kwargs=None,
        filename=None,
        output_dir='.',
    ):
        """Run k-fold CV for each complexity level in n_params_list, checkpointing to CSV after each level."""
        if filename is not None:
            filepath = os.path.join(output_dir, f"k_fold_results_{filename}.csv")

        existing_df = None
        if not force_restart:
            try:
                existing_df = pd.read_csv(filepath)
            except FileNotFoundError:
                pass

        chunks = [] if existing_df is None else [existing_df]

        for n in n_params_list:
            already_done = existing_df is not None and n in existing_df.n_params.unique()
            if not already_done or force_restart:
                self.automatic_parameter_selection(n)
                k_fold_results_df = self.run_k_fold_cross_validation(
                    k_folds,
                    estimate_kwargs,
                    filename=None,
                )
                chunks.append(k_fold_results_df)
                if filename is not None:
                    pd.concat(chunks).to_csv(filepath, index=False, mode='w')
                self.reset_unknown_parameters()

    def load_cross_validation_results(self, filename, dir='.'):
        """Read a previously saved k-fold CV results CSV and return it as a DataFrame."""
        filepath = os.path.join(dir, f"k_fold_results_{filename}.csv")
        if not os.path.exists(filepath):
            print(f"No existing CV results found at: {filepath}")
            return None
        return pd.read_csv(filepath)

    def automatic_parameter_selection(self, n_parameters):
        """Select the top-n_parameters parameters by Hessian ranking and move the rest to known."""
        if self.P is None:
            self.get_smallest_hessian_eigenvalues()
        self.subset_of_unknown_parameters(self.P[:n_parameters])
