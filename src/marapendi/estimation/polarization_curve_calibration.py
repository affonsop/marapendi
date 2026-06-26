from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import qmc
from scipy.linalg import qr, eigvals
from scipy.optimize import differential_evolution, minimize

from ..simulation.conditions import CellConditions, SideConditions
from ..cell.explicit_steady_state import ExplicitSteadyStateModel

@dataclass 
class Parameter: 
    value: float
    key: str = None
    symbol: str = None
    units: str = 'n.d.'
    factor: float = 1

@dataclass
class UnknownParameter(Parameter): 
    initial_guess: float = None
    lower_bound: float = None
    upper_bound: float = None
    is_linear: bool = True 
    

@dataclass 
class BaseModelCalibration: 
    
    def set_params(self, params): 
        self.params = params 

    def set_known_params(self, known_p_list): 
        self.params.update({p.key: p.value for p in known_p_list})

    def set_unknown_params(self, unknown_p_list):
        """
        Define the parameters to be estimated, their bounds and types.

        Parameters
        ----------
        p_list : list of tuples
            Each tuple is (name, (min, max), is_linear, label).
        """
        self.unknown_p_list = unknown_p_list
        self.p_i_min = np.array([p.lower_bound for p in unknown_p_list])
        self.p_i_max = np.array([p.upper_bound for p in unknown_p_list])
        self.p_i_is_linear = np.array([p.is_linear for p in unknown_p_list])
        self.p_i_name = [p.key for p in unknown_p_list]
        self.p_initial_guess = np.array([p.initial_guess for p in unknown_p_list])
        self.params.update({p.key: p.initial_guess for p in unknown_p_list})

    def p_to_theta(self, unknown_p_values):
       
        log_mask = ~self.p_i_is_linear
        theta_i_k = (unknown_p_values - self.p_i_min) / (self.p_i_max - self.p_i_min)

        theta_i_k[log_mask] = ((np.log(unknown_p_values[log_mask]) - np.log(self.p_i_min[log_mask])) / 
                               (np.log(self.p_i_max[log_mask]) - np.log(self.p_i_min[log_mask])))
        
        return theta_i_k

    def theta_to_p(self, theta_k):
        """
        Convert normalized parameters to physical parameters.

        Follows equations 5 and 6 in Goshtasbi et al. (2020).

        Parameters
        ----------
        theta_k : array_like
            Normalized parameters in [0,1].

        Returns
        -------
        p_i_k : ndarray
            Physical parameters.
        
        References
        ----------
        Goshtasbi, A. et al. J. Electrochem. Soc. 167, 044504 (2020).
        """
        p_i_k = np.where(self.p_i_is_linear,
                         self.p_i_min + (self.p_i_max - self.p_i_min) * theta_k,
                         self.p_i_min * np.exp((np.log(self.p_i_max + 1e-20) - np.log(self.p_i_min + 1e-20)) * theta_k))
        return p_i_k

    def compute_y_sim(self, params): 
        pass 

    def compute_residuals(self, params): 
        pass 

    def calculate_local_sensitivity_neighborhood(self, params=None, eps_p=0):
        """
        Compute local sensitivities by finite differences in the neighborhood of the parameters.
        
        Parameters
        ----------
        t : array_like
            Time vector.
        u, x0, p : optional
            Same as above.
        eps_p : float
            Relative difference for derivative estimation by finite difference.

        Returns
        -------
        S : ndarray
            Local sensitivities.
        
        References
        ----------
        Goshtasbi, A. et al. J. Electrochem. Soc. 167, 044504 (2020).
        """
        # Generate normalized samples in parameter space [0,1]
        

        y = []  # to store outputs for each parameter
        for i, p_i in enumerate(self.unknown_p_list):
            y_i = []
            # Make sure to start with a fresh copy of parameter dict
            p_modified = params.copy() if params else self.params.copy()
            unknown_p_values = np.array([p_modified[unknown_p.key] for unknown_p in self.unknown_p_list])
            theta_i = self.p_to_theta(unknown_p_values)[i]
            theta = [theta_i, np.minimum(1,theta_i+eps_p)]
            for k in range(len(theta)):
                # Get the parameter value at theta[k] for this parameter
                p_i_k = self.theta_to_p(theta[k])[i]
                # Update parameter set with this single varied parameter
                p_modified.update({p_i.key: p_i_k})
                # Solve model with updated parameter
                y_i_k = self.compute_y_sim(p_modified)
                y_i.append(y_i_k)
            y.append(y_i)


        # Convert collected outputs to numpy array of shape (n_parameters, n_samples, len(y))
        y = np.array(y)

        # Compute finite differences along sample axis (axis=1)
        dy = np.diff(y, axis=1)
        dtheta = np.diff(theta)  # uniform steps
    
        # Compute derivative dy/dtheta, broadcast over state dimension
        dydtheta = dy / dtheta[:, np.newaxis]

        # Compute normalized sensitivity as in equation 7
        # S = (1 / mean(y)) * mean( dy/dtheta )
        # Added small epsilon in denominator to avoid division by zero
        S = 1 / (1e-20 + np.mean(y, axis=1)) * np.mean(dydtheta, axis=1)

        self.S = S
        return S

    def compute_global_sensitivity(self, params=None,
                                m=8, check_samples=False,
                                rmse_limit=0.3, print_px=False, filename_to_save=None):
        """
        Compute global sensitivities using Sobol sampling.

        Follows equation 8, 9 and 10 in Goshtasbi et al. (2020). 

        Parameters
        ----------
        t, u, x0, p : optional
            Same as above.
        n_samples : int
            Samples for local sensitivity.
        m : int
            Generates 2**m Sobol samples.
        check_samples : bool
            If True, check RMSE against `y_exp`. Similar to Goshtasbi et al. (2020), but uses L2 norm instead.
        y_exp : array_like, optional
            Experimental data for RMSE check.
        rmse_limit : float
            RMSE threshold.
        print_px : bool
            Print parameter sets.
        filename_to_save :  str
            Filename where to save results. Do not save if None. 

        Returns
        -------
        cosPhi_med_ij : ndarray
            Median co-linearity index matrix.
        norm_s_i : ndarray
            Norm of sensitivities.
        S_med, S_std : ndarray
            Median and std of sensitivities.
        S_med_i, S_std_i : ndarray
            Median and std by parameter.
        S_n : ndarray
            Sensitivity samples.
        n_valid : int
            Number of valid samples.

        References
        ----------
        Goshtasbi, A. et al. J. Electrochem. Soc. 167, 044504 (2020).
        """
        n_unknown_p = len(self.unknown_p_list)
        # Generate Sobol samples in the normalized [0,1] space for each parameter
        sampler = qmc.Sobol(d=n_unknown_p, scramble=False,)
        theta_samples = sampler.random_base2(m)

        # Use provided parameter dict, else self.p or empty
        params = params if params else self.params
        
        p_valid = []
        S_n = []
        for n in range(2**m):
            # Transform Sobol sample to actual parameter values
            p_i_n = self.theta_to_p(theta_samples[n])
            # Build parameter dict for this sample
            px = params.copy()
            px.update({p_i: v for p_i, v in zip(self.p_i_name, p_i_n)})

            # Optionally check RMSE to reject unrealistic samples
            if check_samples:
                res = self.compute_residuals(px)
                res = res[~np.isnan(res)]
                isValid = np.sqrt(np.dot(res, res) / len(res)) < rmse_limit
            else:
                isValid = True
                
            if isValid: 
                if print_px:
                    print(px)
                # Compute local sensitivities for this global sample
                s_n = self.calculate_local_sensitivity_neighborhood(px, eps_p=1e-6)
                S_n.append(s_n)
                p_valid.append(px)

        n_valid = len(S_n)
        S_n = np.array(S_n)

        # Compute norms of sensitivities ||S|| for each parameter and sample
        norm_s_i = np.linalg.norm(S_n, axis=-1)

        # Compute pairwise colinearity indices (cosPhi) between parameters
        cosPhi_n = np.ones((len(S_n), n_unknown_p, n_unknown_p)) * np.nan
        # Compute all pairwise dot products for all samples
        dot_product = np.einsum('nij,nkj->nik', S_n, S_n)

        # Compute all pairwise norm products for all samples
        norm_product = np.einsum('ni,nj->nij', norm_s_i, norm_s_i)
        norm_product = np.maximum(norm_product, 1e-12)

        # Compute cosine similarity for all samples
        cosPhi_n = np.abs(dot_product / norm_product)

        # If you only need the upper triangular part (including diagonal):
        cosPhi_n = np.tril(cosPhi_n)
                
        # Store statistics on sensitivity norms and colinearity
        self.norm_s_i = norm_s_i
        self.cosPhi_med_ij = np.median(cosPhi_n, axis=0)
        self.S_med_i = np.median(norm_s_i, axis=0)
        self.S_std_i = np.std(norm_s_i, axis=0)
        self.S_med = np.median(S_n, axis=0)
        self.S_std = np.std(S_n, axis=0)
        self.S_n = S_n
        self.n_valid = n_valid

        # If save results
        if filename_to_save: 
            np.savez(filename_to_save, 
                    cosPhi_med_ij=self.cosPhi_med_ij, 
                    norm_s_i=self.norm_s_i, 
                    S_med=self.S_med, 
                    S_std=self.S_std, 
                    S_med_i=self.S_med_i, 
                    S_std_i=self.S_std_i, 
                    S_n=self.S_n, 
                    n_valid=self.n_valid) 


    def load_global_sensitivity_results(self, filename): 
        """
        Load global sensitivity results from file. 

        Parameters
        ----------
        filname : str
            Filename where results are stored. 
        """
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
        """
        Compute the smallest eigenvalues of successive Hessian approximations 
        constructed from the median sensitivity matrix.

        This helps to analyze parameter identifiability and numerical conditioning
        by progressively adding parameters and observing the smallest eigenvalue
        of the resulting Hessian-like matrix.

        Returns
        -------
        P : ndarray
            Array of parameter indices sorted by importance (from QR decomposition with pivoting).
        min_eigvals : list of float
            List of smallest eigenvalues for each incremental Hessian matrix,
            starting from 1 parameter up to all parameters.
        num_parameters : int
            Total number of parameters considered.

        Notes
        -----
        - Uses QR decomposition with column pivoting to identify influential parameters.
        - The Hessian approximation is built as `H = S_selected x S_selected.T`
        where `S_selected` contains rows of the median sensitivity matrix.
        - Small eigenvalues close to zero indicate collinearity or poor identifiability.

        References
        ----------
        Goshtasbi, A. et al. J. Electrochem. Soc. 167, 044504 (2020).
        Lund, B. F. & Foss, B. A. Automatica (Oxf.) 44, 278–281 (2008).
        """
        num_parameters = len(self.unknown_p_list)

        # Transpose sensitivity matrix to have shape (n_parameters, n_outputs)
        S = self.S_med.copy()
        S = S.transpose()

        # Perform QR decomposition with pivoting to get parameter ordering
        Q, R, P = qr(S, mode='economic', pivoting=True)

        min_eigvals = []
        indices = np.arange(num_parameters)
        for i in range(num_parameters):
            # Select first i+1 parameters based on pivoting order
            selected_indices = indices[np.isin(indices, P[:i+1])]
            
            # Build Hessian-like matrix H = S_selected x S_selected^T
            H = np.matmul(self.S_med[selected_indices,:], self.S_med[selected_indices,:].transpose())
            
            # Compute smallest eigenvalue
            min_eigvals.append(np.min(eigvals(H)))

        self.P = P
        self.min_eigvals = min_eigvals
        self.num_parameters = num_parameters

    def plot_parameter_ranking(self):
        """
        Plot the ranking of parameters based on smallest eigenvalues of the Hessian matrix.

        Parameters
        ----------
        filename : str
            (Currently unused in function, could be for saving figure).
        model : object
            Model containing the list of unknown parameters.

        Returns
        -------
        P : ndarray
            Permutation indices indicating parameter ranking.
        num_parameters : int
            Total number of parameters.
        fig, ax, ax2 : Matplotlib objects
            Figure and axes for further customization or saving.
        """
        n_unknown_p = len(self.unknown_p_list)

        # Compute pivoted QR-based eigenvalue ranking
        self.get_smallest_hessian_eigenvalues()

        # Set up main plot
        fig, ax = plt.subplots(figsize=(12, 2.5))
        
        # Plot the smallest eigenvalues vs parameter ranking
        ax.semilogy(1 + np.arange(n_unknown_p), np.abs(self.min_eigvals), '-s')
        
        # X-axis: ranks with parameter names (ordered by pivoting)
        ax.set_xticks(1 + np.arange(n_unknown_p))
        ax.set_xticklabels([self.unknown_p_list[i].symbol for i in self.P], rotation=45)
        
        # Twin x-axis: showing simply number of selected parameters
        ax2 = ax.twiny()
        ax.set_xlim([0.5, n_unknown_p + 0.5])
        ax2.set_xticks(ax.get_xticks())
        ax2.set_xlim([0.5, n_unknown_p + 0.5])
        ax2.set_xlabel('Number of selected parameters')
        
        # Label left y-axis
        ax.set_xlabel('Ranked parameters')
        ax.set_ylabel('Smallest Hessian\neigenvalue')
        ax.grid()
        
        # Adjust layout
        fig.tight_layout()
        
        return fig, ax, ax2

@dataclass
class SteadyStatePolarizationCurveCalibration(BaseModelCalibration): 
    # Values must be on SI units, 
    conditions_dataset: pd.DataFrame   # dataset with columns case, cell-temperature, pressure-ca, pressure-an, rh-ca, rh-an, st-ca, st-an, min-current-at-st-ca, min-current-at-st-an
    experimental_dataset: pd.DataFrame # dataset with columns case, current_density, voltage, hfr 
    cell_creator: Callable 
    known_parameters: list = field(default_factory=list)
    unknown_parameters: list = field(default_factory=list)
    cell_model: ExplicitSteadyStateModel = field(default_factory=ExplicitSteadyStateModel)

    def __post_init__(self):
        self.params = {}
        self.set_unknown_params(self.unknown_parameters)
        self.set_known_params(self.known_parameters)
        self.full_case_list = self.experimental_dataset['case'].unique()

        self.populate_exp_dataset_conditions()
        self.build_cases_conditions()
        
        self.hfr_mask = {case: np.isfinite(self.get_case_dataset(case)['hfr']) for case in self.full_case_list}
        self.hfr_weight_factor = (
              np.sum(np.isfinite(self.experimental_dataset['voltage']))
             / np.sum(np.isfinite(self.experimental_dataset['hfr'])) 
        )
        
    def solve_case(self, cell, case):
        cond = self.case_conditions[case]
        state = self.cell_model.set_initial_conditions(cell, cond)
        return self.cell_model.solve(cell, cond, state) 
    
    def apply_hfr_weights(self, hfr): 
        return hfr * 1e4 * self.hfr_weight_factor

    

    
    def build_y_sim_cases(self, cell, case_list): 
        return np.concatenate([
            self.build_y_sim(cell, case) for case in case_list
        ])

    def build_y_sim(self, cell, case):
        state = self.solve_case(cell, case)
        hfr = self.cell_model.voltage_model.high_frequency_resistance(cell, state)
        hfr_sim = (
            self.apply_hfr_weights(hfr) * self.hfr_mask[case]
        )
        return np.concatenate([state.cell_voltage, hfr_sim])
    
    def build_y_exp_cases(self, case_list): 
        return np.concatenate([
            self.build_y_exp(case) for case in case_list
        ])
    
    def build_y_exp(self, case): 
        case_dataset = self.get_case_dataset(case)
        hfr = case_dataset['hfr']
        hfr_exp = (
            self.apply_hfr_weights(hfr) * self.hfr_mask[case]
        )
        return np.concatenate([case_dataset['voltage'], hfr_exp])

    def compute_y_sim(self, params=None, cell=None, case_list=[]): 
        if len(case_list) == 0: 
            case_list = self.full_case_list
        if params and not cell: 
            cell = self.cell_creator(params)
        return self.build_y_sim_cases(cell, case_list)

    def compute_residuals(self, params=None, cell=None, case_list=[]):
        if len(case_list) == 0: 
            case_list = self.full_case_list 
        return self.build_y_exp_cases(case_list) - self.compute_y_sim(params, cell, case_list)
     
    def populate_exp_dataset_conditions(self): 
        for side in ('ca', 'an'):
            if f'min-current-at-st-{side}' not in self.conditions_dataset.columns: 
                self.conditions_dataset[f'min-current-at-st-{side}'] =  0
        self.experimental_dataset['current-density'] += 1

        for column in self.conditions_dataset.columns: 
            if column not in self.experimental_dataset.columns: 
                self.experimental_dataset = self.experimental_dataset.merge(self.conditions_dataset[['case', column]], on='case')

    def get_case_dataset(self, case):
        return self.experimental_dataset[self.experimental_dataset['case']==case] 
    
    def build_cases_conditions(self):
        self.case_conditions = {}
        for case in self.full_case_list:
            self.case_conditions[case] = self._make_conditions(case)

    def _make_conditions(self, case):
        case_dataset = self.get_case_dataset(case)
        
        return CellConditions(
            current_density=case_dataset['current-density'].values,
            cell_temperature=case_dataset['cell-temperature'].values,
            ca=SideConditions(
                inlet_temperature=case_dataset['cell-temperature'].values, 
                outlet_pressure=case_dataset['pressure-ca'].values,
                inlet_relative_humidity=case_dataset['rh-ca'].values,
                dry_o2_mole_fraction=0.21, 
                stoichiometry=case_dataset['st-ca'].values * np.maximum(case_dataset['min-current-at-st-ca'].values / case_dataset['current-density'].values, 1),
            ),
            an=SideConditions(
                inlet_temperature=case_dataset['cell-temperature'].values, 
                outlet_pressure=case_dataset['pressure-an'].values,
                inlet_relative_humidity=case_dataset['rh-an'].values,
                dry_h2_mole_fraction=1.0, 
                stoichiometry=case_dataset['st-an'].values * np.maximum(case_dataset['min-current-at-st-an'].values / case_dataset['current-density'].values, 1),
            ),
        )
