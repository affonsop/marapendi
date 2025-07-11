import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
from scipy.stats import qmc
from scipy.linalg import qr, eigvals
class DynamicModel:
    """
    A class to define, simulate, and analyze non-linear dynamic systems in control engineering notation.

    Attributes
    ----------
    f : callable
        State differential equation: x_dot = f(t, x, u, p).
    h : callable
        Output equation: y = h(t, x, u, p).
    p : dict or None
        Dictionary of parameters p.
    x0 : array_like or None
        Initial state vector.
    u : dict or None
        Dictionary of inputs u(t).
    ode_solution_method : str
        Integration method passed to `scipy.integrate.solve_ivp`.
    """

    def __init__(self, f, h, x0=None, p=None, u=None, ode_solution_method='BDF'):
        """
        Initialize the dynamic model.

        Parameters
        ----------
        f : callable
            State differential equation: x_dot = f(t, x, u, p).
        h : callable
            Output equation: y = h(t, x, u, p). 
        x0 : array_like, optional
            Initial state vector.
        p : dict, optional
            Dictionary of parameters p.
        u : dict, optional
            Dictionary of inputs u(t).
        ode_solution_method : str, optional
            ODE solver method (default is 'BDF').
        """
        self.f = f
        self.h = h
        self.p = p
        self.x0 = x0
        self.u = u
        self.ode_solution_method = ode_solution_method

    def set_conditions(self, u):
        """
        Set or update the dictionary of input functions u(t).

        Parameters
        ----------
        u : dict
            Dictionary of inputs.
        """
        self.u = u

    def set_params(self, p):
        """
        Set or update the dictionary of model parameters.

        Parameters
        ----------
        p : dict
            Dictionary of parameters.
        """
        self.p = p

    def set_initial_conditions(self, x0):
        """
        Set or update initial conditions.

        Parameters
        ----------
        x0 : array_like
            Initial state vector.
        """
        self.x0 = x0

    def set_unknown_params(self, p_list):
        """
        Define the parameters to be estimated, their bounds and types.

        Parameters
        ----------
        p_list : list of tuples
            Each tuple is (name, (min, max), is_linear, label).
        """
        self.unknown_p_list = p_list
        self.p_i_min = np.array([p[1][0] for p in p_list])
        self.p_i_max = np.array([p[1][1] for p in p_list])
        self.p_i_isLinear = np.array([p[2] for p in p_list])
        self.p_i_name = [p[0] for p in p_list]
        self.p_i_label = [p[3] for p in p_list]

    def solve(self, time, u=None, x0=None, parameters_dict=None):
        """
        Solve the ODE system over the given time vector.

        Parameters
        ----------
        time : array_like
            Time points for evaluation.
        u : dict, optional
            Input dictionary (overrides self.u).
        x0 : array_like, optional
            Initial conditions (overrides self.x0).
        parameters_dict : dict, optional
            Parameters (overrides self.p).

        Returns
        -------
        t : ndarray
            Time vector.
        x : ndarray
            State trajectories.
        y : ndarray
            Output trajectories.
        """
        parameters_dict = parameters_dict if parameters_dict else self.p
        u = u if u else self.u
        x0 = x0 if x0 else self.x0
        sol = solve_ivp(self.f, (time[0], time[-1]),
                        y0=x0,
                        t_eval=time,
                        args=(u, parameters_dict),
                        method=self.ode_solution_method)
        y = self.h(sol.t, sol.y, u, parameters_dict)
        self.t, self.x, self.y = sol.t, sol.y, y
        return sol.t, sol.y, y

    def residuals(self, y_exp, t, u=None, x0=None, p=None):
        """
        Compute residuals between experimental data and model predictions.

        Parameters
        ----------
        y_exp : array_like
            Experimental output data.
        t : array_like
            Time vector.
        u : dict, optional
            Inputs.
        x0 : array_like, optional
            Initial state.
        p : dict, optional
            Parameters.

        Returns
        -------
        residuals : ndarray
            Differences y_exp - y_model.
        """
        _, _, y = self.solve(t, u, x0, p)
        return y_exp - y

    def estimate(self, y_exp, t, u=None, x0=None, p=None,
                 print_iterations=False, popsize=10, workers=1,
                 ftol=0, atol=0, penalty_threshold=1e-2):
        """
        Estimate unknown parameters by minimizing the mean of squared errors using differential evolution.

        Parameters
        ----------
        y_exp : array_like
            Experimental data.
        t : array_like
            Time vector.
        u, x0, p : optional
            Same as above.
        print_iterations : bool
            If True, print intermediate results.
        popsize, workers, ftol, atol : control evolution settings.
        penalty_threshold : float
            Threshold to penalize large residuals. See Goshtasi et al. (2020)

        Returns
        -------
        sol : OptimizeResult
            Result from `differential_evolution`.
        p_estimated : ndarray
            Estimated parameter values.

        References
        ----------
        Goshtasbi, A. et al. J. Electrochem. Soc. 167, 024518 (2020).
        """
        if not p:
            p = self.p if self.p else {}

        # Define objective function 
        def f(x):
            # Build parameter dict with current candidate
            px = p.copy()
            px.update({p_i[0]: v for p_i, v in zip(self.unknown_p_list, self.theta_to_p(x))})
            res = self.residuals(y_exp, t, u, x0, px)
            if penalty_threshold > 0:
                penalty = np.where(np.abs(res) > penalty_threshold, 10 * (res - penalty_threshold), 0)
            return np.dot(res, res) / len(res) + (np.dot(penalty, penalty) if penalty_threshold > 0 else 0)

        # Define callback function to print results if needed
        def print_res(intermediate_result):
            print('------'*5)
            p_est = self.theta_to_p(intermediate_result.x)
            print('RMSE : {:.1f} mV'.format(1e3 * np.sqrt(intermediate_result.fun)))
            for k, param in enumerate(self.unknown_p_list):
                print(param[0], param[1], '{:.2e}'.format(p_est[k]))
            print('------'*5)
            return intermediate_result.fun < ftol

        sol = differential_evolution(f,
                                     bounds=tuple(([0, 1] for _ in self.unknown_p_list)),
                                     disp=True,
                                     callback=print_res if print_iterations else None,
                                     popsize=popsize, polish=False,
                                     workers=workers, mutation=(0, 1.6),
                                     seed=2, init='sobol', atol=atol)
        return sol, self.theta_to_p(sol.x)

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
        p_i_k = np.where(self.p_i_isLinear,
                         self.p_i_min + (self.p_i_max - self.p_i_min) * theta_k,
                         self.p_i_min * np.exp((np.log(np.maximum(self.p_i_max, 1e-12)) - np.log(np.maximum(self.p_i_min, 1e-12))) * theta_k))
        return p_i_k

    def calculate_local_sensitivity(self, t, u=None, x0=None, p=None, n_samples=7):
        """
        Compute local sensitivities by finite differences.

        Follows equation 7 in Goshtasbi et al. (2020).

        Parameters
        ----------
        t : array_like
            Time vector.
        u, x0, p : optional
            Same as above.
        n_samples : int
            Number of samples per parameter.

        Returns
        -------
        S : ndarray
            Local sensitivities.
        
        References
        ----------
        Goshtasbi, A. et al. J. Electrochem. Soc. 167, 044504 (2020).
        """
        # Generate normalized samples in parameter space [0,1]
        theta = np.linspace(0, 1, n_samples)

        y = []  # to store outputs for each parameter
        for i, p_i in enumerate(self.unknown_p_list):
            y_i = []
            # Make sure to start with a fresh copy of parameter dict
            p_modified = p.copy() if p else self.p.copy()
            for k in range(n_samples):
                # Get the parameter value at theta[k] for this parameter
                p_i_k = self.theta_to_p(theta[k])[i]
                # Update parameter set with this single varied parameter
                p_modified.update({p_i[0]: p_i_k})
                # Solve model with updated parameter
                _, _, y_i_k = self.solve(t, u, x0, p_modified)
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
        S = 1 / (1e-12 + np.mean(y, axis=1)) * np.mean(dydtheta, axis=1)

        self.S = S
        return S


    def compute_global_sensitivity(self, t, u=None, x0=None, p=None, n_samples=7,
                                m=8, check_samples=False, y_exp=None,
                                rmse_limit=0.3, print_px=False):
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
        sampler = qmc.Sobol(d=n_unknown_p, scramble=False)
        theta_samples = sampler.random_base2(m)

        # Use provided parameter dict, else self.p or empty
        p = p if p else self.p if self.p else {}

        S_n = []
        for n in range(2**m):
            # Transform Sobol sample to actual parameter values
            p_i_n = self.theta_to_p(theta_samples[n])
            # Build parameter dict for this sample
            px = p.copy()
            px.update({p_i: v for p_i, v in zip(self.p_i_name, p_i_n)})

            # Optionally check RMSE to reject unrealistic samples
            if check_samples:
                res = self.residuals(y_exp, t, u, x0, px)
                isValid = np.sqrt(np.dot(res, res) / len(res)) < rmse_limit
            else:
                isValid = True

            if isValid:
                if print_px:
                    print(px)
                # Compute local sensitivities for this global sample
                s_n = self.calculate_local_sensitivity(t, u, x0, px, n_samples)
                S_n.append(s_n)

        n_valid = len(S_n)
        S_n = np.array(S_n)

        # Compute norms of sensitivities ||S|| for each parameter and sample
        norm_s_i = np.linalg.norm(S_n, axis=-1)

        # Compute pairwise colinearity indices (cosPhi) between parameters
        cosPhi_n = np.ones((len(S_n), n_unknown_p, n_unknown_p)) * np.nan
        for n in range(n_valid):
            for i in range(n_unknown_p):
                for j in range(n_unknown_p):
                    if i >= j:
                        # Avoid divide by zero by max with small epsilon
                        cosPhi_n[n, i, j] = np.abs(
                            np.dot(S_n[n, j, :], S_n[n, i, :]) 
                            / np.maximum(norm_s_i[n, i] * norm_s_i[n, j], 1e-12)
                        )

        # Store statistics on sensitivity norms and colinearity
        self.norm_s_i = norm_s_i
        self.cosPhi_med_ij = np.median(cosPhi_n, axis=0)
        self.S_med_i = np.median(norm_s_i, axis=0)
        self.S_std_i = np.std(norm_s_i, axis=0)
        self.S_med = np.median(S_n, axis=0)
        self.S_std = np.std(S_n, axis=0)
        self.S_n = S_n
        self.n_valid = n_valid

        return (self.cosPhi_med_ij, self.norm_s_i, self.S_med, self.S_std, 
                self.S_med_i, self.S_std_i, self.S_n, self.n_valid)


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

        return P, min_eigvals, num_parameters
    
    def plot(self, i=0, fig=None, ax=None):
        """
        Plot the i-th state trajectory.

        Parameters
        ----------
        i : int
            Index of state to plot.
        fig, ax : optional
            Matplotlib figure and axis.

        Returns
        -------
        fig, ax : Matplotlib objects.
        """
        if not ax:
            fig, ax = plt.subplots()
        ax.plot(self.t, self.x[i])
        return fig, ax

    def plot_global_sensitivity(self, fig=None, ax=None, cmap='viridis', color='C0',
                                xlabel_angle=45, figsize=(4, 3)):
        """
        Plot global sensitivity metrics.

        Parameters
        ----------
        fig, ax : optional
            Matplotlib figure and axis.
        cmap : str
            Colormap name.
        color : str
            Line color.
        xlabel_angle : float
            Rotation angle of labels.
        figsize : tuple
            Figure size.

        Returns
        -------
        fig, ax : Matplotlib objects.
        """
        # Create figure and axis if not provided
        if not ax:
            fig, ax = plt.subplots(figsize=figsize)
        
        # Generate x positions for each parameter
        xi = np.arange(len(self.unknown_p_list))
        
        # Scatter individual samples (light dots) for each parameter
        for i in xi:
            ax.plot(i * np.ones_like(self.norm_s_i[:, i]), self.norm_s_i[:, i], '.' + color, alpha=0.2)
        
        # Plot median sensitivity values on top, as semilogy line
        ax.semilogy(xi, self.S_med_i, '-s' + color)
        
        # Set x-axis ticks and labels
        ax.set_xticks(xi, labels=self.p_i_label)
        ax.set_xlim(xi[0] - 0.5, xi[-1] + 0.5)
        
        # Rotate x labels if needed
        if xlabel_angle > 0:
            plt.setp(ax.get_xticklabels(), rotation=xlabel_angle, ha="right")
        
        # Label y-axis
        ax.set_ylabel("Median normalized sensitivity")
        fig.tight_layout()
        return fig, ax

    def plot_colinearity_map(self, fig=None, ax=None, cmap='viridis', xlabel_angle=45,
                            figsize=(4, 3), write_text=True):
        """
        Plot the median co-linearity indexes matrix.

        Parameters
        ----------
        fig, ax : optional
            Matplotlib figure and axis.
        cmap : str
            Colormap name.
        xlabel_angle : float
            Rotation angle.
        figsize : tuple
            Figure size.
        write_text : bool
            Annotate cells.

        Returns
        -------
        fig, ax : Matplotlib objects.
        """
        # Create figure and axis if not provided
        if not ax:
            fig, ax = plt.subplots(figsize=figsize)
        
        xi = np.arange(len(self.unknown_p_list))
        
        # Create color mesh of co-linearity index matrix (values between 0 and 1)
        im = ax.pcolormesh(xi, xi, self.cosPhi_med_ij, vmin=0, vmax=1, cmap=plt.colormaps[cmap])
        
        # Write numeric values inside each cell if requested
        if write_text:
            for i in xi:
                for j in xi:
                    ax.text(j, i, f'{self.cosPhi_med_ij[i, j]:.2f}', ha="center", va="center", color="w")
        
        # Add colorbar
        fig.colorbar(im, ax=ax)
        
        # Set x and y tick labels
        ax.set_xticks(xi, labels=self.p_i_label)
        ax.set_yticks(xi, labels=self.p_i_label)
        
        # Put labels on top, disable bottom
        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
        
        # Rotate x labels if needed
        if xlabel_angle > 0:
            plt.setp(ax.get_xticklabels(), rotation=xlabel_angle, ha="center", va='bottom',
                    rotation_mode="default")
        
        # Keep cells square
        plt.gca().set_aspect('equal')
        fig.tight_layout()
        return fig, ax
    
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
        P, min_eigvals, num_parameters = self.get_smallest_hessian_eigenvalues()

        # Set up main plot
        fig, ax = plt.subplots(figsize=(12, 2.5))
        
        # Plot the smallest eigenvalues vs parameter ranking
        ax.semilogy(1 + np.arange(), np.abs(min_eigvals), '-s')
        
        # X-axis: ranks with parameter names (ordered by pivoting)
        ax.set_xticks(1 + np.arange(n_unknown_p))
        ax.set_xticklabels([self.unknown_p_list[i][-1] for i in P], rotation=45)
        
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
        
        return P, num_parameters, fig, ax, ax2

class SteadyStateModel(DynamicModel):
    """
    A sub-class for steady-state models,
    where the output is directly computed from the parameters
    without integrating any dynamic equations.

    Parameters
    ----------
    h : callable
        Measurement function. It should compute the output `y` given the parameters `p`.
    p : dict, optional
        Dictionary of parameters.

    Attributes
    ----------
    h : callable
        The measurement function used to compute the outputs.
    p : dict
        Dictionary of parameters.
    """

    def __init__(self, h, p=None):
        """
        Initialize the steady-state parameter estimation model.

        Parameters
        ----------
        h : callable
            Measurement function `y = h(p)`.
        p : dict, optional
            Dictionary of parameters.
        """
        DynamicModel.__init__(self, f=None, h=h, p=p)

    def solve(self, time, u=None, x0=None, parameters_dict=None):
        """
        Compute the model output at steady state.

        Since this is a steady-state model, it does not integrate any ODEs.
        It simply computes the output `y` directly from the parameters.

        Parameters
        ----------
        time : array-like
            Time vector (not used here but kept for API compatibility).
        u : any, optional
            Input (not used in this steady-state model).
        x0 : any, optional
            Initial conditions (not used).
        parameters_dict : dict, optional
            Dictionary of parameters. If None, uses `self.p`.

        Returns
        -------
        t : int
            Placeholder (0), since time is not evolved in steady state.
        x : int
            Placeholder (0), since there are no state trajectories.
        y : array-like
            Output computed by the measurement function `h`.
        """
        parameters_dict = parameters_dict if parameters_dict else self.p
        y = self.h(parameters_dict)
        return 0, 0, y