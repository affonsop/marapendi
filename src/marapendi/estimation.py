import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
from scipy.stats import qmc

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

        def f(x):
            # Build parameter dict with current candidate
            px = p.copy()
            px.update({p_i[0]: v for p_i, v in zip(self.unknown_p_list, self.theta_to_p(x))})
            res = self.residuals(y_exp, t, u, x0, px)
            if penalty_threshold > 0:
                penalty = np.where(np.abs(res) > penalty_threshold, 10 * (res - penalty_threshold), 0)
            return np.dot(res, res) / len(res) + (np.dot(penalty, penalty) if penalty_threshold > 0 else 0)

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
        theta = np.linspace(0, 1, n_samples)
        y = []
        for i, p_i in enumerate(self.unknown_p_list):
            y_i = []
            p_modified = p.copy() if p else self.p.copy()
            for k in range(n_samples):
                p_i_k = self.theta_to_p(theta[k])[i]
                p_modified.update({p_i[0]: p_i_k})
                _, _, y_i_k = self.solve(t, u, x0, p_modified)
                y_i.append(y_i_k)
            y.append(y_i)
        y = np.array(y)
        dy = np.diff(y, axis=1)
        dtheta = np.diff(theta)
        dydtheta = dy / dtheta[:, np.newaxis]
        S = 1 / (1e-12 + np.mean(y, axis=1)) * np.mean(dydtheta, axis=1)
        self.S = S
        return S

    def compute_global_sensitivity(self, t, u=None, x0=None, p=None, n_samples=7,
                                   m=8, check_samples=False, y_exp=None,
                                   rmse_limit=0.3, print_px=False):
        """
        Compute global sensitivities using Sobol sampling.

        Parameters
        ----------
        t, u, x0, p : optional
            Same as above.
        n_samples : int
            Samples for local sensitivity.
        m : int
            Generates 2**m Sobol samples.
        check_samples : bool
            If True, check RMSE against `y_exp`.
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
        # Sample the parameter space
        sampler = qmc.Sobol(d=n_unknown_p, scramble=False)
        theta_samples = sampler.random_base2(m)
    
        p = p if p else self.p if self.p else {}

        S_n = []
        for n in range(2**m):
            p_i_n = self.theta_to_p(theta_samples[n])
            px = p.copy()
            px.update({p_i: v for p_i, v in zip(self.p_i_name, p_i_n)})
            if check_samples:
                res = self.residuals(y_exp, t, u, x0, px)
                isValid = np.sqrt(np.dot(res, res) / len(res)) < rmse_limit
            else:
                isValid = True
            if isValid:
                if print_px:
                    print(px)
                s_n = self.calculate_local_sensitivity(t, u, x0, px, n_samples)
                S_n.append(s_n)
        n_valid = len(S_n)
        S_n = np.array(S_n)

        # Calculate norm, colinearity index and statistics 
        norm_s_i = np.linalg.norm(S_n, axis=-1)
        cosPhi_n = np.ones((len(S_n), n_unknown_p, n_unknown_p)) * np.nan
        for n in range(n_valid):
            for i in range(n_unknown_p):
                for j in range(n_unknown_p):
                    if i >= j:
                        cosPhi_n[n, i, j] = np.abs(np.dot(S_n[n, j, :], S_n[n, i, :])
                                                    / np.maximum(norm_s_i[n, i] * norm_s_i[n, j], 1e-12))
        self.norm_s_i, self.cosPhi_med_ij = norm_s_i, np.median(cosPhi_n, axis=0)
        self.S_med_i, self.S_std_i = np.median(norm_s_i, axis=0), np.std(norm_s_i, axis=0)
        self.S_med, self.S_std, self.S_n, self.n_valid = np.median(S_n, axis=0), np.std(S_n, axis=0), S_n, n_valid
        return self.cosPhi_med_ij, self.norm_s_i, self.S_med, self.S_std, self.S_med_i, self.S_std_i, self.S_n, self.n_valid

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
        if not ax:
            fig, ax = plt.subplots(figsize=figsize)
        xi = np.arange(len(self.unknown_p_list))
        for i in xi:
            ax.plot(i * np.ones_like(self.norm_s_i[:, i]), self.norm_s_i[:, i], '.' + color, alpha=0.2)
        ax.semilogy(xi, self.S_med_i, '-s' + color)
        ax.set_xticks(xi, labels=self.p_i_label)
        ax.set_xlim(xi[0] - 0.5, xi[-1] + 0.5)
        if xlabel_angle > 0:
            plt.setp(ax.get_xticklabels(), rotation=xlabel_angle, ha="right")
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
        if not ax:
            fig, ax = plt.subplots(figsize=figsize)
        xi = np.arange(len(self.unknown_p_list))
        im = ax.pcolormesh(xi, xi, self.cosPhi_med_ij, vmin=0, vmax=1, cmap=plt.colormaps[cmap])
        if write_text:
            for i in xi:
                for j in xi:
                    ax.text(j, i, f'{self.cosPhi_med_ij[i, j]:.2f}', ha="center", va="center", color="w")
        fig.colorbar(im, ax=ax)
        ax.set_xticks(xi, labels=self.p_i_label)
        ax.set_yticks(xi, labels=self.p_i_label)
        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
        if xlabel_angle > 0:
            plt.setp(ax.get_xticklabels(), rotation=xlabel_angle, ha="center", va='bottom',
                    rotation_mode="default")
        plt.gca().set_aspect('equal')
        fig.tight_layout()
        return fig, ax

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