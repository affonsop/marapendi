from scipy.integrate import solve_ivp
from scipy.stats import qmc
from scipy.optimize import differential_evolution
import numpy as np
import matplotlib.pyplot as plt

class ParameterEstimation:
    def __init__(self, f, h, x0=None, p=None, u=None, method='BDF'):
        self.f = f # x_dot = f(t, x, u, p)
        self.h = h # y = h(t, x, u, p)
        self.p = p
        self.x0 = x0
        self.u = u
        self.method = method

    def set_conditions(self, u):
        self.u = u # Dictionary of u(t)
         
    def set_params(self, p):
        self.p = p # Dictionary of parameters p

    def set_initial_conditions(self, x0):
        self.x0 = x0 # Vector of initial conditions

    def set_unknown_params(self, p_list):
        self.unknown_p_list = p_list 
        self.p_i_min = np.array([p[1][0] for p in self.unknown_p_list])
        self.p_i_max = np.array([p[1][1] for p in self.unknown_p_list])
        self.p_i_isLinear = np.array([p[2] for p in self.unknown_p_list])
        self.p_i_name = [p[0] for p in self.unknown_p_list]
        self.p_i_label = [p[3] for p in self.unknown_p_list]

    def solve(self, time, u=None, x0=None, parameters_dict=None):
        parameters_dict = parameters_dict if parameters_dict else self.p
        u = u if u else self.u
        sol = solve_ivp(self.f,
                  t_span=(time[0],time[-1]),
                  y0=x0 if x0 else self.x0,
                  t_eval=time,
                  args=(u, parameters_dict),
                  method=self.method,
                #   max_step=2
                #   rtol=1e-4,
                #   atol=1e-5,
                )
        y = self.h(sol.t, sol.y, u, parameters_dict)
        self.t = time
        self.x = sol.y
        self.y = y
        return sol.t, sol.y, y,

    def residuals(self, y_exp, t, u=None, x0=None, p=None):
        t, x, y = self.solve(t, u, x0, p)
        return y_exp - y

    def estimate(self, y_exp, t, u=None, x0=None, p=None, print_iterations=False, popsize=10, workers=1, ftol=0, atol=0, penalty_threshold=10e-3):
        if not p:
            if self.p:
                p = self.p
            else: 
                p = {} 
        def f(x): 
            px = p.copy()
            px.update({p[0]: v for p, v in zip(self.unknown_p_list, self.theta_to_p(x))})
            res = self.residuals(y_exp, t, u, x0, px)
            if penalty_threshold > 0: 
                penalty = np.where(np.abs(res) > penalty_threshold, 10 * (res - penalty_threshold), 0)
            return np.dot(res, res) / len(res) + (np.dot(penalty, penalty) if penalty_threshold > 0 else 0)
        
        def print_res(intermediate_result): 
            print('------'*5)
            p = self.theta_to_p(intermediate_result.x)
            print('RMSE : {:.1f} mV'.format(1e3*np.sqrt(intermediate_result.fun)))
            for k, param in enumerate(self.unknown_p_list):
                print(param[0], param[1], '{:.2e}'.format(p[k]))
            print('------'*5)
            if intermediate_result.fun < ftol: 
                return True

        sol = differential_evolution(f, tuple(([0,1] for p in self.unknown_p_list)), disp=True, 
                                     callback=print_res if print_iterations else None, 
                                     popsize=popsize, polish=False, workers=workers, 
                                     mutation=(0,1.6), seed=2, init='sobol',
                                     atol=atol)
        return sol, self.theta_to_p(sol.x)

    def plot(self, i=0, fig=None, ax=None):
        if not ax: 
            fig, ax = plt.subplots(1,1)
        ax.plot(self.t, self.x[i])
        return fig, ax
    
    def theta_to_p(self, theta_k): 
        p_i_k = np.where(self.p_i_isLinear, 
                            self.p_i_min + (self.p_i_max - self.p_i_min) * theta_k, 
                            self.p_i_min * np.exp((np.log(np.maximum(self.p_i_max,1e-12)) - np.log(np.maximum(self.p_i_min,1e-12))) * theta_k)
                            )
        return p_i_k 
        
    def calculate_local_sensitivity(self,  t, u=None, x0=None, p=None, n_samples=7):
        theta = np.linspace(0,1,n_samples) 
        y = []
        for i, p_i in enumerate(self.unknown_p_list): 
            y_i = []
            if p: 
                p_modified = p.copy()
            else: 
                p_modified = self.p.copy()
            p_i_min, p_i_max = p_i[1]
            for k in range(n_samples):
                p_i_k = self.theta_to_p(theta[k])[i]
                p_modified.update({p_i[0]: p_i_k})
                t, x_i_k, y_i_k = self.solve(t, u, x0, p_modified)
                y_i.append(y_i_k)
            y.append(y_i)
        y = np.array(y) # _ikjt
        dy = np.diff(y, axis=1)
        dtheta = np.diff(theta)
        dydtheta = dy / dtheta[:,np.newaxis]
        S = 1/(1e-12 + np.mean(y, axis=1)) * np.mean(dydtheta, axis=1)
        self.S = S
        return S
    
    def compute_global_sensitivity(self,  t, u=None, x0=None, p=None, n_samples=7, m=8, check_samples=False, y_exp=None, res_limit=0.3): 
        # Generates 2**m samples in the space parameter 
        ni = len(self.unknown_p_list)
        sampler = qmc.Sobol(d=ni, scramble=False)
        theta_samples = sampler.random_base2(m)
        if not p: 
            if self.p: 
                p = self.p 
            else: 
                p = {} 

        S_n = []
        for n in range(2**m): 
            p_i_n = self.theta_to_p(theta_samples[n])
            px = p.copy()
            px.update({p_i: v for p_i, v in zip(self.p_i_name, p_i_n)})
            if check_samples:
                res = self.residuals(y_exp, t, u, x0, px)
                isValid = np.sqrt(np.dot(res, res))/len(res) < res_limit 
            else:
                isValid = True
            if isValid: 
                s_n = self.calculate_local_sensitivity(t, u, x0, px, n_samples) 
                S_n.append(s_n) 
        n_valid = len(S_n)
        S_n = np.array(S_n)
        norm_s_i = np.linalg.norm(S_n,axis=(-1)) # Norms of the sensitivity vectors. Matrix with one line for each sample and one column for each parameter. 
        
        cosPhi_n = np.ones((len(S_n), ni, ni)) * np.nan
        for n in range(n_valid): # For each sample n
            for i in range(ni):  # For each parameter i
                for j in range(ni): # For each parameter j
                    if i >= j: 
                        
                        cosPhi_n[n,i,j] = np.abs(np.dot( np.transpose(S_n[n,j,:]),S_n[n,i,:]) / np.maximum((norm_s_i[n,i] * norm_s_i[n,j]),1e-12)) # Maximum to avoid nan when denominator is zero

        self.norm_s_i = norm_s_i
        self.cosPhi_med_ij = np.median(cosPhi_n, axis=0)
        self.S_med_i = np.median(norm_s_i, axis=0)
        self.S_std_i = np.std(norm_s_i, axis=0)

        self.S_med = np.median(S_n, axis=0)
        self.S_std = np.std(S_n, axis=0)
        self.S_n = S_n
        self.n_valid = n_valid 

        return self.cosPhi_med_ij, self.norm_s_i, self.S_med, self.S_std, self.S_med_i, self.S_std_i, self.S_n, self.n_valid
    
    def plot_global_sensitivity(self,fig=None, ax=None, cmap='viridis', color='C0', xlabel_angle=45, figsize=(4,3)): 
        if not ax: 
            fig, ax = plt.subplots(figsize=figsize)
        xi = [i for i in range(len(self.unknown_p_list))]
        for i in range(len(xi)): 
            ax.plot(i * np.ones_like(self.norm_s_i[:,i]), self.norm_s_i[:,i], '.' + color, alpha=0.2)
        ax.semilogy(xi, self.S_med_i, '-s'+ color)
        ax.set_xticks(xi, labels=self.p_i_label)
        if xlabel_angle > 0: 
            plt.setp(ax.get_xticklabels(), rotation=xlabel_angle, ha="right",
                    rotation_mode="anchor")
        ax.set_ylabel("Median normalized sensitivity")
        fig.tight_layout()
        return fig, ax
    
    def plot_colinearity_map(self, fig=None, ax=None, cmap='viridis', xlabel_angle=45, figsize=(4,3)):
        if not ax: 
            fig, ax = plt.subplots(figsize=figsize)
        xi = [i for i in range(len(self.unknown_p_list))]
        im = ax.pcolormesh(xi, xi, self.cosPhi_med_ij[:,:], vmin=0, vmax=1, cmap=plt.colormaps[cmap])
        for i in xi:
            for j in xi:
                text = ax.text(j, i, '{:.2f}'.format(self.cosPhi_med_ij[i,j]),
                            ha="center", va="center", color="w")
        fig.colorbar(im, ax=ax)
        ax.set_xticks(xi, labels=self.p_i_label)
        ax.set_yticks(xi, labels=self.p_i_label)
        if xlabel_angle > 0: 
            plt.setp(ax.get_xticklabels(), rotation=xlabel_angle, ha="right",
                    rotation_mode="anchor")
        plt.gca().set_aspect('equal')
        fig.tight_layout()
        return fig, ax
    
class ParameterEstimationSteadyState(ParameterEstimation): 
    def __init__(self, h, p=None):
        ParameterEstimation.__init__(self, f=None, h=h, p=p)
        
    def solve(self, time, u=None, x0=None, parameters_dict=None):
        parameters_dict = parameters_dict if parameters_dict else self.p
        y = self.h(parameters_dict)
        return 0, 0, y