Parameter estimation
=========================

:class:`~marapendi.estimation.polarization_curve_calibration.SteadyStatePolarizationCurveCalibration`
fits unknown kinetic/transport parameters :math:`p` against measured
polarization curves and HFR by minimising a weighted mean-squared-error cost
function combining both, following Affonso Nobrega et al. (2026):

.. math::

    \mathrm{MSE} = \frac{1}{N_V + N_\mathrm{HFR}}\left[
        \sum_k \left(V_k - \hat V_k\right)^2
        + \frac{N_V}{N_\mathrm{HFR}}\,\frac{\SI{1}{V}}{\SI{1}{\ohm\meter\squared}}
        \sum_k \left(\mathrm{HFR}_k - \widehat{\mathrm{HFR}}_k\right)^2
    \right],

where hatted quantities are experimental and the weight factor
:math:`N_V/N_\mathrm{HFR}` (ratio of voltage to HFR measurement counts,
:attr:`~marapendi.estimation.polarization_curve_calibration.SteadyStatePolarizationCurveCalibration.hfr_weight_factor`)
compensates for voltage measurements typically outnumbering HFR measurements,
and brings the HFR residual to a comparable magnitude/units
(:meth:`~marapendi.estimation.polarization_curve_calibration.SteadyStatePolarizationCurveCalibration.apply_hfr_weights`).
Minimisation
(:meth:`~marapendi.estimation.base_calibration.BaseModelCalibration.estimate`)
uses :func:`scipy.optimize.differential_evolution` by default, over a
normalised parameter space :math:`\theta \in [0,1]^{n_p}`.

Parameter normalisation
----------------------------

Each unknown parameter :math:`p_i` is mapped to :math:`\theta_i \in [0,1]`
before optimisation
(:meth:`~marapendi.estimation.base_calibration.BaseModelCalibration.p_to_theta`).
Parameters with ``is_linear=True`` use linear scaling; those spanning orders
of magnitude (e.g. exchange current densities, permeabilities) use
``is_linear=False`` log scaling:

.. math::

    \theta_i = \frac{p_i - p_i^\mathrm{min}}{p_i^\mathrm{max} - p_i^\mathrm{min}}
    \quad\text{(linear)}, \qquad
    \theta_i = \frac{\ln p_i - \ln p_i^\mathrm{min}}{\ln p_i^\mathrm{max} - \ln p_i^\mathrm{min}}
    \quad\text{(log)}.

Sensitivity analysis
--------------------------

Two sensitivity measures decide which parameters are worth estimating at all,
adapting the global sensitivity analysis methodology of Goshtasbi et al.
(2020a). For a given parameter set, the sensitivity matrix
:math:`\mathbf{S}`, :math:`s_{ij} = \partial y_i/\partial \theta_j` (output
:math:`i` with respect to normalised parameter :math:`j`), is built from
**local sensitivities**
(:meth:`~marapendi.estimation.base_calibration.BaseModelCalibration.calculate_local_sensitivity_neighborhood`).
Rather than Goshtasbi et al.'s finite difference over :math:`n_s=7`
equidistant samples of :math:`\theta_j \in [0,1]` — whose per-sample terms
can still be skewed by draws that would otherwise be discarded for poor
model fit — a single, small forward-difference step
:math:`\Delta\tilde\theta_j = 10^{-6}` is used instead:

.. math::

    s_{ij} = \frac{1}{\tilde y_i}\,
        \frac{\tilde y_i(\tilde\theta_j + \Delta\tilde\theta_j) - \tilde y_i(\tilde\theta_j)}
             {\Delta\tilde\theta_j}.

**Global sensitivity**
(:meth:`~marapendi.estimation.base_calibration.BaseModelCalibration.compute_global_sensitivity`)
repeats this local evaluation over :math:`2^m` Sobol quasi-random samples of
:math:`\theta`-space (samples whose simulated RMSE exceeds a threshold are
discarded), giving :math:`n_k` sensitivity matrices :math:`\mathbf{S}^k`.
The median normalised-sensitivity vector for parameter :math:`i`,

.. math::

    \bar{\mathbf{s}}_i = \mathrm{med}_k \, \lVert \mathbf{s}_i^k \rVert_2
    \qquad \text{(} \texttt{S\_med\_i} \text{)},

and the pairwise co-linearity index between parameters :math:`i` and
:math:`j`,

.. math::

    \psi_{ij}^k = \cos(\phi_{ij}^k) = \frac{\left|\mathbf{s}_i^{k\top} \mathbf{s}_j^k\right|}
        {\lVert\mathbf{s}_i^k\rVert_2\, \lVert\mathbf{s}_j^k\rVert_2}
    \qquad \text{(} \texttt{cosPhi\_med\_ij} \text{)},

together identify parameters that are both weakly influential
(:math:`\bar{\mathbf{s}}_i` small) and hard to distinguish from one another
(:math:`\mathrm{med}_k\,\psi_{ij}^k` close to 1).

Identifiability ranking
----------------------------

Parameters are ranked by how independently they affect the simulated output
via successive orthogonalization, following Lund & Foss (2008) as adapted by
Goshtasbi et al. (2020a)
(:meth:`~marapendi.estimation.base_calibration.BaseModelCalibration.get_smallest_hessian_eigenvalues`).
The median sensitivity matrix :math:`\mathbf{S} \in \mathbb{R}^{n_p \times n_k}`
(rows = parameters, columns = the concatenated, RMSE-filtered global-sensitivity
samples) is QR-decomposed with column pivoting,

.. math::

    \mathbf{S}^\top = QRP,

and the pivot order :math:`P` gives the identifiability ranking directly: the
first pivoted parameter is the most identifiable, and so on. For each prefix
of :math:`k` parameters selected by the pivot order, the smallest eigenvalue
of the local Hessian-proxy matrix :math:`H = \mathbf{S}_{[1:k]} \,
\mathbf{S}_{[1:k]}^\top` is recorded — a small eigenvalue signals that the
selected subset is close to collinear, and therefore poorly identifiable
given the available data. Parameters not selected for estimation are held at
their reference value.

Model-complexity selection: k-fold CV and the 1-SE rule
--------------------------------------------------------------

:meth:`~marapendi.estimation.base_calibration.BaseModelCalibration.run_k_fold_cross_validation_vs_complexity`
re-fits the model for an increasing number of free parameters (ordered by the
identifiability ranking above), estimating each subset with leave-one-out
cross-validation across the calibration operating conditions
(:meth:`~marapendi.estimation.base_calibration.BaseModelCalibration.set_k_folds`
with ``k = len(cases)``) — one round of parameter estimation per condition,
excluded in turn from the training set and used only to evaluate predictive
RMSE. This produces a cross-validated RMSE distribution for each complexity
level. The optimal complexity is then the *simplest* model within one
standard error of the best mean cross-validated RMSE —

.. math::

    n^\ast = \min\left\{ n : \overline{\mathrm{RMSE}}_n \le
        \overline{\mathrm{RMSE}}_{n_\mathrm{best}} + \mathrm{SE}_{n_\mathrm{best}} \right\}

— following Hastie et al. (2009), §7.3
(:func:`~marapendi.estimation.polarization_curve_calibration.optimal_n_1se`).
Additional operating conditions held out entirely from sensitivity analysis,
estimation, and cross-validation can then be used for a final, independent
validation of the selected model complexity. See
:doc:`../user_guide/calibration` for the full pipeline in practice.

References
--------------

Affonso Nobrega, P. et al. *J. Electrochem. Soc.* **173**, 114503 (2026).

Goshtasbi, A., Chen, J., Waldecker, J. R., Hirano, S. & Ersal, T. *J.
Electrochem. Soc.* **167**, 044504 (2020a).

Goshtasbi, A., Chen, J., Waldecker, J. R., Hirano, S. & Ersal, T. *J.
Electrochem. Soc.* **167**, 114513 (2020b).

Lund, B. F. & Foss, B. A. *Comput. Chem. Eng.* **32**, 2338 (2008).

Hastie, T., Tibshirani, R. & Friedman, J. *The Elements of Statistical
Learning*, 2nd ed., Springer (2009).
