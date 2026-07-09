Parameter estimation
=========================

:class:`~marapendi.estimation.polarization_curve_calibration.SteadyStatePolarizationCurveCalibration`
fits unknown kinetic/transport parameters :math:`p` by minimising the mean
squared residual between simulated and measured voltage/HFR,

.. math::

    \min_{\theta \in [0,1]^{n_p}} \; \frac{1}{N}\sum_{k=1}^{N}
        \left(y_\mathrm{exp}^{(k)} - y_\mathrm{sim}^{(k)}(p(\theta))\right)^2,

using :func:`scipy.optimize.differential_evolution` in a normalised parameter
space :math:`\theta`, following the methodology of Goshtasbi et al. (2020).

Parameter normalisation
----------------------------

Each unknown parameter :math:`p_i` is mapped to :math:`\theta_i \in [0,1]`
before optimisation (eqs. 5–6 of Goshtasbi et al. 2020;
:meth:`~marapendi.estimation.base_calibration.BaseModelCalibration.p_to_theta`).
Parameters with ``is_linear=True`` use linear scaling; those with
``is_linear=False`` (e.g. exchange current densities, permeabilities, which
span orders of magnitude) use log scaling:

.. math::

    \theta_i = \frac{p_i - p_i^\mathrm{min}}{p_i^\mathrm{max} - p_i^\mathrm{min}}
    \quad\text{(linear)}, \qquad
    \theta_i = \frac{\ln p_i - \ln p_i^\mathrm{min}}{\ln p_i^\mathrm{max} - \ln p_i^\mathrm{min}}
    \quad\text{(log)}.

Sensitivity analysis
--------------------------

Two sensitivity measures decide which parameters are worth estimating at all.
**Local sensitivity** (eq. 7;
:meth:`~marapendi.estimation.base_calibration.BaseModelCalibration.calculate_local_sensitivity_neighborhood`)
perturbs each normalised parameter by :math:`\varepsilon_p` and measures the
relative response of the simulated output :math:`y`:

.. math::

    S_i = \frac{1}{\bar y}\,\frac{\Delta y}{\Delta \theta_i}.

**Global sensitivity** (eqs. 8–10;
:meth:`~marapendi.estimation.base_calibration.BaseModelCalibration.compute_global_sensitivity`)
repeats this local measure over a Sobol quasi-random sample of :math:`2^m`
points in :math:`\theta`-space, giving a distribution of :math:`S_i` (its
median/std, ``S_med_i``/``S_std_i``) and pairwise correlation between
parameters' sensitivity directions (``cosPhi_med_ij``) rather than a single
point estimate.

Identifiability ranking
----------------------------

Parameters are ranked by how independently they affect the simulated output,
following Goshtasbi et al. (2020) and Lund & Foss (2008)
(:meth:`~marapendi.estimation.base_calibration.BaseModelCalibration.get_smallest_hessian_eigenvalues`).
The (median) sensitivity matrix :math:`S \in \mathbb{R}^{n_p \times n_\mathrm{cases}}`
is QR-decomposed with column pivoting,

.. math::

    S^\top = QRP,

and the pivot order :math:`P` gives the identifiability ranking directly: the
first pivoted column is the most identifiable parameter, and so on. For each
prefix of :math:`k` parameters selected by the pivot order, the smallest
eigenvalue of the local Gram/Hessian-proxy matrix :math:`H = S_{[1:k]} \,
S_{[1:k]}^\top` is recorded — a small eigenvalue signals that the selected
subset is close to collinear (poorly identifiable given the available data).

Model-complexity selection: k-fold CV and the 1-SE rule
--------------------------------------------------------------

:meth:`~marapendi.estimation.base_calibration.BaseModelCalibration.run_k_fold_cross_validation_vs_complexity`
re-fits the model for an increasing number of free parameters (ordered by the
identifiability ranking above) under k-fold cross-validation, producing a
cross-validated RMSE distribution for each complexity level. The optimal
complexity is then the *simplest* model within one standard error of the best
mean cross-validated RMSE —

.. math::

    n^\ast = \min\left\{ n : \overline{\mathrm{RMSE}}_n \le
        \overline{\mathrm{RMSE}}_{n_\mathrm{best}} + \mathrm{SE}_{n_\mathrm{best}} \right\}

— following Hastie et al. (2009), §7.3
(:func:`~marapendi.estimation.polarization_curve_calibration.optimal_n_1se`).
See :doc:`../user_guide/calibration` for the full pipeline in practice.

References
--------------

Goshtasbi, A. et al. *J. Electrochem. Soc.* **167**, 024518 (2020).

Lund, B. F. & Foss, B. A. *Comput. Chem. Eng.* **32**, 2338 (2008).

Hastie, T., Tibshirani, R. & Friedman, J. *The Elements of Statistical
Learning*, 2nd ed., Springer (2009).
