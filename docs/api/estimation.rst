Parameter Estimation
====================

:class:`~marapendi.estimation.base_calibration.BaseModelCalibration` is the abstract
base class for parameter estimation problems.  It handles parameter normalisation
(linear or log scale), subset selection, and k-fold splitting; subclasses supply
:meth:`~marapendi.estimation.base_calibration.BaseModelCalibration.compute_y_sim`
and :meth:`~marapendi.estimation.base_calibration.BaseModelCalibration.compute_residuals`.

:class:`~marapendi.estimation.polarization_curve_calibration.SteadyStatePolarizationCurveCalibration`
is the concrete implementation for fitting to multi-condition polarization and HFR
data.  It uses ``scipy.optimize.differential_evolution`` as the global optimiser and
supports:

- **k-fold cross-validation** (:meth:`~marapendi.estimation.base_calibration.BaseModelCalibration.set_k_folds`)
  to estimate out-of-sample RMSE for each model complexity.
- **1-SE rule** (:func:`~marapendi.estimation.polarization_curve_calibration.optimal_n_1se`)
  to select the simplest model within one standard error of the best-CV model.
- **Complexity sweep** over parameter subsets, accessed via
  :meth:`~marapendi.estimation.base_calibration.BaseModelCalibration.subset_of_unknown_parameters`.

Parameters are declared with :class:`~marapendi.estimation.parameters.Parameter`
(known, fixed value) and :class:`~marapendi.estimation.parameters.UnknownParameter`
(unknown, optimised; carries bounds and initial guess).

Base class
----------

.. autoclass:: marapendi.estimation.base_calibration.BaseModelCalibration
   :members:
   :show-inheritance:

Polarization curve calibration
-------------------------------

.. autoclass:: marapendi.estimation.polarization_curve_calibration.SteadyStatePolarizationCurveCalibration
   :members:
   :show-inheritance:

.. autofunction:: marapendi.estimation.polarization_curve_calibration.optimal_n_1se

.. autofunction:: marapendi.estimation.polarization_curve_calibration.build_rmse_stats_df

Parameter declarations
----------------------

.. autoclass:: marapendi.estimation.parameters.Parameter
   :members:
   :show-inheritance:

.. autoclass:: marapendi.estimation.parameters.UnknownParameter
   :members:
   :show-inheritance:
