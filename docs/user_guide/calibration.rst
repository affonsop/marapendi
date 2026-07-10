Model calibration pipeline
===========================

:class:`~marapendi.estimation.polarization_curve_calibration.SteadyStatePolarizationCurveCalibration`
fits kinetic and transport parameters to multi-condition polarization-curve and
HFR data.  This guide walks through the full pipeline: defining data, choosing
parameters, running the optimiser, cross-validating, and selecting model
complexity with the 1-SE rule.

.. seealso::

   :doc:`/science/parameter_estimation` derives the weighted cost function,
   the sensitivity/co-linearity measures, and the successive-orthogonalization
   identifiability ranking used below, following Affonso Nobrega et al.
   (2026) and Goshtasbi et al. (2020a,b). :doc:`/auto_examples/plot_06_parameter_estimation`
   is a runnable, self-contained version of this pipeline on synthetic data.

Data format
-----------

Two DataFrames are required.

``conditions_dataset`` — one row per experimental condition (case):

+------+------------------+-------------+-------------+-------+-------+-------+-------+
| case | cell-temperature | pressure-ca | pressure-an | rh-ca | rh-an | st-ca | st-an |
+======+==================+=============+=============+=======+=======+=======+=======+
| 1    | 353.15 K         | 1.5e5 Pa    | 1.5e5 Pa    | 0.50  | 0.50  | 2.0   | 1.5   |
+------+------------------+-------------+-------------+-------+-------+-------+-------+
| 2    | 323.15 K         | 2.5e5 Pa    | 2.5e5 Pa    | 0.30  | 0.30  | 2.0   | 1.5   |
+------+------------------+-------------+-------------+-------+-------+-------+-------+

``experimental_dataset`` — one row per measurement point, columns
``case``, ``current-density`` (A/m²), ``voltage`` (V), ``hfr`` (Ω m²).

``hfr`` may be ``NaN`` for points where no HFR measurement is available.

.. code-block:: python

    import numpy as np
    import pandas as pd
    from marapendi.estimation.polarization_curve_calibration import (
        SteadyStatePolarizationCurveCalibration,
        optimal_n_1se,
        build_rmse_stats_df,
    )
    from marapendi.estimation.parameters import Parameter, UnknownParameter
    import marapendi as mrpd

    conditions_df = pd.DataFrame([
        {"case": 1, "cell-temperature": 353.15,
         "pressure-ca": 1.5e5, "pressure-an": 1.5e5,
         "rh-ca": 0.50, "rh-an": 0.50, "st-ca": 2.0, "st-an": 1.5},
        {"case": 2, "cell-temperature": 323.15,
         "pressure-ca": 2.5e5, "pressure-an": 2.5e5,
         "rh-ca": 0.30, "rh-an": 0.30, "st-ca": 2.0, "st-an": 1.5},
    ])

    i_pts = np.array([2e3, 5e3, 10e3, 15e3, 20e3])   # A/m²
    experimental_df = pd.DataFrame([
        {"case": c, "current-density": i,
         "voltage": 0.72 - 0.02 * (i / 1e4),
         "hfr": 5e-5}
        for c in [1, 2] for i in i_pts
    ])

Loading from per-case CSV files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If each condition is stored in a separate CSV (one row per operating point), read
and concatenate them:

.. code-block:: python

    cases = {
        "A": "data/experimental_data_MEA62_condition_A.csv",
        "B": "data/experimental_data_MEA62_condition_B.csv",
    }
    dfs = []
    for case_id, path in cases.items():
        df = pd.read_csv(path, sep=";", skiprows=6)
        dfs.append(pd.DataFrame({
            "case": case_id,
            "current-density": df["I_Pile(A)"] / CELL_AREA,
            "voltage": df["U_Pile(V)"],
            "hfr": np.nan,   # or compute from EIS data
        }))
    experimental_df = pd.concat(dfs, ignore_index=True)

Parameters
----------

:class:`~marapendi.estimation.parameters.Parameter` pins a value.
:class:`~marapendi.estimation.parameters.UnknownParameter` marks a parameter for
optimisation with initial guess and bounds.

``is_linear=True`` applies linear [0, 1] normalisation; ``is_linear=False`` applies
log scaling — use log for parameters that span orders of magnitude (exchange
current density, permeabilities).

.. code-block:: python

    known = [
        Parameter(value=1100.0, key="memb-equiv-weight"),
        Parameter(value=30e-7,  key="elec-resistance"),
    ]
    unknown = [
        UnknownParameter(
            value=2.5e-4, initial_guess=2.5e-4,
            lower_bound=1e-5, upper_bound=1e-2,
            key="i0-c", is_linear=False,
        ),
        UnknownParameter(
            value=1.0, initial_guess=1.0,
            lower_bound=0.1, upper_bound=20.0,
            key="memb-cond-correction", is_linear=True,
        ),
    ]

Cell creator
------------

The ``cell_creator`` callable receives the full parameter dict (known + unknown)
and returns a fresh :class:`~marapendi.cell.fuelcell.FuelCell`.  The optimiser
calls it at each function evaluation.

.. code-block:: python

    liq = mrpd.DarcyTransportModel(J_function_exponent=2)

    def cell_creator(params):
        ionomer = mrpd.PFSAIonomer(
            equivalent_weight=params.get("memb-equiv-weight", 1100),
            conductivity_correction=params.get("memb-cond-correction", 1.0),
        )
        return mrpd.FuelCell(
            area=25e-4,
            electric_resistance=params.get("elec-resistance", 30e-7),
            ca=mrpd.FuelCellSide(
                cl=mrpd.PtCCatalystLayer(
                    ecsa=70e3, platinum_loading=0.4e-2, ionomer=ionomer,
                    reaction=mrpd.ElectrochemicalReaction(
                        reference_exchange_current_density=params.get("i0-c", 2.5e-4),
                        reaction_order=0.54, activation_energy=67e6,
                        reference_activity=1e5, reference_temperature=353.15,
                        number_of_electrons=2, charge_transfer_coeff=0.5,
                    ),
                    thickness=10e-6, thermal_conductivity=0.22,
                    pore_diameter=40e-9, absolute_permeability=1e-13,
                    contact_angle=97., two_phase_transport_model=liq,
                ),
                gdl=mrpd.GasDiffusionLayer(
                    thickness=200e-6, porosity=0.6, contact_angle=120.,
                    effective_gas_diffusion_ratio=0.3, absolute_permeability=1e-12,
                    thermal_conductivity=0.5, two_phase_transport_model=liq,
                ),
                ch=mrpd.FlowChannel(
                    width=1e-3, height=1e-3, length=0.1, n_parallel=20, reactant='o2',
                ),
                has_mpl=False, thermal_contact_resistance=4e-4,
            ),
            an=mrpd.FuelCellSide(
                cl=mrpd.PtCCatalystLayer(thickness=5e-6, two_phase_transport_model=liq),
                gdl=mrpd.GasDiffusionLayer(
                    thickness=200e-6, effective_gas_diffusion_ratio=0.3,
                    thermal_conductivity=0.5, two_phase_transport_model=liq,
                ),
                ch=mrpd.FlowChannel(
                    width=1e-3, height=1e-3, length=0.1, n_parallel=20, reactant='h2',
                ),
                has_mpl=False, thermal_contact_resistance=4e-4,
            ),
            membrane=mrpd.PFSA(ionomer=ionomer, dry_thickness=25e-6),
        )

Calibration object
------------------

.. code-block:: python

    cal = SteadyStatePolarizationCurveCalibration(
        conditions_dataset=conditions_df,
        experimental_dataset=experimental_df,
        cell_creator=cell_creator,
        known_parameters=known,
        unknown_parameters=unknown,
    )

    print("Unknown parameters:", cal.p_i_name)
    print("Initial guess:     ", cal.p_initial_guess)

Inspect residuals at the initial guess
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before running the optimiser, verify that the initial guess is reasonable:

.. code-block:: python

    res = cal.compute_residuals(cal.p_initial_guess)
    # Returns y_exp − y_sim, concatenated across all cases:
    # [voltage_case1, …, voltage_caseN, weighted_hfr_case1, …]
    print("Initial RMSE:", np.sqrt(np.mean(res**2)))

Single-condition optimisation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~marapendi.estimation.base_calibration.BaseModelCalibration.estimate`
calls ``scipy.optimize.differential_evolution`` in the normalised [0, 1]
parameter space:

.. code-block:: python

    sol, p_opt = cal.estimate(
        case_list=cal.full_case_list,
        popsize=10,
        maxiter=200,
        workers=4,           # parallel function evaluations
    )
    print("Optimal parameters:", dict(zip(cal.p_i_name, p_opt)))

k-fold cross-validation
------------------------

Cross-validation estimates generalisation error across unseen conditions.
:meth:`~marapendi.estimation.base_calibration.BaseModelCalibration.set_k_folds`
partitions the condition list into *k* folds; the calibration then trains on all
but one fold and tests on the held-out fold.

.. code-block:: python

    cal.set_k_folds(k=len(cal.full_case_list))   # leave-one-out

    cv_results = cal.run_k_fold_cross_validation(
        estimate_kwargs={"popsize": 10, "maxiter": 150, "workers": 4},
        filename="my_run",        # optional: checkpoint to CSV after each fold
        output_dir="results/",
    )
    print(cv_results)

``cv_results`` is a DataFrame with columns ``fold_id``, ``n_params``,
``computation_time``, ``objective_value``, and one column per unknown parameter.

Model-complexity sweep and the 1-SE rule
-----------------------------------------

Train on a sequence of parameter subsets (ranked by their Hessian sensitivity)
and pick the simplest model within one standard error of the best CV RMSE:

.. code-block:: python

    # First compute global sensitivity to rank parameters
    cal.compute_global_sensitivity(m=6)   # 2**6 = 64 Sobol samples

    # Sweep over 1, 2, 3, … free parameters
    n_params_list = list(range(1, len(unknown) + 1))
    cal.run_k_fold_cross_validation_vs_complexity(
        n_params_list=n_params_list,
        force_restart=False,               # resume if interrupted
        estimate_kwargs={"popsize": 10, "maxiter": 150, "workers": 4},
        filename="complexity_sweep",
        output_dir="results/",
    )

    # Load results and compute 1-SE statistics
    cv_df = cal.load_cross_validation_results("complexity_sweep", dir="results/")
    stats_df = build_rmse_stats_df(cv_df)
    n_opt = optimal_n_1se(stats_df["mean"], stats_df["std"])
    print(f"Optimal complexity: {n_opt} parameters")

Post-processing
---------------

After selecting the optimal complexity, re-fit on all data and simulate:

.. code-block:: python

    cal.automatic_parameter_selection(n_opt)
    _, p_final = cal.estimate(case_list=cal.full_case_list, maxiter=300)

    cell_opt = cell_creator(dict(zip(cal.p_i_name, p_final)))
    for case in cal.full_case_list:
        V_sim, hfr_sim, state = cal.simulate_voltage_and_hfr(cell_opt, case)
        ds = cal.get_case_dataset(case)
        print(f"Case {case}  RMSE_V = "
              f"{np.sqrt(np.mean((ds['voltage'] - V_sim)**2))*1e3:.1f} mV")
