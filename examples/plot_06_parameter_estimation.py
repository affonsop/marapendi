"""
Parameter estimation
====================

:class:`~marapendi.estimation.polarization_curve_calibration.SteadyStatePolarizationCurveCalibration`
fits kinetic and transport parameters to multi-condition polarization and HFR
data.  It uses ``scipy.optimize.differential_evolution`` as the global
optimiser, with k-fold cross-validation and automatic complexity selection
via the 1-SE rule.

This example builds a synthetic 10-condition calibration problem and runs the
**full estimation pipeline**: global sensitivity analysis (Sobol screening +
Hessian-based parameter ranking), a k-fold cross-validation sweep over model
complexity, and the resulting diagnostics (RMSE vs. complexity, parameter
evolution, cross-validation curves, and simulated internal variables).  The
optimiser settings (population size, iterations, Sobol sample count, number
of folds) are kept small so the whole example runs in about a minute; a real
calibration would use larger values.
"""

# %%
# Cell creator
# ------------
#
# ``cell_creator`` receives the current parameter dict and returns a
# :class:`~marapendi.cell.fuelcell.FuelCell`.  Six parameters spanning
# kinetics, transport, and membrane properties are exposed for estimation;
# everything else is held fixed.

import tempfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import marapendi as mrpd
from marapendi.estimation.parameters import UnknownParameter


def cell_creator(params):
    liq = mrpd.DarcyTransportModel(J_function_exponent=0.4)

    cell = mrpd.FuelCell(
        area=25e-4,
        electric_resistance=params['electric-resistance']
    )

    for side in cell.sides:
        side.ch = mrpd.FlowChannel(
            width=0.85e-3,
            height=1e-3,
            length=0.49,
            n_parallel=3,
            reactant="o2" if side is cell.ca else "h2",
            transport_resistance_model=mrpd.ChannelGasResistanceModel(sherwood=3.66, B_ch=1.2)
        )

    for side in cell.sides:
        side.gdl = mrpd.GasDiffusionLayer(
            thickness=params['gdl-thickness'] * 1.4,
            porosity=0.65,
            tortuosity=1.55,
            contact_angle=110.0,
            absolute_permeability=3e-12,
            thermal_conductivity=1.2,
            two_phase_transport_model=liq,
            relative_permeability_exponent=3
        )
        side.mpl = mrpd.MicroPorousLayer(
            thickness=22e-6,
            porosity=0.4,
            tortuosity=3,
            pore_diameter=500e-9,
            contact_angle=130.0,
            absolute_permeability=1e-12,
            thermal_conductivity=0.144,
            two_phase_transport_model=liq,
            relative_permeability_exponent=3
        )

    for side in cell.sides:
        side.thermal_contact_resistance = 1e-4

    orr = mrpd.ElectrochemicalReaction(
        reference_exchange_current_density=params['i0-c'],
        reaction_order=params['gamma-c'],
        activation_energy=42e6,
        reference_activity=1e5,
        reference_temperature=353.15,
        number_of_electrons=2,
        charge_transfer_coeff=0.5,
    )

    nafion = mrpd.PFSAIonomer(
        equivalent_weight=1100.,
        dry_density=1980,
        reference_conductivity=50.,
        residual_conductivity=0.3,
        conductivity_fv_threshold=0.04,
        conductivity_exp=1.5,
        reference_conductivity_temperature=300.,
        conductivity_activation_energy=10.540e6,
        reference_water_absorption_coefficient=1e-5,
        reference_water_absorption_temperature=303.15,
        water_absorption_activation_energy=20e6,
        reference_water_diffusivity=2e-10,
        reference_water_diffusivity_temperature=300.,
        water_diffusivity_activation_energy=20e6,
        vapor_equilibrium_polynomial=[36, -39.85, 17.18, 0.043],
    )

    cell.ca.cl = mrpd.PtCCatalystLayer(
        ecsa=40e3,
        platinum_loading=0.5e-2,
        catalyst_platinum_weight_percent=0.5,
        ionomer_to_carbon_ratio=0.81,
        ionomer=nafion,
        reaction=orr,
        thickness=10e-6,
        tortuosity=3,
        thermal_conductivity=0.18,
        pore_diameter=140e-9,
        carbon_agglomerate_radius=25e-9,
        absolute_permeability=2e-13,
        contact_angle=params['cl-theta'],
        two_phase_transport_model=liq,
        relative_permeability_exponent=3
    )

    cell.an.cl = mrpd.PtCCatalystLayer(
        platinum_loading=0.1e-2,
        ionomer=nafion,
        thickness=7e-6,
        ionomer_to_carbon_ratio=0.57,
        catalyst_platinum_weight_percent=0.2,
        thermal_conductivity=0.18,
        pore_diameter=140e-9,
        absolute_permeability=1e-13,
        contact_angle=params['cl-theta'],
        two_phase_transport_model=liq,
    )

    cell.membrane = mrpd.PFSA(ionomer=nafion, dry_thickness=params['memb-thickness'])
    return cell


# %%
# Operating conditions
# ---------------------
#
# Ten operating conditions spanning temperature, pressure, relative humidity,
# and stoichiometry — enough spread for the sensitivity analysis and
# cross-validation folds below to be meaningful.

conditions_df = pd.DataFrame({
    'case':             list(range(1, 10)),
    'cell-temperature': np.array([80, 50, 80, 80, 80, 90, 50, 80,  65]) + 273.15,
    'pressure-ca':      np.array([1.5, 2.5, 1.5, 2.5, 2.3, 1.5, 1.5, 1.5,  2.0]) * 1e5,
    'pressure-an':      np.array([1.5, 2.5, 1.5, 2.5, 2.5, 1.5, 1.5, 1.5,  2.0]) * 1e5,
    'rh-ca':            np.array([50, 50, 30, 30, 30, 50, 80, 80, 60]) / 100,
    'rh-an':            np.array([50, 50, 30, 30, 50, 50, 80, 80, 60]) / 100,
    'st-ca':            [2.0, 2.0, 2.5, 2.5, 2.0, 2.5, 2.0, 2.5, 2.2],
    'st-an':            [1.5, 1.5, 1.2, 2.0, 1.5, 1.5, 1.5, 2.0, 1.7],
})

# %%
# Synthetic measurements
# -----------------------
#
# Build a cell at a chosen "true" parameter set, run it through
# :class:`~marapendi.cell.explicit_steady_state.ExplicitSteadyStateModel` at
# the current densities in ``_I`` for each case, and add 10 % relative random
# noise to voltage and HFR to emulate test-bench measurement scatter.

_I = np.array([1e3, 3e3, 6e3, 1e4, 1.4e4, 1.8e4])   # A/m²

true_params = {
    "i0-c": 1.5e-4,
    "gamma-c": 0.75,
    "electric-resistance": 12e-7,
    "gdl-thickness": 117e-6,
    "cl-theta": 95.0,
    "memb-thickness": 12e-6,
}
_true_cell = cell_creator(true_params)
_true_model = mrpd.ExplicitSteadyStateModel()
_rng = np.random.default_rng(0)

rows = []
for _, row in conditions_df.iterrows():
    cond = mrpd.CellConditions(
        current_density=_I,
        cell_temperature=row["cell-temperature"],
        ca=mrpd.SideConditions(
            inlet_temperature=row["cell-temperature"],
            outlet_pressure=row["pressure-ca"],
            inlet_relative_humidity=row["rh-ca"],
            dry_o2_mole_fraction=0.21,
            stoichiometry=row["st-ca"],
        ),
        an=mrpd.SideConditions(
            inlet_temperature=row["cell-temperature"],
            outlet_pressure=row["pressure-an"],
            inlet_relative_humidity=row["rh-an"],
            dry_h2_mole_fraction=1.0,
            stoichiometry=row["st-an"],
        ),
    )
    state = _true_model.solve(_true_cell, cond, _true_model.set_initial_conditions(_true_cell, cond))
    hfr = _true_model.voltage_model.high_frequency_resistance(_true_cell, state)

    voltage_noisy = state.cell_voltage * (1 + 0.01 * _rng.standard_normal(len(_I)))
    hfr_noisy = hfr * (1 + 0.01 * _rng.standard_normal(len(_I)))

    for i, v, h in zip(_I, voltage_noisy, hfr_noisy):
        rows.append({"case": row["case"], "current-density": i, "voltage": v, "hfr": h})

experimental_df = pd.DataFrame(rows)

# %%
# Known and unknown parameters
# -----------------------------
#
# :class:`~marapendi.estimation.parameters.Parameter` fixes a value.
# :class:`~marapendi.estimation.parameters.UnknownParameter` marks a parameter
# for optimisation, with initial guess and bounds.
#
# ``is_linear=True`` means the parameter is scaled linearly in [0, 1];
# ``is_linear=False`` means log scaling (appropriate for parameters that span
# orders of magnitude like exchange current density).

known = []
unknown = [
    UnknownParameter(initial_guess=2.5e-4, lower_bound=1e-5,  upper_bound=1e-3,
                      key="i0-c", symbol=r"$i_{0,ca}$", units="A/m²", is_linear=False),
    UnknownParameter(initial_guess=0.70,   lower_bound=0.5,   upper_bound=1.0,
                      key="gamma-c", symbol=r"$\gamma_{ca}$", units="n.d."),
    UnknownParameter(initial_guess=20e-7,  lower_bound=5e-7,  upper_bound=50e-7,
                      key="electric-resistance", symbol=r"$r_{elec}$", units="Ω·m²", factor=1e-7),
    UnknownParameter(initial_guess=140e-6, lower_bound=110e-6, upper_bound=200e-6,
                      key="gdl-thickness", symbol=r"$\delta_{GDL}$", units="µm", factor=1e-6),
    UnknownParameter(initial_guess=100.,   lower_bound=91.,   upper_bound=115.,
                      key="cl-theta", symbol=r"$\theta_{CL}$", units="deg"),
    UnknownParameter(initial_guess=15e-6,  lower_bound=8e-6,  upper_bound=20e-6,
                      key="memb-thickness", symbol=r"$\delta_{mb}$", units="µm", factor=1e-6),
]

# %%
# Calibration object
# ------------------

cal = mrpd.SteadyStatePolarizationCurveCalibration(
    conditions_dataset=conditions_df,
    experimental_dataset=experimental_df,
    cell_creator=cell_creator,
    known_parameters=known,
    unknown_parameters=unknown,
)
n_pts     = len(_I)
n_cases   = len(cal.full_case_list)
n_voltage = n_pts * n_cases



# %%
# Global sensitivity analysis
# -----------------------------
#
# A global sensitivity analysis sample the parameter space (``m=5`` → 32 samples) 
# and calculate normalized sensitivity. The `check_samples` parameter allows to 
# select only samples for which the RMSE is below a certain `rmse_limit`. Here we 
# adopt a very high value (0.1 mV/mOhm.cm2) to make calculations faster. Increasing 
# `rmse_limit` require increasing `m` to ensure a statistically representative number 
# of samples is selected.  In the figure, dots show the values for each sample 
# and squares their median. We can see that the electric resistance is the most sensitive
# parameter, followed by the membrane thickness and the ORR reference exchange current density. 

cal.compute_global_sensitivity(m=5, check_samples=True, rmse_limit=0.1)

fig1, ax1 = mrpd.plot_global_sensitivity(cal, figsize=(5, 3))
ax1.set_ylabel("Normalized sensitivity");

# %% We can also plot the colinearity map, where we can see that the sensitivity
# vectors for the ORR reaction order and the ORR reference exchange current density
# are very colinear, indicating that these two parameters cannot be identified simultaneously
# with the given experimental dataset. Logically, the electric resistance and 
# the membrane thickness are also rather colinear. 

fig2, ax2 = mrpd.plot_colinearity_map(cal, figsize=(5, 4))

# %%
# :func:`~marapendi.estimation.plots.plot_parameter_ranking`
# additionally computes the Hessian-based ranking (``cal.P``) used below to
# select which parameters to estimate at each complexity level.
fig, ax, ax_top = mrpd.plot_parameter_ranking(cal)
fig.set_figwidth(5)


# %%
# k-fold cross-validation vs. complexity
# -----------------------------------------
#
# With 9 cases, ``k=3`` folds leaves 3 held-out cases per fold.  The sweep
# refits the model for every number of estimated parameters from 1 to 6,
# selecting parameters by the Hessian ranking computed above
# (:meth:`~marapendi.estimation.BaseModelCalibration.automatic_parameter_selection`).
# ``estimate_kwargs`` uses a small population and iteration count to keep this
# example fast — a real calibration would use larger values.

cal.set_k_folds(k=3)
print("k-fold assignment:", cal.k_folds)

_cv_dir = tempfile.mkdtemp()
_cv_filename = "plot_07_example"

cal.run_k_fold_cross_validation_vs_complexity(
    n_params_list=list(range(1, cal.n_unkown_p + 1)),
    force_restart=True,
    estimate_kwargs=dict(popsize=6, maxiter=15, rtol=0.05, atol=0, ftol=1e-3),
    filename=_cv_filename,
    output_dir=_cv_dir,
)
cal.reset_unknown_parameters()
cv_results = cal.load_cross_validation_results(filename=_cv_filename, dir=_cv_dir)


# %%
# RMSE vs. complexity 
# ----------------------------------
# The dots correspond to the RMSE for different test cases, coloured by fold.  

fig, ax, voltage_rmse_df, n_opt = mrpd.plot_rmse_vs_complexity(
    cal, cv_results, variable='voltage',
    ylabel='Cell voltage RMSE (mV)', quantity_multiplier=1000,
    use_median=True, figsize=(5, 3), xrotation=45, plot_per_case=False
)
ax.legend(loc=0)
ax.semilogy()
ax.set_ylim([5,500]);

fig, ax, hfr_rmse_df, _ = mrpd.plot_rmse_vs_complexity(
    cal, cv_results, variable='hfr',
    ylabel='HFR RMSE (mΩ·cm²)', quantity_multiplier=1e7,
    use_median=True, figsize=(5, 3), xrotation=45, plot_per_case=False
)
ax.set_ylim([0,100]);

# %%
# Parameter evolution vs. complexity
# -----------------------------------
# The dots correspond to the estimated parameters for different test cases, coloured by fold.  
fig, axes = mrpd.plot_parameter_vs_complexity(cal, cv_results, n_cols=3, figsize=(6, 4),
                                              ref_values=true_params)

# %%
# Cross-validation curves
# -----------------------
# Polarization curves generated with 6 parameters estimated parameter sets for different 
# folds (in each row). Shaded plots correspond to the test cases in the fold (which 
# were not used for calibration). 
n_params = 6
cal.build_cases_conditions(current_density=np.linspace(5e2, 2e4, 60))
fig, ax = mrpd.plot_cross_validation_curves(
    cal, cv_results, n_params,
    variable='voltage', quantity_symbol=r"$V_{cell}$", quantity_unit="V",
    x_label=r"$i$ (A/m$^2$)", uncertainty=0.05, figsize=(12,5)
)
ax[-1,-1].set_ylim([0.45,1]);

# %%
# Internal variables at the optimal complexity
# --------------------------------------------------
# Simulate the full ``CellState`` for all 9 conditions at the optimal
# parameter set using fold 1 and 6 estimated parameters.
# Plot cell voltage, HFR, cathode CL liquid saturation, and
# ionomer water content across the MEA.

fold_id = 1
n_params = 6
fold_results = cv_results[(cv_results.n_params == n_params) & (cv_results.fold_id == fold_id)]
voltage_cases, hfr_cases, state_cases = cal.simulate_for_fold_results(fold_results)

fig, ax = plt.subplots(figsize=(10, 6), nrows=4, ncols=n_cases,
                       sharex=True, sharey='row')

for k, case in enumerate(cal.full_case_list):
    state = state_cases[case]
    i_sim = state.current_density * 1e-4
    case_dataset = cal.get_case_dataset(case)

    ax[0, k].plot(i_sim, state.cell_voltage, f'C{k % 10}', label='Sim.')
    ax[0, k].plot(case_dataset['current-density'] * 1e-4, case_dataset['voltage'],
                  's' + f'C{k % 10}', markersize=2.5, alpha=0.5, label='Exp.')
    ax[0, k].set_ylim([0.45,1])
    ax[1, k].plot(i_sim, hfr_cases[case] * 1e7, f'C{k % 10}', label='Sim.')
    ax[1, k].plot(case_dataset['current-density'] * 1e-4, case_dataset['hfr'] * 1e7,
                  's' + f'C{k % 10}', markersize=2.5, alpha=0.5, label='Exp.')

    ax[2, k].plot(i_sim, 100 * state.ca.cl.liquid_saturation, f'C{k % 10}')

    ax[3, k].plot(i_sim, state.ca.cl.ionomer_water_content, '-.' + f'C{k % 10}', label=r'$\lambda^{ca}_{ion,CL}$')
    ax[3, k].plot(i_sim, state.membrane.water_content, '-' + f'C{k % 10}', label=r'$\lambda^{avg}_{mb}$')
    ax[3, k].plot(i_sim, state.an.cl.ionomer_water_content, '--' + f'C{k % 10}', label=r'$\lambda^{an}_{ion,CL}$')

    ax[0, k].set_title(f"Case {case:.0f}", fontsize=9)
    ax[-1, k].set_xlabel("$i$ (A/cm$^2$)", fontsize=9)

ax[0, 0].set_ylabel("$V_{cell}$ (V)", fontsize=11)
ax[1, 0].set_ylabel(r"HFR (m$\Omega$.cm$^2$)", fontsize=11)
ax[2, 0].set_ylabel(r"$s_l^{ca}$ (%)", fontsize=11)
ax[3, 0].set_ylabel(r"$\lambda$ (n.d.)", fontsize=11)

for row in ax.flat:
    row.grid(True, alpha=0.3)

handles0, labels0 = ax[0, 0].get_legend_handles_labels()
handles3, labels3 = ax[3, 0].get_legend_handles_labels()
fig.legend(handles0, labels0, loc='upper left', bbox_to_anchor=(1.0, 0.95), fontsize=8)
fig.legend(handles3, labels3, loc='upper left', bbox_to_anchor=(1.0, 0.3), fontsize=8)
fig.tight_layout()

plt.show()

