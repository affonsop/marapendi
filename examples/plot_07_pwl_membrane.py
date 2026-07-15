"""
****************************************
Piecewise-linear membrane water balance
****************************************

``marapendi`` provides two membrane water-balance models, both solving the
same steady 1D diffusion + electroosmotic-drag boundary-value problem across
the membrane (see :doc:`/science/water_balance`), but with different closures
for the equilibrium water content at the catalyst-layer interfaces:

* :class:`~marapendi.models.water_balance.membrane_pwl.MembraneWaterBalanceModelPiecewise`
  — the **default** model. The equilibrium isotherm RH(λ) is replaced by a
  piecewise-linear regression, fit once at
  :class:`~marapendi.components.membrane.pem.PFSAIonomer` construction time by
  :meth:`~marapendi.components.membrane.pem.PFSAIonomer.fit_rh_piecewise_linear`. The
  local slope of the active segment gives an equivalent transport resistance
  that is exact on that segment.

* :class:`~marapendi.models.water_balance.membrane.MembraneWaterBalanceModel`
  — the model of Affonso Nobrega et al. (2026), which instead linearises the
  isotherm with a single first-order Taylor expansion around an estimated
  water activity.

This example first inspects the quality of the piecewise-linear fit, then
reuses the cell of :doc:`plot_01_polarization_curve` to compare the
polarization curve, membrane water content, and HFR predicted by both models.
"""

# %%
# Isotherm fit
# ============
#
# :class:`~marapendi.components.membrane.pem.PFSAIonomer` fits a piecewise-linear
# approximation of RH(λ) at ``__post_init__`` time. The fit is cached, so it
# runs at most once per unique (polynomial, number of segments, temperature)
# combination.

import numpy as np
import matplotlib.pyplot as plt
import marapendi as mrpd

ionomer = mrpd.PFSAIonomer()  # fitted at construction time
T_ref = ionomer.pwl_temperature

rh_ref = np.linspace(0.0, 1.0, 500)
lmbd_ref = ionomer.vapor_equilibrium_water_content(rh_ref, T_ref)

n_seg = len(ionomer.pwl_slopes)
print(f"Piecewise-linear regression of RH(lambda_eq)  (T = {T_ref:.0f} K, {n_seg} segments)\n")
print(f"Fitting intervals (RH)     : {np.round(ionomer.fit_rh_breaks, 4).tolist()}")
print(f"Validity intervals (lambda): {np.round(ionomer.lmbd_pwl_breaks, 4).tolist()}")
print("  (boundaries = line intersections, so continuity is guaranteed)\n")
print(f"{'segment':<8} {'validity lambda interval':<26} {'slope':>12} {'intercept':>12}")
for k, (a, b, lo, hi) in enumerate(zip(
    ionomer.pwl_slopes, ionomer.pwl_intercepts,
    ionomer.lmbd_pwl_breaks[:-1], ionomer.lmbd_pwl_breaks[1:],
)):
    print(f"{k:<8} [{lo:.4f}, {hi:.4f}]              {a:>12.6f} {b:>12.6f}")

# %%
# Each segment gives RH as a linear function of λ, RH = a·λ + b, which is the
# form used in the boundary condition of
# :meth:`~marapendi.models.water_balance.membrane_pwl.MembraneWaterBalanceModelPiecewise.solve_membrane_water_balance`
# (see :doc:`/science/water_balance` for the derivation of the equivalent
# sorption coefficient :math:`k_{eq}` from the segment slope).

rh_pwl = ionomer.linear_rh_from_water_content(lmbd_ref)
rms = float(np.sqrt(np.mean((rh_pwl - rh_ref) ** 2)))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

ax = axes[0]
ax.plot(rh_ref, lmbd_ref, "k", lw=1.5, label="Exact", zorder=3)
ax.plot(rh_pwl[:-2], lmbd_ref[:-2], "C0--", lw=1.5, label=f"PWL (RMS={rms * 100:.2f} %)")
ax.scatter(ionomer.rh_pwl_breaks[1:-1], ionomer.lmbd_pwl_breaks[1:-1],
           color="C0", zorder=5, s=60, label="Intersections")
ax.set_ylabel(r"$\lambda_{eq}$ (n.d.)")
ax.set_xlabel("Relative humidity (-)")
ax.set_title(r"RH($\lambda_{eq}$) — continuous piecewise regression")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.4)

lmbd_pwl = ionomer.linear_water_content_from_rh(rh_ref)
rms_lmbd = float(np.sqrt(np.mean((lmbd_pwl - lmbd_ref) ** 2)))

ax = axes[1]
ax.plot(rh_ref, lmbd_pwl - lmbd_ref, "C0", lw=1.2, label=f"RMS={rms_lmbd:.3f} mol/mol")
ax.axhline(0, color="k", lw=0.7)
for rb in ionomer.rh_pwl_breaks[1:-1]:
    ax.axvline(rb, color="C0", lw=0.7, ls="--", alpha=0.6)
ax.set_xlabel("Relative humidity (-)")
ax.set_ylabel(r"$\Delta\lambda_{eq}$ = approx $-$ exact (mol H$_2$O / mol SO$_3^-$)")
ax.set_title("Regression error (dashed = validity boundaries)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.4)

fig.tight_layout()

# %%
# Cell assembly
# =============
#
# We reuse the exact cell of :doc:`plot_01_polarization_curve`, wrapped in a
# helper so it can be instantiated twice (one cell per water-balance model,
# sharing the same fitted ``ionomer``).

orr = mrpd.ElectrochemicalReaction(
    reference_exchange_current_density=1e-3,
    reaction_order=0.8,
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


def _build_cell():
    liq = mrpd.DarcyTransportModel(J_function_exponent=0.4)

    cell = mrpd.FuelCell(area=25e-4, electric_resistance=10e-7)

    for side in cell.sides:
        side.ch = mrpd.FlowChannel(
            width=0.85e-3, height=1e-3, length=0.49, n_parallel=3,
            reactant="o2" if side is cell.ca else "h2",
            transport_resistance_model=mrpd.ChannelGasResistanceModel(sherwood=3.66, B_ch=1.2),
        )
        side.gdl = mrpd.GasDiffusionLayer(
            thickness=117e-6 * 1.4, porosity=0.65, tortuosity=1.55,
            contact_angle=110.0, absolute_permeability=3e-12,
            thermal_conductivity=1.2, two_phase_transport_model=liq,
            relative_permeability_exponent=3, volume_heat_capacity=1.58e6,
        )
        side.mpl = mrpd.MicroPorousLayer(
            thickness=22e-6, porosity=0.4, tortuosity=3, pore_diameter=500e-9,
            contact_angle=130.0, absolute_permeability=1e-12,
            thermal_conductivity=0.144, two_phase_transport_model=liq,
            relative_permeability_exponent=3, volume_heat_capacity=1.98e6,
        )
        side.thermal_contact_resistance = 1e-4

    cell.ca.cl = mrpd.PtCCatalystLayer(
        ecsa=40e3, platinum_loading=0.5e-2, catalyst_platinum_weight_percent=0.5,
        ionomer_to_carbon_ratio=0.81, ionomer=nafion, reaction=orr,
        thickness=10e-6, tortuosity=3, thermal_conductivity=0.18,
        pore_diameter=140e-9, carbon_agglomerate_radius=25e-9,
        absolute_permeability=2e-13, contact_angle=100.0,
        two_phase_transport_model=mrpd.DarcyTransportModel(J_function_exponent=0.4),
        relative_permeability_exponent=3, volume_heat_capacity=1.56e6,
    )
    cell.an.cl = mrpd.PtCCatalystLayer(
        platinum_loading=0.1e-2, ionomer=nafion, thickness=7e-6,
        ionomer_to_carbon_ratio=0.57, catalyst_platinum_weight_percent=0.2,
        thermal_conductivity=0.18, pore_diameter=140e-9,
        absolute_permeability=1e-13, contact_angle=100.0,
        two_phase_transport_model=mrpd.DarcyTransportModel(J_function_exponent=0.4),
        volume_heat_capacity=1.56e6,
    )
    cell.membrane = mrpd.PFSA(ionomer=nafion, dry_thickness=15e-6)
    return cell


T = 273.15 + 71  # K  (71 degC)
i = np.linspace(1, 40000, 200)  # A/m^2

conditions = mrpd.CellConditions(
    current_density=i,
    cell_temperature=T,
    ca=mrpd.SideConditions(
        inlet_temperature=T, outlet_pressure=1.4e5, dry_o2_mole_fraction=0.21,
        inlet_relative_humidity=0.265, stoichiometry=1.6,
        minimum_current_density_for_stoich=0,
    ),
    an=mrpd.SideConditions(
        inlet_temperature=T, outlet_pressure=1.9e5, dry_h2_mole_fraction=1.0,
        inlet_relative_humidity=0.558, stoichiometry=1.4,
        minimum_current_density_for_stoich=0,
    ),
)

# %%
# Model comparison
# =================
#
# The two water-balance closures are swapped in through
# :class:`~marapendi.models.water_balance.water_balance.WaterBalanceModel`'s
# ``membrane_water_balance_model`` argument; everything else about the
# :class:`~marapendi.models.base.explicit_steady_state.ExplicitSteadyStateModel`
# is left at its default.
#
# We can see that the membrane water content is much lower with the Taylor expansion, 
# leading to high HFR and lower cell voltage. 

pwl_model = mrpd.ExplicitSteadyStateModel(
    water_balance_model=mrpd.WaterBalanceModel(
        membrane_water_balance_model=mrpd.MembraneWaterBalanceModelPiecewise()
    )
)
taylor_model = mrpd.ExplicitSteadyStateModel(
    water_balance_model=mrpd.WaterBalanceModel(
        membrane_water_balance_model=mrpd.MembraneWaterBalanceModel()
    )
)

cell_pwl = _build_cell()
cell_taylor = _build_cell()

state_pwl = pwl_model.solve(cell_pwl, conditions, pwl_model.set_initial_conditions(cell_pwl, conditions))
state_taylor = taylor_model.solve(cell_taylor, conditions, taylor_model.set_initial_conditions(cell_taylor, conditions))

i_cm2 = i * 1e-4

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].plot(i_cm2, state_pwl.cell_voltage, "C0-", label="Piecewise-linear")
axes[0].plot(i_cm2, state_taylor.cell_voltage, "C1--", label="Taylor expansion")
axes[0].set_xlabel(r"Current density (A/cm$^2$)")
axes[0].set_ylabel("Cell voltage (V)")
axes[0].set_title("Polarization curve")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(i_cm2, state_pwl.membrane.water_content, "C0-", label="Piecewise-linear")
axes[1].plot(i_cm2, state_taylor.membrane.water_content, "C1--", label="Taylor expansion")
axes[1].set_xlabel(r"Current density (A/cm$^2$)")
axes[1].set_ylabel(r"Mean membrane $\lambda$ (n.d.)")
axes[1].set_title("Membrane water content")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(i_cm2, state_pwl.hfr * 1e7, "C0-", label="Piecewise-linear")
axes[2].plot(i_cm2, state_taylor.hfr * 1e7, "C1--", label="Taylor expansion")
axes[2].set_xlabel(r"Current density (A/cm$^2$)")
axes[2].set_ylabel(r"HFR (m$\Omega$.cm$^2$)")
axes[2].set_title("High-frequency resistance")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

fig.tight_layout()
