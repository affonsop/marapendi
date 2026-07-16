"""
***************************************
Implicit vs explicit steady-state model
***************************************

Both :class:`~marapendi.models.base.explicit_steady_state.ExplicitSteadyStateModel`
and :class:`~marapendi.models.base.implicit_steady_state.ImplicitSteadyStateModel`
compute a steady-state polarization curve from the same cell and conditions,
but differ in how MEA temperature is determined:

+-------------------+----------------------------------------------------------+
| Model             | MEA temperature                                          |
+===================+==========================================================+
| **Explicit**      | Estimated analytically in a single forward pass.         |
+-------------------+----------------------------------------------------------+
| **Implicit**      | Iterated self-consistently with cell voltage via a       |
|                   | vectorised elementwise secant method.                    |
+-------------------+----------------------------------------------------------+

The implicit model is more accurate at high current densities where thermal
feedback is significant; the explicit model is faster.  The current density
range is limited to 3.28 A cm⁻² as the implicit method fails to converge
near the limiting current.

This example uses the same cell and operating conditions as
:ref:`sphx_glr_auto_examples_plot_01_polarization_curve.py`.
"""

# %%
# Cell assembly
# -------------
#
# Identical to :ref:`sphx_glr_auto_examples_plot_01_polarization_curve.py`.

import numpy as np
import matplotlib.pyplot as plt
import marapendi as mrpd

liq = mrpd.DarcyTransportModel(J_function_exponent=0.4)

cell = mrpd.FuelCell(
    area=25e-4,
    electric_resistance=10e-7
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
        thickness=117e-6 * 1.4,
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
    contact_angle=100.0,
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
    contact_angle=100.0,
    two_phase_transport_model=liq,
)

cell.membrane = mrpd.PFSA(ionomer=nafion, dry_thickness=15e-6)


# %%
# Solve with both models
# ----------------------

T   = 273.15 + 71  # K  (71 °C)
dT  = 4
i_arr = np.linspace(1, 32800, 200)  # A/m²

conditions = mrpd.CellConditions(
    current_density=i_arr,
    inlet_cooling_temperature=T - dT / 2,
    outlet_cooling_temperature=T + dT / 2,
    ca=mrpd.SideConditions(
        inlet_temperature=T,
        outlet_pressure=1.4e5,
        dry_o2_mole_fraction=0.21,
        inlet_relative_humidity=0.265,
        stoichiometry=1.6,
    ),
    an=mrpd.SideConditions(
        inlet_temperature=T,
        outlet_pressure=1.9e5,
        dry_h2_mole_fraction=1.0,
        inlet_relative_humidity=0.558,
        stoichiometry=1.4,
    ),
)

exp_model = mrpd.ExplicitSteadyStateModel()
imp_model = mrpd.ImplicitSteadyStateModel()

state_exp = exp_model.solve(cell, conditions, exp_model.set_initial_conditions(cell, conditions))
state_imp = imp_model.solve(cell, conditions, imp_model.set_initial_conditions(cell, conditions))

i_cm2   = i_arr * 1e-4
hfr_exp = exp_model.voltage_model.high_frequency_resistance(cell, state_exp)
hfr_imp = imp_model.voltage_model.high_frequency_resistance(cell, state_imp)


# %%
# Results
# =======
#
# Cell voltage and HFR
# --------------------
#
# The implicit model predicts systematically higher voltage at moderate-to-high
# current densities, where MEA self-heating enhances membrane conductivity.

def _dual_legend(ax, var_labels, colors, loc_vars='upper left', loc_style='upper right', fontsize=7):
    """Split legend into a color-coded variable legend and a line-style (Explicit/Implicit) legend."""
    from matplotlib.lines import Line2D
    var_handles = [Line2D([0], [0], color=c, lw=1.5) for c in colors]
    leg_vars = ax.legend(var_handles, var_labels, loc=loc_vars, fontsize=fontsize)
    ax.add_artist(leg_vars)
    style_handles = [
        Line2D([0], [0], color='k', lw=1.5),
        Line2D([0], [0], color='k', ls='--', lw=1.5),
    ]
    ax.legend(style_handles, ['Explicit', 'Implicit'], loc=loc_style, fontsize=fontsize)


fig, ax = plt.subplots(1, 1, figsize=(4, 3))
ax.plot(i_cm2, state_exp.cell_voltage, 'C0',   lw=1.5)
ax.plot(i_cm2, state_imp.cell_voltage, '--C0', lw=1.5)
ax.set_xlabel("Current density (A/cm$^2$)")
ax.set_ylabel("Cell voltage (V)")
ax.set_ylim(0.2, 1.0)
ax.set_xlim(0, 4)
ax.grid(True)
ax2 = ax.twinx()
ax2.plot(i_cm2, hfr_exp * 1e7, 'C1',   lw=1.5)
ax2.plot(i_cm2, hfr_imp * 1e7, '--C1', lw=1.5)
ax2.set_ylabel(r"HFR (m$\Omega$.cm$^2$)")
ax2.set_ylim(20, 120)
_dual_legend(ax, ['Voltage', 'HFR'], ['C0', 'C1'], loc_vars='upper right', loc_style='lower right')
fig.tight_layout()


# %%
# MEA temperature
# ---------------
#
# The implicit model solves for MEA temperature self-consistently with cell
# voltage; the explicit model computes it analytically in one forward pass.
# The gap grows with ohmic and reaction heat at high current density.

fig, ax = plt.subplots(1, 1, figsize=(4, 3))
ax.plot(i_cm2, state_exp.mea_temperature - 273.15, 'C0',   lw=1.5, label="Explicit")
ax.plot(i_cm2, state_imp.mea_temperature - 273.15, '--C0', lw=1.5, label="Implicit")
ax.set_xlabel("Current density (A/cm$^2$)")
ax.set_ylabel("MEA temperature (°C)")
ax.set_xlim(0, 4)
ax.grid(True)
ax.legend(loc='upper left')
fig.tight_layout()

# %%
# Crossover current density
# -------------------------

i_x_exp = 2 * mrpd.FARADAY_CONSTANT * state_exp.membrane.h2_permeation_flux
i_x_imp = 2 * mrpd.FARADAY_CONSTANT * state_imp.membrane.h2_permeation_flux
fig, ax = plt.subplots(1, 1, figsize=(4, 3))
ax.plot(i_cm2, i_x_exp * 1e-4 * 1e3, 'C0',   lw=1.5, label="Explicit")
ax.plot(i_cm2, i_x_imp * 1e-4 * 1e3, '--C0', lw=1.5, label="Implicit")
ax.grid(True)
ax.set_ylabel("Crossover current density\n(mA/cm$^2$)")
ax.set_xlabel("Current density (A/cm$^2$)")
ax.set_xlim(0, 4)
ax.legend(loc='upper right')
fig.tight_layout()


# %%
# Proton resistance in the cathode CL
# -----------------------------------

fig, ax = plt.subplots(1, 1, figsize=(4, 3))
ax.plot(i_cm2, state_exp.ca.cl.proton_resistance * 1e7, 'C0',   lw=1.5, label="Explicit")
ax.plot(i_cm2, state_imp.ca.cl.proton_resistance * 1e7, '--C0', lw=1.5, label="Implicit")
ax.grid(True)
ax.set_ylabel(r"Proton resistance in the CL (m$\Omega$.cm$^2$)")
ax.set_xlabel("Current density (A/cm$^2$)")
ax.set_xlim(0, 4)
ax.legend(loc='upper right')
fig.tight_layout()


# %%
# Liquid saturation at the cathode
# --------------------------------

fig, ax = plt.subplots(1, 1, figsize=(4, 3))
ax.plot(i_cm2, state_exp.ca.cl.liquid_saturation,  'C0',   lw=1.5)
ax.plot(i_cm2, state_exp.ca.mpl.liquid_saturation, 'C1',   lw=1.5)
ax.plot(i_cm2, state_exp.ca.gdl.liquid_saturation, 'C2',   lw=1.5)
ax.plot(i_cm2, state_imp.ca.cl.liquid_saturation,  '--C0', lw=1.5)
ax.plot(i_cm2, state_imp.ca.mpl.liquid_saturation, '--C1', lw=1.5)
ax.plot(i_cm2, state_imp.ca.gdl.liquid_saturation, '--C2', lw=1.5)
ax.grid(True)
ax.set_ylabel("Cathode water\nsaturation (n.d.)")
ax.set_xlabel("Current density (A/cm$^2$)")
ax.set_xlim(0, 4)
_dual_legend(ax, ['CL', 'MPL', 'GDL'], ['C0', 'C1', 'C2'], fontsize=7)
fig.tight_layout()


# %%
# Membrane and CL ionomer water contents
# --------------------------------------

fig, ax = plt.subplots(1, 1, figsize=(4, 3))
ax.plot(i_cm2, state_exp.ca.cl.ionomer_water_content,   'C0',   lw=1.5)
ax.plot(i_cm2, state_exp.membrane.water_content,        'C1',   lw=1.5)
ax.plot(i_cm2, state_exp.an.cl.ionomer_water_content,   'C2',   lw=1.5)
ax.plot(i_cm2, state_imp.ca.cl.ionomer_water_content,   '--C0', lw=1.5)
ax.plot(i_cm2, state_imp.membrane.water_content,        '--C1', lw=1.5)
ax.plot(i_cm2, state_imp.an.cl.ionomer_water_content,   '--C2', lw=1.5)
ax.set_ylabel("Water content (n.d.)")
ax.set_xlabel("Current density (A/cm$^2$)")
ax.set_xlim(0, 4)
ax.grid(True)
_dual_legend(ax, [r"$\lambda^\mathrm{ion}_{CL,ca}$",
                  r"$\lambda^\mathrm{mb}_{avg}$",
                  r"$\lambda^\mathrm{ion}_{CL,an}$"],
             ['C0', 'C1', 'C2'], fontsize=8)
fig.tight_layout()


# %%
# Cathode water fluxes
# --------------------

fig, ax = plt.subplots(1, 1, figsize=(4, 3))
ax.plot(i_cm2, 1e5 * state_exp.ca.water_flux,          'C0',   lw=1.5)
ax.plot(i_cm2, 1e5 * state_exp.ca.liquid_flux,         'C1',   lw=1.5)
ax.plot(i_cm2, 1e5 * state_exp.ca.membrane_water_flux, 'C2',   lw=1.5)
ax.plot(i_cm2, 1e5 * state_imp.ca.water_flux,          '--C0', lw=1.5)
ax.plot(i_cm2, 1e5 * state_imp.ca.liquid_flux,         '--C1', lw=1.5)
ax.plot(i_cm2, 1e5 * state_imp.ca.membrane_water_flux, '--C2', lw=1.5)
ax.set_ylabel(r"Water flux (µmol/cm$^2$)")
ax.set_xlabel("Current density (A/cm$^2$)")
ax.set_xlim(0, 4)
ax.grid(True)
_dual_legend(ax, [r"$J^\mathrm{ca}_{w}$",
                  r"$J^\mathrm{ca}_{l}$",
                  r"$J^\mathrm{mb}_{w}$"],
             ['C0', 'C1', 'C2'], fontsize=8)
fig.tight_layout()


# %%
# Cathode O2 transport resistance
# --------------------------------

fig, ax = plt.subplots(1, 1, figsize=(4, 3))
ax.plot(i_cm2, state_exp.ca.cl.local_o2_resistance,            'C0--', lw=1.2, label='CL local')
ax.plot(i_cm2, state_exp.ca.cl.gas_transport_resistance['o2'], 'C0',   lw=1.5, label='CL total')
ax.plot(i_cm2, state_exp.ca.mpl.gas_transport_resistance['o2'],'C1',   lw=1.5, label='MPL')
ax.plot(i_cm2, state_exp.ca.gdl.gas_transport_resistance['o2'],'C2',   lw=1.5, label='GDL')
ax.set_ylim([0, 50])
ax.set_ylabel("Cathode O$_2$\ntransport resistance (s/m)")
ax.set_xlabel("Current density (A/cm$^2$)")
ax.set_xlim(0, 4)
ax.grid(True)
ax.legend(loc='upper left', fontsize=7)
fig.tight_layout()

plt.show()
