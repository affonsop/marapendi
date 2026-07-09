"""
*************************************
Quasi-steady simulation — FC-DLC profile
*************************************

The **Fuel Cell Dynamic Load Cycle** (FC-DLC) is the JRC/FCH-JU standard
load profile for PEMFC endurance testing (Tsotridis et al., EUR 27632 EN,
2015, Appendix F).  It is derived from the New European Driving Cycle (NEDC)
and consists of 35 piecewise-constant steps covering 1181 s, including four
urban sub-cycles and one extra-urban sub-cycle.

A quasi-steady simulation treats each time step as an independent steady-state
point: all steps are packed into a single vectorised
:class:`~marapendi.simulation.conditions.CellConditions` array and evaluated
in **one call** — no Python loop required.

This example runs **3 consecutive FC-DLC cycles** using the same cell and
operating conditions as :ref:`sphx_glr_auto_examples_plot_01_polarization_curve.py`.
"""

# %%
# FC-DLC cycle
# ============
#
# :class:`~marapendi.simulation.nedc.NEDCCycle` encapsulates the 35-step
# FC-DLC protocol.  Setting ``i_max = 1.7 A cm⁻²`` scales all load fractions
# to the target maximum current density. This maximum current is defined in the 
# FCH-JU protocol as the one at 0.65 V. Here we use an arbitray value.
# OCV steps (0 % load) are replaced by ``i_min`` to avoid division-by-zero 
# in the model.

import numpy as np
import matplotlib.pyplot as plt
import marapendi as mrpd

nedc = mrpd.NEDCCycle(
        max_current_density=1.7e4,
        time_step=1.0,)

conditions = nedc.conditions(n_cycles=3)

# %% 
# We can check the cycle profile. 
nedc.plot()

# %%
# Cell assembly
# -------------
#
# Identical to :ref:`sphx_glr_auto_examples_plot_01_polarization_curve.py`.

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
# Vectorised quasi-steady solve
# ------------------------------
#
# The FC-DLC current profile is substituted for the polarization-curve sweep;
# all other operating conditions are constant and match example 01.

model = mrpd.ExplicitSteadyStateModel()
state = model.solve(cell, conditions, model.set_initial_conditions(cell, conditions))


# %%
# Results
# =======
#
# Cell voltage and HFR
# --------------------

t_min = conditions.time / 60.

fig, ax = plt.subplots(1, 1, figsize=(10, 3))
ax.plot(t_min, state.cell_voltage, color='C0', lw=1.0, label='$V_{cell}$')
ax.set_xlabel("Time (min)")
ax.set_ylabel("Cell voltage (V)")
ax.set_ylim(0.4, 1.0)
ax.set_xlim(0,conditions.time[-1]/60)
ax.grid(True)

ax2 = ax.twinx()
ax2.plot(t_min, state.hfr * 1e7, color='C1', lw=1.0, label='HFR')
ax2.set_ylabel(r"HFR (m$\Omega$.cm$^2$)")
ax2.set_ylim(0, 500)
ax.legend(loc='upper right')
ax2.legend(loc='lower right')
fig.tight_layout()


# %%
# Crossover current density
# -------------------------

i_x = 2 * mrpd.FARADAY_CONSTANT * state.membrane.h2_permeation_flux
fig, ax = plt.subplots(1, 1, figsize=(10, 3))
ax.plot(t_min, i_x * 1e-4 * 1e3, color='C0', lw=1.0)
ax.grid(True)
ax.set_xlim(0,conditions.time[-1]/60)
ax.set_ylabel("Crossover current density\n(mA/cm$^2$)")
ax.set_xlabel("Time (min)")
fig.tight_layout()


# %%
# Proton resistance in the cathode CL
# -----------------------------------

fig, ax = plt.subplots(1, 1, figsize=(10, 3))
ax.plot(t_min, state.ca.cl.proton_resistance * 1e7, color='C0', lw=1.0)
ax.grid(True)
ax.set_xlim(0,conditions.time[-1]/60)
ax.set_ylabel(r"Proton resistance in the CL (m$\Omega$.cm$^2$)")
ax.set_xlabel("Time (min)")
fig.tight_layout()


# %%
# Liquid saturation at the cathode
# --------------------------------

fig, ax = plt.subplots(1, 1, figsize=(10, 3))
ax.plot(t_min, state.ca.cl.liquid_saturation, color='C0', lw=1.0, label='CL')
ax.plot(t_min, state.ca.mpl.liquid_saturation, color='C1', lw=1.0, label='MPL')
ax.plot(t_min, state.ca.gdl.liquid_saturation, color='C2', lw=1.0, label='GDL')
ax.grid(True)
ax.set_xlim(0,conditions.time[-1]/60)
ax.set_ylabel("Cathode water\nsaturation (n.d.)")
ax.set_xlabel("Time (min)")
ax.legend(loc='upper left')
fig.tight_layout()


# %%
# Membrane and CL ionomer water contents
# --------------------------------------

fig, ax = plt.subplots(1, 1, figsize=(10, 3))
ax.plot(t_min, state.ca.cl.ionomer_water_content,         'C0', lw=1.0, label=r"$\lambda^\mathrm{ion}_{CL,ca}$")
ax.plot(t_min, state.ca.membrane_interface_water_content, '--', lw=1.0, color='C0', label=r"$\lambda^\mathrm{mb}_{ca}$")
ax.plot(t_min, state.membrane.water_content,              'C1', lw=1.0, label=r"$\lambda^\mathrm{mb}_{avg}$")
ax.plot(t_min, state.an.membrane_interface_water_content, '--', lw=1.0, color='C2', label=r"$\lambda^\mathrm{mb}_{an}$")
ax.plot(t_min, state.an.cl.ionomer_water_content,         'C2', lw=1.0, label=r"$\lambda^\mathrm{ion}_{CL,an}$")
ax.set_ylabel("Water content (n.d.)")
ax.set_xlabel("Time (min)")
ax.set_xlim(0,conditions.time[-1]/60)
ax.grid(True)
ax.legend(loc='upper left')
fig.tight_layout()


# %%
# Cathode water fluxes
# --------------------

fig, ax = plt.subplots(1, 1, figsize=(10, 3))
ax.plot(t_min, 1e5 * state.ca.h2o_production,      'C0', lw=1.0, label=r"$i/2F$")
ax.plot(t_min, 1e5 * state.ca.water_flux,           'C1', lw=1.0, label=r"$J^\mathrm{ca}_{w}$")
ax.plot(t_min, 1e5 * state.ca.liquid_flux,          'C2', lw=1.0, label=r"$J^\mathrm{ca}_{l}$")
ax.plot(t_min, 1e5 * state.ca.membrane_water_flux,  'C3', lw=1.0, label=r"$J^\mathrm{mb}_{w}$")
ax.set_ylabel(r"Water flux (µmol/cm$^2$)")
ax.set_xlabel("Time (min)")
ax.set_xlim(0,conditions.time[-1]/60)
ax.legend(loc='upper left')
ax.grid(True)
fig.tight_layout()


# %%
# Cathode O2 transport resistance
# --------------------------------

fig, ax = plt.subplots(1, 1, figsize=(10, 3))
ax.plot(t_min, state.ca.cl.local_o2_resistance,             'C0--', lw=1.0, label='CL local')
ax.plot(t_min, state.ca.cl.gas_transport_resistance['o2'],  'C0',   lw=1.0, label='CL total')
ax.plot(t_min, state.ca.mpl.gas_transport_resistance['o2'], 'C1',   lw=1.0, label='MPL')
ax.plot(t_min, state.ca.gdl.gas_transport_resistance['o2'], 'C2',   lw=1.0, label='GDL')
ax.set_ylim([0, 50])
ax.set_xlim(0,conditions.time[-1]/60)
ax.set_ylabel("Cathode O$_2$\ntransport resistance (s/m)")
ax.set_xlabel("Time (min)")
ax.legend(loc='upper left')
ax.grid(True)
fig.tight_layout()

plt.show()
