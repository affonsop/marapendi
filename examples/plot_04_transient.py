"""
*************************************
Transient simulation — ID-FAST driving cycle
*************************************

:class:`~marapendi.cell.transient.TransientModel` integrates coupled ODEs for
MEA temperature and the membrane water content
profile.

This example drives the cell through one complete **ID-FAST** cycle
(Colombo et al., J. Power Sources 553 (2023) 232246; Table 3 and Fig. 1A)
— a 3925 s load cycle derived from a real automotive fleet dataset. The short 
stop period, where the cathode flux is stopped and a resistance draws current until 
voltage reaches 0.2 V is simulated with a very low O2 mole fraction in the dry cathode
mixture. 


Transient dynamics of MEA temperature, membrane water content, cell voltage,
and high-frequency resistance are compared against the quasi-steady-state (QSS)
prediction at every time step.

The cell is identical to that in
:ref:`sphx_glr_auto_examples_plot_01_polarization_curve.py`.

Reference
---------
Colombo E. et al., "PEMFC performance decay during real-world automotive
operation", J. Power Sources 553 (2023) 232246.
doi:10.1016/j.jpowsour.2022.232246
"""

# %%
# ID-FAST cycle definition
# -------------------------

import time
import numpy as np
import matplotlib.pyplot as plt
import marapendi as mrpd
from marapendi.models.base.transient import TransientModel
from marapendi.models.base.explicit_steady_state import ExplicitSteadyStateModel
from marapendi.simulation.load_cycles.idfast import IDFastCycle


conditions = IDFastCycle()
fig, ax = conditions.plot()

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
# Transient integration
# ----------------------

tr_model = TransientModel(n_memb_mesh=3)
state, x0    = tr_model.set_initial_conditions(cell, conditions(0))

_t0 = time.perf_counter()
sol = tr_model.solve(cell, conditions, t_span=(0, conditions.duration),
                     x0=x0, dense_output=True, method='BDF', max_step=10)
tr_wall_time = time.perf_counter() - _t0
print(f"ODE status: {sol.status}  ({sol.message})")
print(f"Number of ODE steps: {len(sol.t)}")


# %%
# Evaluate diagnostics on a regular time grid
# --------------------------------------------

t_eval = np.linspace(0, conditions.duration, 1000)
diag   = tr_model.evaluate(cell, conditions, t_eval, x_eval=sol.sol(t_eval))


# %%
# Quasi-steady reference
# -----------------------

v          = conditions.get_input_vectors(t_eval)
i_arr      = v['current-density']
p_ca_arr   = v['ca-outlet-pressure']
p_an_arr   = v['an-outlet-pressure']
T_gas_arr  = v['ca-inlet-temperature']
T_cool_in  = v['inlet-cooling-temperature']
T_cool_out = v['outlet-cooling-temperature']
rh_ca_arr  = v['ca-inlet-rh']
rh_an_arr  = v['an-inlet-rh']
st_ca      = v['ca-stoichiometry']
st_an      = v['an-stoichiometry']
x_o2       = v['ca-dry-o2-mole-fraction']

qss_cond = mrpd.CellConditions(
    current_density=i_arr,
    inlet_cooling_temperature=T_cool_in,
    outlet_cooling_temperature=T_cool_out,
    ca=mrpd.SideConditions(
        inlet_temperature=T_gas_arr,
        outlet_pressure=p_ca_arr,
        dry_o2_mole_fraction=x_o2,
        inlet_relative_humidity=rh_ca_arr,
        stoichiometry=st_ca,
    ),
    an=mrpd.SideConditions(
        inlet_temperature=T_gas_arr,
        outlet_pressure=p_an_arr,
        dry_h2_mole_fraction=1.0,
        inlet_relative_humidity=rh_an_arr,
        stoichiometry=st_an,
    ),
)
ss_model = ExplicitSteadyStateModel()
_t0 = time.perf_counter()
ss_state = ss_model.solve(cell, qss_cond,
                          ss_model.set_initial_conditions(cell, qss_cond))
qss_wall_time = time.perf_counter() - _t0
hfr_qss  = ss_model.voltage_model.high_frequency_resistance(cell, ss_state)


# %%
# Results
# =======
#
# Cell voltage and HFR
# ---------------------

t_min = t_eval / 60.

def _vline(ax):
    pass


def _dual_legend(ax, var_labels, colors, loc_vars='upper left', loc_style='upper right', fontsize=7):
    """Split legend into a color-coded variable legend and a line-style (Transient/QSS) legend."""
    from matplotlib.lines import Line2D
    var_handles = [Line2D([0], [0], color=c, lw=1.5) for c in colors]
    leg_vars = ax.legend(var_handles, var_labels, loc=loc_vars, fontsize=fontsize)
    ax.add_artist(leg_vars)
    style_handles = [
        Line2D([0], [0], color='k', lw=1.5),
        Line2D([0], [0], color='k', ls='--', lw=1.5, alpha=0.5),
    ]
    ax.legend(style_handles, ['Transient', 'QSS'], loc=loc_style, fontsize=fontsize)


fig, ax = plt.subplots(1, 1, figsize=(10, 3))
ax.plot(t_min, diag.cell_voltage,       'C0',   lw=1.5)
ax.plot(t_min, ss_state.cell_voltage,   '--C0', lw=1.5, alpha=0.5)
ax.set_xlabel("Time (min)")
ax.set_xlim(t_min[0],t_min[-1])
ax.set_ylabel("Cell voltage (V)")
ax.grid(True)
_vline(ax)
ax2 = ax.twinx()
ax2.plot(t_min, np.asarray(diag.hfr) * 1e7, 'C1',   lw=1.5)
ax2.plot(t_min, hfr_qss * 1e7,              '--C1', lw=1.5, alpha=0.5)
ax2.set_ylabel(r"HFR (m$\Omega$.cm$^2$)")

_dual_legend(ax, ['Voltage', 'HFR'], ['C0', 'C1'], loc_vars='upper right', loc_style='lower right')
fig.tight_layout()


# %%
# MEA temperature
# ---------------

fig, ax = plt.subplots(1, 1, figsize=(10, 3))
ax.plot(t_min, diag.mea_temperature - 273.15,       'C0',   lw=1.5, label="Transient")
ax.plot(t_min, ss_state.mea_temperature - 273.15,   '--C0', lw=1.5, alpha=0.5, label="QSS")
ax.set_xlabel("Time (min)")
ax.set_xlim(t_min[0],t_min[-1])
ax.set_ylabel("MEA temperature (°C)")
ax.grid(True)
ax.legend(loc='upper right')
_vline(ax)
fig.tight_layout()

# %%
# Crossover current density
# -------------------------

i_x_tr  = 2 * mrpd.FARADAY_CONSTANT * np.asarray(diag.membrane.h2_permeation_flux)
i_x_qss = 2 * mrpd.FARADAY_CONSTANT * ss_state.membrane.h2_permeation_flux
fig, ax = plt.subplots(1, 1, figsize=(10, 3))
ax.plot(t_min, i_x_tr  * 1e-4 * 1e3, 'C0',   lw=1.5, label="Transient")
ax.plot(t_min, i_x_qss * 1e-4 * 1e3, '--C0', lw=1.5, alpha=0.5, label="QSS")
ax.grid(True)
ax.set_ylabel("Crossover current density\n(mA/cm$^2$)")
ax.set_xlabel("Time (min)")
ax.set_xlim(t_min[0],t_min[-1])
ax.legend(loc='upper right')
_vline(ax)
fig.tight_layout()


# %%
# Proton resistance in the cathode CL
# -----------------------------------

fig, ax = plt.subplots(1, 1, figsize=(10, 3))
ax.plot(t_min, np.asarray(diag.ca.cl.proton_resistance) * 1e7, 'C0',   lw=1.5, label="Transient")
ax.plot(t_min, ss_state.ca.cl.proton_resistance * 1e7,  '--C0', lw=1.5, alpha=0.5, label="QSS")
ax.grid(True)
ax.set_ylabel(r"Proton resistance in the CL (m$\Omega$.cm$^2$)")
ax.set_xlabel("Time (min)")
ax.set_xlim(t_min[0],t_min[-1])
ax.legend(loc='upper right')
_vline(ax)
fig.tight_layout()


# %%
# Liquid saturation at the cathode
# --------------------------------

fig, ax = plt.subplots(1, 1, figsize=(10, 3))
ax.plot(t_min, np.asarray(diag.ca.cl.liquid_saturation),  'C0',   lw=1.5)
ax.plot(t_min, np.asarray(diag.ca.mpl.liquid_saturation), 'C1',   lw=1.5)
ax.plot(t_min, np.asarray(diag.ca.gdl.liquid_saturation), 'C2',   lw=1.5)
ax.plot(t_min, ss_state.ca.cl.liquid_saturation,           '--C0', lw=1.5, alpha=0.5)
ax.plot(t_min, ss_state.ca.mpl.liquid_saturation,          '--C1', lw=1.5, alpha=0.5)
ax.plot(t_min, ss_state.ca.gdl.liquid_saturation,          '--C2', lw=1.5, alpha=0.5)
ax.grid(True)
ax.set_ylabel("Cathode water\nsaturation (n.d.)")
ax.set_xlabel("Time (min)")
ax.set_xlim(t_min[0],t_min[-1])
_dual_legend(ax, ['CL', 'MPL', 'GDL'], ['C0', 'C1', 'C2'], fontsize=7)
_vline(ax)
fig.tight_layout()


# %%
# Membrane and CL ionomer water contents
# --------------------------------------

fig, ax = plt.subplots(1, 1, figsize=(10, 3))
ax.plot(t_min, np.asarray(diag.ca.cl.ionomer_water_content),         'C0',   lw=1.5)
ax.plot(t_min, np.asarray(diag.membrane.water_content),               'C1',  lw=1.5)
ax.plot(t_min, np.asarray(diag.an.cl.ionomer_water_content),          'C2',  lw=1.5)
ax.plot(t_min, ss_state.ca.cl.ionomer_water_content,                  '--C0', lw=1.5, alpha=0.5)
ax.plot(t_min, ss_state.membrane.water_content,                        '--C1', lw=1.5, alpha=0.5)
ax.plot(t_min, ss_state.an.cl.ionomer_water_content,                   '--C2', lw=1.5, alpha=0.5)
ax.set_ylabel("Water content (n.d.)")
ax.set_xlabel("Time (min)")
ax.set_xlim(t_min[0],t_min[-1])
ax.set_ylim([0,20])
ax.grid(True)
_dual_legend(ax, [r"$\lambda^\mathrm{ion}_{CL,ca}$", 
                  r"$\lambda^\mathrm{mb}_{avg}$", 
                  r"$\lambda^\mathrm{ion}_{CL,an}$"],
             ['C0', 'C1', 'C2'], fontsize=8)
_vline(ax)
fig.tight_layout()


# %%
# Cathode water fluxes
# --------------------

fig, ax = plt.subplots(1, 1, figsize=(10, 3))
ax.plot(t_min, 1e5 * np.asarray(diag.ca.water_flux),          'C0', lw=1.5, label=r"$J^\mathrm{ca}_{w}$")
ax.plot(t_min, 1e5 * np.asarray(diag.ca.liquid_flux),         'C1', lw=1.5, label=r"$J^\mathrm{ca}_{l}$")
ax.plot(t_min, 1e5 * np.asarray(diag.ca.membrane_water_flux), 'C2', lw=1.5, label=r"$J^\mathrm{mb}_{w}$")
ax.plot(t_min, 1e5 * np.asarray(ss_state.ca.water_flux),          'C0--', lw=1.5, alpha=0.5, label=r"$J^\mathrm{ca}_{w}$")
ax.plot(t_min, 1e5 * np.asarray(ss_state.ca.liquid_flux),         'C1--', lw=1.5, alpha=0.5, label=r"$J^\mathrm{ca}_{l}$")
ax.plot(t_min, 1e5 * np.asarray(ss_state.ca.membrane_water_flux), 'C2--', lw=1.5, alpha=0.5, label=r"$J^\mathrm{mb}_{w}$")
ax.set_ylabel(r"Water flux (µmol/cm$^2$)")
ax.set_xlabel("Time (min)")
ax.set_xlim(t_min[0],t_min[-1])
ax.legend(loc='upper left')
ax.grid(True)
_dual_legend(ax, [r"$J^\mathrm{ca}_{w}$", 
                  r"$J^\mathrm{ca}_{l}$", 
                  r"$J^\mathrm{mb}_{w}$"],
             ['C0', 'C1', 'C2'], fontsize=8)
_vline(ax)
fig.tight_layout()


# %%
# Computational performance
# -------------------------
#
# Wall-clock time to simulate the full ID-FAST cycle with each approach, and
# how many times faster than real time (xRT) each one runs.

real_time_s = conditions.duration
tr_speedup  = real_time_s / tr_wall_time
qss_speedup = real_time_s / qss_wall_time

fig, ax = plt.subplots(1, 1, figsize=(5, 3))
labels = ['Transient', 'QSS']
times  = [tr_wall_time, qss_wall_time]
speedups = [tr_speedup, qss_speedup]
bars = ax.bar(labels, times, color=['C0', 'C1'])
for bar, speedup in zip(bars, speedups):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{speedup:,.0f}x RT", ha='center', va='bottom', fontsize=9)
ax.set_ylabel("Computational time (s)")
ax.set_ylim([0,10])
ax.set_title(f"Simulated cycle duration: {conditions.duration / 60:.1f} min")
fig.tight_layout()


plt.show()
