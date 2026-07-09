"""
Multi-condition polarization curves
=====================================

Simulate polarization curves across a range of operating conditions
(temperature, pressure, inlet relative humidity) to understand the
sensitivity of cell performance to each parameter.
"""

# %%
# Cell assembly
# -------------
#
# Identical to :ref:`sphx_glr_auto_examples_plot_01_polarization_curve.py`.
import marapendi as mrpd
import matplotlib.pyplot as plt
import numpy as np 

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
# We define the model and a helper function to generate conditions for each case

i = np.linspace(500, 22000, 50)
model = mrpd.ExplicitSteadyStateModel()

def _solve(T=353.15, p=1.5e5, rh=0.5):
    cond = mrpd.CellConditions(
        current_density=i, cell_temperature=T,
        ca=mrpd.SideConditions(
            inlet_temperature=T, outlet_pressure=p,
            dry_o2_mole_fraction=0.21, inlet_relative_humidity=rh, stoichiometry=2.0,
        ),
        an=mrpd.SideConditions(
            inlet_temperature=T, outlet_pressure=p,
            dry_h2_mole_fraction=1.0, inlet_relative_humidity=rh, stoichiometry=1.5,
        ),
    )
    st = model.solve(cell, cond, model.set_initial_conditions(cell, cond))
    st._hfr = model.voltage_model.high_frequency_resistance(cell, st)
    return st

# %%
# Effect of temperature
# ----------------------

temperatures = [333.15, 353.15, 368.15]   # 60, 80, 95 °C
colors = ["C0", "C1", "C2"]
i_cm2 = i * 1e-4

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for T, c in zip(temperatures, colors):
    st = _solve(T=T, p=1.5e5, rh=0.5)
    lbl = f"{T - 273.15:.0f} °C"
    axes[0].plot(i_cm2, st.cell_voltage, "-", color=c, label=lbl)
    axes[1].plot(i_cm2, st._hfr * 1e7, "-", color=c, label=lbl)

axes[0].set_xlabel("Current density (A cm⁻²)")
axes[0].set_ylabel("Cell voltage (V)")
axes[0].legend()
axes[0].grid(True)

axes[1].set_xlabel("Current density (A cm⁻²)")
axes[1].set_ylabel("HFR (mΩ cm²)")
axes[1].legend()
axes[1].grid(True)

fig.tight_layout()

# %%
# Effect of pressure
# -------------------

pressures = [1.0e5, 1.5e5, 2.5e5]   # 1.0, 1.5, 2.5 bara
p_labels = ["1.0 bar", "1.5 bar", "2.5 bar"]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for p, c, lbl in zip(pressures, colors, p_labels):
    st = _solve(T=353.15, p=p, rh=0.5)
    axes[0].plot(i_cm2, st.cell_voltage, "-", color=c, label=lbl)
    axes[1].plot(i_cm2, st._hfr * 1e7, "-", color=c, label=lbl)

axes[0].set_xlabel("Current density (A cm⁻²)")
axes[0].set_ylabel("Cell voltage (V)")
axes[0].legend()
axes[0].grid(True)

axes[1].set_xlabel("Current density (A cm⁻²)")
axes[1].set_ylabel("HFR (mΩ cm²)")
axes[1].legend()
axes[1].grid(True)

fig.tight_layout()

# %%
# Effect of inlet relative humidity
# ----------------------------------

rh_values = [0.2, 0.5, 0.9]
rh_labels = ["RH = 20 %", "RH = 50 %", "RH = 90 %"]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for rh, c, lbl in zip(rh_values, colors, rh_labels):
    st = _solve(T=353.15, p=1.5e5, rh=rh)
    axes[0].plot(i_cm2, st.cell_voltage, "-", color=c, label=lbl)
    axes[1].plot(i_cm2, st._hfr * 1e7, "-", color=c, label=lbl)

axes[0].set_xlabel("Current density (A cm⁻²)")
axes[0].set_ylabel("Cell voltage (V)")
axes[0].legend()
axes[0].grid(True)

axes[1].set_xlabel("Current density (A cm⁻²)")
axes[1].set_ylabel("HFR (mΩ cm²)")
axes[1].legend()
axes[1].grid(True)

fig.tight_layout()
plt.show()