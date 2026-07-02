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

import numpy as np
import matplotlib.pyplot as plt
import marapendi as mrpd

liq = mrpd.DarcyTransportModel(J_function_exponent=2)
ionomer = mrpd.PFSAIonomer(equivalent_weight=1100, dry_density=1980)

def _make_cell():
    return mrpd.FuelCell(
        area=25e-4,
        electrical_resistance=30e-7,
        ca=mrpd.FuelCellSide(
            cl=mrpd.PtCCatalystLayer(
                ecsa=70e3, platinum_loading=0.4e-2, ionomer=ionomer,
                reaction=mrpd.ElectrochemicalReaction(
                    reference_exchange_current_density=2.5e-4,
                    reaction_order=0.54, activation_energy=67e6,
                    reference_activity=1e5, reference_temperature=353.15,
                    number_of_electrons=2, charge_transfer_coeff=0.5,
                ),
                thickness=10e-6, thermal_conductivity=0.22,
                pore_diameter=40e-9, absolute_permeability=1e-13, contact_angle=97.,
                two_phase_transport_model=liq,
            ),
            gdl=mrpd.GasDiffusionLayer(
                thickness=200e-6, porosity=0.6, contact_angle=120.,
                effective_gas_diffusion_ratio=0.3, absolute_permeability=1e-12,
                thermal_conductivity=0.5, two_phase_transport_model=liq,
            ),
            ch=mrpd.FlowChannel(
                width=1e-3, height=1e-3, length=0.1, n_parallel=20, reactant="o2"
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
                width=1e-3, height=1e-3, length=0.1, n_parallel=20, reactant="h2"
            ),
            has_mpl=False, thermal_contact_resistance=4e-4,
        ),
        membrane=mrpd.PFSA(ionomer=ionomer, dry_thickness=25e-6),
    )


i_arr = np.linspace(500, 22000, 50)
model = mrpd.ExplicitSteadyStateModel()

def _solve(T, p, rh):
    cell = _make_cell()
    cond = mrpd.CellConditions(
        current_density=i_arr, cell_temperature=T,
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
i_cm2 = i_arr * 1e-4

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for T, c in zip(temperatures, colors):
    st = _solve(T=T, p=1.5e5, rh=0.5)
    lbl = f"{T - 273.15:.0f} °C"
    axes[0].plot(i_cm2, st.cell_voltage, "-", color=c, label=lbl)
    axes[1].plot(i_cm2, st._hfr * 1e4, "-", color=c, label=lbl)

axes[0].set_xlabel("Current density (A cm⁻²)")
axes[0].set_ylabel("Cell voltage (V)")
axes[0].set_title("Effect of temperature on V–i")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel("Current density (A cm⁻²)")
axes[1].set_ylabel("HFR (mΩ cm²)")
axes[1].set_title("Effect of temperature on HFR")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

fig.tight_layout()

# %%
# Effect of pressure
# -------------------

pressures = [1.0e5, 1.5e5, 2.5e5]   # 1.0, 1.5, 2.5 bara
p_labels = ["1.0 bar", "1.5 bar", "2.5 bar"]

fig, ax = plt.subplots(figsize=(6, 4))
for p, c, lbl in zip(pressures, colors, p_labels):
    st = _solve(T=353.15, p=p, rh=0.5)
    ax.plot(i_cm2, st.cell_voltage, "-", color=c, label=lbl)

ax.set_xlabel("Current density (A cm⁻²)")
ax.set_ylabel("Cell voltage (V)")
ax.set_title("Effect of outlet pressure on V–i")
ax.legend()
ax.grid(True, alpha=0.3)
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
    axes[1].plot(i_cm2, st.membrane.water_content, "-", color=c, label=lbl)

axes[0].set_xlabel("Current density (A cm⁻²)")
axes[0].set_ylabel("Cell voltage (V)")
axes[0].set_title("Effect of humidity on V–i")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel("Current density (A cm⁻²)")
axes[1].set_ylabel("Mean λ_membrane (mol/mol)")
axes[1].set_title("Membrane water content")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

fig.tight_layout()
