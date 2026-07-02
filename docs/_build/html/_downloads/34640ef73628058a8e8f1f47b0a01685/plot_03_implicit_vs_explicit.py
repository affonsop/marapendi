"""
Implicit vs explicit steady-state model
========================================

Both :class:`~marapendi.cell.explicit_steady_state.ExplicitSteadyStateModel`
and :class:`~marapendi.cell.implicit_steady_state.ImplicitSteadyStateModel`
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

The implicit model is more accurate at high current densities where the
thermal feedback is significant; the explicit model is faster.
"""

# %%
# Cell assembly
# -------------

import numpy as np
import matplotlib.pyplot as plt
import marapendi as mrpd
from marapendi.cell.implicit_steady_state import ImplicitSteadyStateModel

liq = mrpd.DarcyTransportModel(J_function_exponent=2)
ionomer = mrpd.PFSAIonomer(equivalent_weight=1100, dry_density=1980)

cell = mrpd.FuelCell(
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

# %%
# Solve with both models
# ----------------------

T = 353.15
i_arr = np.linspace(500, 22000, 50)

conditions = mrpd.CellConditions(
    current_density=i_arr,
    cell_temperature=T,
    ca=mrpd.SideConditions(
        inlet_temperature=T, outlet_pressure=1.5e5,
        dry_o2_mole_fraction=0.21, inlet_relative_humidity=0.5, stoichiometry=2.0,
    ),
    an=mrpd.SideConditions(
        inlet_temperature=T, outlet_pressure=1.5e5,
        dry_h2_mole_fraction=1.0, inlet_relative_humidity=0.5, stoichiometry=1.5,
    ),
)

exp_model = mrpd.ExplicitSteadyStateModel()
imp_model = ImplicitSteadyStateModel()

state_exp = exp_model.solve(cell, conditions, exp_model.set_initial_conditions(cell, conditions))
state_imp = imp_model.solve(cell, conditions, imp_model.set_initial_conditions(cell, conditions))

i_cm2 = i_arr * 1e-4

# %%
# Polarization curve comparison
# ------------------------------

fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))

axes[0].plot(i_cm2, state_exp.cell_voltage, "o-", ms=4, label="Explicit")
axes[0].plot(i_cm2, state_imp.cell_voltage, "s--", ms=4, label="Implicit")
axes[0].set_xlabel("Current density (A cm⁻²)")
axes[0].set_ylabel("Cell voltage (V)")
axes[0].set_title("Polarization curve")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(i_cm2, state_exp.mea_temperature - 273.15, "o-", ms=4, label="Explicit")
axes[1].plot(i_cm2, state_imp.mea_temperature - 273.15, "s--", ms=4, label="Implicit")
axes[1].axhline(T - 273.15, color="k", lw=0.8, ls=":", label="Stack T")
axes[1].set_xlabel("Current density (A cm⁻²)")
axes[1].set_ylabel("T_MEA (°C)")
axes[1].set_title("MEA temperature")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

hfr_exp = exp_model.voltage_model.high_frequency_resistance(cell, state_exp)
hfr_imp = imp_model.voltage_model.high_frequency_resistance(cell, state_imp)
axes[2].plot(i_cm2, hfr_exp * 1e4, "o-", ms=4, label="Explicit")
axes[2].plot(i_cm2, hfr_imp * 1e4, "s--", ms=4, label="Implicit")
axes[2].set_xlabel("Current density (A cm⁻²)")
axes[2].set_ylabel("HFR (mΩ cm²)")
axes[2].set_title("High-frequency resistance")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

fig.tight_layout()

# %%
# Voltage difference
# ------------------
#
# The voltage gap is small at moderate current but grows at high current
# where the MEA temperature rise becomes significant.

fig, ax = plt.subplots(figsize=(6, 3.5))
dV_mV = (state_imp.cell_voltage - state_exp.cell_voltage) * 1e3
ax.plot(i_cm2, dV_mV, "o-", ms=4, color="C3")
ax.axhline(0, color="k", lw=0.8, ls="--")
ax.set_xlabel("Current density (A cm⁻²)")
ax.set_ylabel("ΔV = V_implicit − V_explicit (mV)")
ax.set_title("Voltage difference between models")
ax.grid(True, alpha=0.3)
fig.tight_layout()
