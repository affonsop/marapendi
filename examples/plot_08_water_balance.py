"""
Water balance vs. cell temperature
===================================

:class:`~marapendi.simulation.state.GasFlowState` bundles the per-species
molar flow rates (+ liquid water) at one point of a cell side, and converts
to/from :class:`~marapendi.simulation.conditions.SideConditions`. When
:class:`~marapendi.models.base.explicit_steady_state.ExplicitSteadyStateModel`
solves a single operating point, it populates
``state.ca.inlet_gas_flow_state`` / ``outlet_gas_flow_state`` (and the same
for ``state.an``) automatically: the inlet flow implied by the stoichiometry
+ dry composition + RH inputs, and the corresponding outlet flow from a mass
balance (reactant consumed, product water added as vapor and/or liquid).

This example sweeps the cell temperature and plots the resulting water
balance — theoretical water production against the modelled
cathode/anode inlet and outlet water mass flow rates — reproducing the kind
of bar chart typically used to check a test bench's water balance closes
(produced ≈ outlet − inlet, summed over both sides).
"""

# %%
# Cell assembly
# =============
#
# Identical to :ref:`sphx_glr_auto_examples_plot_01_polarization_curve.py`.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import marapendi as mrpd
from marapendi.models.thermo.gas import index_h2ov
from marapendi.models.thermo.constants import WATER_MOLECULAR_WEIGHT, FARADAY_CONSTANT

cell = mrpd.FuelCell(
    area=25e-4,
    electric_resistance=10e-7
)

liq = mrpd.DarcyTransportModel(J_function_exponent=0.4)

for side in cell.sides:
    side.ch = mrpd.FlowChannel(
        width=0.85e-3, height=1e-3, length=0.49, n_parallel=3,
        reactant="o2" if side is cell.ca else "h2",
        transport_resistance_model=mrpd.ChannelGasResistanceModel(sherwood=3.66, B_ch=1.2)
    )
    side.gdl = mrpd.GasDiffusionLayer(
        thickness=117e-6 * 1.4, porosity=0.65, tortuosity=1.55,
        contact_angle=110.0, absolute_permeability=3e-12,
        thermal_conductivity=1.2, two_phase_transport_model=liq,
        relative_permeability_exponent=3, volume_heat_capacity=1.58e6
    )
    side.mpl = mrpd.MicroPorousLayer(
        thickness=22e-6, porosity=0.4, tortuosity=3, pore_diameter=500e-9,
        contact_angle=130.0, absolute_permeability=1e-12,
        thermal_conductivity=0.144, two_phase_transport_model=liq,
        relative_permeability_exponent=3, volume_heat_capacity=1.98e6
    )
    side.thermal_contact_resistance = 1e-4

orr = mrpd.ElectrochemicalReaction(
    reference_exchange_current_density=1e-3, reaction_order=0.8,
    activation_energy=42e6, reference_activity=1e5,
    reference_temperature=353.15, number_of_electrons=2,
    charge_transfer_coeff=0.5,
)
nafion = mrpd.PFSAIonomer(
    equivalent_weight=1100., dry_density=1980,
    reference_conductivity=50., residual_conductivity=0.3,
    conductivity_fv_threshold=0.04, conductivity_exp=1.5,
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
    ecsa=40e3, platinum_loading=0.5e-2, catalyst_platinum_weight_percent=0.5,
    ionomer_to_carbon_ratio=0.81, ionomer=nafion, reaction=orr,
    thickness=10e-6, tortuosity=3, thermal_conductivity=0.18,
    pore_diameter=140e-9, carbon_agglomerate_radius=25e-9,
    absolute_permeability=2e-13, contact_angle=100.0,
    two_phase_transport_model=liq, relative_permeability_exponent=3,
    volume_heat_capacity=1.56e6
)
cell.an.cl = mrpd.PtCCatalystLayer(
    platinum_loading=0.1e-2, ionomer=nafion, thickness=7e-6,
    ionomer_to_carbon_ratio=0.57, catalyst_platinum_weight_percent=0.2,
    thermal_conductivity=0.18, pore_diameter=140e-9,
    absolute_permeability=1e-13, contact_angle=100.0,
    two_phase_transport_model=liq, volume_heat_capacity=1.56e6
)
cell.membrane = mrpd.PFSA(ionomer=nafion, dry_thickness=15e-6)

# %%
# Operating conditions
# =====================
#
# One current density, swept over cell temperature. ``inlet_relative_humidity``
# and ``stoichiometry`` are held fixed on each side; only ``cell_temperature``
# (and, for consistency, ``inlet_temperature``) changes across the sweep.

I_OP = 10000.  # A/m2
T_CELL_C = np.array([70., 75., 80., 85.])  # deg C
T_CELL = T_CELL_C + 273.15

model = mrpd.ExplicitSteadyStateModel()

water_produced = np.zeros_like(T_CELL)
water_inlet_ca = np.zeros_like(T_CELL)
water_inlet_an = np.zeros_like(T_CELL)
water_outlet_ca = np.zeros_like(T_CELL)
water_outlet_an = np.zeros_like(T_CELL)

for k, T in enumerate(T_CELL):
    conditions = mrpd.CellConditions(
        current_density=I_OP,
        cell_temperature=T,
        ca=mrpd.SideConditions(
            inlet_temperature=T, outlet_pressure=1.4e5,
            dry_o2_mole_fraction=0.21, inlet_relative_humidity=0.5,
            stoichiometry=2.0,
        ),
        an=mrpd.SideConditions(
            inlet_temperature=T, outlet_pressure=1.9e5,
            dry_o2_mole_fraction=0., dry_h2_mole_fraction=1.0,
            inlet_relative_humidity=0.5, stoichiometry=1.5,
        ),
    )
    state = model.set_initial_conditions(cell, conditions)
    state = model.solve(cell, conditions, state)

    # kmol/s -> g/min
    to_g_per_min = WATER_MOLECULAR_WEIGHT * 1000. * 60.

    water_produced[k] = I_OP * cell.area / (2 * FARADAY_CONSTANT) * to_g_per_min
    water_inlet_ca[k] = state.ca.inlet_gas_flow_state.gas_species_molar_flow_rates[index_h2ov] * to_g_per_min
    water_inlet_an[k] = state.an.inlet_gas_flow_state.gas_species_molar_flow_rates[index_h2ov] * to_g_per_min
    water_outlet_ca[k] = (
        state.ca.outlet_gas_flow_state.gas_species_molar_flow_rates[index_h2ov]
        + state.ca.outlet_gas_flow_state.liquid_molar_flow_rate
    ) * to_g_per_min
    water_outlet_an[k] = (
        state.an.outlet_gas_flow_state.gas_species_molar_flow_rates[index_h2ov]
        + state.an.outlet_gas_flow_state.liquid_molar_flow_rate
    ) * to_g_per_min

# %%
# .. attention::
#
#    Water production, split between the two sides via the mass balance in
#    :meth:`~marapendi.simulation.state.GasFlowState.consume`, always sums to
#    the theoretical value: ``(outlet_ca + outlet_an) - (inlet_ca + inlet_an)
#    == water_produced`` at every temperature (up to floating-point
#    precision) — the water balance closes by construction.

# %%
# Water balance bar chart
# ========================
#
# One stacked column per temperature for the inlet side (anode inlet +
# cathode inlet + theoretical production) and one for the outlet side
# (anode outlet + cathode outlet) — since the balance closes exactly, the
# two columns in each pair have the same total height.

def _lighten(color, fraction=0.5):
    """Blend *color* with white by *fraction* (0 = unchanged, 1 = white) --
    a lighter solid color, as opposed to alpha transparency which would let
    the background grid show through the bar."""
    r, g, b = mcolors.to_rgb(color)
    return (1 - fraction) * r + fraction, (1 - fraction) * g + fraction, (1 - fraction) * b + fraction

labels = [f"{t:g}" for t in T_CELL_C]
x = np.arange(len(T_CELL))
width = 0.35

fig, ax = plt.subplots(1, 1, figsize=(5, 4))
ax.set_axisbelow(True)

bottom = np.zeros_like(T_CELL)
for name, values, color in [
    ("Anode inlet", water_inlet_an, "C1"),
    ("Cathode inlet", water_inlet_ca, "C0"),
    ("Produced (theoretical)", water_produced, "C2"),
]:
    bars = ax.bar(x - width / 2, values, width, bottom=bottom, label=name, color=color)
    ax.bar_label(bars, fmt="%.2f", label_type="center", fontsize=8)
    bottom = bottom + values

bottom = np.zeros_like(T_CELL)
for name, values, color in [
    ("Anode outlet", water_outlet_an, "C1"),
    ("Cathode outlet", water_outlet_ca, "C0"),
]:
    bars = ax.bar(x + width / 2, values, width, bottom=bottom, label=name, color=_lighten(color))
    ax.bar_label(bars, fmt="%.2f", label_type="center", fontsize=8)
    bottom = bottom + values

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_xlabel(r"$T_{cell}$ (°C)")
ax.set_ylabel("Water mass flow rate (g/min)")
ax.legend(loc="upper left", fontsize=8)
ax.grid(True, axis='y')
fig.tight_layout()

plt.show()
