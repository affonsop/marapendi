"""
******************
Polarization curve
******************

Assemble a PEM fuel cell from first principles and compute a steady-state
polarization curve with a single vectorised call.

``marapendi`` separates the *description* of the cell (a tree of dataclasses
holding geometry and material parameters) from the *physics* (model objects
that receive the cell and operating conditions at solve time).  This example
walks through both steps and plots the resulting V–i curve, power density,
high-frequency resistance, and individual loss contributions.
"""

# %%
# Cell assembly
# =============
#
# The cell component details is defined in :class:`~marapendi.components.cell.fuelcell.FuelCell`. 
# It contains two :class:`~marapendi.components.cell.cell.CellSide` objects and a 
# :class:`~marapendi.components.membrane.pem.PFSA` membrane. 
# When :class:`~marapendi.components.cell.fuelcell.FuelCell` is initialized, the cell :attr:`~marapendi.components.cell.fuelcell.FuelCell.area`
# and the lumped specific :attr:`~marapendi.components.cell.fuelcell.FuelCell.electric_resistance` are defined. 
#
# .. attention::
#
#    All inputs must be in SI units (e.g. ``20e-4`` m² for a 20 cm² cell).
#    Molar quantities use kmol instead of mol.
 
import numpy as np
import matplotlib.pyplot as plt
import marapendi as mrpd

cell = mrpd.FuelCell(
    area=25e-4,
    electric_resistance=10e-7 
)

# %%
# The anode and cathode each own a catalyst layer, a gas diffusion layer, and a
# flow channel. Those can be defined after the :class:`~marapendi.components.cell.fuelcell.FuelCell` creation. 
# 
# Each :class:`~marapendi.components.porous_layers.porous_layers.PorousLayer` requires a :attr:`~marapendi.components.porous_layers.porous_layers.PorousLayer.two_phase_transport_model`, defining a capillary
# pressure saturation relation, and methods to calculate liquid water transport. 
# Here we use the default :class:`~marapendi.models.darcy.DarcyTransportModel`.

liq = mrpd.DarcyTransportModel(J_function_exponent=0.4)


# %% 
# Flow channels
# ------------- 
#
# :class:`~marapendi.components.channel.flow_channels.FlowChannel` describes the 
# geometry of the flowfield, with ``width``, ``height``, and ``length`` for 
# one channel and ``n_parallel`` the number of parallel channels. 
# ``reactant`` identifies which gas species is consumed (O2 at the cathode,
# H2 at the anode for a PEM fuel cell).
for side in cell.sides:
    side.ch = mrpd.FlowChannel(
        width=0.85e-3,                    
        height=1e-3,                   
        length=0.49,                 
        n_parallel=3,              
        reactant="o2" if side is cell.ca else "h2",
        transport_resistance_model=mrpd.ChannelGasResistanceModel(sherwood=3.66, B_ch=1.2)
    )

# %%
# .. tip::
# 
#   The :attr:`~marapendi.components.cell.fuelcell.FuelCell.sides` property iterates over
#   ``cell.ca`` and ``cell.an``, which is convenient when both sides share the
#   same component geometry.

# %%
# Gas diffusion layers
# --------------------
#
# :class:`~marapendi.components.porous_layers.porous_layers.GasDiffusionLayer` is a
# :class:`~marapendi.components.porous_layers.porous_layers.PorousLayer` with defaults
# suitable for a carbon-fibre paper or woven GDL.
for side in cell.sides:
    side.gdl = mrpd.GasDiffusionLayer(
        thickness=117e-6 * 1.4,              
        porosity=0.65,                 
        tortuosity=1.55,                
        contact_angle=110.0,           
        absolute_permeability=3e-12,   
        thermal_conductivity=1.2,      
        two_phase_transport_model=liq,
        relative_permeability_exponent=3,
        volume_heat_capacity=1.58e6
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
        relative_permeability_exponent=3,
        volume_heat_capacity=1.98e6
    )
# %%
# .. attention::
#
#    A :class:`~marapendi.components.porous_layers.porous_layers.MicroPorousLayer` can also be defined using the 
#    :attr:`~marapendi.components.cell.cell.CellSide.mpl` attribute of :class:`~marapendi.components.cell.cell.CellSide.


# %%
# Thermal contact resistance
# --------------------------
# We also add a lumped specific thermal contact 
# resistance representing the sum of contact resistances for 
# each side. 
for side in cell.sides:
    side.thermal_contact_resistance = 1e-4 


# %%
# Cathode catalyst layer
# ----------------------
#
# :class:`~marapendi.components.porous_layers.catalyst_layers.PtCCatalystLayer` describes
# a carbon-supported platinum catalyst layer where ORR or HER take place.
#
# The electrochemical kinetics are encapsulated in
# :class:`~marapendi.thermo.electrochemistry.ElectrochemicalReaction`.
# 
# :class:`~marapendi.components.membrane.pem.PFSAIonomer` characterises the ionomer material 
# used in the catalyst layers and/or the membrane.
# Here we model Aquivion® (EW = 790 g/mol), a short-side-chain PFSA. For simplicity
# we consider the same ionomer is used for the membrane and the CL.  

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
    relative_permeability_exponent=3,
    volume_heat_capacity=1.56e6
)

# %%
# Anode catalyst layer
# --------------------
#
# The hydrogen oxidation reaction (HOR) is orders of magnitude faster than
# the ORR. In the current version, the activation overpotential of the anode is not
# considered for PEMFC, so that no kinetic sub-model is needed.

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
    volume_heat_capacity=1.56e6
)

# %%
# Membrane
# --------
#
# :class:`~marapendi.components.membrane.pem.PFSA` membrane combines an ionomer material
# (already defined above) with the membrane geometry. 

cell.membrane = mrpd.PFSA(ionomer=nafion, dry_thickness=15e-6)


# %%
# Operating conditions
# ====================
#
# :class:`~marapendi.simulation.conditions.CellConditions` bundles the
# electrical load, cell temperature and operating gas-supply conditions on each side.  
# All fields accept 1D numpy arrays, so the full polarization curve is obtained in a
# single vectorised call. 
#
# ``stoichiometry`` is computed here from a fixed volumetric flow rate
# (5 NL/min air, 2 NL/min H₂) so it varies with current, reproducing a
# constant-flow-rate test-bench protocol.

T = 273.15 + 71  # K  (71 °C)
i = np.linspace(1, 40000, 200)  # A/m²

conditions = mrpd.CellConditions(
    current_density=i,
    cell_temperature=T,
    ca=mrpd.SideConditions(
        inlet_temperature=T,
        outlet_pressure=1.4e5,
        dry_o2_mole_fraction=0.21,
        inlet_relative_humidity=0.265,
        stoichiometry = 1.6,
        minimum_current_density_for_stoich=0,
    ),
    an=mrpd.SideConditions(
        inlet_temperature=T,
        outlet_pressure=1.9e5,
        dry_h2_mole_fraction=1.0,
        inlet_relative_humidity=0.558,
        stoichiometry=1.4,
        minimum_current_density_for_stoich=0,
    ),
)

# %%
# .. attention::
#
#    When numpy arrays are used to describe operating conditions, they must have the same length
#    as the current density. Scalar values are broadcast. 

# %%
# Model definition and solution
# ============================= 
# 
# Here we define an :class:`~marapendi.models.base.explicit_steady_state.ExplicitSteadyStateModel` using default voltage, thermal, water balance and
# gas transport submodels. Those use default implementations available in `marapendi`, but 
# submodels can be easily changed by creating subclases derived from the original models. 

model = mrpd.ExplicitSteadyStateModel(
    voltage_model=mrpd.VoltageModel(),
    thermal_model=mrpd.ThermalModel(),
    water_balance_model=mrpd.WaterBalanceModel(),
    gas_transport_model=mrpd.GasTransportModel()
    )

# %%
# We need to initialize a state :class:`~marapendi.simulation.state.CellState` object,
# which is used to solve the `model`. All the results are stored in the 
# :class:`~marapendi.simulation.state.CellState` returned by :attr:`marapendi`.

state = model.set_initial_conditions(cell, conditions)
state = model.solve(cell, conditions, state)

# %%
# Results
# =======
# 
# Polarization curve and HFR
# --------------------------
# We can verify the results for the polarization curve simulation by accessing :attr:`~marapendi.simulation.state.CellState.cell_voltage` 
# and :attr:`~marapendi.simulation.state.CellState.hfr` attributes of :class:`~marapendi.simulation.state.CellState`.  

fig, ax = plt.subplots(1, 1, figsize=(4,3))
ax.plot(state.current_density * 1e-4, state.cell_voltage, "-", color="C0", label='$V_{cell}$')
ax.set_xlabel("Current density (A/cm$^2$)")
ax.set_ylabel("Cell voltage (V)")
ax.set_ylim(0.2, 1.0)
ax.set_xlim(0,4)
ax.grid(True)

ax2 = ax.twinx() 
ax2.plot(state.current_density * 1e-4, state.hfr * 1e7, "-", color="C1", label='HFR')
ax2.set_ylabel(r"HFR (m$\Omega$.cm$^2$)")
ax2.set_ylim(20, 120)
ax.legend(loc='upper right')
ax2.legend(loc='lower right')

fig.tight_layout()



# %%
# Internal variables
# ------------------
# Accessing different internal variables is also straightforward with the :class:`~marapendi.simulation.state.CellState` object. 

fig, ax = plt.subplots(1, 1, figsize=(4,3))
i_x = (2 * mrpd.FARADAY_CONSTANT) * state.membrane.h2_permeation_flux
ax.plot(state.current_density * 1e-4, i_x * 1e-4 * 1e3, "-", color="C0", label='$V_{cell}$')
ax.grid(True)
ax.set_ylabel("Crossover current density\n(mA/cm$^2$)")
ax.set_xlabel("Current density (A/cm$^2$)")
ax.set_xlim(0,4)
fig.tight_layout()


fig, ax = plt.subplots(1, 1, figsize=(4,3))
ax.plot(state.current_density * 1e-4, state.ca.cl.proton_resistance * 1e7, "-", color="C0", label='$V_{cell}$')
ax.grid(True)
ax.set_ylabel("Proton resistance in the CL\n(mA/cm$^2$)")
ax.set_xlabel("Current density (A/cm$^2$)")
ax.set_xlim(0,4)
fig.tight_layout()


fig, ax = plt.subplots(1, 1, figsize=(4,3))
ax.plot(state.current_density * 1e-4, state.ca.cl.liquid_saturation, color="C0", label='CL')
ax.plot(state.current_density * 1e-4, state.ca.mpl.liquid_saturation, color="C1", label='MPL')
ax.plot(state.current_density * 1e-4, state.ca.gdl.liquid_saturation, color="C2", label='GDL')
ax.grid(True)
ax.set_ylabel("Cathode water\nsaturation (n.d.)")
ax.set_xlabel("Current density (A/cm$^2$)")
ax.legend(loc='upper left')
ax.set_xlim(0,4)
fig.tight_layout()


fig, ax = plt.subplots(1, 1, figsize=(4,3))
plt.plot(state.current_density * 1e-4,state.ca.cl.ionomer_water_content, 'C0', label=r"$\lambda^{ion}_{CL,ca}$")
plt.plot(state.current_density * 1e-4,state.ca.membrane_interface_water_content, '--C0', label=r"$\lambda^{mb}_{ca}$")
plt.plot(state.current_density * 1e-4,state.membrane.water_content, 'C1', label=r"$\lambda^{mb}_{avg}$")
plt.plot(state.current_density * 1e-4,state.an.membrane_interface_water_content, '--C2', label=r"$\lambda^{mb}_{avg}$")
plt.plot(state.current_density * 1e-4,state.an.cl.ionomer_water_content, '-C2', label=r"$\lambda^{ion}_{CL,an}$")
ax.set_ylabel("Water content (n.d.)")
ax.set_xlabel("Current density (A/cm$^2$)")
ax.grid(True)
ax.legend(loc='lower left')
ax.set_xlim(0,4)
fig.tight_layout()


fig, ax = plt.subplots(1, 1, figsize=(4,3))
plt.plot(state.current_density * 1e-4,1e5 * state.ca.h2o_production, 'C0', label=r"$i/2F$")
plt.plot(state.current_density * 1e-4,1e5 * state.ca.water_flux, 'C1', label=r"$J^{ca}_{w}$")
plt.plot(state.current_density * 1e-4,1e5 * state.ca.liquid_flux, 'C2', label=r"$J^{ca}_{l}$")
plt.plot(state.current_density * 1e-4,1e5 * state.ca.membrane_water_flux, 'C3', label=r"$J^{mb}_{w}$")
ax.set_ylabel("Water flux (µmol/cm$^2$)")
ax.set_xlabel("Current density (A/cm$^2$)")
ax.legend(loc='upper left')
ax.grid(True)
ax.set_xlim(0,4)
fig.tight_layout()

fig, ax = plt.subplots(1, 1, figsize=(4,3))
plt.plot(state.current_density * 1e-4,state.ca.cl.local_o2_resistance, 'C0--', label='CL local')
plt.plot(state.current_density * 1e-4,state.ca.cl.gas_transport_resistance['o2'], 'C0', label='CL total')
plt.plot(state.current_density * 1e-4,state.ca.mpl.gas_transport_resistance['o2'], 'C1', label='MPL')
plt.plot(state.current_density * 1e-4,state.ca.gdl.gas_transport_resistance['o2'], 'C2', label='GDL')
ax.set_ylim([0,50])
ax.set_ylabel("Cathode O$_2$\ntransport resistance (s/m)")
ax.set_xlabel("Current density (A/cm$^2$)")
ax.legend(loc='upper left')
ax.grid(True)
ax.set_xlim(0,4)
fig.tight_layout()

plt.show()