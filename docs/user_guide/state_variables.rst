Inspecting state variables
===========================

After a solve, every physical quantity **marapendi** tracked along the way
is available on the returned :class:`~marapendi.simulation.state.CellState`
— nothing is discarded once the solver has moved on. This page is an
exhaustive reference of what is available on each state object, so you know
what to look for without reading the solver source.

.. code-block:: python

   model = mrpd.ExplicitSteadyStateModel()
   state = model.set_initial_conditions(cell, conditions)
   state = model.solve(cell, conditions, state)

   state.cell_voltage                     # CellState
   state.ca.cl.gas.relative_humidity      # CellSideState -> CatalystLayerState -> GasState
   state.membrane.water_content_profile   # MembraneState

.. seealso::

   :doc:`/architecture` for how ``simulation.state`` mirrors the shape of
   ``components``, and the full autodoc (every method included) at
   :class:`~marapendi.simulation.state.CellState`,
   :class:`~marapendi.simulation.state.CellSideState`,
   :class:`~marapendi.simulation.state.CatalystLayerState`,
   :class:`~marapendi.simulation.state.LayerState`,
   :class:`~marapendi.simulation.state.MembraneState`,
   :class:`~marapendi.simulation.state.FlowChannelState`,
   :class:`~marapendi.simulation.state.GasFlowState` and
   :class:`~marapendi.simulation.state.GasState`.

CellState
---------

Returned by ``model.solve(...)`` — the full state of the cell at one
operating point.

.. code-block:: python

   state.ca, state.an                     # CellSideState, cathode/anode
   state.membrane                         # MembraneState

   state.current_density                  # Current density (A/m^2)
   state.temperature                      # Cell (stack) temperature (K)
   state.cell_voltage                     # Cell voltage (V)
   state.E_rev                            # Reversible (Nernst) voltage (V)
   state.eta_act                          # Activation overpotential (V)
   state.eta_ohm                          # Ohmic overpotential (V)

   state.thermal_resistance               # MEA-to-coolant thermal resistance (K m^2/W)
   state.mea_temperature                  # MEA temperature (K)
   state.mea_temperature_increase         # MEA temperature rise above `temperature` (K)
   state.mea_water_molar_volume           # Molar volume of water at mea_temperature (m^3/kmol)
   state.heat_release                     # Total heat release rate (W/m^2)

   state.crossover_current                # Equivalent current from H2 crossover (A/m^2)
   state.hfr                              # High-frequency resistance (Ohm m^2), set by evaluate()
   state.ode_solution                     # Raw scipy OdeResult, set by TransientModel.solve() (else None)

   state.sides                            # (ca, an), for `for side in state.sides: ...`
   state.side_layers                      # All layer states across both sides, GDL-to-CL order
   state.layers                           # side_layers plus the membrane

CellSideState
-------------

State of one side (cathode or anode), held at ``state.ca``/``state.an``.

.. code-block:: python

   side = state.ca                        # or state.an

   side.cl                                # CatalystLayerState
   side.gdl, side.mpl                     # LayerState (mpl is None if the cell has no MPL)
   side.ch                                # FlowChannelState

   side.h2ov_transport_resistance         # Channel-to-CL water vapor transport resistance (s/m)
   side.reactant_transport_resistance     # Channel-to-CL reactant transport resistance (s/m)
   side.reactant_consumption              # Reactant consumption rate (kmol/(m^2 s))
   side.gas_transport_resistance          # dict of per-species channel-to-CL resistances (s/m)

   side.h2o_production                    # Water production rate at the CL (kmol/(m^2 s))
   side.s_relax                           # Relaxation liquid saturation (water-balance model)
   side.membrane_water_flux               # Water flux to/from the membrane at this side (kmol/(m^2 s))
   side.water_flux, side.liquid_flux, side.vapor_flux  # Total/liquid/vapor water fluxes (kmol/(m^2 s))
   side.gas_flux                          # Net gas-phase molar flux (kmol/(m^2 s))

   side.inlet_gas_flow_state              # GasFlowState, set by set_gas_flow_states
   side.outlet_gas_flow_state             # GasFlowState, set by set_gas_flow_states

   side.rh_at_cl_without_crossover        # RH at the CL before crossover correction
   side.estimated_water_content           # Membrane water content at this side's interface
   side.estimated_water_content_derivative  # d(estimated_water_content)/d(RH or lambda)
   side.liquid_eq_water_content           # Liquid-equilibrium water content at this side
   side.vapor_eq_water_content            # Vapor-equilibrium water content at this side

   side.is_liquid_equilibrated            # Liquid- (s > 0) vs. vapor-equilibrated interface
   side.alpha                             # Non-dimensional water-transport parameter (Ferrara et al. 2018)
   side.peclet_over_modified_biot         # Non-dimensional water-transport parameter
   side.biot_number                       # Non-dimensional water-transport parameter

   side.porous_layers                     # [gdl, mpl, cl], GDL-to-CL order, excluding any None
   side.layers                            # [ch, gdl, mpl, cl], channel-to-CL order, excluding any None

LayerState
----------

State of a single porous layer (GDL or MPL); also the base class of
:class:`~marapendi.simulation.state.CatalystLayerState`.

.. code-block:: python

   layer = state.ca.gdl                   # or .mpl, .cl (any LayerState)

   layer.gas                              # GasState of this layer
   layer.temperature, layer.pressure      # Pass-through onto gas.temperature/gas.pressure (K, Pa)
   layer.RT                               # Pass-through onto gas.RT
   layer.saturation_pressure              # Pass-through onto gas.saturation_pressure
   layer.diffusion_temp_and_pressure_correction  # Pass-through onto the same gas property

   layer.liquid_saturation                # Liquid water saturation (0 to 1)
   layer.non_wetting_saturation           # Saturation of the non-wetting phase
   layer.capillary_pressure               # Capillary pressure (Pa)
   layer.breakthrough_pressure            # Capillary breakthrough pressure at this temperature (Pa)
   layer.saturation_flow_resistance       # Two-phase flow resistance at the current saturation (s/m)
   layer.non_wetting_flux                 # Non-wetting phase (liquid) flux through the layer (kmol/(m^2 s))
   layer.downstream_saturation            # Liquid saturation at the downstream face
   layer.upstream_saturation              # Liquid saturation at the upstream face
   layer.downstream_capillary_pressure    # Capillary pressure at the downstream face (Pa)
   layer.electrolyte_saturation           # Electrolyte-phase saturation (electrolyte-flooded layers)
   layer.gas_transport_resistance         # dict of per-species diffusion resistances (s/m)

CatalystLayerState
------------------

Extends :class:`~marapendi.simulation.state.LayerState` with catalyst-layer-specific state.

.. code-block:: python

   cl = state.ca.cl                       # CatalystLayerState

   cl.ionomer_water_content               # Ionomer water content lambda at the CL (mol H2O / mol SO3-)
   cl.overpotential                       # Local activation overpotential (V)
   cl.water_film_thickness                # Liquid water film thickness on the catalyst surface (m)
   cl.proton_resistance                   # Ionomer proton transport resistance within the CL (Ohm m^2)
   cl.theta_catalyst                      # Fractional catalyst surface coverage (e.g. surface oxide)
   cl.local_o2_resistance                 # Local (ionomer film) O2 transport resistance (s/m)
   cl.eq_water_content                    # Equilibrium water content at the CL/membrane interface
   cl.membrane_interface_water_content    # Water content at the CL side of the membrane interface

MembraneState
-------------

State of the membrane, held at ``state.membrane``.

.. code-block:: python

   memb = state.membrane                  # MembraneState

   memb.temperature                       # Membrane temperature (K)
   memb.saturation_pressure               # Saturation pressure of water at `temperature` (Pa)
   memb.density                           # Dry membrane density (kg/m^3)
   memb.water_content                     # Mean (through-plane-averaged) water content lambda (mol/mol)
   memb.water_flux                        # Net water flux through the membrane (kmol/(m^2 s))
   memb.h2_permeation_flux                # H2 crossover flux through the membrane (kmol/(m^2 s))
   memb.proton_resistance                 # Through-plane proton resistance (Ohm m^2)

   memb.eod_speed                         # Electro-osmotic drag speed (m/s)
   memb.absorption_coefficient            # Water absorption/desorption coefficient at the surface
   memb.water_diffusivity                 # Water self-diffusion coefficient in the membrane (m^2/s)
   memb.water_diffusion_resistance        # Through-plane water diffusion resistance (s/m)
   memb.vapor_equilibrium_saturation_water_content  # Where vapor/liquid-equilibrium isotherms meet

   memb.peclet_number                     # Non-dimensional Peclet number (Ferrara et al. 2018)
   memb.ePe, memb.ePexi, memb.xi          # Non-dimensional profile intermediates/mesh coordinate

   memb.water_content_profile             # Through-plane water content profile lambda(xi)
   memb.water_content_derivative_profile  # d(lambda)/d(xi) profile, same mesh
   memb.diffusion_flux                    # Diffusive internal water flux profile (kmol/(m^2 s))
   memb.eod_flux                          # Electro-osmotic-drag internal water flux profile
   memb.water_net_flux                    # Net internal water flux profile

FlowChannelState
----------------

State of a flow channel (cathode or anode), held at ``state.ca.ch``/``state.an.ch``.

.. code-block:: python

   ch = state.ca.ch                       # or state.an.ch

   ch.gas                                 # GasState of the channel
   ch.temperature, ch.pressure            # Pass-through onto gas.temperature/gas.pressure (K, Pa)
   ch.RT                                  # Pass-through onto gas.RT
   ch.saturation_pressure                 # Pass-through onto gas.saturation_pressure
   ch.diffusion_temp_and_pressure_correction  # Pass-through onto the same gas property

   ch.inlet_gas_flow_rate                 # Inlet volumetric gas flow rate (m^3/s)
   ch.inlet_liquid_flow_rate              # Inlet volumetric liquid water flow rate (m^3/s)
   ch.inlet_liquid_saturation             # Liquid saturation of the inlet flow, if any
   ch.inlet_stoichiometry                 # Reactant stoichiometry actually delivered at the inlet
   ch.gas_transport_resistance            # dict of per-species channel gas-transport resistances (s/m)

GasFlowState
------------

Gas + liquid flow state at one point (inlet or outlet) of a cell side — a
mass-flow-rate representation, as opposed to the intensive
:class:`~marapendi.simulation.state.GasState` composition.

.. code-block:: python

   flow = state.ca.inlet_gas_flow_state   # GasFlowState

   flow.temperature                       # Flow temperature (K)
   flow.pressure                          # Flow pressure (Pa)
   flow.gas_species_molar_flow_rates      # Per-species molar flow rate (O2, N2, H2, H2O), shape (4,) (kmol/s)
   flow.liquid_molar_flow_rate            # Molar flow rate of liquid water (kmol/s)

   flow.RT                                # GAS_CONSTANT * temperature
   flow.gas_molar_flow_rate               # Total gas-phase molar flow rate (kmol/s)
   flow.gas                               # GasState synthesized on every access

GasState
--------

Composition and thermodynamic state of a gas mixture, held at every
``.gas`` attribute above (and directly for :class:`~marapendi.simulation.state.GasFlowState`).

.. code-block:: python

   gas = state.ca.cl.gas                  # GasState, any .gas attribute above

   gas.X                                  # Mole fractions (O2, N2, H2, H2O), shape (4,)
   gas.temperature                        # Gas temperature (K)
   gas.pressure                           # Gas pressure (Pa)

   gas.RT                                 # GAS_CONSTANT * temperature
   gas.saturation_pressure                # Saturation pressure of water at `temperature` (Pa), cached
   gas.diffusion_temp_and_pressure_correction  # Fick's law adjustment T^1.5 / P, cached

   gas.vapor_pressure                     # Partial pressure of water vapor (Pa)
   gas.vapor_concentration                # Concentration of water vapor (kmol/m^3)
   gas.saturation_concentration           # Saturation concentration of water vapor (kmol/m^3)
   gas.relative_humidity                  # Relative humidity (0 to 1)
   gas.mixture_molecular_weight           # Mean molecular weight of the gas mixture (kg/kmol)
   gas.concentration                      # Total molar concentration of the gas mixture (kmol/m^3)
   gas.density                            # Mass density of the gas mixture (kg/m^3)
   gas.mixture_kinematic_viscosity        # Kinematic viscosity, mole-weighted average (m^2/s)

   gas.species_mole_fraction('h2o')       # Mole fraction of a species ('o2', 'n2', 'h2', 'h2o')
   gas.species_partial_pressure('h2o')    # Partial pressure of a species (Pa)
   gas.species_concentration('h2o')       # Concentration of a species (kmol/m^3)
   gas.species_kinematic_viscosity('h2o')  # Kinematic viscosity of the pure species (m^2/s)
   gas.species_diffusion_coefficient('h2o')  # Binary diffusion coefficient in the mixture (m^2/s)
