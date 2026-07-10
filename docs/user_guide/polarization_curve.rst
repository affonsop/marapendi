Steady-state polarization curve
================================

This guide covers the full workflow for a single steady-state simulation:
assembling the cell, defining operating conditions, running the solver, and
inspecting every variable in the output state.

.. seealso::

   The physics implemented by each model referenced here is documented in
   :doc:`/science/index` — in particular :doc:`/science/water_balance` (membrane
   water transport), :doc:`/science/membrane_correlations` (ionomer
   correlations), :doc:`/science/two_phase_flow` and :doc:`/science/gas_transport`
   (liquid water and species transport), :doc:`/science/catalyst_layer`, and
   :doc:`/science/cell_voltage`.
   :doc:`/auto_examples/plot_01_polarization_curve` is the runnable version of
   the cell assembled below; :doc:`/auto_examples/plot_07_pwl_membrane`
   compares the two water-balance models mentioned at the end of this page.

Cell assembly
-------------

A :class:`~marapendi.cell.fuelcell.FuelCell` is a pure component tree — it
holds geometry and material parameters only; all physics lives in the model
objects. Assemble the cell once and reuse it across many conditions.

.. code-block:: python

    import numpy as np
    import marapendi as mrpd

    liq     = mrpd.DarcyTransportModel(J_function_exponent=2)
    ionomer = mrpd.PFSAIonomer(equivalent_weight=1100, dry_density=1980)

    cell = mrpd.FuelCell(
        area=25e-4,                   # m²
        electric_resistance=30e-7,  # Ω m²
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
                pore_diameter=40e-9, absolute_permeability=1e-13,
                contact_angle=97., two_phase_transport_model=liq,
            ),
            gdl=mrpd.GasDiffusionLayer(
                thickness=200e-6, porosity=0.6, contact_angle=120.,
                effective_gas_diffusion_ratio=0.3, absolute_permeability=1e-12,
                thermal_conductivity=0.5, two_phase_transport_model=liq,
            ),
            ch=mrpd.FlowChannel(
                width=1e-3, height=1e-3, length=0.1, n_parallel=20, reactant='o2',
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
                width=1e-3, height=1e-3, length=0.1, n_parallel=20, reactant='h2',
            ),
            has_mpl=False, thermal_contact_resistance=4e-4,
        ),
        membrane=mrpd.PFSA(ionomer=ionomer, dry_thickness=25e-6),
    )

Operating conditions
--------------------

:class:`~marapendi.simulation.conditions.CellConditions` accepts scalar values
(single point) or numpy arrays (vectorised: all points evaluated in one call).

.. code-block:: python

    i_arr = np.linspace(500, 22000, 40)   # A/m²
    T = 353.15                             # K

    conditions = mrpd.CellConditions(
        current_density=i_arr,
        cell_temperature=T,
        ca=mrpd.SideConditions(
            inlet_temperature=T,
            outlet_pressure=1.5e5,        # Pa
            dry_o2_mole_fraction=0.21,
            inlet_relative_humidity=0.5,
            stoichiometry=2.0,
        ),
        an=mrpd.SideConditions(
            inlet_temperature=T,
            outlet_pressure=1.5e5,
            dry_h2_mole_fraction=1.0,
            inlet_relative_humidity=0.5,
            stoichiometry=1.5,
        ),
    )

Two-step solve
--------------

.. code-block:: python

    model = mrpd.ExplicitSteadyStateModel()
    state = model.set_initial_conditions(cell, conditions)
    state = model.solve(cell, conditions, state)

    # Polarization curve
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(i_arr * 1e-4, state.cell_voltage)
    ax.set_xlabel("Current density (A cm⁻²)")
    ax.set_ylabel("Cell voltage (V)")

:meth:`~marapendi.models.base.explicit_steady_state.ExplicitSteadyStateModel.set_initial_conditions`
populates a :class:`~marapendi.simulation.state.CellState` with starting values (equilibrium
water content at the inlet RH, analytical temperature estimate). ``solve`` updates it in
place and returns the same object.

Implicit model
--------------

:class:`~marapendi.models.base.implicit_steady_state.ImplicitSteadyStateModel` iterates
MEA temperature and cell voltage to self-consistency.  The API is identical:

.. code-block:: python

    from marapendi.models.base.implicit_steady_state import ImplicitSteadyStateModel

    imp = ImplicitSteadyStateModel()
    state_imp = imp.solve(cell, conditions,
                          imp.set_initial_conditions(cell, conditions))

The implicit model is more accurate at high current densities where the thermal
feedback is significant.  The explicit model is faster and sufficient for
calibration.

Accessing state variables
--------------------------

The :class:`~marapendi.simulation.state.CellState` mirrors the component tree.  After
``solve`` every field is populated:

.. code-block:: python

    # --- Cell-level ---
    state.cell_voltage            # V,   shape == i_arr.shape
    state.E_rev                   # V,   Nernst (reversible) voltage
    state.eta_act                 # V,   activation overpotential  (V = E_rev - eta_act - eta_ohm)
    state.eta_ohm                 # V,   ohmic overpotential
    state.mea_temperature         # K,   MEA temperature
    state.mea_temperature_increase # K,  T_MEA - T_stack
    state.current_density         # A/m²
    state.crossover_current       # A/m², equivalent H₂ crossover current

    # HFR is computed on demand (not stored automatically by the steady-state solver)
    hfr = model.voltage_model.high_frequency_resistance(cell, state)  # Ω m²

    # --- Cathode side ---
    state.ca.cl.temperature       # K,   catalyst layer temperature
    state.ca.cl.liquid_saturation # –,   liquid saturation
    state.ca.cl.ionomer_water_content  # mol/mol
    state.ca.cl.membrane_interface_water_content  # mol/mol
    state.ca.cl.proton_resistance # Ω m², CL proton resistance
    state.ca.cl.relative_humidity # –
    state.ca.cl.pressure          # Pa
    state.ca.cl.overpotential     # V

    state.ca.gdl.liquid_saturation
    state.ca.gdl.relative_humidity

    state.ca.ch.gas.X             # mole fractions [O2, N2, H2, H2O]
    state.ca.ch.pressure          # Pa
    state.ca.ch.temperature       # K

    state.ca.h2ov_transport_resistance   # m² s / mol, water vapour resistance
    state.ca.reactant_transport_resistance  # m² s / mol

    # --- Membrane ---
    state.membrane.water_content          # mol/mol, mean λ
    state.membrane.water_content_profile  # array, λ(ξ) through-plane profile
    state.membrane.proton_resistance      # Ω m²
    state.membrane.water_flux             # mol / (m² s)
    state.membrane.eod_speed             # –, electroosmotic drag coefficient
    state.membrane.diffusion_flux         # mol / (m² s)
    state.membrane.eod_flux               # mol / (m² s)

    # Anode side (same structure as cathode)
    state.an.cl.liquid_saturation
    state.an.cl.ionomer_water_content
    # …

.. note::

   Iteration helpers let you loop over all layers without naming ``ca`` / ``an``
   explicitly:

   .. code-block:: python

       for side in state.sides:          # (state.ca, state.an)
           print(side.cl.liquid_saturation)

       for layer in state.layers:        # all porous layers + membrane
           print(layer.temperature)

Overpotential breakdown
------------------------

The voltage balance is ``V = E_rev - η_act - η_ohm`` exactly:

.. code-block:: python

    fig, ax = plt.subplots()
    ax.stackplot(
        i_arr * 1e-4,
        [np.abs(state.eta_act), np.abs(state.eta_ohm)],
        labels=["Activation (η_act)", "Ohmic (η_ohm)"],
        colors=["C3", "C2"], alpha=0.75,
    )
    ax.plot(i_arr * 1e-4, state.E_rev - state.cell_voltage, "k--", label="Total loss")
    ax.set_xlabel("Current density (A cm⁻²)")
    ax.set_ylabel("Overpotential (V)")
    ax.legend()

Swapping the membrane water-balance model
-----------------------------------------

Both steady-state models use
:class:`~marapendi.models.water_balance.membrane_pwl.MembraneWaterBalanceModelPiecewise`
by default — the piecewise-linear isotherm closure described in
:doc:`/science/membrane_correlations`.  To use the first-order linear
expansion from Affonso Nobrega et al. (2026) instead (see
:doc:`/science/water_balance` for the derivation of both closures, and
:doc:`/auto_examples/plot_07_pwl_membrane` for a side-by-side comparison):

.. code-block:: python

    from marapendi.models.water_balance.water_balance import WaterBalanceModel
    from marapendi.models.water_balance.membrane import MembraneWaterBalanceModel

    model = mrpd.ExplicitSteadyStateModel(
        water_balance_model=WaterBalanceModel(
            membrane_water_balance_model=MembraneWaterBalanceModel()
        )
    )
