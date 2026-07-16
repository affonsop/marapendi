Steady-state polarization curve
================================

This guide covers the full workflow for a single steady-state simulation:
assembling the cell, defining operating conditions, running the solver, and
inspecting every variable in the output state.

.. seealso::

   The physics implemented by each model referenced here is documented in
   :doc:`/science/index`, in particular :doc:`/science/water_balance` (membrane
   water transport), :doc:`/science/membrane_correlations` (ionomer
   correlations), :doc:`/science/two_phase_flow` and :doc:`/science/gas_transport`
   (liquid water and species transport), :doc:`/science/catalyst_layer`, and
   :doc:`/science/cell_voltage`.
   :doc:`/auto_examples/plot_01_polarization_curve` is the runnable version of
   the cell assembled below; :doc:`/auto_examples/plot_07_pwl_membrane`
   compares the two water-balance models mentioned at the end of this page.

Cell assembly
-------------

A :class:`~marapendi.components.cell.fuelcell.FuelCell` is a pure component tree. It
holds geometry and material parameters only; all physics lives in the model
objects. Assemble the cell once and reuse it across many conditions.

.. note:: 
    All quantities are defined in SI units, except moles which are always in kmol.

.. code-block:: python

    import numpy as np
    import marapendi as mrpd

    liq     = mrpd.DarcyTransportModel(J_function_exponent=2)
    ionomer = mrpd.PFSAIonomer(equivalent_weight=1100, dry_density=1980)

    cell = mrpd.FuelCell(
        area=25e-4,                
        electric_resistance=30e-7,  
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

    i = np.linspace(500, 22000, 40) # A/m2
    conditions = mrpd.CellConditions(
        current_density=i, 
        cell_temperature=353.15,
        ca=mrpd.SideConditions(
            inlet_temperature=T,
            outlet_pressure=1.5e5,      
            dry_o2_mole_fraction=0.21,
            inlet_relative_humidity=0.5,
            stoichiometry=2.0,
        ),
        an=mrpd.SideConditions(
            inlet_temperature=353.15,
            outlet_pressure=1.5e5,
            dry_h2_mole_fraction=1.0,
            inlet_relative_humidity=0.5,
            stoichiometry=1.5,
        ),
    )

Solving the steady-state model
--------------------------------------

.. code-block:: python

    model = mrpd.ExplicitSteadyStateModel()
    state = model.set_initial_conditions(cell, conditions)
    state = model.solve(cell, conditions, state)


:meth:`~marapendi.models.base.explicit_steady_state.ExplicitSteadyStateModel.set_initial_conditions`
populates a :class:`~marapendi.simulation.state.CellState` with starting values (equilibrium
water content at the inlet RH, analytical temperature estimate). ``solve`` updates it in
place and returns the same object.

Plotting the polarization curve
--------------------------------------

.. code-block:: python

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(i * 1e-4, state.cell_voltage)
    ax.set_xlabel("Current density (A/cm$^2$)")
    ax.set_ylabel("Cell voltage (V)")


Implicit model
--------------

:class:`~marapendi.models.base.implicit_steady_state.ImplicitSteadyStateModel` iterates
MEA temperature and cell voltage to self-consistency.  The API is identical:

.. code-block:: python

    imp = mrpd.ImplicitSteadyStateModel()
    state_imp = imp.solve(cell, conditions,
                          imp.set_initial_conditions(cell, conditions))

The implicit model is more accurate at high current densities where the thermal
feedback is significant.  See :doc:`/auto_examples/plot_03_implicit_vs_explicit` for 
a comparison. 

Accessing state variables
--------------------------

The :class:`~marapendi.simulation.state.CellState` mirrors the component tree. 
See :doc:`/user_guide/state_variables` for a complete list of quantities stored in `state`.

.. note::

   Iteration helpers let you loop over all layers without naming ``ca`` / ``an``
   explicitly:

   .. code-block:: python

       for side in state.sides:          # (state.ca, state.an)
           print(side.cl.liquid_saturation)

       for layer in state.layers:        # all porous layers + membrane
           print(layer.temperature)
