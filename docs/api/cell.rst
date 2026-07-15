Cell
====

:class:`~marapendi.components.cell.fuelcell.FuelCell` is the root of the component tree.
It assembles a cathode and an anode :class:`~marapendi.components.cell.fuelcell.FuelCellSide`
(each owning a catalyst layer, GDL, optional MPL, and flow channel) and a
:class:`~marapendi.components.membrane.membrane_base.Membrane`.  The cell object holds only
geometry and material parameters — no physics.

All physics are evaluated by a model object passed the cell at solve time::

    model = ExplicitSteadyStateModel()
    state = model.set_initial_conditions(cell, conditions)
    state = model.solve(cell, conditions, state)

The :class:`~marapendi.simulation.state.CellState` returned by ``solve`` mirrors the
structure of :class:`~marapendi.components.cell.fuelcell.FuelCell` but holds runtime
physical quantities (voltages, temperatures, transport resistances, water
contents, saturation).

FuelCell
--------

.. autoclass:: marapendi.components.cell.fuelcell.FuelCell
   :members:
   :show-inheritance:

.. autoclass:: marapendi.components.cell.fuelcell.FuelCellSide
   :members:
   :show-inheritance:

State
-----

.. autoclass:: marapendi.simulation.state.CellState
   :members:
   :show-inheritance:

.. autoclass:: marapendi.simulation.state.CellSideState
   :members:
   :show-inheritance:

.. autoclass:: marapendi.simulation.state.CatalystLayerState
   :members:
   :show-inheritance:

.. autoclass:: marapendi.simulation.state.MembraneState
   :members:
   :show-inheritance:

.. autoclass:: marapendi.simulation.state.LayerState
   :members:
   :show-inheritance:

.. autoclass:: marapendi.simulation.state.FlowChannelState
   :members:
   :show-inheritance:
