.. _package_structure:

Package structure
=================

marapendi is organized into four subpackages under ``src/marapendi/``:

.. code-block:: text

    src/marapendi/
    ├── components/        # Cell components (static parameters)
    │   ├── cell/          # FuelCell, FuelCellSide, ElectrolyzerCell
    │   ├── channel/       # FlowChannel (geometry, gas-transport model)
    │   ├── membrane/      # PFSA, PAP85, Ionomer, PFSAIonomer, PAPIonomer
    │   └── porous/        # PorousLayer, GasDiffusionLayer, MicroPorousLayer,
    │       └── cl/        # CatalystLayer, PtCCatalystLayer, PorousTransferLayer
    ├── models/            # Physics models (stateless or own internal state)
    │   ├── cell/          # ExplicitSteadyStateModel, VoltageModel, ThermalModel,
    │   │                  # GasTransportModel, MembraneWaterBalanceModel
    │   ├── porous/        # PorousGasResistanceModel, DarcyTransportModel
    │   ├── channel/       # ChannelGasResistanceModel, BakerChannelGasResistanceModel
    │   ├── membrane/      # HydrogenPermeationModel
    │   ├── degradation/   # PtDissolution, PlatinumOxideFormation
    │   ├── gas.py         # GasState, GasModel
    │   ├── water.py       # Water property correlations
    │   ├── electrochemistry.py  # ElectrochemicalReaction, calculate_reversible_cell_voltage
    │   └── constants.py   # Physical constants
    ├── simulation/        # State containers and load cycles
    │   ├── state.py       # CellState, CellSideState, LayerState, ...
    │   └── load_cycles.py # LoadCycle
    └── estimation/        # Parameter estimation
        ├── estimation.py  # DynamicModel, SteadyStateModel
        └── cross_validation.py

Design philosophy
-----------------

**Separation of static parameters and runtime state**

marapendi follows a strict separation:

* **Components** (``components/``) hold *static* parameters: geometry, material
  constants, transport-model objects.  They are dataclass instances that do not
  change during a simulation.
* **State** (``simulation/state.py``) holds *runtime* values: temperatures, gas
  compositions, water contents, fluxes, overpotentials.  State objects are plain
  dataclasses with no physics methods.
* **Models** (``models/``) contain the physics.  Every model method takes
  ``(component, state)`` as its first two arguments — it reads static data from
  the component and reads/writes runtime values on the state.

The entry point for a simulation is :class:`~marapendi.components.cell.fuelcell.FuelCell`,
which owns the component tree and a :class:`~marapendi.simulation.state.CellState`.
The ``compute_ui_curve`` method delegates the full solve to
:class:`~marapendi.models.cell.explicit_steady_state.ExplicitSteadyStateModel`.

**No delegate pattern**

Physics methods on model objects must be called with explicit ``(component, state)``
arguments:

.. code-block:: python

    # correct
    resistance = layer.transport_resistance_model.gas_transport_resistance(layer, state, 'o2')

    # never do this (removed)
    resistance = layer.gas_transport_resistance(state, 'o2')

This keeps components as passive data containers and makes the data flow explicit.

**GasState / GasModel**

:class:`~marapendi.models.gas.GasState` stores only the mole-fraction array ``X``
for the four species (O₂, N₂, H₂, H₂O).  Temperature and pressure live on the
surrounding :class:`~marapendi.simulation.state.LayerState` or
:class:`~marapendi.simulation.state.FlowChannelState`.

:class:`~marapendi.models.gas.GasModel` provides pure static methods that take a
``state`` object (anything with ``.gas``, ``.temperature``, ``.pressure``):

.. code-block:: python

    pp_o2 = mrpd.GasModel.species_partial_pressure(layer_state, 'o2')
    D_o2  = mrpd.GasModel.species_diffusion_coefficient(layer_state, 'o2')
    rh    = mrpd.GasModel.relative_humidity(layer_state)

Subpackage responsibilities
---------------------------

components/
~~~~~~~~~~~

Contains only dataclass definitions.  Each class stores the material and
geometry parameters that are fixed once the cell has been assembled.  No
physics calculations live here.

models/
~~~~~~~

All physics is here.  Sub-namespaces:

* ``cell/`` — top-level orchestration (``ExplicitSteadyStateModel``), voltage,
  thermal, gas-transport and water-balance models.
* ``porous/`` — Bruggeman/Knudsen gas diffusion (``PorousGasResistanceModel``),
  capillary two-phase transport (``DarcyTransportModel``).
* ``channel/`` — Sherwood-number channel gas-transport model.
* ``membrane/`` — hydrogen permeation correlation.
* ``degradation/`` — Pt dissolution kinetics.

simulation/
~~~~~~~~~~~

:class:`~marapendi.simulation.state.CellState` mirrors the component tree
(``ca``/``an`` sides, each with a catalyst layer, GDL/MPL, and flow channel,
plus a membrane).  Iteration helpers such as ``.sides``, ``.porous_layers``,
and ``.layers`` let model code loop over all layers without hardcoding names.

estimation/
~~~~~~~~~~~

:class:`~marapendi.estimation.estimation.SteadyStateModel` wraps a steady-state
model function ``h(params) -> array`` and provides global parameter estimation
via ``scipy.optimize.differential_evolution``.
:class:`~marapendi.estimation.estimation.DynamicModel` does the same for ODE
systems.
