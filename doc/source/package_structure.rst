.. _package_structure:

Package structure
=================

marapendi is organized into subpackages under ``src/marapendi/``:

.. code-block:: text

    src/marapendi/
    ├── cell/              # Top-level cell objects and physics models
    │   ├── fuelcell.py             # FuelCell, FuelCellSide
    │   ├── aem_electrolyzer.py     # ElectrolyzerCell, ElectrolyzerCellSide
    │   ├── cell.py                 # PEMFuelCell, AEMElectrolyzer (assembled)
    │   ├── state.py                # CellState, CellSideState, LayerState, …
    │   ├── explicit_steady_state.py # ExplicitSteadyStateModel
    │   ├── voltage.py              # VoltageModel
    │   ├── thermal.py              # ThermalModel
    │   ├── gas_transport.py        # GasTransportModel
    │   └── water_balance.py        # MembraneWaterBalanceModel
    ├── membrane/          # Membrane and ionomer materials
    │   ├── ionomer.py              # Ionomer (base class)
    │   ├── pem.py                  # PFSAIonomer, NafionD2020
    │   ├── aem.py                  # PAPIonomer
    │   ├── membrane.py             # Membrane, PFSA, AEM, FAA3, PAP85, …
    │   └── membrane_permeation_models.py  # HydrogenPermeationModel
    ├── porous_layers/     # Porous transport layers and catalyst layers
    │   ├── porous_layers.py        # PorousLayer, GasDiffusionLayer, MicroPorousLayer
    │   ├── catalyst_layers.py      # CatalystLayer, PtCCatalystLayer, PorousTransferLayer
    │   ├── darcy.py                # DarcyTransportModel
    │   └── diffusion.py            # PorousGasResistanceModel
    ├── channel/           # Flow channels
    │   └── flow_channels.py        # FlowChannel
    │   └── baker.py                # BakerChannelGasResistanceModel
    ├── thermo/            # Thermodynamic properties and constants
    │   ├── constants.py            # Physical constants (FARADAY_CONSTANT, …)
    │   ├── electrochemistry.py     # ElectrochemicalReaction, calculate_reversible_cell_voltage
    │   ├── gas.py                  # GasState, GasModel
    │   └── water.py                # Water property correlations
    ├── electrolyte/       # Electrolyte solutions (AEM/alkaline systems)
    │   ├── electrolyte.py          # ElectrolyteSolution
    │   └── koh.py                  # KOHSolution
    ├── degradation/       # Degradation models
    │   └── degradation.py          # PtDissolution, PlatinumOxideFormation
    ├── simulation/        # Load cycle helpers
    │   └── load_cycles.py          # LoadCycle
    ├── estimation/        # Parameter estimation
    │   ├── estimation.py           # DynamicModel, SteadyStateModel
    │   └── cross_validation.py
    └── tools.py           # Shared utilities (arrhenius_term, …)

Design philosophy
-----------------

**Cell objects own the component tree**

:class:`~marapendi.cell.fuelcell.FuelCell` is the top-level object a user
constructs.  It owns the complete component tree (catalyst layers, GDL/MPL,
flow channels, membrane) and exposes a simple ``compute_ui_curve`` API that
delegates the full solve to
:class:`~marapendi.cell.explicit_steady_state.ExplicitSteadyStateModel`.

**GasState / GasModel**

:class:`~marapendi.thermo.gas.GasState` stores only the mole-fraction array ``X``
for the four species (O₂, N₂, H₂, H₂O).  Temperature and pressure live on the
surrounding :class:`~marapendi.cell.state.LayerState` or
:class:`~marapendi.cell.state.FlowChannelState`.

:class:`~marapendi.thermo.gas.GasModel` provides pure static methods that take a
``state`` object (anything with ``.gas``, ``.temperature``, ``.pressure``):

.. code-block:: python

    pp_o2 = mrpd.GasModel.species_partial_pressure(layer_state, 'o2')
    D_o2  = mrpd.GasModel.species_diffusion_coefficient(layer_state, 'o2')
    rh    = mrpd.GasModel.relative_humidity(layer_state)

Subpackage responsibilities
---------------------------

cell/
~~~~~

Top-level cell assembly classes (:class:`~marapendi.cell.fuelcell.FuelCell`,
:class:`~marapendi.cell.aem_electrolyzer.ElectrolyzerCell`) plus all physics
models:

* ``explicit_steady_state.py`` — full polarization-curve solve.
* ``voltage.py`` — reversible voltage, activation, ohmic and concentration
  overpotentials.
* ``thermal.py`` — MEA heat-transfer and temperature model.
* ``gas_transport.py`` — CL and GDL gas-transport resistances.
* ``water_balance.py`` — membrane water-content profile and net water flux.
* ``state.py`` — runtime state dataclasses (``CellState``, ``CellSideState``,
  ``LayerState``, ``MembraneState``, ``FlowChannelState``).

membrane/
~~~~~~~~~

Ionomer and membrane material models.  ``ionomer.py`` defines the abstract
:class:`~marapendi.membrane.ionomer.Ionomer` base; ``pem.py`` provides
:class:`~marapendi.membrane.pem.PFSAIonomer` (Nafion-family); ``aem.py``
provides :class:`~marapendi.membrane.aem.PAPIonomer`.  ``membrane.py`` adds
geometry (thickness, area) and permeation models on top of the ionomer.

porous_layers/
~~~~~~~~~~~~~~

Porous transport layers: GDL, MPL, and catalyst layers.
``darcy.py`` implements the capillary two-phase transport model;
``diffusion.py`` implements Bruggeman/Knudsen gas diffusion.

estimation/
~~~~~~~~~~~

:class:`~marapendi.estimation.estimation.SteadyStateModel` wraps a steady-state
model function ``h(params) -> array`` and provides global parameter estimation
via ``scipy.optimize.differential_evolution``.
:class:`~marapendi.estimation.estimation.DynamicModel` does the same for ODE
systems.
