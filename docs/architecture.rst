Architecture
=========================

**marapendi** separates the *description* of a cell from the *calculations*
performed on it, and keeps the runtime *state* of a simulation separate from
both. Every sub-model is an ordinary Python class, so that subclassing can be used
to define new models from existing ones. See :doc:`user_guide/extending_models`
for how to override a single correlation or swap a whole sub-model.

Components
----------
    Dataclasses holding the static, measurable properties of a cell's
    components (geometry, porosity, permeability, ionomer parameters,
    catalyst loading, ...), together with the correlation models that turn
    these properties into transport and electrochemical quantities. A
    :class:`~marapendi.components.cell.fuelcell.FuelCell` assembles a cathode and anode
    :class:`~marapendi.components.cell.fuelcell.FuelCellSide` (each with a 
    :class:`~marapendi.components.porous_layers.catalyst_layers.CatalystLayer`,
    a :class:`~marapendi.components.porous_layers.porous_layers.GasDiffusionLayer`, 
    an optional :class:`~marapendi.components.porous_layers.porous_layers.MicroPorousLayer` and 
    a :class:`~marapendi.components.channel.flow_channels.FlowChannel`) and a
    :class:`~marapendi.components.membrane.membrane_base.Membrane`: 

    .. code-block:: python

       cell = mrpd.FuelCell(
           area=25e-4, 
           ca=mrpd.FuelCellSide(
               cl=mrpd.PtCCatalystLayer(thickness=10e-6, platinum_loading=0.4e-2),
               gdl=mrpd.GasDiffusionLayer(thickness=200e-6, porosity=0.6),
               ch=mrpd.FlowChannel(width=1e-3, height=1e-3, length=0.1, reactant='o2'),
           ), 
           an=mrpd.FuelCellSide(
               cl=mrpd.PtCCatalystLayer(thickness=5e-6, platinum_loading=0.1e-2),
               gdl=mrpd.GasDiffusionLayer(thickness=200e-6, porosity=0.6),
               ch=mrpd.FlowChannel(width=1e-3, height=1e-3, length=0.1, reactant='h2'),
           ),
           membrane=mrpd.PFSA(dry_thickness=25e-6),
       )

    See :doc:`installation` for a complete, runnable version with realistic
    parameter values.

Simulation
----------
    Runtime data structures for one operating point.
    :mod:`~marapendi.simulation.state` holds dataclasses mirroring the shape
    of :class:`~marapendi.components.cell.fuelcell.FuelCell` but with the
    *physical variables* (temperature, pressure, gas composition, saturation,
    water content, fluxes, ...) instead:

    .. code-block:: python

       model = mrpd.ExplicitSteadyStateModel()
       state = model.set_initial_conditions(cell, conditions)
       state = model.solve(cell, conditions, state)

       state.cell_voltage                     # cell voltage (V)
       state.membrane.water_content_profile   # Membrane water content
       state.ca.cl.gas.relative_humidity      # RH in the cathode catalyst layer
       state.an.gdl.temperature               # anode GDL temperature (K)

       # list every physical variable tracked for one layer
       import dataclasses
       [f.name for f in dataclasses.fields(state.ca.cl)]

    ``cell`` is the :class:`~marapendi.components.cell.fuelcell.FuelCell` built
    above; see :doc:`installation` for how to build matching ``conditions``.

    :mod:`~marapendi.simulation.conditions` defines the operating-condition
    inputs (:class:`~marapendi.simulation.conditions.CellConditions`,
    :class:`~marapendi.simulation.conditions.SideConditions`), and
    :mod:`~marapendi.simulation.load_cycles` holds
    :class:`~marapendi.simulation.load_cycles.LoadCycle` and the standardised
    ID-FAST/FC-DLC driving cycles used to drive the transient model in time.

Models
------
    Orchestration classes that combine a :class:`~marapendi.components.cell.cell.Cell`
    and a :class:`~marapendi.simulation.state.CellState` to compute the cell's
    behaviour. ``models.base`` holds the top-level solvers —
    :class:`~marapendi.models.base.explicit_steady_state.ExplicitSteadyStateModel` and
    :class:`~marapendi.models.base.implicit_steady_state.ImplicitSteadyStateModel`
    for steady-state evaluation, :class:`~marapendi.models.base.transient.TransientModel`
    for the coupled MEA-temperature/water-profile ODEs — while
    :mod:`~marapendi.models.thermal`, :mod:`~marapendi.models.voltage`,
    :mod:`~marapendi.models.gas_transport_resistance`, ``models.water_balance``,
    :mod:`~marapendi.models.darcy` and :mod:`~marapendi.models.diffusion` provide the
    thermal, voltage, gas-transport, membrane water-balance and two-phase-transport
    sub-models each solver is built from. Also includes stateless physics building blocks 
    (water thermodynamics, electrochemical
    kinetics, gas mixture properties) used by the components and models above.

``estimation``
    :class:`~marapendi.estimation.polarization_curve_calibration.SteadyStatePolarizationCurveCalibration`
    fits kinetic and transport parameters to multi-condition polarization and HFR data
    using differential-evolution global optimisation, with k-fold cross-validation and
    automatic model-complexity selection via the 1-SE rule.

``interop``
    A thin adapter (:mod:`marapendi.interop.simulink_bridge`) exposing
    :class:`~marapendi.models.base.transient.TransientModel` to MATLAB through
    plain scalars, lists, and dicts. It backs the ``TransientPEMFC`` Simulink
    block (``matlab/transient_pemfc/``, see :doc:`user_guide/simulink_block`),
    which calls the live Python model at every solver step — no physics is
    re-implemented in MATLAB, and cell parameters can be supplied either as a
    dotted Python builder path or as a plain MATLAB struct.
