Architecture
=========================

**marapendi** separates the *description* of a cell from the *calculations*
performed on it, and keeps the runtime *state* of a simulation separate from
both. Every sub-model is an ordinary Python class, so the framework is meant
to be subclassed rather than configured — see :doc:`user_guide/extending_models`
for how to override a single correlation or swap a whole sub-model.

``components`` (:mod:`marapendi.cell`, :mod:`marapendi.porous_layers`, :mod:`marapendi.membrane`, :mod:`marapendi.channel`, :mod:`marapendi.electrolyte`)
    Dataclasses holding the static, measurable properties of a cell's
    components (geometry, porosity, permeability, ionomer parameters,
    catalyst loading, ...), together with the correlation models that turn
    these properties into transport and electrochemical quantities. A
    :class:`~marapendi.cell.fuelcell.FuelCell` assembles a cathode and anode
    :class:`~marapendi.cell.fuelcell.FuelCellSide` (each with a catalyst layer,
    gas diffusion layer, optional microporous layer and flow channel) and a
    :class:`~marapendi.membrane.membrane_base.Membrane`.

``state`` (:mod:`marapendi.simulation.state`)
    Dataclasses mirroring the shape of :class:`~marapendi.cell.fuelcell.FuelCell`
    but holding the *physical variables* (temperature, pressure, gas composition,
    saturation, water content, fluxes, ...) at one operating point. State objects
    are pure data: no physics lives here.

``models`` (:mod:`marapendi.models`)
    Orchestration classes that combine a :class:`~marapendi.cell.fuelcell.FuelCell`
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
    sub-models each solver is built from.

``correlations`` (:mod:`marapendi.models.thermo`, :mod:`marapendi.simulation`)
    Stateless physics building blocks (water thermodynamics, electrochemical
    kinetics, gas mixture properties, operating conditions) used by the
    components and models above. :mod:`marapendi.simulation` also holds
    :class:`~marapendi.simulation.load_cycles.LoadCycle` and the standardised
    ID-FAST/FC-DLC driving cycles used to drive the transient model in time.

``estimation`` (:mod:`marapendi.estimation`)
    :class:`~marapendi.estimation.polarization_curve_calibration.SteadyStatePolarizationCurveCalibration`
    fits kinetic and transport parameters to multi-condition polarization and HFR data
    using differential-evolution global optimisation, with k-fold cross-validation and
    automatic model-complexity selection via the 1-SE rule.

``interop`` (:mod:`marapendi.interop`)
    A thin adapter (:mod:`marapendi.interop.simulink_bridge`) exposing
    :class:`~marapendi.models.base.transient.TransientModel` to MATLAB through
    plain scalars, lists, and dicts. It backs the ``TransientPEMFC`` Simulink
    block (``matlab/transient_pemfc/``, see :doc:`user_guide/simulink_block`),
    which calls the live Python model at every solver step — no physics is
    re-implemented in MATLAB, and cell parameters can be supplied either as a
    dotted Python builder path or as a plain MATLAB struct.

The steady-state model and the underlying correlations implement the model
described in Affonso Nobrega et al., *J. Electrochem. Soc.* **173**, 114503
(2026), to which this documentation defers for the physical and mathematical
description of the cell model.
