Using the TransientPEMFC Simulink block
========================================

:class:`~marapendi.models.base.transient.TransientModel` can be dropped into a
Simulink model as a single masked block, ``TransientPEMFC``. The block does
not re-implement the physics in MATLAB — it calls the live Python model at
every solver step through MATLAB's ``py.`` interface, so editing the model
means editing the Python source under ``src/marapendi/`` as usual; the block
always reflects whatever is currently on disk.

The block is driven directly by inlet
:class:`~marapendi.simulation.state.GasFlowState`\ s (cathode + anode) plus
current density and cell temperature — the natural signal for a
system-level model where upstream compressor/humidifier/coolant blocks
already produce flow states, not a stoichiometry spec. It outputs the full
diagnostics (:class:`~marapendi.simulation.state.CellState`), the
corresponding outlet ``GasFlowState``\ s (from a mass balance), heat release
rate, and cell voltage.

The block and the scripts that build it live under ``matlab/transient_pemfc/``
in the repository — they are not included in the PyPI package, so a local
clone is required (see :doc:`../installation`).

.. note::

   This has been built and validated in MATLAB R2025a on macOS: a 300 s
   constant-input transient run through the generated ``TransientPEMFC.slx``
   matches :meth:`~marapendi.models.base.transient.TransientModel.solve`/
   :meth:`~marapendi.models.base.transient.TransientModel.evaluate` run
   directly in Python exactly — cell voltage, MEA temperature, all 6 ODE
   states, heat release rate, and both outlet ``GasFlowState``\ s' species
   flow rates (the residual difference from Simulink's own ``ode15s`` vs.
   SciPy's ``BDF``, both at ``rtol=1e-3``, is below floating-point display
   precision at the operating points tested).

How it's wired
---------------

Level-2 MATLAB S-Functions cannot reliably declare bus-typed ports from
M-code across MATLAB releases, so the S-Function core
(``transient_pemfc_sfun.m``) uses plain vector ports, and the masked
subsystem does the bus conversion with standard blocks::

    CaInlet, AnInlet (bus in)
        -> Bus Selector (7 scalar signals each)
        -> Mux (1x7 vector each)
        -> TransientPEMFC_core (S-Function: 6 continuous states, plain vector I/O)
        -> Demux (27 scalars) + raw profile vector
        -> Bus Creator (-> CellStateBus)
        -> CellState (bus out)
    TransientPEMFC_core -> Demux (7) -> Bus Creator (-> GasFlowStateBus) -> CaOutlet, AnOutlet (bus out)
    TransientPEMFC_core -> HeatRelease, CellVoltage, x (plain ports)

The S-Function itself calls
``marapendi.interop.simulink_bridge.cell_derivative()``/``.cell_diagnostics()``
each step — a thin adapter module (``src/marapendi/interop/simulink_bridge.py``)
that exposes :class:`~marapendi.models.base.transient.TransientModel` through
plain scalars, lists, and dicts, since those are the only types MATLAB's
``py.`` interface marshals unambiguously. It also caches the Python
``FuelCell`` object handle (and the last computed
:class:`~marapendi.simulation.state.CellState`) across steps (Simulink
continuous-state Dwork vectors only hold numeric data, so the handle is
kept in an id-indexed registry instead, ``transient_pemfc_registry.m``).

This relies on :class:`~marapendi.models.base.transient.TransientModel`
populating ``state.ca.outlet_gas_flow_state``/``state.an.outlet_gas_flow_state``
(and the same for ``an``) and ``state.heat_release`` automatically — see
:meth:`~marapendi.models.base.explicit_steady_state.ExplicitSteadyStateModel.set_gas_flow_states` and
:class:`~marapendi.simulation.state.GasFlowState` for how that mass balance
works, and ``tests/test_gas_flow_state.py`` for how it's validated
(Faraday's law, water stoichiometry, N2 inertness) directly in Python.

One-time setup
---------------

1. Make sure the Python environment you'll point MATLAB at has
   ``marapendi``'s dependencies installed (``numpy``, ``scipy``) and can
   ``import marapendi`` once ``src/`` is on ``sys.path`` — no ``pip install``
   of the package itself is required.

2. In MATLAB, from ``matlab/transient_pemfc/``:

   .. code-block:: matlab

      cd path/to/marapendi/matlab/transient_pemfc
      pyenv_setup('/path/to/marapendi/.venv/bin/python');  % or omit the arg to use the current pyenv()

   MATLAB can only load one Python version per session — if a different
   interpreter is already loaded in-process, restart MATLAB first. A
   "Python version X.Y is not supported" warning is non-fatal in recent
   MATLAB releases; only worry about it if ``import marapendi`` itself
   fails.

3. Build the model:

   .. code-block:: matlab

      build_transient_block                                  % uses the current pyenv()
      build_transient_block('/path/to/.venv/bin/python')     % or switch interpreter first

   This writes ``TransientPEMFC.slx`` next to the build scripts.

4. Try the runnable example:

   .. code-block:: matlab

      build_example_transient_pemfc                                  % uses the current pyenv()
      build_example_transient_pemfc('/path/to/.venv/bin/python')     % or switch interpreter first

   Writes ``example_transient_pemfc.slx`` (copies the ``TransientPEMFC``
   block into a small top-level model): fixed inlet ``GasFlowState``\ s
   sized for stoichiometric excess at the higher of two current densities, a
   ``Step`` block drives ``CurrentDensity`` from 10000 to 20000 A/m² at
   t=100 s, scoped on cell voltage, MEA temperature, heat release rate, and
   the cathode outlet water content. Open it and press Run to see the
   transient step response end to end without building any wiring yourself.

Block interface
-----------------

- **Inputs**

  - ``CaInlet``, ``AnInlet`` — bus, type ``GasFlowStateBus``:
    ``temperature``, ``pressure``, and per-species molar flow rates ``o2``,
    ``n2``, ``h2``, ``h2o`` (kmol/s), plus ``liquid``, mirroring
    :class:`~marapendi.simulation.state.GasFlowState`.
  - ``CurrentDensity`` (A/m²), ``CellTemperature`` (K) — scalars.

- **Outputs**

  - ``CellState`` — a bus of type ``CellStateBus`` with the flattened
    diagnostics :meth:`~marapendi.models.base.transient.TransientModel.evaluate`
    returns (27 scalar fields — cell voltage, HFR, heat release,
    catalyst-layer water content/saturation/proton resistance, water
    fluxes, and so on — plus ``membrane_water_content_profile``).
  - ``CaOutlet``, ``AnOutlet`` — bus, type ``GasFlowStateBus``, the outlet
    flow from a mass balance
    (:meth:`~marapendi.simulation.state.GasFlowState.consume`): reactant
    consumed, product water added as vapor and/or liquid.
  - ``HeatRelease`` (W/m²), ``CellVoltage`` (V) — scalars, duplicating
    ``CellState.heat_release``/``.cell_voltage`` as standalone ports for
    convenience (e.g. wiring straight to a scope without a Bus Selector).
  - ``x`` — the raw 6-element ODE state vector, ``[T_mea; lambda_1..lambda_5]``,
    normalised the same way as
    :meth:`~marapendi.models.base.transient.TransientModel.f_transient`.

- **Mask parameters**

  - ``n_memb_mesh`` — membrane finite-volume node count. Changing it requires
    re-running ``build_transient_block`` (which rebuilds the buses and the
    Bus Selector/Demux/Bus Creator wiring to match), not just editing the mask.
  - ``cellBuilderExpr`` — either a dotted path to a zero-argument Python
    function returning a :class:`~marapendi.components.cell.fuelcell.FuelCell`
    (default: ``marapendi.interop.simulink_bridge.build_default_cell``, the
    same cell assembled in :doc:`../auto_examples/plot_01_polarization_curve`),
    or the name of a MATLAB function (no ``.``) returning a struct of cell
    parameters — no Python editing required to change geometry/materials.
    For the latter, copy ``matlab/transient_pemfc/cell_params_template.m``
    (e.g. to ``my_cell_params.m``), edit the values, and set
    ``cellBuilderExpr`` to ``'my_cell_params'``. The struct fields/defaults
    mirror :func:`~marapendi.interop.simulink_bridge.default_cell_params`,
    and ``call_python_builder.m`` dispatches on whether the string contains
    a ``.`` to tell the two forms apart.
  - ``x0`` — initial ODE state. ``build_transient_block.m`` computes one for a
    nominal operating point automatically; override it for a different
    starting condition using
    ``py.marapendi.interop.simulink_bridge.cell_initial_state(...)`` from MATLAB.

Some ``CellState`` fields (for example ``membrane_water_flux``,
``membrane_proton_resistance``, ``an_cl_proton_resistance``) can show up as
``NaN`` at certain operating points. That is not a marshalling bug — it means
the corresponding Python ``CellState`` attribute is ``None`` there, since not
every field is populated by every code path through the model's internal
``_eval_state``.

Known limitations
--------------------

- **Not real-time, not parallel.** Every derivative evaluation, and every
  *major-time-step* output evaluation, round-trips into Python and holds
  the GIL, so Rapid Accelerator, multicore, and Simulink Compiler/Coder
  targets are not supported — run in Normal or plain Accelerator mode.
  ``Outputs`` already applies the standard Simulink fix for expensive output
  blocks (``block.IsMajorTimeStep``): it only calls into Python on major
  time steps and reuses the cached ``CellState`` on minor steps (used
  internally by the variable-step solver, not part of the reported
  trajectory). Measured on a 300 s constant-input scenario, the block runs
  roughly 5x slower wall-clock than calling
  :meth:`~marapendi.models.base.transient.TransientModel.solve` directly in
  Python — not primarily because of MATLAB/Python marshalling overhead or a
  more expensive solver (`Derivatives` call count is the same order as
  Python's `f_transient` evaluation count), but because Simulink still
  evaluates the diagnostics pipeline once per major step via ``Outputs``,
  while ``solve()`` only runs it once, vectorised, at the very end. See
  ``matlab/transient_pemfc/README.md`` for the exact numbers, and for a
  documented negative result on trying to close that gap further (an
  investigated `return_state=True` cache-sharing optimization that turned
  out not to pay off against `ode15s`).
- **Cell parameters are not a runtime signal.** The ``FuelCell`` is built
  once (in the S-Function's ``Start`` callback) from ``cellBuilderExpr`` and
  reused for every step, matching how
  :class:`~marapendi.models.base.transient.TransientModel` treats ``cell`` in
  Python. To sweep cell parameters, point ``cellBuilderExpr`` at a different
  builder function and rebuild.

Validating a change
----------------------

Because the block calls the same :class:`~marapendi.models.base.transient.TransientModel`
code path, any mismatch against a direct Python run means a marshalling or
wiring bug, not a physics difference. To check:

1. Run the scenario directly in Python:

   .. code-block:: python

      import marapendi as mrpd
      from marapendi.interop import simulink_bridge as sb

      cell = sb.build_default_cell()
      model = mrpd.TransientModel(n_memb_mesh=5)
      ca_flow = dict(temperature=344.15, pressure=140000., o2=1.05e-7, n2=3.90e-7, h2=0., h2o=3.25e-8, liquid=0.)
      an_flow = dict(temperature=344.15, pressure=190000., o2=0., n2=0., h2=3.6e-7, h2o=3.8e-8, liquid=0.)
      cond = sb._cell_conditions_from_flows(ca_flow, an_flow, 10000., 344.15)
      sol = model.solve(cell, cond, t_span=(0, 300), rtol=1e-3)
      print(sol.diagnostics.cell_voltage[-1], sol.diagnostics.mea_temperature[-1], sol.y[:, -1])

2. In Simulink, load ``TransientPEMFC.slx``, replace the ``CaInlet``/
   ``AnInlet`` inports with ``Constant`` blocks carrying the same operating
   point (MATLAB structs matching ``ca_flow``/``an_flow`` above, typed via
   ``'OutDataTypeStr', 'Bus: GasFlowStateBus'``), and ``CurrentDensity``/
   ``CellTemperature`` constants, set ``Solver=ode15s``, ``RelTol=1e-3``,
   ``StopTime=300``, and run.
3. Compare ``CellState.cell_voltage``, ``CellState.mea_temperature``, and
   ``x`` at the final time step against the Python values — they should
   agree to solver tolerance.

See ``matlab/transient_pemfc/README.md`` in the repository for the full file
listing and further detail.
