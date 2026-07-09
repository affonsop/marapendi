Using the TransientPEMFC Simulink block
========================================

:class:`~marapendi.models.base.transient.TransientModel` can be dropped into a
Simulink model as a single masked block, ``TransientPEMFC``. The block does not
re-implement the physics in MATLAB — it calls the live Python model at every
solver step through MATLAB's ``py.`` interface, so editing the model means
editing the Python source under ``src/marapendi/`` as usual; the block always
reflects whatever is currently on disk.

The block and the scripts that build it live under ``matlab/transient_pemfc/``
in the repository.

.. note::

   This has been built and validated in MATLAB R2025a on macOS: a 300 s
   step-current transient run through the generated ``TransientPEMFC.slx``
   matches :meth:`~marapendi.models.base.transient.TransientModel.solve` run
   directly in Python to about 6 significant figures (the residual
   difference is expected solver noise — Simulink's ``ode15s`` vs. SciPy's
   ``BDF``, both at ``rtol=1e-3``).

How it's wired
---------------

Level-2 MATLAB S-Functions cannot reliably declare bus-typed ports from
M-code across MATLAB releases, so the S-Function core
(``transient_pemfc_sfun.m``) uses plain vector ports, and the masked
subsystem does the bus conversion with standard blocks::

    CellConditions (bus in)
        -> Bus Selector (24 scalar signals)
        -> Mux (1x24 vector)
        -> TransientPEMFC_core (S-Function: 6 continuous states, plain vector I/O)
        -> Demux (26 scalars) + raw profile vector
        -> Bus Creator (-> CellStateBus)
        -> CellState (bus out)
    TransientPEMFC_core -> x (raw ODE state vector, plain port)

The S-Function itself calls
``marapendi.interop.simulink_bridge.derivative()``/``.diagnostics()`` each
step — a thin adapter module (``src/marapendi/interop/simulink_bridge.py``)
that exposes :class:`~marapendi.models.base.transient.TransientModel` through
plain scalars, lists, and dicts, since those are the only types MATLAB's
``py.`` interface marshals unambiguously. It also caches the Python
``FuelCell`` object handle across steps (Simulink continuous-state Dwork
vectors only hold numeric data, so the handle is kept in an id-indexed
registry instead, ``transient_pemfc_registry.m``).

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

Block interface
-----------------

- **Input** — ``CellConditions``, a bus of type ``CellConditionsBus``
  mirroring :class:`~marapendi.simulation.conditions.CellConditions`
  (``current_density``, ``cell_temperature``, and a ``ca``/``an``
  :class:`~marapendi.simulation.conditions.SideConditions` sub-bus each).
- **Outputs**

  - ``CellState`` — a bus of type ``CellStateBus`` with the flattened
    diagnostics :meth:`~marapendi.models.base.transient.TransientModel.evaluate`
    returns: cell voltage, HFR, membrane water-content profile, catalyst-layer
    water content/saturation/proton resistance, water fluxes, and so on.
  - ``x`` — the raw 6-element ODE state vector, ``[T_mea; lambda_1..lambda_5]``,
    normalised the same way as
    :meth:`~marapendi.models.base.transient.TransientModel.f_transient`.

- **Mask parameters**

  - ``n_memb_mesh`` — membrane finite-volume node count. Changing it requires
    re-running ``build_transient_block`` (which rebuilds the buses and the
    Bus Selector/Demux/Bus Creator wiring to match), not just editing the mask.
  - ``cellBuilderExpr`` — dotted path to a zero-argument Python function
    returning a :class:`~marapendi.cell.fuelcell.FuelCell`. Defaults to
    ``marapendi.interop.simulink_bridge.build_default_cell``, the same cell
    assembled in :doc:`../auto_examples/plot_01_polarization_curve`. Point
    this at your own builder function to simulate a different cell.
  - ``x0`` — initial ODE state. ``build_transient_block.m`` computes one for a
    nominal operating point automatically; override it for a different
    starting condition using
    ``py.marapendi.interop.simulink_bridge.initial_state(...)`` from MATLAB.

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
  trajectory). Measured on the 300 s validation scenario below, the block
  still runs about 3.4x slower wall-clock than calling
  :meth:`~marapendi.models.base.transient.TransientModel.solve` directly in
  Python — not because of MATLAB/Python marshalling overhead (~15% per
  call) or a more expensive solver, but because Simulink evaluates the full
  diagnostics pipeline once per major step via ``Outputs``, while
  ``solve()`` only runs it once, vectorised, at the very end. See
  ``matlab/transient_pemfc/README.md`` for the numbers.
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
      from marapendi.interop.simulink_bridge import build_default_cell

      cell = build_default_cell()
      model = mrpd.TransientModel(n_memb_mesh=5)
      cond = mrpd.CellConditions(
          current_density=20000., cell_temperature=344.15,
          ca=mrpd.SideConditions(inlet_temperature=344.15, outlet_pressure=1.4e5,
                                  dry_o2_mole_fraction=0.21, inlet_relative_humidity=0.265,
                                  stoichiometry=1.6),
          an=mrpd.SideConditions(inlet_temperature=344.15, outlet_pressure=1.9e5,
                                  dry_h2_mole_fraction=1.0, inlet_relative_humidity=0.558,
                                  stoichiometry=1.4),
      )
      sol = model.solve(cell, cond, t_span=(0, 300), rtol=1e-3)
      print(sol.diagnostics.cell_voltage[-1], sol.diagnostics.mea_temperature[-1], sol.y[:, -1])

2. In Simulink, load ``TransientPEMFC.slx``, replace the ``CellConditions``
   inport with a ``Constant`` block carrying the same operating point (a
   MATLAB struct shaped like ``CellConditionsBus``, typed via
   ``'OutDataTypeStr', 'Bus: CellConditionsBus'``), set
   ``Solver=ode15s``, ``RelTol=1e-3``, ``StopTime=300``, and run.
3. Compare ``CellState.cell_voltage``, ``CellState.mea_temperature``, and
   ``x`` at the final time step against the Python values — they should
   agree to about 5-6 significant figures.

See ``matlab/transient_pemfc/README.md`` in the repository for the full file
listing and further detail.
