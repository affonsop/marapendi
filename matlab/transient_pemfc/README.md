# TransientPEMFC Simulink block

Wraps `marapendi.models.base.transient.TransientModel` as a Simulink block by
calling the live Python model from a MATLAB S-Function (via MATLAB's `py.`
interface) — no physics is re-implemented in MATLAB. Editing the model?
Edit the Python source under `src/marapendi/`; the block always calls
whatever is currently on disk.

The block is driven directly by inlet
`marapendi.simulation.state.GasFlowState`s (cathode + anode) plus current
density and cell temperature — the natural signal for a system-level model
where upstream compressor/humidifier/coolant blocks already produce flow
states, not a stoichiometry spec. It outputs the full diagnostics
(`CellState`), the corresponding outlet `GasFlowState`s (from a mass
balance), heat release rate, and cell voltage.

Built and validated end-to-end against the Python model in MATLAB R2025a
(macOS): a 300 s constant-input transient run through `TransientPEMFC.slx`
matches `TransientModel.solve()`/`evaluate()` run directly in Python exactly
— cell voltage, MEA temperature, all 6 ODE states, heat release rate, and
both outlet `GasFlowState`s' species flow rates (the residual difference
from Simulink's own `ode15s` vs. SciPy's `BDF`, both at `rtol=1e-3`, is
below floating-point display precision at the operating points tested).

## Ports

- **Inputs**
  - `CaInlet`, `AnInlet` — bus, type `GasFlowStateBus` (`temperature`,
    `pressure`, and per-species molar flow rates `o2`, `n2`, `h2`, `h2o`
    (kmol/s), plus `liquid`, mirroring
    `marapendi.simulation.state.GasFlowState`).
  - `CurrentDensity` (A/m²), `CellTemperature` (K) — scalars.
- **Outputs**
  - `CellState` — bus, type `CellStateBus`: the flattened diagnostics
    `TransientModel.evaluate()` returns (27 scalar fields — cell voltage,
    HFR, heat release, catalyst-layer water content/saturation/proton
    resistance, water fluxes, and so on — plus
    `membrane_water_content_profile`).
  - `CaOutlet`, `AnOutlet` — bus, type `GasFlowStateBus`, the outlet flow
    from a mass balance (`GasFlowState.consume`): reactant consumed,
    product water added as vapor and/or liquid.
  - `HeatRelease` (W/m²), `CellVoltage` (V) — scalars, duplicating
    `CellState.heat_release`/`.cell_voltage` as standalone ports for
    convenience (e.g. wiring straight to a scope without a Bus Selector).
  - `x` — raw 6-element ODE state vector, `[T_mea; λ_1..λ_5]`, normalised
    the same way as `TransientModel.f_transient`.
- **Mask parameters**
  - `n_memb_mesh` — membrane finite-volume node count. Changing it requires
    re-running `build_transient_block` (which rebuilds the buses and the
    Bus Selector/Demux/Bus Creator wiring to match), not just editing the mask.
  - `cellBuilderExpr` — either a dotted path to a zero-argument Python
    function returning a `FuelCell` (default:
    `marapendi.interop.simulink_bridge.build_default_cell`, the same cell
    assembled in `examples/plot_01_polarization_curve.py`), or the name of a
    MATLAB function (no `.`, so `call_python_builder.m` can tell the two
    apart) returning a struct of cell parameters. For the latter, copy
    `cell_params_template.m` (e.g. to `my_cell_params.m`), edit the values,
    and set `cellBuilderExpr` to `'my_cell_params'` — no Python editing
    needed to change geometry/material parameters. See
    `default_cell_params()` in `simulink_bridge.py` for the full field list;
    the two are kept in sync by hand.
  - `x0` — initial ODE state. `build_transient_block.m` computes one for a
    nominal operating point automatically; override it for a different
    starting condition using
    `py.marapendi.interop.simulink_bridge.cell_initial_state(...)` from MATLAB.

Some `CellState` fields (e.g. `membrane_water_flux`, `membrane_proton_resistance`,
`an_cl_proton_resistance`) can show up as `NaN` at certain operating points —
that's not a marshalling bug, it means the corresponding Python `CellState`
attribute is `None` there (not every field is populated by every code path
through `_eval_state`).

## How it's wired

Level-2 MATLAB S-Functions cannot reliably declare bus-typed ports from
M-code (every documented-looking port property for this — `BusDataType`,
`BusOutputAsStruct`, `IsBus`, `Datatype` — turned out to be read-only in
R2025a), so `TransientPEMFC_core` (`transient_pemfc_sfun.m`) uses plain
vector ports instead, and the surrounding masked subsystem does the bus
conversion with standard blocks:

```
CaInlet, AnInlet (bus in)
    -> Bus Selector (7 scalar signals each, order = gasflow_field_order())
    -> Mux (1x7 vector each)
    -> TransientPEMFC_core (S-Function: 6 continuous states, plain vector I/O)
    -> Demux (27 scalars) + raw profile vector
    -> Bus Creator (-> CellStateBus)
    -> CellState (bus out)
TransientPEMFC_core -> Demux (7) -> Bus Creator (-> GasFlowStateBus) -> CaOutlet, AnOutlet (bus out)
TransientPEMFC_core -> HeatRelease, CellVoltage, x (plain ports, no bus needed)
```

`transient_pemfc_sfun.m` itself calls
`marapendi.interop.simulink_bridge.cell_derivative()`/`.cell_diagnostics()`
each step; it caches the `FuelCell` Python object handle (and the last
computed `CellState`) in a Dwork-indexed registry (`transient_pemfc_registry.m`)
since Dwork vectors can only hold numeric data, not a `py.object` or struct.

`Outputs` only calls `cell_diagnostics()` on **major time steps** —
Simulink's standard pattern for expensive output blocks
(`block.IsMajorTimeStep`). Minor time steps, which the variable-step solver
uses internally for error estimation/interpolation and which are not part
of the reported trajectory, reuse the last computed `CellState` from the
registry instead of paying another Python round trip. See the performance
numbers below.

## Files

- `simulink_bridge.py` (in `src/marapendi/interop/`) — flat scalar/list/dict
  adapter around `TransientModel`. This is what the S-Function actually
  calls; it is the single place the field list is defined.
  `cell_initial_state`/`cell_derivative`/`cell_diagnostics` build the
  equivalent `CellConditions` internally via `GasFlowState(...).to_side_conditions()`
  (`stoichiometry=0`, so the solver doesn't add a stoichiometric flow on top
  of the fully-specified inlet).
- `pyenv_setup.m` — points MATLAB's `pyenv()` at the repo's Python
  environment and puts `src/` on `sys.path`.
- `create_buses.m` — defines `GasFlowStateBus`, `CellStateBus`
  (`Simulink.Bus` objects) in the base workspace.
- `gasflow_field_order.m` / `state_scalar_field_order.m` — the fixed field
  orders shared between the bus definitions, the Bus Selector/Creator
  wiring, and the S-Function's vector ports. Change a field list in one
  place (and in `simulink_bridge.py`'s `GASFLOW_FIELDS`/`_state_to_dict`),
  not several.
- `transient_pemfc_sfun.m` — the Level-2 MATLAB S-Function core.
- `transient_pemfc_registry.m` — id -> `py.object`/cache store.
- `vec2flowdict.m`, `diagdict2flowvec.m`, `diagdict2scalarvec.m`,
  `pylist2mat.m`, `matstruct2pydict.m`, `call_python_builder.m` — marshalling
  helpers.
- `cell_params_template.m` — struct-shaped mirror of `default_cell_params()`
  in `simulink_bridge.py`; copy and edit to define a cell entirely from
  MATLAB, without touching Python. See `cellBuilderExpr` below.
- `build_transient_block.m` — programmatic model builder; running it
  produces `TransientPEMFC.slx`.
- `build_example_transient_pemfc.m` — builds `example_transient_pemfc.slx`,
  a runnable demo (step change in current density, scoped outputs) — see
  `docs/user_guide/simulink_block.rst`.

This relies on `marapendi.models.base.transient.TransientModel` populating
`state.ca.outlet_gas_flow_state`/`state.an.outlet_gas_flow_state` and
`state.heat_release` automatically — see `set_gas_flow_states` in
`src/marapendi/simulation/state.py` and `ThermalModel.temperature_rate_of_change`
in `src/marapendi/models/thermal.py`.

## One-time setup

1. Make sure the repo's Python environment has `marapendi`'s dependencies
   installed (numpy, scipy) and can `import marapendi` when `src/` is on
   `sys.path` (that's all `pyenv_setup.m` does — no `pip install` needed).
2. In MATLAB:
   ```matlab
   cd path/to/marapendi/matlab/transient_pemfc
   pyenv_setup('/path/to/marapendi/.venv/bin/python');  % or omit the arg to use the current pyenv()
   ```
   MATLAB can only load one Python version per session — if a `pyenv()` was
   already loaded (in-process) with a different interpreter, restart MATLAB
   first. A "Python version X.Y is not supported" warning is non-fatal in
   recent MATLAB releases (tested against Python 3.13 on R2025a, officially
   unsupported but working) — only worry about it if `import marapendi`
   itself fails.
3. Build the model:
   ```matlab
   build_transient_block                                    % uses current pyenv()
   build_transient_block('/path/to/.venv/bin/python')       % or switch interpreter first
   ```
4. Try the runnable example — see `docs/user_guide/simulink_block.rst`.

## Known limitations

- **Not real-time, not parallel.** Every derivative evaluation, and every
  *major-time-step* output evaluation, round-trips into Python and holds
  the GIL. Rapid Accelerator, multicore, and Simulink Compiler/Coder
  targets are not supported — run in Normal or plain Accelerator mode.

  Measured on a 300 s constant-input scenario (`n_memb_mesh=5`,
  `ode15s`/`BDF`, both `rtol=1e-3`, MATLAB R2025a, Python 3.13, mean of 5
  warmed-up runs):

  | | time |
  |---|---|
  | `TransientModel.solve()` in Python | 69 ms (104 `f_transient` evaluations) |
  | `sim('TransientPEMFC')` in Simulink | 368 ms |

  So the block is roughly **5.4x slower wall-clock** than calling
  `TransientModel.solve()` directly, for a run that only takes a fraction of
  a second either way. This is **not** the solver being intrinsically more
  expensive (`Derivatives` was called **88** times, the same order as
  Python's 104 `f_transient` evaluations), and **not** primarily MATLAB↔Python
  marshalling overhead — it's call *count*: `Outputs` still calls into
  Python **71** times (major steps only — the standard Simulink fix for
  expensive output blocks via `block.IsMajorTimeStep`, cutting what would
  otherwise be ~206 unconditional calls), on top of which
  `TransientModel.solve()` computes the equivalent diagnostics **once**,
  vectorised, at the very end. Total Python round trips: **159** in
  Simulink (88 `Derivatives` + 71 `Outputs`) vs. **105** in Python (104
  `f_transient` + 1 vectorised `evaluate()`) — expect cost to scale with
  major-step count, not internal solver-stage count.

  **A tempting further optimization that doesn't pay off:** `f_transient`
  has a `return_state=True` option that returns the full `CellState`
  computed internally alongside `dxdt`, so `Derivatives` and `Outputs`
  could in principle share one physics pass instead of two. Tried and
  measured: caching that state and reusing it in `Outputs` when `(t, x)`
  matches the last `Derivatives` call gave **0** cache hits against
  `ode15s` — `t` always matches exactly, but `x` doesn't (off by ~1e-8 to
  ~5e-3), because the solver's last `Derivatives` call is at the
  corrector's final Newton iterate, not the converged state `Outputs`
  reports. `return_state` is kept on `TransientModel.f_transient` as a
  reusable capability (e.g. for a fixed-step or custom integration harness
  where the caller controls `(t, x)` alignment), but the S-Function calls
  plain `cell_derivative()`/`cell_diagnostics()`, since caching bought no
  round-trip reduction here.
- **Fixed mesh size at build time.** `n_memb_mesh` sizes the S-Function's
  continuous-state vector and the Bus Selector/Demux/Bus Creator wiring, so
  changing it means re-running `build_transient_block`, not just editing
  the mask.
- **Cell parameters are not a runtime signal.** `FuelCell` geometry/materials
  are built once (`Start` callback) from `cellBuilderExpr` and reused for
  every step, matching how `TransientModel` treats `cell` in Python. To sweep
  cell parameters, point `cellBuilderExpr` at a different builder — either a
  MATLAB struct function (copy `cell_params_template.m`) or a Python one —
  and rebuild.

## Validating a change

Because the block calls the same `TransientModel` code path, a mismatch
would mean a marshalling/wiring bug, not a physics difference. To check:

1. In Python, run a reference scenario directly:
   ```python
   import marapendi as mrpd
   from marapendi.interop import simulink_bridge as sb

   cell = sb.build_default_cell()
   model = mrpd.TransientModel(n_memb_mesh=5)
   ca_flow = dict(temperature=344.15, pressure=140000., o2=1.05e-7, n2=3.90e-7, h2=0., h2o=3.25e-8, liquid=0.)
   an_flow = dict(temperature=344.15, pressure=190000., o2=0., n2=0., h2=3.6e-7, h2o=3.8e-8, liquid=0.)
   cond = sb._cell_conditions_from_flows(ca_flow, an_flow, 10000., 344.15)
   sol = model.solve(cell, cond, t_span=(0, 300), rtol=1e-3)
   print(sol.diagnostics.cell_voltage[-1], sol.diagnostics.mea_temperature[-1], sol.y[:, -1])
   ```
2. In Simulink, load `TransientPEMFC.slx`, replace the `CaInlet`/`AnInlet`
   inports with `Constant` blocks carrying the same operating point (MATLAB
   structs matching `ca_flow`/`an_flow` above, typed via
   `'OutDataTypeStr', 'Bus: GasFlowStateBus'`), and `CurrentDensity`/
   `CellTemperature` constants, set `Solver=ode15s`, `RelTol=1e-3`,
   `StopTime=300`, and run.
3. Compare `CellState.cell_voltage`, `CellState.mea_temperature`, and `x`
   at the final time step against the Python values — they should agree to
   solver tolerance.
