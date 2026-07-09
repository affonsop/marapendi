# TransientPEMFC Simulink block

Wraps `marapendi.models.base.transient.TransientModel` as a Simulink block by
calling the live Python model from a MATLAB S-Function (via MATLAB's `py.`
interface) — no physics is re-implemented in MATLAB. Editing the model?
Edit the Python source under `src/marapendi/`; the block always calls
whatever is currently on disk.

Built and validated end-to-end against the Python model in MATLAB R2025a
(macOS): a 300 s step-current transient run through the generated
`TransientPEMFC.slx` matches `TransientModel.solve()` run directly in Python
to ~6 significant figures (the residual difference is expected solver
noise — Simulink's `ode15s` vs. SciPy's `BDF`, both at `rtol=1e-3`).

## How it's wired

Level-2 MATLAB S-Functions cannot reliably declare bus-typed ports from
M-code (every documented-looking port property for this — `BusDataType`,
`BusOutputAsStruct`, `IsBus`, `Datatype` — turned out to be read-only in
R2025a), so `TransientPEMFC_core` (`transient_pemfc_sfun.m`) uses plain
vector ports instead, and the surrounding masked subsystem does the bus
conversion with standard blocks:

```
CellConditions (bus in)
    -> Bus Selector (24 scalar signals, order = cond_field_order())
    -> Mux (1x24 vector)
    -> TransientPEMFC_core (S-Function: 6 continuous states, plain vector I/O)
    -> Demux (26 scalars, order = state_scalar_field_order()) + raw profile vector
    -> Bus Creator (-> CellStateBus)
    -> CellState (bus out)
TransientPEMFC_core -> x (raw ODE state vector, plain port, no bus needed)
```

`transient_pemfc_sfun.m` itself calls
`marapendi.interop.simulink_bridge.derivative()`/`.diagnostics()` each step;
it caches the `FuelCell` Python object handle (and the last computed
`CellState`) in a Dwork-indexed registry (`transient_pemfc_registry.m`)
since Dwork vectors can only hold numeric data, not a `py.object` or struct.

`Outputs` only calls `diagnostics()` on **major time steps** — Simulink's
standard pattern for expensive output blocks
(`block.IsMajorTimeStep`). Minor time steps, which the variable-step solver
uses internally for error estimation/interpolation and which are not part
of the reported trajectory, reuse the last computed `CellState` from the
registry instead of paying another Python round trip. See the performance
numbers below.

## Files

- `simulink_bridge.py` (in `src/marapendi/interop/`) — flat scalar/list/dict
  adapter around `TransientModel`. This is what the S-Function actually
  calls; it is the single place the field list is defined.
- `pyenv_setup.m` — points MATLAB's `pyenv()` at the repo's Python
  environment and puts `src/` on `sys.path`.
- `create_buses.m` — defines `SideConditionsBus`, `CellConditionsBus`,
  `CellStateBus` (`Simulink.Bus` objects) in the base workspace.
- `cond_field_order.m` / `state_scalar_field_order.m` — the fixed field
  orders shared between the bus definitions, the Bus Selector/Creator
  wiring, and the S-Function's vector ports. Change a field list in one
  place (and in `simulink_bridge.py`), not several.
- `transient_pemfc_sfun.m` — the Level-2 MATLAB S-Function core.
- `transient_pemfc_registry.m` — id -> `py.object` cache (see above).
- `vec2conddict.m`, `diagdict2scalarvec.m`, `pylist2mat.m`,
  `busconditions2dict.m`, `call_python_builder.m` — marshalling helpers.
- `build_transient_block.m` — programmatic model builder; running it
  produces `TransientPEMFC.slx`.

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
   This creates `TransientPEMFC.slx` next to this README, containing a single
   masked `TransientPEMFC` block with:
   - **Input**: `CellConditions` (bus, type `CellConditionsBus`)
   - **Outputs**: `CellState` (bus, type `CellStateBus`, flattened
     diagnostics — voltage, HFR, membrane water-content profile, catalyst
     layer water content/saturation/proton resistance, water fluxes, …) and
     `x` (raw 6-element ODE state vector, `[T_mea; λ_1..λ_5]`, normalised the
     same way as `TransientModel.f_transient`).
   - **Mask parameters**: `n_memb_mesh` (must match `create_buses`'s value —
     changing it requires rebuilding the buses and block), `cellBuilderExpr`
     (dotted path to a zero-arg Python function returning a `FuelCell`;
     default `marapendi.interop.simulink_bridge.build_default_cell`, the same
     cell as `examples/plot_01_polarization_curve.py`), `x0` (initial ODE
     state — recomputed for a nominal operating point by
     `build_transient_block.m`; override for a different starting condition
     using `py.marapendi.interop.simulink_bridge.initial_state(...)`).

Some `CellState` fields (e.g. `membrane_water_flux`, `membrane_proton_resistance`,
`an_cl_proton_resistance`) show up as `NaN` at certain operating points —
that's not a marshalling bug, it means the corresponding Python
`CellState` attribute is `None` there (not every field is populated by every
code path through `_eval_state`).

## Known limitations

- **Not real-time, not parallel.** Every derivative and output evaluation
  round-trips into Python and holds the GIL. Rapid Accelerator, multicore,
  and Simulink Compiler/Coder targets are not supported — run in Normal or
  plain Accelerator mode.

  Measured on the 300 s step-current scenario from the validation section
  below (`n_memb_mesh=5`, `ode15s`/`BDF`, both `rtol=1e-3`, MATLAB R2025a,
  Python 3.13, mean of 5 warmed-up runs):

  | | time |
  |---|---|
  | `TransientModel.solve()` in Python | 96 ms (137 `f_transient` evaluations) |
  | `sim('TransientPEMFC')` in Simulink | 326 ms |
  | — of which `InitFcn` (`pyenv_setup`+`create_buses`, once per `sim()` call) | 52 ms |

  So the block is roughly **3.4x slower wall-clock** than calling
  `TransientModel.solve()` directly, for a run that only takes ~0.1-0.3 s
  either way. This is **not** the solver being intrinsically more expensive,
  and **not** MATLAB↔Python marshalling overhead — both are minor
  contributors:

  - Solver cost is comparable: instrumenting the S-Function to count calls
    over the same run gives **128** `Derivatives` evaluations for
    `ode15s`/`RelTol=1e-3`, close to Python `BDF`'s **137** `f_transient`
    evaluations at the same `rtol`.
  - Marshalling overhead per call is modest: a single `derivative()` round
    trip via MATLAB's `py.` interface takes ~0.60 ms, against ~0.52 ms for
    the equivalent bare `model.f_transient(...)` call in Python — about 15%.

  The dominant cost is call *count*, specifically from `Outputs`, and the
  block already applies the standard fix for it: `TransientModel.solve()`
  calls the full diagnostics pipeline (`evaluate()`) **once**, vectorised,
  over the accepted output points, while a naive Simulink S-Function would
  call the block's `Outputs` method — which round-trips into Python and
  runs that same full pipeline — at every major *and minor* solver step.
  `Outputs` here checks `block.IsMajorTimeStep` and reuses the cached
  `CellState` on minor steps (used internally for error
  estimation/interpolation, not part of the reported trajectory) instead of
  calling Python again. That cuts `Outputs`' Python round trips from **206**
  (every call) to **79** (major steps only) on this scenario — total Python
  round trips **207** (128 `Derivatives` + 79 `Outputs`) vs. **138** in
  Python (137 `f_transient` + 1 vectorised `evaluate()`), and wall time from
  452 ms down to 326 ms. The remaining ~1.5x call-count gap (207 vs. 138) is
  inherent to the architecture — Simulink evaluates `Outputs` once per major
  step in addition to `Derivatives`, where `TransientModel.solve()` defers
  all diagnostics to one vectorised pass at the end — so expect cost to keep
  scaling with major-step count, not internal solver-stage count.

  **A tempting further idea that doesn't pay off here:** `f_transient` has a
  `return_state=True` option (and `simulink_bridge.step()` exposes it) that
  returns the full `CellState` computed internally alongside `dxdt`, so
  `Derivatives` and `Outputs` could in principle share one physics pass
  instead of two. It was tried and measured: caching that state and reusing
  it in `Outputs` when `(t, x)` matches the last `Derivatives` call gave
  **0/79 cache hits** against `ode15s` on this scenario. Instrumenting
  showed why — `t` always matches exactly, but `x` almost never does (the
  gap ranged ~1e-8 to ~5e-3), because the solver's last `Derivatives` call
  is at the corrector's final Newton iterate, not the converged state
  `Outputs` reports. `return_state`/`step()` are kept in the model and
  bridge as a reusable capability (e.g. for a fixed-step or custom
  integration harness where the caller controls exactly when `(t, x)`
  repeats), but the S-Function itself calls plain `derivative()`/
  `diagnostics()`, since caching bought no round-trip reduction here and
  would only add complexity.
- **Fixed mesh size at build time.** `n_memb_mesh` sizes the S-Function's
  continuous-state vector and the Bus Selector/Demux/Bus Creator wiring, so
  changing it means re-running `build_transient_block` (which calls
  `create_buses` itself), not just editing the mask.
- **Cell parameters are not a runtime signal.** `FuelCell` geometry/materials
  are built once (`Start` callback) from `cellBuilderExpr` and reused for
  every step, matching how `TransientModel` treats `cell` in Python. To sweep
  cell parameters, point `cellBuilderExpr` at a different builder function
  and rebuild, or add a new mask parameter that mutates the cell object
  in `Start` before caching it.

## Validating against the Python model

Since the block calls the same `TransientModel` code path, a mismatch would
mean a marshalling/wiring bug, not a physics difference. To reproduce the
validation run:

1. In Python:
   ```python
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
   ```
2. In Simulink, load `TransientPEMFC.slx`, replace the `CellConditions`
   inport with a `Constant` block carrying the same operating point (a
   MATLAB struct matching `CellConditionsBus`'s shape, typed via
   `'OutDataTypeStr', 'Bus: CellConditionsBus'`), set
   `Solver=ode15s, RelTol=1e-3, StopTime=300`, and run.
3. Compare `CellState.cell_voltage`, `CellState.mea_temperature`, and `x`
   at the final time step against the Python values — they should agree to
   ~5-6 significant figures (solver-dependent residual, not a bug).
