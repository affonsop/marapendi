# marapendi

**marapendi** is a Python framework for physics-based modelling of proton-exchange
membrane (PEM) fuel cells.

The model is zero-dimensional (single operating point). Both steady-state and
transient formulations are provided. The steady-state solver computes cell voltage
as a function of current density, accounting for activation, ohmic and
mass-transport losses, membrane water balance, and two-phase liquid-water transport
in the porous layers. The transient solver integrates coupled ODEs for the MEA
temperature and membrane water-content profile under time-varying operating
conditions.

**marapendi** offers:

- The basis for the implementation of 0D (and up to 1D) physics-based models of
  PEM/AEM fuel cells and electrolyzers.
- Very fast steady-state and transient 0D cell models — low enough computational
  cost to make sensitivity analysis, parameter estimation and cross-validation
  practical.
- An easy-to-use API for defining, calibrating and simulating cell models in a
  few lines of code (see the [Quick start](#quick-start) below).
- Pre-defined correlations and sub-models for heat transfer, reaction kinetics,
  two-phase transport, membrane water balance, ohmic losses, and more (see
  [`docs/science/`](docs/science)).
- Straightforward extension to new models: sub-models are ordinary Python
  classes, so overriding one correlation or swapping a whole sub-model is a
  matter of subclassing.
- Multiple runnable examples demonstrating **marapendi**'s capabilities (see
  [Examples](#examples) below).
- Detailed documentation and openly readable, commented code, designed for
  transparency and easy understanding.
- A transient 0D model available as a MATLAB/Simulink S-function block (see
  [MATLAB / Simulink](#matlab--simulink) below).

## Features

- **PEM fuel cell** — polarization curves from first principles (Butler-Volmer kinetics,
  Nernst equation, membrane water transport, GDL/MPL liquid saturation)
- **Two steady-state model variants** — `ExplicitSteadyStateModel` (one forward pass, fast) and
  `ImplicitSteadyStateModel` (self-consistent MEA temperature via vectorised secant iteration);
  both accept full current-density arrays in a single call
- **Piecewise-linear membrane water balance** — `MembraneWaterBalanceModelPiecewise` fits
  the PFSA equilibrium isotherm RH(λ) with a piecewise-linear regression; this is the
  default model used in the steady-state solvers
- **Transient model** — `TransientModel` integrates coupled MEA-temperature and membrane
  water-content ODEs (Ferrara et al., 2018) under time-varying conditions via
  `scipy.integrate.solve_ivp`; `solve()` auto-attaches a `diagnostics` `CellState`
  (voltage, HFR, water contents, saturation) at each internal time step, and `evaluate()`
  lets you re-sample any dense-output trajectory at arbitrary times
- **Parameter estimation** — `SteadyStatePolarizationCurveCalibration` fits kinetic and
  transport parameters to experimental polarization and HFR data using differential-evolution
  global optimisation, with k-fold cross-validation and automatic model-complexity selection
- **Clean component/state separation** — static cell parameters and runtime state are
  kept strictly separate, making the data flow explicit
- **MATLAB/Simulink block** — `TransientPEMFC` (`matlab/transient_pemfc/`) drives
  `TransientModel` from Simulink, calling the live Python source at every solver
  step; cell parameters can be supplied as a dotted Python builder path or a plain
  MATLAB struct (see `matlab/transient_pemfc/README.md`)

## Installation

Requires Python 3.10+.

### With conda (recommended)

```bash
conda env create -f ci/conda_env.yml
conda activate marapendi
pip install -e .
```

### With pip

```bash
pip install -e .
```

## Quick start

```python
import numpy as np
import marapendi as mrpd

# --- Build the cell ---
liq = mrpd.DarcyTransportModel(J_function_exponent=2)

cell = mrpd.FuelCell(
    area=25e-4,
    electric_resistance=30e-7,
    ca=mrpd.FuelCellSide(
        cl=mrpd.PtCCatalystLayer(
            ecsa=70e3, platinum_loading=0.4e-2, ionomer=mrpd.PFSAIonomer(),
            reaction=mrpd.ElectrochemicalReaction(
                reference_exchange_current_density=2.5e-4,
                reaction_order=0.54, activation_energy=67e6,
                reference_activity=1e5, reference_temperature=353.15,
                number_of_electrons=2, charge_transfer_coeff=0.5,
            ),
            thickness=10e-6, two_phase_transport_model=liq,
        ),
        gdl=mrpd.GasDiffusionLayer(
            thickness=200e-6, porosity=0.6, contact_angle=120.,
            tortuosity=2.0, absolute_permeability=1e-12,
            thermal_conductivity=0.5, two_phase_transport_model=liq,
        ),
        ch=mrpd.FlowChannel(width=1e-3, height=1e-3, length=0.1, n_parallel=20, reactant='o2'),
        thermal_contact_resistance=4e-4,
    ),
    an=mrpd.FuelCellSide(
        cl=mrpd.PtCCatalystLayer(thickness=5e-6, two_phase_transport_model=liq),
        gdl=mrpd.GasDiffusionLayer(
            thickness=200e-6, porosity=0.6, contact_angle=120.,
            tortuosity=2.0, absolute_permeability=1e-12,
            thermal_conductivity=0.5, two_phase_transport_model=liq,
        ),
        ch=mrpd.FlowChannel(width=1e-3, height=1e-3, length=0.1, n_parallel=20, reactant='h2'),
        thermal_contact_resistance=4e-4,
    ),
    membrane=mrpd.PFSA(
        ionomer=mrpd.PFSAIonomer(equivalent_weight=1100, dry_density=1980),
        dry_thickness=25e-6,
    ),
)

# --- Operating conditions ---
T = 353.15
conditions = mrpd.CellConditions(
    current_density=np.linspace(1e3, 2e4, 20),   # A/m²
    cell_temperature=T,
    ca=mrpd.SideConditions(
        inlet_temperature=T, outlet_pressure=1.5e5,
        dry_o2_mole_fraction=0.21, inlet_relative_humidity=0.5, stoichiometry=2.0,
    ),
    an=mrpd.SideConditions(
        inlet_temperature=T, outlet_pressure=1.5e5,
        dry_h2_mole_fraction=1.0, inlet_relative_humidity=0.5, stoichiometry=1.5,
    ),
)

# --- Polarization curve (explicit model, one vectorised call) ---
model = mrpd.ExplicitSteadyStateModel()
state = model.set_initial_conditions(cell, conditions)
state = model.solve(cell, conditions, state)
# state.cell_voltage   — array of voltages (V)
# state.mea_temperature — array of MEA temperatures (K)

# --- Implicit model (self-consistent T_MEA, fully vectorised) ---
imp_model = mrpd.ImplicitSteadyStateModel()
state = imp_model.set_initial_conditions(cell, conditions)
state = imp_model.solve(cell, conditions, state)

# --- Transient model (time-varying conditions) ---
from marapendi.models.base.transient import TransientModel

cond_0 = mrpd.CellConditions(          # single operating point for initial conditions
    current_density=np.atleast_1d(1e4),
    cell_temperature=T,
    ca=mrpd.SideConditions(inlet_temperature=T, outlet_pressure=1.5e5,
                            dry_o2_mole_fraction=0.21, stoichiometry=2.0,
                            inlet_relative_humidity=0.5),
    an=mrpd.SideConditions(inlet_temperature=T, outlet_pressure=1.5e5,
                            dry_h2_mole_fraction=1.0, stoichiometry=1.5,
                            inlet_relative_humidity=0.5),
)

tr_model = TransientModel(n_memb_mesh=5)
sol = tr_model.solve(cell, cond_0, t_span=(0, 3600))
# sol.y[0]           → T_MEA(t)  [K]
# sol.y[1:]          → λ(ξ, t)   [mol H2O / mol site]
# sol.diagnostics    → CellState with .cell_voltage, .hfr, .membrane.water_content, …

# Pass a callable for time-varying conditions:
def conditions_t(t):
    i = 5e3 if t < 1800 else 2e4   # step change at 30 min
    return mrpd.CellConditions(current_density=np.atleast_1d(i), cell_temperature=T,
                                ca=cond_0.ca, an=cond_0.an)

sol = tr_model.solve(cell, conditions_t, t_span=(0, 3600))
```

## Package structure

```
src/marapendi/
├── cell/          # Component tree (no physics)
│   ├── cell.py                   # Cell, CellSide (base classes)
│   └── fuelcell.py                # FuelCell, FuelCellSide
├── simulation/    # Runtime state and operating conditions (pure data)
│   ├── state.py                   # CellState, CellSideState, LayerState, GasFlowState, …
│   ├── conditions.py              # CellConditions, SideConditions
│   └── load_cycles/               # LoadCycle, standardised ID-FAST/FC-DLC driving cycles
├── models/        # Physics: orchestration + sub-models
│   ├── base/
│   │   ├── explicit_steady_state.py # ExplicitSteadyStateModel
│   │   ├── implicit_steady_state.py # ImplicitSteadyStateModel (self-consistent T_MEA)
│   │   └── transient.py             # TransientModel (ODE for T_MEA + membrane water profile)
│   ├── water_balance/
│   │   ├── water_balance.py         # WaterBalanceModel (orchestration)
│   │   ├── membrane_pwl.py          # MembraneWaterBalanceModelPiecewise (default)
│   │   └── membrane.py              # MembraneWaterBalanceModel (Affonso Nobrega et al. 2026)
│   ├── thermo/                      # GasState, GasModel, water properties, constants
│   ├── thermal.py                   # ThermalModel
│   ├── voltage.py                   # VoltageModel
│   ├── darcy.py                     # DarcyTransportModel (two-phase liquid water)
│   ├── diffusion.py                 # PorousGasDiffusionModel (Fickian + Knudsen)
│   └── gas_transport_resistance.py  # GasTransportModel (channel-to-CL resistance network)
├── membrane/      # Membrane and ionomer materials
│   ├── ionomer_base.py            # Ionomer (abstract base class)
│   ├── pem.py                     # PFSAIonomer, PFSA, NafionD2020
│   ├── aem.py                     # PAPIonomer, AEM, PAP85
│   └── membrane_base.py           # Membrane (composes an Ionomer instance)
├── porous_layers/ # GasDiffusionLayer, MicroPorousLayer, PtCCatalystLayer, …
├── channel/       # FlowChannel, ChannelGasResistanceModel, BakerChannelGasResistanceModel
├── electrolyte/   # ElectrolyteSolution, KOHSolution
├── degradation/   # PtDissolution, PlatinumOxideFormation
├── estimation/    # BaseModelCalibration, SteadyStatePolarizationCurveCalibration, Parameter
└── interop/       # simulink_bridge.py — thin adapter for the MATLAB Simulink block
```

## Examples

Each example is a self-contained, runnable Python script under `examples/`,
rendered with its output plots at `docs/auto_examples/` when the docs are
built (see [Documentation](#documentation) below).

| Script | Description |
|---|---|
| [examples/plot_01_polarization_curve.py](examples/plot_01_polarization_curve.py) | Assemble a cell, simulate a polarization curve, plot V–i and HFR |
| [examples/plot_02_quasi_steady.py](examples/plot_02_quasi_steady.py) | Replay a test-bench log with a vectorised quasi-steady simulation; compare simulated and measured voltage over time |
| [examples/plot_03_implicit_vs_explicit.py](examples/plot_03_implicit_vs_explicit.py) | Compare explicit and implicit steady-state models on voltage, MEA temperature, and HFR |
| [examples/plot_04_transient.py](examples/plot_04_transient.py) | Transient simulation over an ID-FAST driving cycle; compare against the quasi-steady-state prediction |
| [examples/plot_05_multi_condition.py](examples/plot_05_multi_condition.py) | Simulate and compare polarization curves across multiple experimental conditions |
| [examples/plot_06_parameter_estimation.py](examples/plot_06_parameter_estimation.py) | Fit kinetic and transport parameters to multi-condition experimental data; k-fold cross-validation and complexity selection |
| [examples/plot_07_pwl_membrane.py](examples/plot_07_pwl_membrane.py) | Compare the piecewise-linear (standard) and paper-model (Affonso Nobrega et al. 2026) membrane water balance models |
| [examples/plot_08_water_balance.py](examples/plot_08_water_balance.py) | Water balance vs. cell temperature: theoretical water production against modelled inlet/outlet water flows |
| [examples/plot_09_membrane_correlations.py](examples/plot_09_membrane_correlations.py) | Plot membrane/ionomer correlations (conductivity, isotherm, permeability, diffusivity) vs. water content at several temperatures |

## MATLAB / Simulink

`matlab/transient_pemfc/` builds a `TransientPEMFC` Simulink block that calls
`TransientModel` directly through MATLAB's `py.` interface — no physics is
re-implemented in MATLAB, so the block always reflects whatever is on disk in
`src/marapendi/`. See `matlab/transient_pemfc/README.md` and the
[Simulink block guide](docs/user_guide/simulink_block.rst) for setup and usage.

## Running the tests

```bash
pytest
```

## Documentation

Build the Sphinx docs from the `docs/` directory:

```bash
cd docs
make html
open _build/html/index.html
```

Start at [`docs/installation.rst`](docs/installation.rst) for a runnable
quick-start snippet, [`docs/user_guide/`](docs/user_guide) for task-oriented
guides, and [`docs/science/`](docs/science) for the governing equations and
literature references behind every model.

## Reference

Pedro Affonso Nobrega, Christophe Morin, Anass Chabani, Mathias Herlé,
"A zero-dimensional PEM fuel cell model with self-consistent MEA temperature
and membrane water balance", *J. Electrochem. Soc.* **173**, 114503 (2026).

## Author

Pedro Affonso Nobrega, pedro.affonso_nobrega@minesparis.psl.eu

## License

MIT License
