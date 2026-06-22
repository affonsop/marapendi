# marapendi

**marapendi** is a Python framework for physics-based modelling of proton-exchange
membrane (PEM) and anion-exchange membrane (AEM) electrochemical cells, including
PEM fuel cells and AEM water electrolyzers.

The model is zero-dimensional (single operating point) and steady-state. It solves
for cell voltage as a function of current density, accounting for activation,
ohmic and mass-transport losses, membrane water balance, and two-phase liquid-water
transport in the porous layers.

## Features

- **PEM fuel cell** — polarization curves from first principles (Butler-Volmer kinetics,
  Nernst equation, membrane water transport, GDL/MPL liquid saturation)
- **Two steady-state model variants** — `ExplicitSteadyStateModel` (one forward pass, fast) and
  `ImplicitSteadyStateModel` (self-consistent MEA temperature via vectorised secant iteration);
  both accept full current-density arrays in a single call
- **AEM electrolyzer** — analogous model with AEM membrane (PAP family) and KOH electrolyte
- **Parameter estimation** — differential-evolution global optimizer wrapped in
  `SteadyStateModel` for fitting to experimental data
- **Clean component/state separation** — static cell parameters and runtime state are
  kept strictly separate, making the data flow explicit

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
    electrical_resistance=30e-7,
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
            effective_gas_diffusion_ratio=0.3, absolute_permeability=1e-12,
            thermal_conductivity=0.5, two_phase_transport_model=liq,
        ),
        ch=mrpd.FlowChannel(width=1e-3, height=1e-3, length=0.1, n_parallel=20, reactant='o2'),
        has_mpl=False, thermal_contact_resistance=4e-4,
    ),
    an=mrpd.FuelCellSide(
        cl=mrpd.PtCCatalystLayer(thickness=5e-6, two_phase_transport_model=liq),
        gdl=mrpd.GasDiffusionLayer(
            thickness=200e-6, porosity=0.6, contact_angle=120.,
            effective_gas_diffusion_ratio=0.3, absolute_permeability=1e-12,
            thermal_conductivity=0.5, two_phase_transport_model=liq,
        ),
        ch=mrpd.FlowChannel(width=1e-3, height=1e-3, length=0.1, n_parallel=20, reactant='h2'),
        has_mpl=False, thermal_contact_resistance=4e-4,
    ),
    membrane=mrpd.PFSA(
        ionomer=mrpd.PFSAIonomer(equivalent_weight=1100, dry_density=1980),
        dry_thickness=25e-6,
    ),
    use_eq_water_content_for_ionomer=True,
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
# state.cell_voltage — array of voltages (V)
# state.mea_temperature — array of MEA temperatures (K)

# --- Implicit model (self-consistent T_MEA, fully vectorised) ---
imp_model = mrpd.ImplicitSteadyStateModel()
state = imp_model.set_initial_conditions(cell, conditions)
state = imp_model.solve(cell, conditions, state)
```

## Package structure

```
src/marapendi/
├── cell/          # FuelCell, ElectrolyzerCell, and all physics models
│   ├── fuelcell.py             # FuelCell, FuelCellSide
│   ├── aem_electrolyzer.py     # ElectrolyzerCell, ElectrolyzerCellSide
│   ├── state.py                # CellState, CellSideState, LayerState, …
│   ├── explicit_steady_state.py # ExplicitSteadyStateModel
│   ├── implicit_steady_state.py # ImplicitSteadyStateModel (self-consistent T_MEA)
│   ├── voltage.py              # VoltageModel
│   ├── thermal.py              # ThermalModel
│   ├── gas_transport.py        # GasTransportModel
│   └── water_balance.py        # MembraneWaterBalanceModel
├── membrane/      # Membrane and ionomer materials
│   ├── ionomer_base.py         # Ionomer (abstract base class)
│   ├── pem.py                  # PFSAIonomer, PFSA, NafionD2020
│   ├── aem.py                  # PAPIonomer, AEM, PAP85
│   ├── membrane_base.py        # Membrane (composes an Ionomer instance)
│   └── permeation.py           # HydrogenPermeationModel
├── porous_layers/ # GasDiffusionLayer, MicroPorousLayer, CatalystLayer, …
├── channel/       # FlowChannel, ChannelGasResistanceModel, BakerChannelGasResistanceModel
├── thermo/        # GasState, GasModel, water properties, constants
├── electrolyte/   # ElectrolyteSolution, KOHSolution
├── degradation/   # PtDissolution, PlatinumOxideFormation
├── simulation/    # OperatingConditions, DynamicOperatingConditions, LoadCycle
└── estimation/    # SteadyStateModel, DynamicModel
```

## Example notebooks

| Notebook | Description |
|---|---|
| `notebooks/01_polarization_curve.ipynb` | Simulate a polarization curve, plot V–i and HFR |
| `notebooks/02_parameter_estimation.ipynb` | Fit kinetic parameters to data |
| `notebooks/03_quasi_steady_simulation_monocell.ipynb` | Replay a test-bench log sample-by-sample; compare simulated and measured cell voltage over time |

## Running the tests

```bash
pytest
```

## Documentation

Build the Sphinx docs from the `doc/` directory:

```bash
cd doc
make html
open _build/html/index.html
```

## Author

Pedro Affonso Nobrega, pedro.affonso_nobrega@minesparis.psl.eu

## License

MIT License

## Project status

In development
