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
        equivalent_weight=1100, dry_density=1980, dry_thickness=25e-6,
        water_balance_model=mrpd.MembraneWaterBalanceModel(),
    ),
    use_eq_water_content_for_ionomer=True,
)

# --- Operating conditions ---
T = 353.15
ca_cond = mrpd.OperatingConditions(
    inlet_temperature=T, inlet_pressure=1.5e5, outlet_pressure=1.5e5,
    dry_o2_mole_fraction=0.21, inlet_relative_humidity=0.5, stoichiometry=2.0,
)
an_cond = mrpd.OperatingConditions(
    inlet_temperature=T, inlet_pressure=1.5e5, outlet_pressure=1.5e5,
    dry_h2_mole_fraction=1.0, inlet_relative_humidity=0.5, stoichiometry=1.5,
)

# --- Polarization curve ---
i = np.linspace(1e3, 2e4, 20)   # A/m²
V = cell.compute_ui_curve(i, T, ca_cond, an_cond)
```

## Package structure

```
src/marapendi/
├── components/        # Cell components (static parameters only)
│   ├── cell/          # FuelCell, FuelCellSide, ElectrolyzerCell
│   ├── channel/       # FlowChannel
│   ├── membrane/      # PFSA, PAP85, PFSAIonomer, PAPIonomer
│   └── porous/        # GasDiffusionLayer, MicroPorousLayer, PtCCatalystLayer
├── models/            # Physics (stateless, take component + state arguments)
│   ├── cell/          # ExplicitSteadyStateModel, VoltageModel, ThermalModel, ...
│   ├── porous/        # PorousGasResistanceModel, DarcyTransportModel
│   ├── channel/       # ChannelGasResistanceModel
│   ├── gas.py         # GasState, GasModel
│   ├── water.py       # Water property correlations
│   └── electrochemistry.py  # ElectrochemicalReaction, reversible voltage
├── simulation/        # State containers (runtime values)
│   └── state.py       # CellState, CellSideState, LayerState, ...
└── estimation/        # Parameter estimation
    └── estimation.py  # SteadyStateModel, DynamicModel
```

Key design principle: **components** hold static parameters; **state** holds runtime
values; **models** contain the physics and always take `(component, state)` arguments.

## Example notebooks

| Notebook | Description |
|---|---|
| `notebooks/01_polarization_curve.ipynb` | Simulate a polarization curve, plot V–i and HFR |
| `notebooks/02_parameter_estimation.ipynb` | Fit kinetic parameters to data |

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
