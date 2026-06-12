# marapendi

marapendi is a framework for modelling proton and anion exchange membrane electrochemical cell devices, such as fuel cells and water electrolyzers.

## Structure

- `marapendi.cell`, `marapendi.state` — dataclasses describing a cell's components (catalyst layers, porous layers, membrane, flow channels, ionomer) and the physical state at one operating point.
- `marapendi.model`, `marapendi.water_balance`, `marapendi.transport`, `marapendi.voltage`, `marapendi.thermal` — orchestration of the steady-state cell model.
- `marapendi.water`, `marapendi.electrochemistry`, `marapendi.membrane_permeation_models`, `marapendi.transport_models`, `marapendi.gas`, `marapendi.conditions`, `marapendi.constants` — underlying physics correlations.
- `marapendi.estimation` — parameter estimation and cross-validation against experimental polarization data.

The steady-state model implements the model described in Affonso Nobrega et al., *J. Electrochem. Soc.* 173, 114503 (2026).

`marapendi.dynamic` is a separate, transient-capable implementation inspired by Yang et al. (2019), currently under evaluation.

See the [documentation](docs/index.rst) for details.

## Installation

You can simply clone or download the template from https://git.persee.mines-paristech.fr/pedro.affonso_nobrega/marapedi.git and copy the files and folders in the project you want to start.


### Python environment creation
Ensure Conda is initiated in your shell.

To create the environment, run in the package directory:

```bash
$ conda env create -f ./ci/conda_env.yml
```

Then activate the environment: 

```bash
$ conda activate marapendi
```
### Editable installation 
To be able to modify marapendi your local marapendi version (developer mode), use an editable installation. In the package directory, run: 

```bash
$ pip install -e . 
```

## Getting started 

Example notebooks can be found in the `notebooks` folder. 
## Author

Pedro Affonso Nobrega, pedro.affonso_nobrega@minesparis.psl.eu

## License

MIT Licence

## Project status

In development