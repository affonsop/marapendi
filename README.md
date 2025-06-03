# marapendi

marapendi is a framework for modelling anion and proton exchange membrane electrochemical cell devices, such as water electrolyzers and fuel cells. 

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