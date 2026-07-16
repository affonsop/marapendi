# marapendi

**marapendi** is a Python framework for physics-based modelling of proton-exchange
membrane (PEM) fuel cells.

**marapendi** offers:

- **A basis for building fuel cell and electrolyzer models** — pre-defined
  correlations and sub-models for heat transfer, reaction kinetics,
  two-phase transport, membrane water balance, ohmic losses, and more (see
  [`docs/science/`](docs/science)), with a focus on 0D PEM/AEM models.
  Sub-models are ordinary Python classes, so overriding a single correlation
  or swapping a whole sub-model is a matter of subclassing.
- **Fast, practical calibration, validation and simulation** — ready-to-use,
  fast steady-state and transient 0D cell models, low-cost enough to make
  sensitivity analysis, parameter estimation and cross-validation
  practical, through an easy-to-use API for defining, calibrating and
  simulating cell models in a few lines of code (see the
  [Quick start](#quick-start) below).
- **Open, readable and well-documented code** — multiple runnable examples
  demonstrating **marapendi**'s capabilities (see [Examples](#examples)
  below), backed by detailed documentation and openly readable, commented
  code designed for transparency and easy understanding.
- **Cross-platform support** — a transient 0D model available as a
  MATLAB/Simulink S-function block (see
  [MATLAB / Simulink](#matlab--simulink) below). Linking with the
  [VirtualFCS](https://github.com/Virtual-FCS/VirtualFCS) Modelica library is
  also planned for a future release.

## Repository
marapendi is developed on GitLab at
[git.persee.minesparis.psl.eu/matpro/marapendi](https://git.persee.minesparis.psl.eu/matpro/marapendi),
and mirrored to [github.com/affonsop/marapendi](https://github.com/affonsop/marapendi)
for external contributions (issues, pull requests). Releases are published to
[PyPI](https://pypi.org/project/marapendi/). 

## Documentation

The full documentation is available at:
[pages.persee.minesparis.psl.eu/matpro/marapendi](https://pages.persee.minesparis.psl.eu/matpro/marapendi).

## Author

Pedro Affonso Nobrega, pedro.affonso_nobrega@minesparis.psl.eu

## License

MIT License
