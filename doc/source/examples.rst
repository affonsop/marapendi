.. _examples:

Examples
========

The ``notebooks/`` directory contains Jupyter notebooks that demonstrate the
main use cases:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Notebook
     - Description
   * - ``01_polarization_curve.ipynb``
     - Build a PEMFC, sweep current density, plot polarization curve and HFR.
       Includes a sensitivity study on inlet relative humidity.
   * - ``02_parameter_estimation.ipynb``
     - Fit kinetic parameters (exchange current density, charge-transfer
       coefficient) to synthetic polarization-curve data using
       ``SteadyStateModel`` and differential evolution.
   * - ``03_quasi_steady_simulation_monocell.ipynb``
     - Load a real test-bench time-series log, replay each sample through the
       steady-state model using the logged temperatures, pressures, relative
       humidities and stoichiometries, and compare simulated vs. measured cell
       voltage over time.

Running the notebooks
---------------------

Activate the marapendi conda environment and launch Jupyter::

    conda activate marapendi
    jupyter notebook notebooks/
