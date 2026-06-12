User Guide
==========

End-to-end examples showing how to assemble cell models, set initial
conditions, and run the steady-state solution:

- ``notebooks/01_cell_assembly_and_polarization_curve.ipynb`` — assembling a
  :class:`~marapendi.cell.Cell` and computing a steady-state polarization
  curve with :class:`~marapendi.model.ExplicitSteadyStateModel`.
- ``notebooks/02_custom_h2_permeation_model.ipynb`` — customizing a
  correlation model by subclassing (a temperature-independent hydrogen
  permeation model) and comparing it to the default.
- ``notebooks/03_quasi_steady_simulation_monocell.ipynb`` — turning a
  test-bench log into a sequence of quasi-steady operating points and
  comparing them to the steady-state model.
- ``notebooks/parameter_estimation_Affonso_Nobrega_et_al_2026_JES.ipynb`` —
  the full parameter-estimation and cross-validation workflow.
