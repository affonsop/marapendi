"""
marapendi: framework for modelling anion and proton exchange membrane
electrochemical cell devices, such as water electrolyzers and fuel cells.

The top-level package holds the reference cell model: static component
dataclasses (:mod:`marapendi.cell`, :mod:`marapendi.catalyst_layers`,
:mod:`marapendi.porous_layers`, :mod:`marapendi.membrane`, ...), their
per-simulation state (:mod:`marapendi.state`), and the orchestration models
(:mod:`marapendi.model`, :mod:`marapendi.water_balance`,
:mod:`marapendi.transport`, :mod:`marapendi.voltage`,
:mod:`marapendi.thermal`).

:mod:`marapendi.dynamic` holds an independent, transient-capable cell model
with a partially overlapping (and not yet unified) set of components and
correlations.
"""
