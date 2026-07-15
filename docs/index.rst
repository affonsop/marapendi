.. title:: marapendi

.. image:: /_static/marapendi-full-bleu.png
   :width: 600px
   :align: center
   :alt: marapendi logo

**marapendi** is a Python framework for physics-based modelling of proton-exchange
membrane (PEM) and anion-exchange membrane (AEM) electrochemical cells. The current
release targets PEM fuel cells; AEM electrolyzer support is planned for a future version.

**marapendi** offers:

- The basis for the implementation of 0D (and up to 1D) physics-based models of
  PEM/AEM fuel cells and electrolyzers.
- Very fast steady-state and transient 0D cell models — low enough computational
  cost to make sensitivity analysis, parameter estimation and cross-validation
  practical (see :doc:`science/parameter_estimation`).
- An easy-to-use API for defining, calibrating and simulating cell models in a
  few lines of code (see :doc:`installation` for a runnable example).
- Pre-defined correlations and sub-models for heat transfer, reaction kinetics,
  two-phase transport, membrane water balance, ohmic losses, and more (see
  :doc:`science/index`).
- Straightforward extension to new models: sub-models are ordinary Python
  classes, so overriding one correlation or swapping a whole sub-model is a
  matter of subclassing (see :doc:`user_guide/extending_models`).
- Multiple runnable examples demonstrating **marapendi**'s capabilities (see
  :doc:`auto_examples/index`).
- Detailed documentation and openly readable, commented code, designed for
  transparency and easy understanding.
- A transient 0D model available as a MATLAB/Simulink S-function block (see
  :doc:`user_guide/simulink_block`).

Architecture
------------

**marapendi** is structured and written to make the implementation of models
and sub-models, and their use for simulation and parameter estimation, easy.
It separates the *description* of a cell (``components``) from the
*calculations* performed on it (``models``), and keeps the runtime *state* of
a simulation separate from both — every sub-model is an ordinary Python
class, meant to be subclassed rather than configured. See :doc:`architecture`
for the full breakdown of each subpackage.

Table of contents
-----------------
.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   architecture

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/index

.. toctree::
   :maxdepth: 2
   :caption: Science

   science/index

.. toctree::
   :maxdepth: 2
   :caption: Examples

   auto_examples/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
