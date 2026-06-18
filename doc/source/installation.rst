.. _installation:

Installation
============

Prerequisites
-------------

* Python 3.10 or later
* Conda (Anaconda or Miniconda) **or** a standard Python virtual environment

From source (recommended)
--------------------------

Clone the repository::

    git clone <repository-url>
    cd marapendi

Create and activate the conda environment::

    conda env create -f ci/conda_env.yml
    conda activate marapendi

Install in editable (developer) mode::

    pip install -e .

Running the test suite
----------------------

From the repository root::

    pytest

All tests should pass.  The baseline polarization-curve tests
(``tests/test_polarization_curves_baseline.py``) are the reference for
numerical correctness and must not be modified.

Building the documentation
--------------------------

Additional dependencies: ``sphinx``, ``sphinx-napoleon``.

From the ``doc/`` folder::

    make html

Open ``doc/_build/html/index.html`` in a browser.
