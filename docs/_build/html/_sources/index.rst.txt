marapendi
=========

**marapendi** is a Python framework for modelling anion and proton exchange
membrane (AEM/PEM) electrochemical cell devices, including water electrolyzers
and fuel cells.

Philosophy
----------
The philosophy of **marapendi** is that `components` store physical (measureable) 
properties of cell components (*e.g.* membranes, catalyst layers, diffusion layers, flow fields, etc.). 
That can include parameters of well-established correlations (*e.g.* for membrane conductivity).  

`models` store all kind of sub-models for calculating needed quantities and physical variables. Those are
grouped under a `Model` class. Sub-models typically include voltage models, membrane models, gas and liquid
transport models, heat transfer models and so on. Sub-models are defined as classes which can be inherited 
to make user-defined models. 

`estimation` contains a number of methods allowing for sensitivity analysis and parameter estimation. 

`materials` is a database of pre-defined common materials. 
 

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/components
   api/models
   api/materials
   api/simulation
   api/estimation
   api/tools

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
