Models
======

Stateless mathematical models for ionomer physics, membrane transport,
catalyst-layer electrokinetics, gas-phase transport, voltage, water
thermodynamics, transient dynamics, and degradation.

Each model class is a pure strategy object: it accepts component
dataclasses as arguments and returns computed quantities without storing
state.  Component dataclasses (see :doc:`components`) carry the physical
parameters; model classes carry the equations.

See the :doc:`../user_guide/polarization_curve` guide for a worked example.

.. automodule:: marapendi.models.model
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: marapendi.models.transient
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: marapendi.models.membrane
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: marapendi.models.catalyst_layer
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: marapendi.models.transport
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: marapendi.models.voltage
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: marapendi.models.water
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: marapendi.models.electrochemistry
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: marapendi.models.degradation
   :members:
   :undoc-members:
   :show-inheritance:
