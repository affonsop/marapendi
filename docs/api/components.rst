Components
==========

Physical building blocks of an electrochemical cell, and the per-simulation
state that mirrors them.

Component classes are dataclasses that hold geometric and material
parameters, together with the correlation methods that turn those
parameters (and a state object) into transport and electrochemical
quantities. Computation across components is orchestrated by
:class:`~marapendi.model.CellModel` (see :doc:`models`).

.. automodule:: marapendi.cell
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: marapendi.state
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: marapendi.catalyst_layers
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: marapendi.porous_layers
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: marapendi.membrane
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: marapendi.flow_channels
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: marapendi.ionomer
   :members:
   :undoc-members:
   :show-inheritance:
