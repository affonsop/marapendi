"""
CellBaseModel — BaseModel subclass for a single PEM/AEM cell simulation.

Classes
-------
CellBaseModel
    Owns all physics strategy objects as named fields and wires a
    ``TransientCellModel`` into ``submodels`` automatically.
"""

import numpy as np
from dataclasses import dataclass, field

from marapendi.models.model import BaseModel
from marapendi.models.transient import TransientCellModel
from marapendi.models.transport import PorousGasResistanceModel, DarcyTransportModel
from marapendi.models.membrane import MembraneModel
from marapendi.models.catalyst_layer import CatalystLayerModel
from marapendi.models.voltage import VoltageModel


@dataclass
class CellBaseModel(BaseModel):
    """
    ``BaseModel`` pre-configured for a single PEM/AEM cell simulation.

    Owns all physics strategy objects as named fields and wires the
    ``TransientCellModel`` into ``submodels`` automatically — no manual
    ``submodels`` or ``input_fns`` dicts required.

    ``BaseModel.__post_init__`` then:

    * registers ``transient_transport_model.get_inputs`` in ``input_fns``
      so the current density flows from the model into the ODE dispatcher;
    * injects ``self`` into ``transient_transport_model.base_model`` so
      the model can resolve physics objects.

    Parameters
    ----------
    transient_transport_model : TransientCellModel
        The ODE submodel.  Set ``current_density`` on it to control the
        applied current (scalar or ``f(t)`` callable).
    gas_diffusion_model : PorousGasResistanceModel
    darcy_transport_model : DarcyTransportModel
    memb_model : MembraneModel
    cl_model : CatalystLayerModel
    voltage_model : VoltageModel

    Example
    -------
    ::

        model = TransientCellModel(cell=cell, current_density=5000.)
        base = CellBaseModel(
            transient_transport_model=model,
            memb_model=PFSAModel(),
            cl_model=PtCCatalystLayerModel(),
        )
        y0  = base.initial_state(cell_temperature=353.15, ...)
        sol = solve_ivp(base.rates_of_change, t_span=(0, 100), y0=y0)
    """

    transient_transport_model: TransientCellModel = field(
        default_factory=TransientCellModel)
    gas_diffusion_model: PorousGasResistanceModel = field(
        default_factory=PorousGasResistanceModel)
    darcy_transport_model: DarcyTransportModel = field(
        default_factory=DarcyTransportModel)
    memb_model: MembraneModel = field(default_factory=MembraneModel)
    cl_model: CatalystLayerModel = field(default_factory=CatalystLayerModel)
    voltage_model: VoltageModel = field(default_factory=VoltageModel)

    def __post_init__(self):
        # Wire named field → submodels dict, then let BaseModel do the rest
        # (slice computation, get_inputs registration, base_model injection).
        self.submodels = {'cell': self.transient_transport_model}
        super().__post_init__()

    def initial_state(self, **kwargs) -> np.ndarray:
        """Build the initial state from flat keyword arguments.

        All kwargs are forwarded directly to
        ``TransientCellModel.initial_state`` — no nesting needed::

            base.initial_state(cell_temperature=353.15, ca_rh=0.7, ...)
        """
        return super().initial_state(cell=kwargs)
