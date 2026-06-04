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
from marapendi.models.gas import GasMixtureModel
from marapendi.models.catalyst_layer import CatalystLayerModel
from marapendi.models.voltage import VoltageModel
from marapendi.models.thermal import ThermalModel


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
    gas_model: GasMixtureModel = field(default_factory=GasMixtureModel)
    cl_model: CatalystLayerModel = field(default_factory=CatalystLayerModel)
    voltage_model: VoltageModel = field(default_factory=VoltageModel)
    thermal_model: ThermalModel = field(default_factory=ThermalModel)

    def __post_init__(self):
        # Wire named field → submodels dict, then let BaseModel do the rest
        # (slice computation, get_inputs registration, base_model injection).
        self.submodels = {'transient_transport': self.transient_transport_model}
        super().__post_init__()

    def initial_state(self, **kwargs) -> np.ndarray:
        """Build the initial state from flat keyword arguments.

        All kwargs are forwarded directly to
        ``TransientCellModel.initial_state`` — no nesting needed::

            base.initial_state(cell_temperature=353.15, ca_rh=0.7, ...)
        """
        return super().initial_state(transient_transport=kwargs)

    def postprocess(self, y_flat: np.ndarray, i_density: float = None):
        """Post-process a state array into a populated ``CellState``.

        Unpacks and un-normalises the state, then calls
        ``_compute_derived_quantities`` and ``_compute_voltage`` to fill in
        all thermodynamic and electrochemical fields.

        Parameters
        ----------
        y_flat : np.ndarray, shape (n_states,) or (n_states, m)
            Normalised state vector or time-series matrix (e.g. ``sol.y``).
        i_density : float, optional
            Applied current density [A m⁻²] for voltage computation.
            Defaults to the model's current ``current_density`` value
            (evaluated at *t* = 0 for callable current profiles).

        Returns
        -------
        CellState
            Snapshot with ``V_cell``, ``eta_act``, ``eta_ohm``, species
            concentrations, temperatures, and saturation fields populated.
        """
        model = self.transient_transport_model
        if i_density is None:
            i_density = model.get_inputs(0.)['i']
        cell_y = self.split_state(y_flat)['transient_transport']
        x = (
            cell_y.reshape(model.n_layers, model.n_variables, -1)
            * model.norm_factor[..., np.newaxis]
        )
        state = model._compute_derived_quantities(x, i_density)
        self.gas_model.compute_state(state, model.cell, model, self.gas_diffusion_model)
        self.cl_model.compute_local_o2_partial_pressure(state, model.cell, self.memb_model)
        self.voltage_model.compute_cell_voltage(state, model.cell, self.memb_model, self.cl_model)

        cell = model.cell
        m_dim = x.shape[-1]
        state.C   = np.ones((model.n_layers, model.n_variables, m_dim))
        state.R   = np.full((model.n_layers, model.n_variables, m_dim), np.inf)
        state.S   = np.zeros((model.n_layers, model.n_variables, m_dim))
        state.phi = x.copy()
        state.J   = np.zeros((model.n_layers + 1, model.n_variables, m_dim))
        state.J_des = None
        state.S_lv  = None

        self.memb_model.update_transport_matrices(state, cell, model)
        self.darcy_transport_model.update_transport_matrices(state, cell, model)
        self.gas_diffusion_model.update_transport_matrices(state, cell, model, self.gas_model)
        self.thermal_model.update_transport_matrices(state, cell, model, self.memb_model)

        eff_R = (state.R[:-1] + state.R[1:]) / 2
        eff_R[-1, model.i_T] += cell.thermal_resistance / 2
        eff_R[ 0, model.i_T] += cell.thermal_resistance / 2
        np.nan_to_num(state.R, copy=False, nan=np.inf, posinf=np.inf)
        np.nan_to_num(eff_R,   copy=False, nan=np.inf, posinf=np.inf)
        state.J[1:-1] = -(state.phi[1:] - state.phi[:-1]) / eff_R
        state.eff_R   = eff_R
        self.memb_model.add_eod_flux(state, cell, model)
        return state
