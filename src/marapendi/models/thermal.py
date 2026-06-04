"""
ThermalModel — heat transport strip for the TransientCellModel matrix system.

Classes
-------
ThermalModel
    Fills state.R[:,i_T], state.C[:,i_T], state.S[:,i_T].
    Must be called after MembraneModel (needs J_des) and
    DarcyTransportModel (needs S_lv).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import cantera as ct

from marapendi.models.electrochemistry import enthalpy_condensation, std_formation_entropy_h2ol

if TYPE_CHECKING:
    from marapendi.models.transient import TransientCellModel
    from marapendi.components.cell_state import CellState

__all__ = ['ThermalModel']


@dataclass
class ThermalModel:
    """
    Thermal transport model for a PEM/AEM cell layer stack.

    Fills the temperature (i_T) strip of the C, R, S transport matrices.
    Stateless strategy object — geometry and materials come from the Cell.

    Parameters passed to update_transport_matrices
    -----------------------------------------------
    memb_model :
        Ionomer model providing ``heat_of_adsorption`` for the
        desorption enthalpy correction.
    """

    def update_transport_matrices(
        self,
        state: CellState,
        cell,
        tm: TransientCellModel,
        memb_model,
    ) -> None:
        """
        Fill state.R[:,i_T], state.C[:,i_T], state.S[:,i_T] in-place.

        Reads state.S_lv (set by DarcyTransportModel) and state.J_des
        (set by MembraneModel); must be called after both.

        Parameters
        ----------
        state : CellState
        cell : Cell
        tm : TransientCellModel
            Provides variable indices (i_T) and domain masks.
        memb_model :
            Ionomer model with ``heat_of_adsorption`` method.
        """
        i_T = tm.i_T
        i   = state.iF * ct.faraday

        # --- Resistance ---
        state.R[:, i_T, ...] = cell.thickness / cell.bulk_thermal_conductivity
        for ch in (cell.ca.ch, cell.an.ch):
            state.R[ch.ix, i_T, ...] = 2 * ch.height / ch.bulk_thermal_conductivity

        # --- Capacity ---
        state.C[:, i_T, ...] = cell.bulk_density * cell.bulk_specific_heat_capacity

        # --- Sources: ohmic / activation / GDL losses ---
        S_T_losses = np.zeros_like(state.T)
        S_T_losses[cell.memb.ix,   ...] = state.eta_memb * i
        S_T_losses[cell.ca.cl.ix,  ...] = (
            state.eta_act - std_formation_entropy_h2ol / (2 * ct.faraday)
        ) * i
        S_T_losses[cell.ca.gdl.ix, ...] = state.eta_gdl * i
        S_T_losses[cell.an.gdl.ix, ...] = state.eta_gdl * i

        h_vl = enthalpy_condensation(state.T)
        state.S[:, i_T, ...] += S_T_losses / cell.thickness + state.S_lv * h_vl

        # --- Sources: water desorption enthalpy ---
        H_ad = memb_model.heat_of_adsorption(state.T, cell.memb)
        for side in (cell.an, cell.ca):
            state.S[side.cl.ix, i_T, ...] -= (
                state.J_des[side.cl.ix, ...] / side.cl.thickness * H_ad[side.cl.ix, ...]
            )
