
"""
Module providing a Darcy transport model.
"""
from dataclasses import dataclass
import numpy as np
from ..thermo.constants import GAS_CONSTANT

@dataclass
class DarcyTransportModel:
    """
    Model for calculating non-wetting phase transport in porous layers using a Darcy-based approach.

    Attributes
    ----------
    J_function_exponent : float
        Exponent in the non-wetting phase capillary pressure-saturation relation.
    """
    J_function_exponent: float = 2

    def saturation_from_capillary_pressure(self, layer_or_state, capillary_pressure):
        """Compute the non-wetting saturation from capillary pressure using the J-function relation.

        Parameters
        ----------
        layer_or_state : PorousLayer or LayerState
            Object providing ``breakthrough_pressure`` (Pa).
        capillary_pressure : float
            Capillary pressure (Pa).

        Returns
        -------
        float
            Non-wetting saturation (0–1).
        """
        return np.minimum(
            (capillary_pressure / layer_or_state.breakthrough_pressure) ** (1. / self.J_function_exponent),
            1,
        )

    def capillary_pressure_from_saturation(self, layer_or_state, non_wetting_saturation):
        """Compute the capillary pressure from non-wetting saturation using the inverse J-function.

        Parameters
        ----------
        layer_or_state : PorousLayer or LayerState
            Object providing ``breakthrough_pressure`` (Pa).
        non_wetting_saturation : float
            Non-wetting saturation level (0–1).

        Returns
        -------
        float
            Capillary pressure (Pa).
        """
        return (non_wetting_saturation ** self.J_function_exponent) * layer_or_state.breakthrough_pressure

    def calculate_non_wetting_saturation(self, layer, layer_state, upstream_capillary_pressure=0, mask=None):
        """
        Calculate the non-wetting saturation distribution across the porous layer.

        Parameters
        ----------
        layer : PorousLayer
            Porous layer being analyzed (provides static geometry and permeability).
        layer_state : LayerState
            Runtime state for this layer; reads ``non_wetting_flux``,
            ``saturation_flow_resistance``, and ``breakthrough_pressure``;
            writes ``upstream_saturation``, ``downstream_saturation``,
            ``non_wetting_saturation``, and ``downstream_capillary_pressure``.
        upstream_capillary_pressure : array-like, optional
            Capillary pressure at the upstream boundary (Pa, default is 0).
        mask : array-like of bool, optional
            Boolean mask selecting which elements to update; all elements updated when ``None``.
        """
        if mask is None:
            mask = np.ones_like(layer_state.non_wetting_flux, dtype=bool)

        q = layer.relative_permeability_exponent
        n = self.J_function_exponent
        exponent = 1.0 / (q + n)

        us = self.saturation_from_capillary_pressure(
            layer_state, upstream_capillary_pressure[mask]
        )

        flux = np.maximum(0.0, layer_state.non_wetting_flux[mask])

        ds = (layer_state.saturation_flow_resistance * flux * (q + n) / n) ** exponent

        s_down = np.clip(us + ds, 0.0, 0.9)

        s_avg = (s_down - us) * ((q + n) / (q + n + 1)) + us
        s_avg = np.clip(s_avg, 0.0, 0.9)

        cp_down = self.capillary_pressure_from_saturation(layer_state, s_down)

        layer_state.upstream_saturation[mask] = us
        layer_state.downstream_saturation[mask] = s_down
        layer_state.non_wetting_saturation[mask] = s_avg
        layer_state.downstream_capillary_pressure[mask] = cp_down

    def calculate_equivalent_flow_resistance(self, cell_side):
        """Compute flow resistance for each porous layer on *cell_side* and return the total.

        Parameters
        ----------
        cell_side : FuelCellSide
            Cell side whose ``porous_layers`` list is iterated.

        Returns
        -------
        float
            Sum of ``equivalent_flow_resistance`` across all porous layers.
        """
        total = 0.0
        for layer in cell_side.porous_layers:
            layer.equivalent_flow_resistance = layer.saturation_flow_resistance
            total = total + layer.equivalent_flow_resistance
        return total
