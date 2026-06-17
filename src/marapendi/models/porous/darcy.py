
"""
Module providing a Darcy transport model.
"""
from dataclasses import dataclass
import numpy as np
from ..constants import GAS_CONSTANT

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

    def saturation_from_capillary_pressure(self, layer, capillary_pressure):
        """
        Compute the non-wetting saturation from capillary pressure using J-function relation.

        Parameters
        ----------
        layer : PorousLayer
            Porous layer for which to compute the value.
        capillary_pressure : float
            Capillary pressure (Pa).

        Returns
        -------
        float
            Estimated water saturation.
        """
        return np.minimum((capillary_pressure / layer.capillary_pressure_J_ratio) ** (1. / self.J_function_exponent),1)

    def capillary_pressure_from_saturation(self, layer, non_wetting_saturation):
        """
        Compute the capillary pressure from non-wetting phase saturation using inverse J-function relation.

        Parameters
        ----------
        layer : PorousLayer
            Porous layer for which to compute the value.
        water_saturation : float
            Water saturation level (0-1).

        Returns
        -------
        float
            Capillary pressure (Pa).
        """
        return (non_wetting_saturation ** self.J_function_exponent) * layer.capillary_pressure_J_ratio

    def calculate_non_wetting_saturation(self, layer, non_wetting_flux, upstream_capillary_pressure=0, mask=None): 
        """
        Calculate the non-wetting saturation distribution across the porous layer due to non-wetting phase flux.

        Parameters
        ----------
        layer : PorousLayer
            Porous layer being analyzed.
        non_wetting_flux : float
            Non-wetting phase molar flux (kmol/m²/s).
        upstream_capillary_pressure : float, optional
            Capillary pressure at the upstream boundary (default is 0).

        Updates
        -------
        layer.upstream_saturation : float
            Saturation at upstream side of the layer.
        layer.downstream_saturation : float
            Saturation at downstream side of the layer.
        layer.non_wetting_saturation : float
            Average or effective saturation in the layer.
        layer.downstream_capillary_pressure : float
            Capillary pressure at the downstream side of the layer.
        """
        
        if mask is None:
            mask = np.ones_like(non_wetting_flux, dtype=bool)

        q = layer.relative_permeability_exponent
        n = self.J_function_exponent
        exponent = 1.0 / (q + n)

        # ---- masked views (single extraction) ----
        us = self.saturation_from_capillary_pressure(
            layer, upstream_capillary_pressure[mask]
        )

        flux = np.maximum(0.0, non_wetting_flux[mask])

        # ---- downstream saturation ----
        ds = (layer.saturation_flow_resistance * flux * (q+n)/n) ** exponent

        s_down = np.clip(us + ds, 0.0, 0.9)

        # ---- average saturation ----
        s_avg = (s_down - us) * ((q + n) / (q + n + 1)) + us
        s_avg = np.clip(s_avg, 0.0, 0.9)

        # ---- capillary pressure ----
        cp_down = self.capillary_pressure_from_saturation(layer, s_down)

        # ---- write back once ----
        layer.upstream_saturation[mask] = us
        layer.downstream_saturation[mask] = s_down
        layer.non_wetting_saturation[mask] = s_avg
        layer.downstream_capillary_pressure[mask] = cp_down

    def calculate_equivalent_flow_resistance(self, cell_side):
        """Compute saturation-dependent flow resistance for each porous layer; return total."""
        total = 0.0
        for layer in cell_side.porous_layers:
            layer.equivalent_flow_resistance = layer.saturation_flow_resistance
            total = total + layer.equivalent_flow_resistance
        return total
