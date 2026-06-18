
"""
Module providing a model for gas diffusion in porous layers.
"""
from dataclasses import dataclass
import numpy as np
from ..thermo.constants import GAS_CONSTANT
from ..thermo.gas import GasModel, species_indexes, molecular_weights
    
@dataclass
class PorousGasDiffusionModel:
    """
    Porous media gas transport model with water saturation and Knudsen corrections.

    Attributes
    ----------
    water_saturation_exponent : float
        Exponent for empirical water saturation correction.
    """
    water_saturation_exponent: float = 3.0

    def water_saturation_correction(self, water_saturation):
        """
        Compute correction factor for effective diffusivity.

        Parameters
        ----------
        water_saturation : float
            Water saturation [-].

        Returns
        -------
        float
            Correction factor [-].
        """
        return np.clip(1 - water_saturation, 1e-6, 1) ** self.water_saturation_exponent
    
    def molecular_diffusion_effective_length(self, layer, water_saturation=0):
        """
        Compute effective diffusion length with saturation correction.

        Parameters
        ----------
        layer : PorousLayer
            Layer with thickness and effective diffusion ratio.
        water_saturation : float, optional
            Water saturation [-].

        Returns
        -------
        float
            Effective diffusion length [m].
        """
        return layer.thickness / layer.effective_gas_diffusion_ratio / self.water_saturation_correction(water_saturation)
    
    def molecular_diffusion_resistance(self, layer, diffusion_coefficient, water_saturation=0):
        """
        Compute molecular diffusion resistance.

        Parameters
        ----------
        layer : PorousLayer
            Layer properties.
        diffusion_coefficient : float
            Diffusion coefficient [m2/s].
        water_saturation : float, optional
            Water saturation [-].

        Returns
        -------
        float
            Diffusion resistance [s/m].
        """
        return self.molecular_diffusion_effective_length(layer, water_saturation) / diffusion_coefficient
    
    def knudsen_diffusivity(self, layer, temperature, molecular_weight):
        """
        Compute Knudsen diffusivity in the porous layer.

        Parameters
        ----------
        layer : PorousLayer
            Contains pore diameter.
        temperature : float
            Temperature [K].
        molecular_weight : float
            Molecular weight [kg/mol].

        Returns
        -------
        float
            Knudsen diffusivity [m2/s].
        """
        return layer.pore_diameter / 3 * np.sqrt(8 * GAS_CONSTANT * temperature / molecular_weight / np.pi)
    
    def total_diffusion_resistance(self, layer, temperature, diffusion_coefficient, molecular_weight, water_saturation):
        """
        Compute total diffusion resistance (molecular + Knudsen).

        Parameters
        ----------
        layer : PorousLayer
            Layer properties.
        temperature : float
            Temperature [K].
        diffusion_coefficient : float
            Molecular diffusion coefficient [m2/s].
        molecular_weight : float
            Molecular weight [kg/mol].
        water_saturation : float
            Water saturation [-].

        Returns
        -------
        float
            Total resistance [s/m].
        """
        return self.molecular_diffusion_effective_length(layer, water_saturation) * (
            1/diffusion_coefficient + 1/self.knudsen_diffusivity(layer, temperature, molecular_weight))

    def gas_transport_resistance(self, layer, state, species: str = 'o2') -> float:
        """Total diffusion resistance for *species* through *layer* given *state* (s/m)."""
        return self.total_diffusion_resistance(
            layer,
            state.temperature,
            GasModel.species_diffusion_coefficient(state, species),
            molecular_weights[species_indexes[species]],
            state.non_wetting_saturation,
        )
