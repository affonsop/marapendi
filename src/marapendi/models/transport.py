"""
Module providing gas and liquid transport models.
"""
from dataclasses import dataclass
import numpy as np
import cantera as ct
from marapendi.models.water import water_surface_tension



@dataclass 

class ChannelGasResistanceModel:
    """
    Channel gas transport resistance model using Sherwood number approach.
    Similar equations are used by Kim et al. (2022). 

    Attributes
    ----------
    sherwood : float
        Sherwood number for mass transfer.
    B_ch : float
        Coefficient for convection resistance.
    
    References
    ----------
    Kim, H. et al. Int. J. Heat Mass Transf. 183, 122106 (2022).
    """
    sherwood: float = 4.0
    B_ch: float = 1.0

    def molecular_diffusion_resistance(self, channel, diffusion_coefficient):
        """
        Compute molecular diffusion resistance using Sherwood number.

        Parameters
        ----------
        channel : FlowChannel
            Channel geometry.
        diffusion_coefficient : float
            Diffusion coefficient [m2/s].

        Returns
        -------
        float
            Channel diffusion resistance [s/m].
        """
        return 1/self.sherwood * channel.hydraulic_diameter / diffusion_coefficient
    
    def convection_resistance(self, channel, volume_flow_rate):
        """
        Compute convection resistance with channel geometry.
        Assumes that the channel gas concentration is equal to the average between 
        inlet and outlet values.  

        Parameters
        ----------
        channel : FlowChannel
            Channel geometry.
        volume_flow_rate : float
            Volumetric flow rate [m3/s].

        Returns
        -------
        float
            Channel convection resistance [s/m].
        """
        return self.B_ch * channel.length * channel.width * (1 + 1/channel.channel_land_ratio) / 2 * channel.n_parallel / (volume_flow_rate + 1e-12)

    def total_resistance(self, channel, diffusion_coefficient, volume_flow_rate):
        """
        Compute total resistance combining diffusion and convection.

        Parameters
        ----------
        channel : FlowChannel
            Channel geometry.
        diffusion_coefficient : float
            Diffusion coefficient [m2/s].
        volume_flow_rate : float
            Volumetric flow rate [m3/s].

        Returns
        -------
        float
            Channel total resistance [s/m].
        """
        return (self.molecular_diffusion_resistance(channel, diffusion_coefficient) +
                self.convection_resistance(channel, volume_flow_rate))

    
@dataclass
class PorousGasResistanceModel:
    """
    Porous media gas transport model with water saturation and Knudsen corrections.

    Attributes
    ----------
    n_s : float
        Exponent for empirical water saturation correction.
    """
    n_s: float = 3.0

    def species_diffusion_coefficient(self, temperature, pressure, x_h2=0):
        """
        Calculate the binary diffusion coefficient for a given species in the gas phase.

        Uses empirical correlations based on reference values adjusted for
        temperature and pressure. Data from Vetter and Schumacher (2019).

    
        Returns
        -------
        float
            The adjusted diffusion coefficient [m^2/s].

        Reference
        ----------
        Vetter, R. & Schumacher, J. O. Comput. Phys. Commun. 234, 223–234 (2019).
        """
        ("O2", "N2", "H2", "H2O")
        reference_diffusion_coeff =np.where(x_h2[...,np.newaxis] <= 0, np.array([0.28e-4, 0.28e-4, 0.28e-4, 0.36e-4])[np.newaxis, ...] ,
                                    np.array([1.24e-4, 1.24e-4, 1.24e-4, 1.24e-4])[np.newaxis, ...])


        # Apply temperature and pressure correction
        # Fick's law adjustment: D ~ T^1.5 / P
        diffusion_coeff = reference_diffusion_coeff[...,np.newaxis] * (temperature ** 1.5 / np.maximum(pressure, 1.) * 15.0682)[:,np.newaxis,...]
        return diffusion_coeff

    def water_saturation_correction(self, s, n_s):
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
        return np.clip(1 - s, 1e-6, 1) ** n_s

    def molecular_diffusion_effective_length(self, thickness, eps_p, tort, s, n_s):
        return thickness * tort / eps_p / self.water_saturation_correction(s, n_s)

    def molecular_diffusion_resistance(self, D_g_k, thickness, eps_p, tort, s, n_s):
        return self.molecular_diffusion_effective_length(thickness, eps_p, tort, s, n_s) / D_g_k

    def knudsen_diffusivity(self, T, d_p, M_k):
        return d_p / 3 * np.sqrt(8 * ct.gas_constant * T / np.maximum(M_k, 1e-30) / np.pi)

    def total_diffusion_resistance(self, T, s, D_g_k, M_k, thickness, eps_p, tort, d_p, n_s):
        """
        Compute total diffusion resistance (molecular + Knudsen).

        Returns
        -------
        float
            Total resistance [s/m].
        """
        return self.molecular_diffusion_effective_length(thickness, eps_p, tort, s, n_s) * (
            1/D_g_k + 1/self.knudsen_diffusivity(T, d_p, M_k))

@dataclass    
class DarcyTransportModel: 
    """
    Model for calculating non-wetting phase transport in porous layers using a Darcy-based approach.

    Attributes
    ----------
    J_function_exponent : float
        Exponent in the non-wetting phase capillary pressure-saturation relation.
    """
 

    def calculate_liquid_darcy_flow_resistance(self, s, nu_l, thickness, K_abs, n_rel):
        # on a mass basis
        return ((thickness * nu_l) /
                (K_abs * np.maximum(1e-3, s) ** n_rel))

    def saturation_from_capillary_pressure(self, layer, p_c, p_b, m, n):
        """
        Van Genuchten model
        """
        return 1 - (1 + (p_c/p_b)**n)**(-m)

    def capillary_pressure_from_saturation(self, s, p_b, m, n):
        """
        Van Genuchten model
        """
        s = np.clip(s, 1e-3, 1 - 1e-6)
        return  p_b * ((1 - s)**(-1/m) - 1)**(1/n)
