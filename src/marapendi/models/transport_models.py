"""
Module providing gas and liquid transport models.
"""
from dataclasses import dataclass
import numpy as np
import cantera as ct
from marapendi.components.water import water_surface_tension
from marapendi.tools.tools import arrhenius_term, polyval_vec


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
class MembraneWaterTransportModel:
    ''''
    See Wei et al. (2023)
    '''
    def water_vol_fraction(self, lmbd, V_w, V_ion):
        lmbd_V_w = lmbd * V_w
        return lmbd_V_w / (V_ion + lmbd_V_w)
    
    def diffusion_coefficient(self, lmbd, f_v, T, darken_num, darken_den, alpha_lmbd, E_act, T_ref=303.15):
        return polyval_vec(darken_num[:,::-1], lmbd) / polyval_vec(darken_den[:,::-1], lmbd) * alpha_lmbd * f_v * arrhenius_term(E_act, T, T_ref)

    def sorption_coefficient(self, f_v, T, k_des, E_act, T_ref=303.15):
        return k_des * f_v * arrhenius_term(E_act, T, T_ref)
    
    def calculate_membrane_water_resistance(self, D_lmbd, thickness, eps_ion, c_ion, tort_ion):
        D_eff = D_lmbd * c_ion * eps_ion / tort_ion
        return thickness / D_eff
    
    def equilibrium_water_content(self, rh, sorption_coeffs):
        rh = np.clip(rh, 0, 1)
        return polyval_vec(sorption_coeffs[:,::-1], rh)

    def liquid_equilibrium_water_content(self, reference_liquid_water_content):
        return reference_liquid_water_content
    
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
        return d_p / 3 * np.sqrt(8 * ct.gas_constant * T / M_k / np.pi)

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
        s = np.clip(s, 1e-3, 1)
        return  p_b * ((1 - s)**(-1/m) - 1)**(1/n)
