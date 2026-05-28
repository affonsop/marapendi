"""
Module providing gas and liquid transport models.
"""
from dataclasses import dataclass
import numpy as np
import cantera as ct
from marapendi.components.water import water_surface_tension
from marapendi.tools.tools import arrhenius_term 

from dataclasses import dataclass
import numpy as np
import cantera as ct

@dataclass 
class BakerChannelGasResistanceModel:
    """
    Empirical model for gas transport resistance in flow channels. 
    Based on the work of Baker et al. (2009). 

    Attributes
    ----------
    A_ch : float
        Coefficient for diffusion resistance.
    B_ch : float
        Coefficient for convection resistance.

    References
    ----------
    Baker et al. J. Electrochem. Soc. 156, B991 (2009).
    """
    A_ch: float = 1.0
    B_ch: float = 1.0

    def molecular_diffusion_resistance(self, channel, diffusion_coefficient):
        """
        Compute diffusion resistance in the channel.

        Parameters
        ----------
        channel : FlowChannel
            Flow channel object with geometry.
        diffusion_coefficient : float
            Molecular diffusion coefficient [m2/s].

        Returns
        -------
        float
            Channel diffusion resistance [s/m].
        """
        return self.A_ch * channel.half_width / diffusion_coefficient
    
    def convection_resistance(self, channel, volume_flow_rate):
        """
        Compute convection resistance in the channel.

        Parameters
        ----------
        channel : FlowChannel
            Flow channel object with geometry.
        volume_flow_rate : float
            Volumetric flow rate [m3/s].

        Returns
        -------
        float
            Channel convection resistance [s/m].
        """
        return self.B_ch * channel.length / channel.half_width * channel.total_flow_section / (volume_flow_rate + 1e-12)

    def total_resistance(self, channel, diffusion_coefficient, volume_flow_rate):
        """
        Compute total gas transport resistance (diffusion + convection).

        Parameters
        ----------
        channel : FlowChannel
            Flow channel object.
        diffusion_coefficient : float
            Molecular diffusion coefficient [m2/s].
        volume_flow_rate : float
            Volumetric flow rate [m3/s].

        Returns
        -------
        float
            Total channel resistance [s/m].
        """
        return (self.molecular_diffusion_resistance(channel, diffusion_coefficient) +
                self.convection_resistance(channel, volume_flow_rate))

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
    def water_vol_fraction(self, water_content, water_molar_volume, ionomer_molar_volume): 
        adsorbed_water_molar_volume = water_content * water_molar_volume 
        return adsorbed_water_molar_volume / (ionomer_molar_volume + adsorbed_water_molar_volume)
    
    def diffusion_coefficient(self, water_content, water_vol_fraction, temperature, darken_num, darken_den, diffusivity, activation_energy, reference_temperature=303.15): 
        return np.polyval(darken_num[::-1], water_content) / np.polyval(darken_den[::-1], water_content)  * diffusivity * water_vol_fraction * arrhenius_term(activation_energy, temperature, reference_temperature) 

    def sorption_coefficient(self, water_vol_fraction, temperature, desorption_coefficent, activation_energy, reference_temperature=303.15): 
        return desorption_coefficent * water_vol_fraction * arrhenius_term(activation_energy, temperature, reference_temperature)
    
    def calculate_membrane_water_resistance(self, diffusion_coefficient, thickness, ionomer_vol_fraction, ionomer_concentration, ionomer_tortuosity):
        return (
            thickness
            / (
               diffusion_coefficient
                * ionomer_concentration 
                * ionomer_vol_fraction 
                / ionomer_tortuosity)
        )

@dataclass
class PorousGasResistanceModel:
    """
    Porous media gas transport model with water saturation and Knudsen corrections.

    Attributes
    ----------
    water_saturation_exponent : float
        Exponent for empirical water saturation correction.
    """
    water_saturation_exponent: float = 3.0

    def water_saturation_correction(self, water_saturation, water_saturation_exponent):
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
        return np.clip(1 - water_saturation, 1e-6, 1) ** water_saturation_exponent
    
    def molecular_diffusion_effective_length(self, thickness, porosity, tortuosity, water_saturation, water_saturation_exponent):
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
        return thickness * tortuosity / porosity / self.water_saturation_correction(water_saturation, water_saturation_exponent)
    
    def molecular_diffusion_resistance(self, diffusion_coefficient, thickness, porosity, tortuosity, water_saturation, water_saturation_exponent):
        """
        Compute molecular diffusion resistance.

        Returns
        -------
        float
            Diffusion resistance [s/m].
        """
        return self.molecular_diffusion_effective_length(thickness, porosity, tortuosity, water_saturation, water_saturation_exponent) / diffusion_coefficient
    
    def knudsen_diffusivity(self, temperature, pore_diameter, molecular_weight):
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
  
        return pore_diameter / 3 * np.sqrt(8 * ct.gas_constant * temperature / molecular_weight / np.pi)
    
    def total_diffusion_resistance(self, temperature, water_saturation, diffusion_coefficient, molecular_weight, thickness, porosity, tortuosity, pore_diameter, water_saturation_exponent):
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
        return self.molecular_diffusion_effective_length(thickness, porosity, tortuosity, water_saturation, water_saturation_exponent) * (
            1/diffusion_coefficient + 1/self.knudsen_diffusivity(temperature, pore_diameter, molecular_weight))

@dataclass    
class DarcyTransportModel: 
    """
    Model for calculating non-wetting phase transport in porous layers using a Darcy-based approach.

    Attributes
    ----------
    J_function_exponent : float
        Exponent in the non-wetting phase capillary pressure-saturation relation.
    """
 

    def calculate_liquid_darcy_flow_resistance(self, temperature, saturation, thickness, water_kinematic_viscosity, absolute_permeability, relative_permeability_exponent):
        # on a mass basis 
        return ((thickness * water_kinematic_viscosity) / 
                (absolute_permeability * np.maximum(1e-3,saturation) ** relative_permeability_exponent))

    def saturation_from_capillary_pressure(self, layer, capillary_pressure, breakthrough_pressure, m, n):
        """
        Van Genuchten model
        """
        return 1 - (1 + (capillary_pressure/breakthrough_pressure)**n)**(-m)

    def capillary_pressure_from_saturation(self, saturation, breakthrough_pressure, m, n):
        """
        Van Genuchten model
        """
        saturation = np.clip(saturation, 1e-3, 1)
        return  breakthrough_pressure * ((1 - saturation)**(-1/m) - 1)**(1/n)

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
