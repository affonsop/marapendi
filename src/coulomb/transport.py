"""
Module providing gas and liquid transport models.
"""
from dataclasses import dataclass
import numpy as np
import cantera as ct

from .tools import sigmoid
from .water import water_density, water_dynamic_viscosity, water_surface_tension

@dataclass 
class ChannelGasResistanceModel: 
    A_ch: float = 1.0
    B_ch: float = 1.0

    def molecular_diffusion_resistance(self, channel, diffusion_coefficient): 
        return self.A_ch * channel.half_width / diffusion_coefficient
    
    def convection_resistance(self, channel, volume_flow_rate): 
        return self.B_ch * channel.length / channel.half_width * channel.total_flow_section / volume_flow_rate

    def total_resistance(self, channel, diffusion_coefficient, volume_flow_rate): 
        return (self.molecular_diffusion_resistance(channel, diffusion_coefficient) +
                self.convection_resistance(channel, volume_flow_rate)) 


@dataclass
class PorousGasResistanceModel:
    water_saturation_exponent: float = 3.0

    def water_saturation_correction(self, water_saturation):
        return (1 - water_saturation) ** self.water_saturation_exponent
    
    def molecular_diffusion_effective_length(self, layer, water_saturation=0):
        return layer.thickness / layer.effective_gas_diffusion_ratio / self.water_saturation_correction(water_saturation)
    
    def molecular_diffusion_resistance(self, layer, diffusion_coefficient, water_saturation=0):
        return self.molecular_diffusion_effective_length(layer, water_saturation) / diffusion_coefficient
    
    def knudsen_diffusivity(self,layer, temperature, molecular_weight):
        return layer.pore_diameter / 3 * np.sqrt(8 * ct.gas_constant * temperature / molecular_weight / np.pi)
    
    def total_diffusion_resistance(self, layer, temperature, diffusion_coefficient, molecular_weight, water_saturation):
        return self.molecular_diffusion_resistance(layer, diffusion_coefficient, water_saturation) + layer.thickness / self.knudsen_diffusivity(layer, temperature, molecular_weight)


@dataclass
class PorousLiquidTransportModel: 
    critical_damkholer: float = 1.
    dry_wet_transition_parameter: float = 10
    wet_saturation: float = 0.4 

    def vapor_transport_resistance(self, cell_side): 
        return cell_side.calculate_gas_transport_resistance('h2o')
    
    def calculate_damkholer_number(self, cell_side, water_injection_flux): 
        cl_sat_concentration = cell_side.cl.get_saturation_concentration()
        ch_vapor_concentration = cell_side.ch.get_vapor_concentration()
        max_vapor_removal_flux = (cl_sat_concentration - ch_vapor_concentration) / cell_side.h2ov_transport_resistance
        return water_injection_flux / max_vapor_removal_flux
        
    def calculate_water_saturation(self, cell_side, water_injection_flux): 
        damkholer = self.calculate_damkholer_number(cell_side, water_injection_flux)
        return self.wet_saturation * sigmoid(damkholer, self.critical_damkholer, self.dry_wet_transition_parameter)

@dataclass    
class DarcyLiquidTransportModel: 
    dry_wet_transition_parameter: float = 0.2


    def vapor_transport_resistance(self, cell_side): 
        return cell_side.calculate_gas_transport_resistance('h2o')
    
    def calculate_damkholer_number(self, cell_side, water_injection_flux): 
        cl_sat_concentration = cell_side.cl.get_saturation_concentration()
        ch_vapor_concentration = cell_side.ch.get_vapor_concentration()
        max_vapor_removal_flux = (cl_sat_concentration - ch_vapor_concentration) / cell_side.h2ov_transport_resistance
        return water_injection_flux / max_vapor_removal_flux

    def calculate_water_saturation(self, cell_side): 
        #damkholer = self.calculate_damkholer_number(cell_side, water_injection_flux)

        #return self.wet_saturation *  cb.sigmoid(water_injection_flux, max_vapor_removal_flux * self.critical_damkholer, self.dry_wet_transition_parameter)
        return np.minimum(0.9,(cell_side.liquid_flux * cell_side.calculate_equivalent_flow_resistance()) ** self.dry_wet_transition_parameter)