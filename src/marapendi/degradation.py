"""
Module providing classes to model degradation in electrochemical cells.
"""
from dataclasses import dataclass, field
import numpy as np 
import cantera as ct 
from .tools import potential_activation 

@dataclass 
class PtDissolution: 
    k1: float = 3.43e-12 # kmol/m2.s
    k2: float = 1.36e-10 # kmol/m2.s
    k3: float = 3.2e-23  # kmol/m2.s
    sigma_Pt = 2.37 # surface tension Pt cristallite J/m2 
    sigma_PtO = 1.00 # surface tension PtO
    platinum_density: float = 21450. # kg/m3
    platinum_molecular_weight: float =  195.08 # kg/kmol
    platinum_oxide_molecular_weight: float = 211.08 # kg/kmol
    platinum_oxide_density: float =  14100. # kg/m3
    platinum_dissolution_reference_potential: float = 1.188 # V
    platinum_oxide_formation_reference_potential: float = 0.98 # V 
    platinum_dissolution_transfer_coeff_ca: float = 0.5 
    platinum_dissolution_transfer_coeff_an: float = 0.5
    dissolved_platinum_reference_concentration: float = 1. # kmol / m3
    proton_reference_concentration: float = 1. # kmol / m3
    platinum_oxide_formation_transfer_coeff_ca: float = 0.15 
    platinum_oxide_formation_transfer_coeff_an: float = 0.35
    omega_platinum_oxide_formation: float = 30e6
    def platinum_surface_tension_potential_shift(self, particle_radius): 
        return self.sigma_Pt * self.platinum_molecular_weight / self.platinum_density / particle_radius 
    
    def platinum_oxide_surface_tension_potential_shift(self, particle_radius): 
        return self.sigma_PtO * self.platinum_oxide_molecular_weight / self.platinum_oxide_density / particle_radius 
    
    def platinum_dissolution_equilibrium_potential(self, particle_radius): 
        return (self.platinum_dissolution_reference_potential -
                self.platinum_surface_tension_potential_shift(particle_radius) / (2 * ct.faraday))
    
    def platinum_oxide_formation_equilibrium_potential(self, particle_radius): 
        return (self.platinum_oxide_formation_reference_potential + 
                (self.platinum_oxide_surface_tension_potential_shift(particle_radius) - 
                self.platinum_surface_tension_potential_shift(particle_radius)) / (2 * ct.faraday))
    

    def platinum_dissolution_rate_of_reaction(self, 
                                              dissolved_platinum_concentration, 
                                              platinum_oxide_coverage, 
                                              potential, temperature, particle_radius): 
        potential_difference = (potential - 
                                self.platinum_dissolution_equilibrium_potential(particle_radius))
        dissolved_platinum_concentration_ratio = (
            dissolved_platinum_concentration / 
            self.dissolved_platinum_reference_concentration
        )
        print(potential_difference)
        return self.k1 * np.maximum(1.-platinum_oxide_coverage,0) * (
            
            
            - 
            dissolved_platinum_concentration_ratio * potential_activation(
                self.platinum_dissolution_transfer_coeff_ca, 
                2, temperature, -potential_difference
            )   
            )
    
    def platinum_oxide_formation_rate_of_reaction(self, 
                                              dissolved_platinum_concentration, 
                                              platinum_oxide_coverage, 
                                              proton_concentration,
                                              potential, temperature, particle_radius): 
        potential_difference = (potential - 
                                self.platinum_oxide_formation_equilibrium_potential(particle_radius))
        proton_concentration_ratio = (
            proton_concentration / 
            self.proton_reference_concentration
        )
        potential_correction = self.omega_platinum_oxide_formation  * platinum_oxide_coverage / (2 * ct.faraday * self.platinum_oxide_formation_transfer_coeff_an)
        return self.k2 * (
            potential_activation(
                self.platinum_oxide_formation_transfer_coeff_an, 
                2, temperature, 
                potential_difference - potential_correction
            ) - 
            platinum_oxide_coverage * potential_activation(
                self.platinum_oxide_formation_transfer_coeff_ca, 
                2, temperature, -potential_difference
            ) * proton_concentration_ratio ** 2
            )   
    # def platinum_oxide_dissolution_rate_of_reaction(self, ): 