"""
Module providing classes to model porous layers in electrochemical cells.
"""
from dataclasses import dataclass, field
import numpy as np 
import cantera as ct

from .gas_composition import GasComposition, index_h2, index_o2, index_h2ov, species_indexes
from .transport_models import PorousGasResistanceModel, DarcyTransportModel
from .water import water_kinematic_viscosity, water_surface_tension, water_molecular_weight

@dataclass
class PorousLayer():
    """
    Represents a porous layer in a fuel cell or electrolyzer, defining its
    properties related to gas transport, permeability, capillarity, and thermal behavior.

    Attributes
    ----------
    thickness : float
        Thickness of the porous layer in meters (default is 1e-3 m).
    gas : GasComposition
        Gas composition object representing the gas properties in the layer.
    porosity : float
        Porosity of the layer (0 < porosity < 1).
    effective_gas_diffusion_ratio : float
        Ratio accounting for effective gas diffusion through the porous medium (default is 1).
    pore_diameter : float
        Average pore diameter in meters (default is large, so Knudsen diffusion negligible).
    transport_resistance_model : PorousGasResistanceModel
        Model used to calculate gas transport resistance.
    two_phase_transport_model : DarcyTransportModel
        Model used to compute liquid water flow and capillarity.
    non_wetting_saturation : float
        Fraction of the pore volume occupied by non-wetting phase (0 to 1).
    thermal_conductivity : float
        Thermal conductivity in W/(m·K).
    absolute_permeability : float
        Absolute permeability in m².
    relative_permeability_exponent : float
        Exponent for relative permeability model (default is 3, often used for cubic relationship).
    contact_angle : float
        Contact angle for liquid water in degrees (default is 120°).
    non_wetting_phase : str
        Non-wetting phase (liquid or gas).

    Notes
    -----
    This class internally computes quantities like capillary pressure scaling
    and liquid flow resistance based on current temperature, saturation, and geometry.
    """

    thickness: float = 1e-3
    gas: GasComposition = field(default_factory=GasComposition)
    porosity: float = 1
    effective_gas_diffusion_ratio: float = 1
    pore_diameter: float=1e12
    transport_resistance_model: PorousGasResistanceModel = field(default_factory=PorousGasResistanceModel)
    two_phase_transport_model: DarcyTransportModel = field(default_factory=DarcyTransportModel)
    non_wetting_saturation: float = 0
    thermal_conductivity: float = 1e12 
    absolute_permeability: float = 1e6
    relative_permeability_exponent: float = 3
    contact_angle: float = 120. 
    non_wetting_phase: str = 'water'
    wetting_phase: str = 'gas'

    def __post_init__(self): 
        self.sqrt_abs_permeability_porosity = np.sqrt(self.absolute_permeability * self.porosity)
        self.cosinus_contact_angle = np.abs(np.cos(np.pi / 180 * self.contact_angle))
        self.capillary_pressure_J_ratio = 1
        self.saturation_flow_resistance = 1
        self.set_gas_temperature_and_pressure(self.gas.temperature, self.gas.pressure)

    def o2_mole_fraction(self):
        """
        Returns the mole fraction of oxygen (O₂) in the gas phase.

        Returns
        -------
        float
            Oxygen mole fraction.
        """
        return self.gas.X[..., index_o2]

    def h2_mole_fraction(self):
        """
        Returns the mole fraction of hydrogen (H₂) in the gas phase.

        Returns
        -------
        float
            Hydrogen mole fraction.
        """
        return self.gas.X[..., index_h2]

    def species_mole_fraction(self, species):
        """
        Returns the mole fraction of a given species in the gas phase.

        Parameters
        ----------
        species : str
            The chemical species (e.g., 'o2', 'h2', 'h2o', 'n2').

        Returns
        -------
        float
            Mole fraction of the specified species.
        """
        return self.gas.X[..., species_indexes[species]]

    def species_diffusion_coefficient(self, species):
        """
        Returns the diffusion coefficient of a given species in the gas phase.

        Parameters
        ----------
        species : str
            The chemical species for which the diffusion coefficient is needed.

        Returns
        -------
        float
            Diffusion coefficient of the specified species (m²/s).
        """
        return self.gas.calculate_species_diffusion_coefficient(species)

    def species_molecular_weight(self, species):
        """
        Returns the molecular weight of a given species.

        Parameters
        ----------
        species : str
            The chemical species for which the molecular weight is needed.

        Returns
        -------
        float
            Molecular weight of the specified species (g/mol).
        """
        return self.gas.molecular_weights[species_indexes[species]]

    def gas_temperature(self):
        """
        Returns the temperature of the gas phase.

        Returns
        -------
        float
            Gas temperature in Kelvin (K).
        """
        return self.temperature

    def gas_pressure(self):
        """
        Returns the pressure of the gas phase.

        Returns
        -------
        float
            Gas pressure in Pascals (Pa).
        """
        return self.pressure
    
    def set_gas_composition(self, dry_o2_mole_fraction, dry_h2_mole_fraction, relative_humidity):
        """
        Sets the gas composition in terms of oxygen, hydrogen, and humidity.

        Parameters
        ----------
        dry_o2_mole_fraction : float
            Mole fraction of oxygen in dry gas.
        dry_h2_mole_fraction : float
            Mole fraction of hydrogen in dry gas.
        relative_humidity : float
            Relative humidity (0 to 1).
        """
        self.gas.set_composition(dry_o2_mole_fraction, dry_h2_mole_fraction, relative_humidity)

    def set_gas_pressure(self, pressure):
        """
        Sets the gas pressure.

        Parameters
        ----------
        pressure : float
            Gas pressure in Pascals (Pa).
        """
        self.pressure = pressure
        self.gas.set_pressure(pressure)

    def set_gas_temperature(self, temperature):
        """
        Sets the gas temperature.

        Parameters
        ----------
        temperature : float
            Gas temperature in Kelvin (K).
        """
        self.temperature = temperature
        self.RT = ct.gas_constant * self.temperature
        self.gas.set_temperature(temperature)
        self.capillary_pressure_J_ratio = (water_surface_tension(self.temperature) * self.cosinus_contact_angle) / np.sqrt(self.absolute_permeability / self.porosity)
        self.saturation_flow_resistance = self.calculate_saturation_flow_resistance()

    def set_gas_temperature_and_pressure(self, temperature, pressure):
        """
        Sets both the gas temperature and pressure.

        Parameters
        ----------
        temperature : float
            Gas temperature in Kelvin (K).
        pressure : float
            Gas pressure in Pascals (Pa).
        """
        self.set_gas_temperature(temperature)
        self.pressure = pressure
        self.gas.set_temperature_and_pressure(temperature, pressure)

    def species_concentration(self, species):
        """
        Computes the concentration of a given species.

        Parameters
        ----------
        species : str
            The chemical species (e.g., 'o2', 'h2', 'h2o', 'n2').

        Returns
        -------
        float
            Species concentration in mol/m³.
        """
        return self.gas.X[..., species_indexes[species]] * self.gas.pressure / self.RT

    def species_partial_pressure(self, species):
        """
        Computes the partial pressure of a given species.

        Parameters
        ----------
        species : str
            The chemical species (e.g., 'o2', 'h2', 'h2o', 'n2').

        Returns
        -------
        float
            Partial pressure in Pascals (Pa).
        """
        return self.gas.X[..., species_indexes[species]] * self.pressure

    def vapor_pressure(self):
        """
        Computes the vapor pressure of water in the gas mixture.

        Returns
        -------
        float
            Vapor pressure in Pascals (Pa).
        """
        return self.gas.X[..., index_h2ov] * self.pressure

    def relative_humidity(self):
        """
        Returns the relative humidity of the gas mixture.

        Returns
        -------
        float
            Relative humidity (0 to 1).
        """
        return self.gas.relative_humidity

    def saturation_pressure(self):
        """
        Returns the saturation pressure of water in the gas mixture.

        Returns
        -------
        float
            Saturation pressure in Pascals (Pa).
        """
        return self.gas.saturation_pressure

    def saturation_concentration(self):
        """
        Computes the saturation concentration of water vapor.

        Returns
        -------
        float
            Saturation concentration in mol/m³.
        """
        return self.saturation_pressure() / self.RT

    def vapor_concentration(self):
        """
        Computes the water vapor concentration.

        Returns
        -------
        float
            Vapor concentration in mol/m³.
        """
        return self.vapor_pressure() / self.RT
    
    def gas_transport_resistance(self, species='o2'):
        """
        Computes the gas transport resistance for a given species.

        Parameters
        ----------
        species : str, optional
            The gas species for which the transport resistance is computed 
            (default is 'o2').

        Returns
        -------
        float
            Gas transport resistance in s/m.
        """
        return self.transport_resistance_model.total_diffusion_resistance(
            self, 
            self.temperature, 
            self.species_diffusion_coefficient(species), 
            self.species_molecular_weight(species), 
            self.non_wetting_saturation
        )

    def thermal_resistance(self):
        """
        Computes the thermal resistance of the layer.

        Returns
        -------
        float
            Thermal resistance in m²·K/W.
        """
        return self.thickness / self.thermal_conductivity

    def calculate_saturation_flow_resistance(self, electrolyte=None):
        """
        Computes the resistance to non-wetting phase flow of the layer. 
        Based on a water saturation gradient.

        Returns
        -------
        float
            Saturation flow resistance in s.m²/mol.
        """
        if self.non_wetting_phase == 'water': 
            non_wetting_kinematic_viscosity = water_kinematic_viscosity(self.temperature)
            non_wetting_molecular_weight = water_molecular_weight
            non_wetting_surface_tension = water_surface_tension(self.temperature)
        elif self.non_wetting_phase == 'gas': 
            non_wetting_kinematic_viscosity = self.gas.mixture_kinematic_viscosity 
            non_wetting_molecular_weight = self.gas.mixture_molecular_weight
            non_wetting_surface_tension = water_surface_tension(self.temperature) if self.wetting_phase == 'water' else electrolyte.surface_tension
            
        return ((self.thickness * non_wetting_kinematic_viscosity * non_wetting_molecular_weight) / 
                    (self.sqrt_abs_permeability_porosity * self.cosinus_contact_angle * non_wetting_surface_tension))
    

    def saturation_from_capillary_pressure(self, capillary_pressure):
        """
        Computes the non-wetting phase saturation given a capillary pressure
        using the layer's two-phase transport model.

        Parameters
        ----------
        capillary_pressure : float
            Capillary pressure in Pascals (Pa).

        Returns
        -------
        float
            Non-wetting phase saturation (0 to 1).
        """
        return self.two_phase_transport_model.saturation_from_capillary_pressure(self, capillary_pressure) 

    def capillary_pressure_from_saturation(self, saturation):
        """
        Computes the capillary pressure given a liquid saturation
        using the layer's liquid transport model.

        Parameters
        ----------
        saturation : float
            Liquid saturation (0 to 1).

        Returns
        -------
        float
            Capillary pressure in Pascals (Pa).
        """
        return self.two_phase_transport_model.capillary_pressur_from_saturation(self, saturation)

