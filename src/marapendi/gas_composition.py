"""
Module providing a class to handle gas composition and properties. 
"""
import numpy as np
import cantera as ct
from marapendi.water import water_saturation_pressure

gas = ct.Solution("gri30.yaml")
species_list = ("O2", "N2", "H2", "H2O")
selected_species_dict = {sp.name: sp for sp in gas.species() if sp.name in species_list}
selected_species = [selected_species_dict[sp] for sp in species_list]
species_names = [sp.lower() for sp in species_list]
species_indexes = dict(zip(species_names, (0,1,2,3)))
index_o2, index_n2, index_h2, index_h2ov = 0, 1, 2, 3

class GasComposition:
    """
    A class for managing and calculating gas compositions, including temperature, 
    pressure, and relative humidity effects on the gas phase.

    The gas is composed of O2, N2, H2 and H2O. 

    Attributes:
    -----------
    gas : ct.Solution
        Cantera `Solution` object for the gas mixture with selected species.
    relative_humidity : float
        The relative humidity of the gas mixture.
    saturation_pressure : float
        The saturation pressure of water at the current temperature in Pascals (Pa).

    Methods:
    --------
    set_pressure(new_pressure):
        Set the pressure of the gas mixture and update the relative humidity.
    
    set_temperature(new_temperature):
        Set the temperature of the gas mixture, updating the saturation pressure and relative humidity.
    
    set_temperature_and_pressure(temperature, pressure):
        Set both the temperature and pressure of the gas mixture.

    pressure():
        Get the current pressure of the gas mixture in Pascals (Pa).
    
    temperature():
        Get the current temperature of the gas mixture in Kelvin (K).

    set_composition(dry_o2_mole_fraction, dry_h2_mole_fraction, relative_humidity):
        Set the composition of the gas mixture, including dry gas oxygen and hydrogen mole fractions, 
        and relative humidity.
    """

    def __init__(self, temperature=300, pressure=1e5):
        """
        Initialize the gas composition with a given temperature and pressure.

        Parameters:
        -----------
        temperature : float, optional, default=300
            Temperature of the gas mixture in Kelvin (K).
        pressure : float, optional, default=1e5
            Pressure of the gas mixture in Pascals (Pa).
        """
        self.X = np.zeros(shape=(4,))
        self.set_temperature_and_pressure(temperature, pressure)
        self.set_composition(1,0,0)
        self.molecular_weights = np.array([32., 28., 2., 18.])

    def set_pressure(self, new_pressure: float):
        """
        Set the pressure of the gas mixture and adjust the relative humidity.

        Parameters:
        -----------
        new_pressure : float
            The new pressure in Pascals (Pa).
        """
        self.relative_humidity *= new_pressure / self.pressure
        self.pressure = new_pressure

    def set_temperature(self, new_temperature: float):
        """
        Set the temperature of the gas mixture and adjust the relative humidity.

        Parameters:
        -----------
        new_temperature : float
            The new temperature in Kelvin (K).
        """
        self.temperature = new_temperature
        self.calculate_relative_humidity()

    def set_temperature_and_pressure(self, temperature: float, pressure: float):
        """
        Set both the temperature and pressure of the gas mixture.

        Parameters:
        -----------
        temperature : float
            The temperature in Kelvin (K).
        pressure : float
            The pressure in Pascals (Pa).
        """
        self.temperature = temperature
        self.pressure = pressure
        self.calculate_relative_humidity()

    def pressure(self) -> float:
        """
        Get the current pressure of the gas mixture.

        Returns:
        --------
        float
            Pressure in Pascals (Pa).
        """
        return self.pressure

    def temperature(self) -> float:
        """
        Get the current temperature of the gas mixture.

        Returns:
        --------
        float
            Temperature in Kelvin (K).
        """
        return self.temperature

    def vapor_pressure(self) -> float:
        """
        Get the current vapor partial pressure in the gas mixture.

        Returns:
        --------
        float
            Vapor partial pressure in Pa.
        """
        return self.X[...,index_h2ov] * self.pressure

    def calculate_relative_humidity(self) -> float: 
        """
        Calculate the current relative humidity in the gas mixture.

        Returns:
        --------
        float
            Relative humidity between 0 and 1.
        """
        self.saturation_pressure = water_saturation_pressure(self.temperature)
        self.relative_humidity = self.vapor_pressure() / self.saturation_pressure
        return self.relative_humidity
    
    def concentration(self): 
        return self.pressure / (ct.gas_constant * self.temperature)
    
    def get_relative_humidity(self) -> float: 
        """
        Get the current relative humidity in the gas mixture.

        Returns:
        --------
        float
            Relative humidity between 0 and 1.
        """
        return self.relative_humidity
            
    def set_composition(self, dry_o2_mole_fraction: float, dry_h2_mole_fraction: float, relative_humidity: float):
        """
        Set the composition of the gas mixture.

        The mole fraction of water vapor is calculated based on the relative humidity 
        and the saturation pressure at the current temperature. The wet gas composition 
        is updated accordingly. 

        Parameters:
        -----------
        dry_o2_mole_fraction : float
            Mole fraction of oxygen in the dry gas mixture.
        dry_h2_mole_fraction : float
            Mole fraction of hydrogen in the dry gas mixture.
        relative_humidity : float
            Relative humidity of the gas mixture.
        """
        dry_mole_fractions = np.zeros_like(self.X)
        dry_mole_fractions[...,index_o2] = dry_o2_mole_fraction
        dry_mole_fractions[...,index_h2] = dry_h2_mole_fraction
        dry_mole_fractions[...,index_n2] = 1 - dry_o2_mole_fraction - dry_h2_mole_fraction


        self.saturation_pressure = water_saturation_pressure(self.temperature)
        h2o_mole_fraction = relative_humidity *  self.saturation_pressure / self.pressure

        vapor_mole_fractions = np.zeros_like(self.X)
        vapor_mole_fractions[...,index_h2ov] = h2o_mole_fraction

        self.X = dry_mole_fractions * (1 - vapor_mole_fractions[...,index_h2ov, np.newaxis]) + vapor_mole_fractions
        self.relative_humidity = self.pressure * vapor_mole_fractions[...,index_h2ov] / self.saturation_pressure
      
    
    def species_diffusion_coefficient(self, species):
        """
        Calculate the binary diffusion coefficient for a given species in the gas phase.

        Uses empirical correlations based on reference values adjusted for
        temperature and pressure. Data from Vetter and Schumacher (2019).

        Parameters
        ----------
        species : str
            The chemical species ('o2', 'h2', or 'h2o') for which to compute the diffusion coefficient.

        Returns
        -------
        float
            The adjusted diffusion coefficient [m^2/s].

        Reference
        ----------
        Vetter, R. & Schumacher, J. O. Comput. Phys. Commun. 234, 223–234 (2019).
        """
        # Set the reference diffusion coefficient based on species
        if species == 'o2':
            reference_diffusion_coeff = 0.28e-4  # [m^2/s]
        elif species == 'h2':
            reference_diffusion_coeff = 1.24e-4
        elif species == 'h2o':
            # If H2 is present, assume H2-H2O; else O2-H2O
            if np.max(self.X[..., index_h2]) > 0:
                reference_diffusion_coeff = 1.24e-4
            else:
                reference_diffusion_coeff = 0.36e-4

        # Apply temperature and pressure correction
        # Fick's law adjustment: D ~ T^1.5 / P
        return reference_diffusion_coeff * (self.temperature / 353.15) ** 1.5 * (100000 / self.pressure)