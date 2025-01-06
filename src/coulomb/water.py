"""
Module providing a water thermodynamic and physical properties. 
Based on Cantera's Water object. 
See: https://cantera.org/documentation/docs-3.0/sphinx/html/cython/importing.html#cantera.Water
"""

import cantera as ct
import numpy as np

h2o_phase = ct.Water()

def water_saturation_pressure(temperature):
    """
    Calculate the saturation pressure of water at a given temperature.

    Parameters:
    -----------
    temperature : float
        Temperature in Kelvin (K).

    Returns:
    --------
    float
        Saturation pressure of water in Pascals (Pa).
    """
    h2o = ct.SolutionArray(h2o_phase, np.shape(temperature))
    h2o.TQ = temperature, 0  # Set temperature and vapor quality
    return h2o.P_sat


def water_saturation_concentration(temperature):
    """
    Calculate the saturation concentration of water vapor in the gas phase.

    Parameters:
    -----------
    temperature : float
        Temperature in Kelvin (K).

    Returns:
    --------
    float
        Saturation concentration of water vapor in kmol/m³.
    """
    return water_saturation_pressure(temperature) / (ct.gas_constant * temperature)


def water_dew_point(vapor_pressure):
    """
    Calculate the dew point temperature of water given its partial pressure.

    Parameters:
    -----------
    vapor_pressure : float
        Partial pressure of water vapor in Pascals (Pa).

    Returns:
    --------
    float
        Dew point temperature in Kelvin (K).
    """
    h2o = ct.SolutionArray(h2o_phase, np.shape(vapor_pressure))
    h2o.PQ = vapor_pressure, 0  # Set pressure and vapor quality
    return h2o.T


def water_dynamic_viscosity(temperature=300):
    """
    Calculate the dynamic viscosity of water at a given temperature.

    Parameters:
    -----------
    temperature : float, optional, default=300
        Temperature in Kelvin (K). Default is 300 K.

    Returns:
    --------
    float
        Dynamic viscosity of water in Pascal-seconds (Pa·s).
    """
    h2o = ct.SolutionArray(h2o_phase, np.shape(temperature))
    h2o.TQ = temperature, 0  # Set temperature and vapor quality
    return h2o.viscosity

def water_density(temperature=300): 
    """
    Calculate the density of water at a given temperature.

    Parameters:
    -----------
    temperature : float, optional, default=300
        Temperature in Kelvin (K). Default is 300 K.

    Returns:
    --------
    float
        Density of water in kg/m³.
    """
    h2o = ct.SolutionArray(h2o_phase, np.shape(temperature))
    h2o.TQ = temperature, 0 
    return h2o.density_mass

def water_molar_volume(temperature=300): 
    """
    Calculate the molar volume of water at a given temperature.

    Parameters:
    -----------
    temperature : float, optional, default=300
        Temperature in Kelvin (K). Default is 300 K.

    Returns:
    --------
    float
        Molar volume of water m³/kmol.
    """
    h2o = ct.SolutionArray(h2o_phase, np.shape(temperature))
    h2o.TQ = temperature, 0 
    return h2o.volume_mole

class WaterProperties:
    """
    A class to hold and compute thermophysical properties of water.

    Attributes:
    -----------
    h2o : ct.Water
        An instance of Cantera's `Water` class for thermophysical property calculations.
    density : float
        The density of water in kg/m³.
    dynamic_viscosity : float
        The dynamic viscosity of water in Pa·s.
    molar_volume : float
        The molar volume of water in m³/mol.
    saturation_pressure : float
        The saturation pressure of water in Pascals (Pa).

    Methods:
    --------
    set_temperature(temperature: float):
        Set the temperature of water and update its properties.

    Notes:
    ------
    This class uses Cantera's `Water` class to compute water's properties at 
    a given temperature. Ensure Cantera is installed and properly configured.
    """

    def __init__(self, temperature: float = 300):
        """
        Initialize the WaterProperties class with a given temperature.

        Parameters:
        -----------
        temperature : float
            The temperature in Kelvin (K) at which the water properties are computed.

        Notes:
        ------
        The `set_temperature` method is called during initialization to set the water 
        temperature and compute the corresponding properties.
        """
        self.h2o = ct.SolutionArray(ct.Water(), np.shape(temperature))
        self.set_temperature(temperature)

    def set_temperature(self, temperature: float):
        """
        Set the temperature of water and update its thermophysical properties.

        Parameters:
        -----------
        temperature : float
            The temperature in Kelvin (K) to set the water properties.

        Updates:
        --------
        - density : float
        - dynamic_viscosity : float
        - molar_volume : float
        - saturation_pressure : float

        Example:
        --------
        >>> water = WaterProperties(temperature=373.15)
        >>> print(water.density)
        958.366
        """
        self.h2o.TQ = temperature, 0  # Set the temperature and quality (0 for liquid water)
        self.density = self.h2o.density_mass  # kg/m³
        self.dynamic_viscosity = self.h2o.viscosity  # Pa·s
        self.molar_volume = self.h2o.volume_mole  # m³/mol
        self.saturation_pressure = self.h2o.P_sat  # Pa


