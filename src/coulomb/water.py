"""
Module providing a water thermodynamic and physical properties. 
Based on Cantera's Water object. 
See: https://cantera.org/documentation/docs-3.0/sphinx/html/cython/importing.html#cantera.Water
"""

import cantera as ct
import numpy as np

h2o_phase = ct.Water()

water_molecular_weight = h2o_phase.molecular_weights[0]

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
    Tcelsius = temperature - 273.15
    return 611.21 * np.exp((18.678 - Tcelsius /  234.5) * (Tcelsius / (257.14 + Tcelsius)))
    # return np.piecewise(
    #     Tcelsius, [Tcelsius > 0, Tcelsius <= 0], [
    #         lambda Tcelsius: 611.21 * np.exp((18.678 - Tcelsius /  234.5) * (Tcelsius / (257.14 + Tcelsius))), 
    #         lambda Tcelsius: 611.15 * np.exp((23.036 - Tcelsius /  333.7) * (Tcelsius / (279.82 + Tcelsius)))
    #     ])
 
    # h2o = ct.SolutionArray(h2o_phase, np.shape(temperature))
    # h2o.TQ = temperature, 0  # Set temperature and vapor quality
    # return h2o.P_sat


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

def water_kinematic_viscosity(temperature=300):
    """
    Calculate the kinemattic viscosity of water at a given temperature.

    Parameters:
    -----------
    temperature : float, optional, default=300
        Temperature in Kelvin (K). Default is 300 K.

    Returns:
    --------
    float
        Kinematic viscosity of water in m2/s.
    """
    return water_dynamic_viscosity(temperature) / water_density(temperature)

def water_surface_tension(temperature=300): 
    return 0.076 - 1.677e-4 * (temperature - 273.15)  # N/m

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
    # Source : https://onlinelibrary.wiley.com/doi/pdf/10.1002/9780470516430.app3
    T_Celsius = temperature - 273.15
    return 1001.3 - 0.155 * T_Celsius - 2.658e-3 * T_Celsius ** 2 # kg/m3

    # h2o = ct.SolutionArray(h2o_phase, np.shape(temperature))
    # h2o.TQ = temperature, 0 
    # return h2o.density_mass

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
    # h2o = ct.SolutionArray(h2o_phase, np.shape(temperature))
    # h2o.TQ = temperature, 0 
    return  h2o_phase.molecular_weights[0] / water_density(temperature)

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


