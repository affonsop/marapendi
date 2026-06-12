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
    Dynamic viscosity of liquid water [Pa·s] via the Vogel equation.

    Replaces a Cantera SolutionArray call (~18 µs) with a direct formula
    (~1.5 µs). Error vs. Cantera: <0.1 % at 20 °C, <1 % at 80 °C, <1.5 % at 100 °C.

    Parameters:
    -----------
    temperature : float or ndarray
        Temperature in Kelvin (K).

    Returns:
    --------
    float or ndarray
        Dynamic viscosity [Pa·s].
    """
    return 2.414e-5 * 10 ** (247.8 / (temperature - 140.0))

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
    return np.polyval([- 2.658e-3, - 0.155, 1001.3], T_Celsius) # kg/m3

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

def o2_water_diffusivity(temperature=300): 
    """
    Calculate the O2 diffusivity in liquid water at a given temperature.
    Uses value at 298 K from Tsimpanogiannis et al. (2021), table 11.
    Parameters:
    -----------
    temperature : float, optional, default=300
        Temperature in Kelvin (K). Default is 300 K.

    Returns:
    --------
    float
        O2 diffusivity in liquid water in m2/s.
    """
    return 4.6e-7 * np.exp(-0.155e4/temperature)
