"""
Module providing a water thermodynamic and physical properties. 
Based on Cantera's Water object. 
See: https://cantera.org/documentation/docs-3.0/sphinx/html/cython/importing.html#cantera.Water
"""

import cantera as ct

h2o = ct.Water()

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
    h2o.TQ = temperature, 0  # Set temperature and vapor quality
    return h2o.viscosity
