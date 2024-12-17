"""
Module providing electrochemistry functions. 
"""

import numpy as np
import cantera as ct

h2o2 = ct.Solution('h2o2.yaml')
h2 = h2o2.species('h2').thermo
o2 = h2o2.species('o2').thermo
h2ol = ct.Solution('water.yaml', name='liquid_water').species(0).thermo
h2os = ct.Solution('water.yaml', name='ice').species(0).thermo

STD_PRESSURE = 1e5
STD_TEMPERATURE = 298.15

std_formation_enthalpy_h2ol = (h2ol.h(STD_TEMPERATURE) -
                               (h2.h(STD_TEMPERATURE) +
                                0.5 * o2.h(STD_TEMPERATURE)))
std_formation_entropy_h2ol = (h2ol.s(STD_TEMPERATURE) -
                               (h2.s(STD_TEMPERATURE) +
                                0.5 * o2.s(STD_TEMPERATURE)))
std_formation_gibbs_h2ol = (std_formation_enthalpy_h2ol -
                            STD_TEMPERATURE * std_formation_entropy_h2ol)


def calculate_reversible_cell_voltage(
    temperature,
    partial_pressure_o2,
    partial_pressure_h2
):
    """
    Calculate the reversible cell voltage of a hydrogen fuel cell or electrolyser, 
    using Nernst equation.

    Parameters:
    -----------
    temperature : float
        Temperature of the cell in Kelvin (K).
    partial_pressure_o2 : float
        Partial pressure of oxygen (O₂) in Pascals (Pa).
    partial_pressure_h2 : float
        Partial pressure of hydrogen (H₂) in Pascals (Pa).
 
    Returns:
    --------
    float
        Reversible cell voltage in Volts (V).

    Notes:
    ------
    The function computes the reversible cell voltage using the thermodynamic relationship:
        E_rev = (-ΔG° - ΔS°(T - T°) + RT ln(Q)) / (2F)
    where Q is the reaction quotient based on the partial pressures of H₂ and O₂.

    Constants:
    ----------
    - R: Universal gas constant, 8.314 J/(mol·K).
    - F: Faraday constant, 96485 C/mol.

    Example:
    --------
    >>> reversible_cell_voltage(T=300, p_o2=2e5, p_h2=1e5)
    1.1901
    """
    gibbs_formation_h2ol = (- std_formation_gibbs_h2ol -
                             std_formation_entropy_h2ol * (temperature - STD_TEMPERATURE))

    activity_o2 = partial_pressure_o2 / STD_PRESSURE
    activity_h2 = partial_pressure_h2 / STD_PRESSURE
    activities_ratio = activity_o2 * activity_h2 ** 0.5

    reversible_cell_voltage = (gibbs_formation_h2ol +
                               ct.gas_constant * temperature *
                               np.log(activities_ratio)) / (2 * ct.faraday)

    return reversible_cell_voltage

def calculate_tafel_overpotential(
        current_density,
        exchange_current_density,
        temperature,
        number_of_electrons,
        charge_transfer_coeff):
    """
    Calculate the Tafel overpotential for an electrochemical reaction.

    Parameters:
    -----------
    current_density : float
        The operating current density in Amperes per square meter (A/m²).
    exchange_current_density : float
        The exchange current density in Amperes per square meter (A/m²).
    temperature : float
        Temperature in Kelvin (K).
    number_of_electrons : int
        Number of electrons transferred in the electrochemical reaction.
    charge_transfer_coefficient : float
        Symmetry factor or charge transfer coefficient (dimensionless).

    Returns:
    --------
    float
        The Tafel overpotential in Volts (V).

    Example:
    --------
    >>> calculate_tafel_overpotential(
    ...     current_density=1e4, 
    ...     exchange_current_density=1e-3, 
    ...     temperature=298.15, 
    ...     number_of_electrons=2, 
    ...     charge_transfer_coeff=0.5
    ... )
    0.17985
    """
    tafel_slope = (ct.gas_constant * temperature /
                   (number_of_electrons * charge_transfer_coeff * ct.faraday))
    return tafel_slope * np.log(current_density / exchange_current_density)
