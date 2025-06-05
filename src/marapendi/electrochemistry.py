"""
Module providing electrochemistry functions. 
"""
from dataclasses import dataclass
import numpy as np
import cantera as ct
from marapendi.tools import calculate_arrhenius_term

h2o2 = ct.Solution('gri30.yaml')
h2 = h2o2.species('H2').thermo
o2 = h2o2.species('O2').thermo
h2ov = h2o2.species('H2O').thermo
h2ol = ct.Solution('water.yaml', name='liquid_water').species(0).thermo
h2os = ct.Solution('water.yaml', name='ice').species(0).thermo

STD_PRESSURE = 1e5
STD_TEMPERATURE = 298.15

std_formation_enthalpy_h2ov = (h2ov.h(STD_TEMPERATURE) -
                               (h2.h(STD_TEMPERATURE) +
                                0.5 * o2.h(STD_TEMPERATURE)))
std_formation_entropy_h2ov = (h2ov.s(STD_TEMPERATURE) -
                               (h2.s(STD_TEMPERATURE) +
                                0.5 * o2.s(STD_TEMPERATURE)))

std_formation_gibbs_h2ov = (std_formation_enthalpy_h2ov -
                            STD_TEMPERATURE * std_formation_entropy_h2ov)

std_formation_enthalpy_h2ol = (h2ol.h(STD_TEMPERATURE) -
                               (h2.h(STD_TEMPERATURE) +
                                0.5 * o2.h(STD_TEMPERATURE)))
std_formation_entropy_h2ol = (h2ol.s(STD_TEMPERATURE) -
                               (h2.s(STD_TEMPERATURE) +
                                0.5 * o2.s(STD_TEMPERATURE)))
std_formation_gibbs_h2ol = (std_formation_enthalpy_h2ol -
                            STD_TEMPERATURE * std_formation_entropy_h2ol)


def h2_hhv(temperature):
    """
    Calculate the hydrogen higher heating value voltage.

    Parameters:
    -----------
    temperature : float
        Temperature of the cell in Kelvin (K).
    
    Returns:
    --------
    float
        Hydrogen higher heating value voltage in Volts (V).
    """
    return h2ol.h(temperature) - 0.5 * o2.h(temperature) - h2.h(temperature)

h2_hhv = np.vectorize(h2_hhv)

def h2_lhv(temperature):
    """
    Calculate the hydrogen lower heating value voltage.

    Parameters:
    -----------
    temperature : float
        Temperature of the cell in Kelvin (K).
    
    Returns:
    --------
    float
        Hydrogen lower heating value voltage in Volts (V).
    """
    return h2ov.h(temperature) - 0.5 * o2.h(temperature) - h2.h(temperature)
h2_lhv = np.vectorize(h2_lhv)

def calculate_reversible_cell_voltage(
    temperature,
    activities_ratio):
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
        E_rev = (-ΔG° + ΔS°(T - T°) + RT ln(Q)) / (2F)
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
    gibbs_formation_h2ol = (- std_formation_gibbs_h2ol +
                             std_formation_entropy_h2ol * (temperature - STD_TEMPERATURE))

    reversible_cell_voltage = (gibbs_formation_h2ol +
                               ct.gas_constant * temperature *
                               np.log(activities_ratio)) / (2 * ct.faraday)

    return reversible_cell_voltage


def calculate_tafel_slope(
        temperature,
        number_of_electrons,
        charge_transfer_coeff):
    """
    Calculate the Tafel slope for an electrochemical reaction.

    Parameters:
    -----------
    temperature : float
        Temperature in Kelvin (K).
    number_of_electrons : int
        Number of electrons transferred in the electrochemical reaction.
    charge_transfer_coefficient : float
        Symmetry factor or charge transfer coefficient (dimensionless).

    Returns:
    --------
    float
        The Tafel slope in Volts (V/decade).
    """
    return 2.303 * (ct.gas_constant * temperature /
            (charge_transfer_coeff * ct.faraday))

def calculate_tafel_overpotential(
        current_density,
        exchange_current_density,
        temperature,
        number_of_electrons,
        charge_transfer_coeff):
    """
    Calculate the Tafel overpotential for an electrochemical reaction.
    This is an approximation of the Butler-Volmer is valid for high 
    overpotentials (> 0.1 V). 

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
                   (charge_transfer_coeff * ct.faraday))
    return tafel_slope * (np.asinh(current_density / exchange_current_density / 2) if charge_transfer_coeff == 0.5
                          else np.log(np.maximum(current_density / exchange_current_density,1)))


@dataclass
class ElectrochemicalReaction:
    """
    A dataclass representing the parameters of an electrochemical reaction.

    Attributes:
    -----------
    reference_exchange_current_density : float
        The reference exchange current density in Amperes per square meter (A/m²).
    activation_energy : float, optional, default=0.0741
        The activation energy of the reaction in Joules (J/kmol). 
    reaction_order : float, optional, default=1.0
        The reaction order with respect to the reactant activity (dimensionless).
    reference_activity : float, optional, default=1.0
        The reference reactant activity (dimensionless). Typically set to 1 for 
        normalized activities or standard concentrations.
    reference_temperature : float, optional, default=300
        The reference temperature in Kelvin (K), for the reference exchange current density. 
    number_of_electrons : int, optional, default=2
        Number of electrons transferred in the electrochemical reaction.
    charge_transfer_coeff : float, optional, default=0.5
        Symmetry factor or charge transfer coefficient (dimensionless).

    Methods:
    --------
    exchange_current_density(temperature, reactant_activity):
        Calculate the exchange current density for the reaction under specific conditions.

    tafel_overpotential(current_density, temperature, reactant_activity):
        Calculate the Tafel overpotential for the reaction under specific conditions.
 

    Example:
    --------
    >>> reaction = ElectrochemicalReaction(
    ...     reference_exchange_current_density=1e-4,
    ...     activation_energy=50e6,
    ...     number_of_electrons=2,
    ...     charge_transfer_coeff=0.5
    ... )
    """
    reference_exchange_current_density: float = 0.0741
    activation_energy: float = 67.e6
    reaction_order: float = 0.54
    reference_activity: float = 1.e5
    reference_temperature: float = 353.15
    number_of_electrons: int = 2
    charge_transfer_coeff: float = 0.5

    def exchange_current_density(self, temperature, reactant_activity):
        """
        Calculate the exchange current density for the electrochemical reaction.

        Parameters:
        -----------
        temperature : float
            Temperature in Kelvin (K).
        reactant_activity : float
            Activity of the reactant (dimensionless), typically a ratio or concentration.

        Returns:
        --------
        float
            The exchange current density in Amperes per square meter (A/m²).

        Notes:
        ------
        This method uses the `calculate_exchange_current_density` function and passes the 
        dataclass instance (`self`) to encapsulate the reaction parameters.

        Example:
        --------
        >>> reaction = ElectrochemicalReaction(
            reference_exchange_current_density=1e-4, 
            activation_energy=50e6
        )
        >>> i0 = reaction.exchange_current_density(temperature=310, reactant_activity=0.5)
        """
        return calculate_exchange_current_density(temperature, reactant_activity, self)
    
    def tafel_slope(self, temperature):
        """
        Calculate the Tafel slope for the electrochemical reaction.

        Parameters:
        -----------
        temperature : float
            Temperature in Kelvin (K).

        Returns:
        --------
        float
            The Tafel slope in Volts (V/decade).
        """
        return calculate_tafel_slope(temperature,
                                     self.number_of_electrons,
                                     self.charge_transfer_coeff)

    def tafel_overpotential(self, current_density, temperature, reactant_activity):
        """
        Calculate the Tafel overpotential for the electrochemical reaction.

        Parameters:
        -----------
        current_density : float
            The operating current density in Amperes per square meter (A/m²).
        temperature : float
            Temperature in Kelvin (K).
        reactant_activity : float
            Activity of the reactant (dimensionless), typically a ratio or concentration.

        Returns:
        --------
        float
            The Tafel overpotential in Volts (V).

        Example:
        --------
        >>> reaction = ElectrochemicalReaction(
        ...     reference_exchange_current_density=1e-4,
        ...     activation_energy=50e6,
        ...     number_of_electrons=2,
        ...     charge_transfer_coeff=0.5
        ... )
        >>> eta = reaction.tafel_overpotential(
            current_density=1e-3, temperature=310, reactant_activity=0.5
        )
        """
        exchange_current_density = self.exchange_current_density(temperature,reactant_activity)
        return calculate_tafel_overpotential(
            current_density,
            exchange_current_density,
            temperature,
            self.number_of_electrons, 
            self.charge_transfer_coeff)

def calculate_exchange_current_density(
    temperature: float,
    reactant_activity: float,
    params: ElectrochemicalReaction):

    """
    Calculate the exchange current density for an electrochemical reaction with a single reactant. 

    Parameters:
    -----------
    temperature : float
        Temperature in Kelvin (K).
    reactant_activity : float
        Activity of the reactant (dimensionless), typically a ratio or concentration.
    params : ElectrochemicalReaction
        An instance of the `ElectrochemicalReaction` dataclass containing the reaction parameters:
        - reference_exchange_current_density : float
            The reference exchange current density in Amperes per square meter (A/m²).
        - activation_energy : float
            Activation energy in Joules (J).
        - reaction_order : float
            Reaction order with respect to the reactant activity (dimensionless).
        - reference_activity : float
            Reference reactant activity (dimensionless). Defaults to 1.0.
        - reference_temperature : float
            Reference temperature in Kelvin (K). Defaults to 300 K.

    Returns:
    --------
    float
        The exchange current density in Amperes per square meter (A/m²).

    """
    activity_correction = np.maximum(reactant_activity / params.reference_activity, 
                                     1e-12) ** params.reaction_order
    arrhenius_term = calculate_arrhenius_term(params.activation_energy,
                                              temperature,
                                              params.reference_temperature)
    return params.reference_exchange_current_density * activity_correction * arrhenius_term
