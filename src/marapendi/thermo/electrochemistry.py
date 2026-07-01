"""
Electrochemistry correlations: reversible voltage, Butler-Volmer kinetics.

:func:`calculate_reversible_cell_voltage` computes the Nernst equilibrium
cell voltage. :class:`ElectrochemicalReaction` encapsulates Butler-Volmer
kinetics for a single reaction step (ORR or HOR) parameterised by exchange
current density, activation energy, reaction order and charge-transfer
coefficient.
"""
from dataclasses import dataclass
import numpy as np
from marapendi.tools import arrhenius_term
from marapendi.thermo.constants import (
    GAS_CONSTANT,
    FARADAY_CONSTANT,
    STD_TEMPERATURE,
    STD_FORMATION_GIBBS_H2OL,
    STD_FORMATION_ENTROPY_H2OL,
    H2_LHV_COEFFS,
    H2_HHV_COEFFS,
)

STD_PRESSURE = 1e5

std_formation_gibbs_h2ol = STD_FORMATION_GIBBS_H2OL
std_formation_entropy_h2ol = STD_FORMATION_ENTROPY_H2OL


def h2_hhv(temperature):
    """
    Higher heating value voltage of hydrogen at a given temperature.

    Parameters
    ----------
    temperature : float
        Temperature in K.

    Returns
    -------
    float
        HHV voltage in V.
    """
    return np.polyval(H2_HHV_COEFFS, temperature)


def h2_lhv(temperature):
    """
    Lower heating value value of hydrogen at a given temperature.

    Parameters
    ----------
    temperature : float
        Temperature in K.

    Returns
    -------
    float
        LHV voltage in J/kmol.
    """
    return np.polyval(H2_LHV_COEFFS, temperature)


def calculate_reversible_cell_voltage(temperature, activities_ratio):
    """
    Reversible (Nernst) cell voltage of a hydrogen fuel cell or electrolyser.

    Parameters
    ----------
    temperature : float
        Cell temperature in K.
    activities_ratio : float
        Reaction quotient Q = a_H2 * a_O2^0.5 / a_H2O (dimensionless).
        Typically computed from partial pressures normalised by a reference pressure.

    Returns
    -------
    float
        Reversible cell voltage in V.

    Notes
    -----
    Uses the thermodynamic relation::

        E_rev = (-ΔG° + ΔS°(T - T°) + RT ln Q) / (2F)

    where ΔG° and ΔS° are the standard Gibbs energy and entropy of formation
    of liquid water, and F is the Faraday constant.
    """
    gibbs_formation_h2ol = (- std_formation_gibbs_h2ol +
                             std_formation_entropy_h2ol * (temperature - STD_TEMPERATURE))

    reversible_cell_voltage = (gibbs_formation_h2ol +
                               GAS_CONSTANT * temperature *
                               np.log(activities_ratio)) / (2 * FARADAY_CONSTANT)

    return reversible_cell_voltage


def calculate_tafel_slope(temperature, number_of_electrons, charge_transfer_coeff):
    """
    Calculate the Tafel slope.

    Parameters
    ----------
    temperature : float
        Temperature in K.
    number_of_electrons : int
        Number of electrons transferred per reaction event.
    charge_transfer_coeff : float
        Charge-transfer (symmetry) coefficient (dimensionless).

    Returns
    -------
    float
        Tafel slope in V/decade.
    """
    return 2.303 * (GAS_CONSTANT * temperature /
            (number_of_electrons * charge_transfer_coeff * FARADAY_CONSTANT))


def calculate_tafel_overpotential(
        current_density,
        exchange_current_density,
        temperature,
        number_of_electrons,
        charge_transfer_coeff):
    """
    Activation overpotential from Butler-Volmer or Tafel kinetics.

    Uses the symmetric Butler-Volmer form (``arcsinh``) when
    ``charge_transfer_coeff == 0.5``, and the Tafel approximation
    (valid for |η| > ~0.1 V) otherwise.

    Parameters
    ----------
    current_density : float
        Operating current density in A/m².
    exchange_current_density : float
        Exchange current density i₀ in A/m².
    temperature : float
        Temperature in K.
    number_of_electrons : int
        Number of electrons transferred per reaction event.
    charge_transfer_coeff : float
        Charge-transfer coefficient (dimensionless).

    Returns
    -------
    float
        Activation overpotential in V.
    """
    tafel_slope = (GAS_CONSTANT * temperature /
                   (number_of_electrons * charge_transfer_coeff * FARADAY_CONSTANT))
    return tafel_slope * (np.asinh(current_density / exchange_current_density / 2)
                          if charge_transfer_coeff == 0.5
                          else np.log(np.maximum(current_density / exchange_current_density, 1)))


def calculate_linear_overpotential(
        current_density,
        exchange_current_density,
        temperature,
        number_of_electrons,
        charge_transfer_coeff):
    """
    Linear (low-overpotential) approximation of Butler-Volmer kinetics.

    Valid when |η| ≪ RT/(nF), i.e. η < ~0.01 V.

    Parameters
    ----------
    current_density : float
        Operating current density in A/m².
    exchange_current_density : float
        Exchange current density i₀ in A/m².
    temperature : float
        Temperature in K.
    number_of_electrons : int
        Number of electrons transferred per reaction event.
    charge_transfer_coeff : float
        Charge-transfer coefficient (dimensionless).

    Returns
    -------
    float
        Linear activation overpotential in V.
    """
    tafel_slope = (GAS_CONSTANT * temperature /
                   (number_of_electrons * charge_transfer_coeff * FARADAY_CONSTANT))
    return tafel_slope * current_density / exchange_current_density


@dataclass
class ElectrochemicalReaction:
    """
    Parameters of a single-reactant electrochemical reaction (ORR or HOR).

    Stores the kinetic constants required to evaluate exchange current density
    and activation overpotential via Butler-Volmer / Tafel expressions.

    Parameters
    ----------
    reference_exchange_current_density : float
        Reference exchange current density at ``reference_temperature`` and
        ``reference_activity`` in A/m² (per unit geometric area).
    activation_energy : float
        Arrhenius activation energy in J/kmol.
    reaction_order : float
        Reaction order with respect to the reactant activity (dimensionless).
    reference_activity : float
        Reactant activity at the reference state in Pa (or the same units as
        the activity passed to the kinetic methods).
    reference_temperature : float
        Temperature at which ``reference_exchange_current_density`` is defined, in K.
    number_of_electrons : int
        Electrons transferred per elementary step.
    charge_transfer_coeff : float
        Charge-transfer (symmetry) coefficient (dimensionless).
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
        Exchange current density at the given temperature and reactant activity.

        Parameters
        ----------
        temperature : float
            Temperature in K.
        reactant_activity : float
            Reactant activity (same units as ``reference_activity``).

        Returns
        -------
        float
            Exchange current density i₀ in A/m².
        """
        return calculate_exchange_current_density(temperature, reactant_activity, self)

    def tafel_slope(self, temperature):
        """
        Tafel slope at the given temperature.

        Parameters
        ----------
        temperature : float
            Temperature in K.

        Returns
        -------
        float
            Tafel slope in V.
        """
        return calculate_tafel_slope(temperature,
                                     self.number_of_electrons,
                                     self.charge_transfer_coeff)

    def tafel_overpotential(self, current_density, temperature, reactant_activity):
        """
        Activation overpotential for the given operating conditions.

        Parameters
        ----------
        current_density : float
            Operating current density in A/m².
        temperature : float
            Temperature in K.
        reactant_activity : float
            Reactant activity (same units as ``reference_activity``).

        Returns
        -------
        float
            Activation overpotential in V.
        """
        i0 = self.exchange_current_density(temperature, reactant_activity)
        return calculate_tafel_overpotential(
            current_density, i0, temperature,
            self.number_of_electrons, self.charge_transfer_coeff)

    def linear_overpotential(self, current_density, temperature, reactant_activity):
        """
        Linear (low-overpotential) activation overpotential.

        Parameters
        ----------
        current_density : float
            Operating current density in A/m².
        temperature : float
            Temperature in K.
        reactant_activity : float
            Reactant activity (same units as ``reference_activity``).

        Returns
        -------
        float
            Linear activation overpotential in V.
        """
        i0 = self.exchange_current_density(temperature, reactant_activity)
        return calculate_linear_overpotential(
            current_density, i0, temperature,
            self.number_of_electrons, self.charge_transfer_coeff)


def calculate_exchange_current_density(
        temperature: float,
        reactant_activity: float,
        params: ElectrochemicalReaction):
    """
    Exchange current density for a single-reactant reaction.

    Combines an Arrhenius temperature correction with a power-law activity
    dependence.

    Parameters
    ----------
    temperature : float
        Temperature in K.
    reactant_activity : float
        Reactant activity (same units as ``params.reference_activity``).
    params : ElectrochemicalReaction
        Kinetic parameters for the reaction.

    Returns
    -------
    float
        Exchange current density i₀ in A/m².
    """
    activity_correction = np.maximum(reactant_activity / params.reference_activity,
                                     1e-12) ** params.reaction_order
    arrhenius = arrhenius_term(params.activation_energy,
                               temperature,
                               params.reference_temperature)
    return params.reference_exchange_current_density * activity_correction * arrhenius
