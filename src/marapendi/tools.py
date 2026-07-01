"""
General-purpose auxiliary functions shared across marapendi modules.
"""

import math
import numpy as np
from marapendi.thermo.constants import GAS_CONSTANT, FARADAY_CONSTANT


def arrhenius_term(activation_energy, temperature, reference_temperature):
    """
    Arrhenius temperature-correction factor relative to a reference temperature.

    Parameters
    ----------
    activation_energy : float
        Activation energy in J/kmol.
    temperature : float
        Current temperature in K.
    reference_temperature : float
        Reference temperature at which the pre-exponential factor is defined, in K.

    Returns
    -------
    float
        Dimensionless correction factor (1 at reference_temperature).
    """
    exponent = activation_energy / GAS_CONSTANT * (1/reference_temperature - 1/temperature)
    if np.ndim(exponent) == 0:
        return math.exp(float(exponent))
    return np.exp(exponent)


def potential_activation(transfer_coefficient, electron_number, temperature, potential_difference):
    """
    Butler-Volmer exponential factor for a given overpotential.

    Parameters
    ----------
    transfer_coefficient : float
        Charge-transfer coefficient (dimensionless).
    electron_number : int
        Number of electrons transferred per elementary step.
    temperature : float
        Temperature in K.
    potential_difference : float
        Overpotential in V.

    Returns
    -------
    float
        Dimensionless exponential activation factor.
    """
    return np.exp(transfer_coefficient * electron_number * potential_difference * FARADAY_CONSTANT / (GAS_CONSTANT * temperature))


def sigmoid(x, x_inflection, slope_parameter):
    """
    Logistic sigmoid function.

    Parameters
    ----------
    x : float or array-like
        Input value.
    x_inflection : float
        Value of x at which the output equals 0.5.
    slope_parameter : float
        Controls the steepness of the transition. Larger values give a sharper step.

    Returns
    -------
    float or ndarray
        Output in the open interval (0, 1).
    """
    return 1/(1 + np.exp(-slope_parameter * (x - x_inflection)))
