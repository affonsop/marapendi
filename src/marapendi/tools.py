"""
Module providing some useful auxiliary functions.
"""

import math
import numpy as np
from marapendi.thermo.constants import GAS_CONSTANT, FARADAY_CONSTANT

def arrhenius_term(
        activation_energy,
        temperature,
        reference_temperature
):
    exponent = activation_energy / GAS_CONSTANT * (1/reference_temperature - 1/temperature)
    if np.ndim(exponent) == 0:
        return math.exp(float(exponent))
    return np.exp(exponent)

def potential_activation(
        transfer_coefficient,
        electron_number,
        temperature,
        potential_difference):
    return np.exp(transfer_coefficient * electron_number * potential_difference * FARADAY_CONSTANT / (GAS_CONSTANT * temperature))


def sigmoid(x, x_inflection, slope_parameter):
    return 1/(1 + np.exp(-slope_parameter * (x - x_inflection)))