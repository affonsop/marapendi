"""
Module providing some useful auxiliary functions. 
"""

import cantera as ct
import numpy as np

def arrhenius_term(
        activation_energy,
        temperature,
        reference_temperature
):
    
    return np.exp(activation_energy / ct.gas_constant * (1/reference_temperature - 1/temperature)) 

def potential_activation(
        transfer_coefficient, 
        electron_number, 
        temperature, 
        potential_difference): 
    return np.exp(transfer_coefficient * electron_number * potential_difference * ct.faraday / (ct.gas_constant * temperature))


def sigmoid(x, x_inflection, slope_parameter):
    return 1/(1 + np.exp(-slope_parameter * (x - x_inflection)))