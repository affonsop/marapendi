"""
Module providing some useful auxiliary functions. 
"""

import cantera as ct
import numpy as np

def calculate_arrhenius_term(
        activation_energy,
        temperature,
        reference_temperature
):
    
    return np.exp(activation_energy / ct.gas_constant * (1/reference_temperature - 1/temperature)) 