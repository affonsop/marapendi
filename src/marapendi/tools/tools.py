"""
Module providing some useful auxiliary functions. 
"""

import cantera as ct
import numpy as np
from dataclasses import is_dataclass
from typing import Dict, Any, get_origin, get_args
 

class Updatable: 
    def update_from_dict(self, params: Dict[str, Any]) -> None:
        """Recursively update cell properties from a nested dictionary."""
        for key, value in params.items():
            if hasattr(self, key):     
                attr = getattr(self, key)
            
            if isinstance(value, dict):
                # If it's a dataclass, recurse into it
                if is_dataclass(attr) and not isinstance(attr, type):
                    attr.update_from_dict(value)
                else:
                    raise TypeError(f"Cannot update '{key}' with dict")
            else:
                # Set scalar values directly
                setattr(self, key, value)

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

def polyval_vec(coeffs, xs):
    """
    Vectorised polyval: evaluate one polynomial per row.

    Parameters
    ----------
    coeffs : (N, D) array  — N polynomials, each of degree D-1,
                             coefficients in descending order (like np.polyval)
    xs     : (N,)   array  — one evaluation point per polynomial

    Returns
    -------
    (N,) array of evaluated values
    """
   
    # ── Horner's method (faster, more numerically stable) ──────────────────
    coeffs = np.asarray(coeffs, dtype=float)
    xs     = np.asarray(xs,     dtype=float)
    result = np.zeros((len(xs), 1))
    
    for col in coeffs.T: # iterate over coefficient columns
        result = result * xs + col[:,np.newaxis]
    return result