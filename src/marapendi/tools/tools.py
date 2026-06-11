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
    Vectorised polyval: evaluate N polynomials, each at one or more points.

    Parameters
    ----------
    coeffs : (N, D) array
        N polynomials of degree D-1; coefficients in descending order
        (highest degree first, like ``np.polyval``).
    xs : (N,) or (N, m) array
        Evaluation points.  A 1-D input of shape ``(N,)`` is treated as
        ``(N, 1)`` — one point per polynomial.  A 2-D input of shape
        ``(N, m)`` evaluates each polynomial at m points simultaneously.

    Returns
    -------
    (N, m) array
        ``result[i, j] = poly_i(xs[i, j])``.

    Notes
    -----
    The Horner step is ``result = result * xs + coeff_column``.  When xs is
    1-D the step becomes an ``(N, 1) × (N,)`` outer product, giving a wrong
    ``(N, N)`` result.  Promoting xs to at least 2-D with ``atleast_2d``
    keeps the multiplication element-wise.
    """
    coeffs = np.asarray(coeffs, dtype=float)
    xs     = np.atleast_2d(np.asarray(xs, dtype=float))  # (N, m)

    N = coeffs.shape[0]
    if xs.shape == (N, 1):
        # m=1: a pure-Python Horner loop is faster than numpy here, since
        # the arrays involved are too small to amortise numpy's per-call
        # overhead. Same evaluation order as the vectorised loop below, so
        # results are bit-identical.
        xs_flat = xs[:, 0].tolist()
        coeffs_list = coeffs.tolist()
        result = np.empty((N, 1))
        for i in range(N):
            x = xs_flat[i]
            row = coeffs_list[i]
            acc = row[0]
            for c in row[1:]:
                acc = acc * x + c
            result[i, 0] = acc
        return result

    result = np.zeros((len(xs), 1))
    for col in coeffs.T:                   # iterate over coefficient columns
        result = result * xs + col[:, np.newaxis]
    return result