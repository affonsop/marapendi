"""
Gas-mixture viscosity model.

Classes
-------
GasMixtureModel
    Computes mixture dynamic viscosity from per-species viscosity
    polynomials obtained from Cantera's GRI 3.0 transport database.

Cantera stores viscosity transport data as a polynomial in ``log(T)`` whose
evaluated value equals ``sqrt(μ_k) · T^{-1/4}``, so the dynamic viscosity is::

    p_k(log T) = sqrt(μ_k) · T^{-1/4}
    μ_k        = p_k(log T)² · sqrt(T)

Mixture viscosity uses the mole-fraction / √M weighting rule
(accurate for non-polar species at moderate pressures):

    μ_mix = Σ_k [ x_k √M_k μ_k ] / Σ_k [ x_k √M_k ]
"""
from dataclasses import dataclass

import cantera as ct
import numpy as np

from marapendi.tools.tools import polyval_vec


@dataclass
class GasMixtureModel:
    """
    Mixture dynamic viscosity for a user-specified list of species.

    Parameters
    ----------
    species : list[str]
        Species names as recognised by GRI 3.0 (case-insensitive).

    Attributes
    ----------
    viscosity_polynomials : ndarray, shape (n_species, n_coeffs)
        Transport polynomial coefficients in **descending** degree order
        (suitable for Horner evaluation), one row per species.
    """

    species: list

    def __post_init__(self):
        self.i = {s: i for i, s in enumerate(self.species)}

        # Build a fresh GRI 3.0 solution to query transport polynomials.
        gas_ref = ct.Solution("gri30.yaml")

        # get_viscosity_polynomial returns [a0, a1, …] in ascending degree
        # order.  Reverse to descending for Horner's method.
        self.viscosity_polynomials = np.array([
            gas_ref.get_viscosity_polynomial(
                gas_ref.species_index(sp.upper())
            )[::-1]
            for sp in self.species
        ])  # (n_species, n_coeffs)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def calculate_mixture_kinematic_viscosity(
        self,
        T: np.ndarray,
        x_k: np.ndarray,
        M_k: np.ndarray,
    ) -> np.ndarray:
        """Mixture dynamic viscosity [Pa·s] via mole-fraction/√M weighting.

        Parameters
        ----------
        T : ndarray, shape (n_layers, n_measurements)
            Temperature [K].
        x_k : ndarray, shape (n_layers, n_species, n_measurements)
            Mole fractions (need not sum to 1; only relative weights matter).
        M_k : ndarray, shape (n_species,)
            Molar masses [kg/kmol or g/mol — consistent units throughout].

        Returns
        -------
        ndarray, shape (n_layers, n_measurements)
            Mixture dynamic viscosity [Pa·s].
        """
        n_layers, n_meas = T.shape
        n_species = self.viscosity_polynomials.shape[0]
        logT_flat = np.log(T).ravel()   # (n_layers * n_meas,)

        # Evaluate all n_species polynomials at every (layer, meas) point.
        # polyval_vec accepts (N, m) xs: row i = polynomial i, column j = point j.
        # Broadcasting the same n_lm points across all n_species rows — no
        # need to repeat the coefficient matrix.
        xs_2d  = np.tile(logT_flat, (n_species, 1))          # (n_species, n_lm)
        p_flat = polyval_vec(self.viscosity_polynomials, xs_2d)  # (n_species, n_lm)

        # Reshape to (n_layers, n_species, n_meas)
        p_logT = p_flat.reshape(n_species, n_layers, n_meas).transpose(1, 0, 2)

        sqrtT = np.sqrt(T)[:, np.newaxis, :]   # (n_layers, 1, n_measurements)
        mu_k = p_logT ** 2 * sqrtT             # (n_layers, n_species, n_measurements)

        # Mole-fraction / √M weighted average
        sqrt_M = np.sqrt(M_k)[np.newaxis, :, np.newaxis]  # (1, n_species, 1)
        weights = x_k * sqrt_M                             # (n_layers, n_species, n_measurements)

        mu_mix = (
            np.sum(weights * mu_k, axis=1)
            / np.sum(weights, axis=1)
        )  # (n_layers, n_measurements)

        return mu_mix
