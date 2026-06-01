"""
CellState — snapshot of all derived thermodynamic quantities at a given
instant, produced by ``TransientCellModel._compute_derived_quantities``.

Storing both full per-layer arrays and pre-computed layer slices means
every downstream method (``_compute_resistances``, ``_compute_fluxes``,
``VoltageModel`` methods, …) can receive a single ``state`` object instead
of an unpacked tuple of 10+ arrays.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class CellState:
    """
    Thermodynamic state derived from the ODE state vector ``x``.

    Full arrays have shape ``(n_layers, ...)``.
    Sliced fields (suffixed with a layer label) have shape ``(...,)``
    and are numpy views into the corresponding full array.

    Parameters
    ----------
    lmbd : ndarray
        Ionomer water content [–].
    T : ndarray
        Temperature [K].
    cg_k : ndarray, shape (n_layers, n_species, ...)
        Gas-phase molar concentrations [kmol m⁻³].
    s : ndarray
        Liquid water saturation [–].
    iF : ndarray or float
        Faradaic current density normalised by Faraday constant [kmol m⁻² s⁻¹].
    p_g : ndarray
        Total gas pressure [Pa].
    p_g_k : ndarray, shape (n_layers, n_species, ...)
        Species partial pressures [Pa].
    D_g_k : ndarray, shape (n_layers, n_species, ...)
        Gas-phase diffusion coefficients per species.
    c_sat : ndarray
        Water vapour saturation concentration [kmol m⁻³].
    c_v : ndarray
        Water vapour concentration [kmol m⁻³].
    rh : ndarray
        Relative humidity [–].
    rho_l : ndarray
        Liquid water density [kg m⁻³].
    nu_l : ndarray
        Liquid water kinematic viscosity [m² s⁻¹].
    M_k : ndarray, shape (n_layers, n_species, ...)
        Species molar-mass fractions [kg kmol⁻¹].
    f_v : ndarray
        Ionomer water volume fraction [–].

    Pre-sliced convenience fields
    ------------------------------
    T_memb, T_ca_cl, T_an_cl : ndarray
    f_v_memb, f_v_ca_cl : ndarray
    lmbd_ca_cl : ndarray
    p_h2 : ndarray  — H2 partial pressure at anode CL
    p_o2_ca_cl : ndarray  — O2 partial pressure at cathode CL
    """

    # ---- normalised ODE state vector ----
    x:     np.ndarray  # shape (n_layers, n_variables, m)

    # ---- full arrays ----
    lmbd:  np.ndarray
    T:     np.ndarray
    cg_k:  np.ndarray
    s:     np.ndarray
    iF:    object          # float or ndarray
    p_g:   np.ndarray
    p_g_k: np.ndarray
    D_g_k: np.ndarray
    c_sat: np.ndarray
    c_v:   np.ndarray
    rh:    np.ndarray
    rho_l: np.ndarray
    nu_l:  np.ndarray
    M_k:   np.ndarray
    f_v:   np.ndarray

    # ---- pre-sliced fields ----
    T_memb:    np.ndarray
    T_ca_cl:   np.ndarray
    T_an_cl:   np.ndarray
    f_v_memb:  np.ndarray
    f_v_ca_cl: np.ndarray
    lmbd_ca_cl: np.ndarray
    p_h2:      np.ndarray
    p_o2_ca_cl: np.ndarray
