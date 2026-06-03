"""
Ionomer component dataclasses.

Classes
-------
Ionomer : Base dataclass holding ionomer material parameters.
PFSAIonomer : Parameter dataclass for perfluorosulfonic acid ionomers (e.g. Nafion).

These classes carry physical and electrochemical parameters only.
All computation (conductivity, permeability, water uptake, drag
coefficients, etc.) is implemented in the corresponding model classes
in :mod:`marapendi.models.membrane`.
"""

from dataclasses import dataclass, field
import numpy as np
import cantera as ct

from marapendi.tools.tools import Updatable


@dataclass
class Ionomer(Updatable):
    """
    Base dataclass storing material parameters for an ionomer phase.

    These parameters are consumed by the model classes in
    :mod:`marapendi.models.membrane` (e.g. ``IonomerModel``,
    ``MembraneModel``, ``PFSAModel``) to compute derived quantities such
    as charge conductivity, water diffusivity, and sorption isotherms.

    Attributes
    ----------
    rho_dry_ion : float
        Dry density of the ionomer [kg/m³].
    EW_ion : float
        Equivalent weight [kg/kmol].
    darken_num_ion, darken_den_ion : np.ndarray
        Polynomial coefficients (numerator / denominator) for the Darken
        correction to water diffusivity.
    sorption_coeffs_ion : np.ndarray
        Polynomial coefficients for the equilibrium water-content isotherm.
    lmbd_liq_ref_ion : float
        Reference liquid-equilibrium water content [mol H₂O / mol SO₃⁻].
    D_lmbd_ref_ion : float
        Reference water diffusivity [m²/s].
    k_des_ref_ion : float
        Reference desorption rate constant [m/s].
    sigma_ref_ion : float
        Reference ionic conductivity [S/m].
    T_ref_sigma_ion, T_ref_D_ion, T_ref_des_ion : float
        Reference temperatures for conductivity, diffusivity, and
        desorption Arrhenius corrections [K].
    E_act_ion : float
        Activation energy for ionic conductivity [J/kmol].
    f_v_perc_ion : float
        Percolation threshold for conductivity (volume fraction) [-].
    n_sigma_ion : float
        Exponent in the percolation conductivity model.
    c_ion : float
        Molar concentration of ionomer [kmol/m³] — computed in
        ``__post_init__`` from ``rho_dry_ion / EW_ion``.
    V_ion : float
        Molar volume of dry ionomer [m³/kmol] — computed in
        ``__post_init__``.
    charge_ion : str
        Charge carrier type (``'proton'`` or ``'hydroxide'``).
    """
    rho_dry_ion: float = field(default=None)
    EW_ion: float = field(default=None)
    darken_num_ion: np.ndarray = field(default=None)
    darken_den_ion: np.ndarray = field(default=None)
    sorption_coeffs_ion: np.ndarray = field(default=None)
    lmbd_liq_ref_ion: float = field(default=None)
    D_lmbd_ref_ion: float = field(default=None)
    k_des_ref_ion: float = field(default=None)
    sigma_ref_ion: float = 1e-12
    T_ref_sigma_ion: float = field(default=None)
    T_ref_D_ion: float = field(default=None)
    T_ref_des_ion: float = field(default=None)
    E_act_ion: float = field(default=None)
    E_act_cond_ion: float = field(default=None)
    f_v_perc_ion: float = field(default=None)
    n_sigma_ion: float = field(default=None)
    c_ion: float = field(default=None)
    V_ion: float = field(default=None)
    charge_ion: str = field(default='proton')

    def __post_init__(self):
        """Compute ``c_ion`` and ``V_ion`` from density and equivalent weight.
        Skipped when either field has not been supplied yet."""
        if self.rho_dry_ion is not None and self.EW_ion is not None:
            self.c_ion = self.rho_dry_ion / self.EW_ion  # kmol/m³
            self.V_ion = 1. / self.c_ion  # m³/kmol
        if not self.E_act_cond_ion: 
            self.E_act_cond_ion = self.E_act_ion


@dataclass
class PFSAIonomer(Ionomer):
    """
    Parameter dataclass for perfluorosulfonic acid ionomers (e.g. Nafion).

    Inherits all fields from :class:`Ionomer`.  The corresponding
    computation methods (``o2_permeability``, ``h2_permeability``,
    ``calculate_electroosmotic_drag_coefficient``) live in
    :class:`marapendi.models.membrane.PFSAModel`.
    """

    def o2_permeability(self, f_v, T=353.15):
        RT = ct.gas_constant * T
        return (6.74e-15 * np.exp(-21280e3/RT) + f_v * 50.5e-15 * np.exp(-20470e3/RT))

    def h2_permeability(self, T: float, f_v: float) -> float:
        RT = ct.gas_constant * T
        return (15.7e-15 * np.exp(-20280e3/RT) + f_v * 45e-15 * np.exp(-18930e3/RT))

    def calculate_electroosmotic_drag_coefficient(self, T, lmbd):
        return (0.02 * T - 3.86) / 22.5 * lmbd
