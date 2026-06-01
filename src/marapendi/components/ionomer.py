"""
Module providing classes to model ionomers in catalyst layers.

Classes
-------
CatalystLayerIonomer : Base class for ionomer properties in catalyst layers.
PFSAIonomer : Represents a perfluorosulfonic acid ionomer (e.g. Nafion).
PAPIonomer : Represents a poly(aryl piperidinium) (PA) ionomer.

Each class provides methods to compute water content-dependent properties
like proton or hydroxide conductivity, oxygen permeability, and equilibrium hydration.
"""

from dataclasses import dataclass, field
import numpy as np
import cantera as ct

from marapendi.models.electrochemistry import enthalpy_condensation
from marapendi.tools.tools import arrhenius_term, Updatable
from .water import water_molecular_weight, water_molar_volume, water_density


@dataclass 
class Ionomer(Updatable): 
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
    f_v_perc_ion: float = field(default=None)
    n_sigma_ion: float = field(default=None)
    c_ion: float = field(default=None)
    V_ion: float = field(default=None)
    charge_ion: str = field(default='proton')

    def __post_init__(self):
        """
        Compute derived properties of the membrane after initialization.
        Skipped when dry_density or equiv_weight have not been supplied yet.
        """
        if self.rho_dry_ion is not None and self.EW_ion is not None:
            self.c_ion = self.rho_dry_ion / self.EW_ion  # kmol/m³
            self.V_ion = 1. / self.c_ion  # m³/kmol
    
    def wet_density(self, lmbd, T):
        water_mass = water_molecular_weight * lmbd
        return self.EW_ion + water_mass / (self.EW_ion / self.bulk_density + water_mass / water_density(T))

    def heat_of_adsorption(self, T):
        return enthalpy_condensation(T)

    def wet_expansion_factor(self, lmbd, T):
        water_mass = water_molecular_weight * lmbd
        return 1 + self.rho_dry_ion * water_mass / self.EW_ion / water_density(T)

    def charge_conductivity(self, f_v, T, charge):
        charge_conductivity = self.sigma_ref_ion * np.maximum(0.01, f_v - self.f_v_perc_ion) ** self.n_sigma_ion
        return (charge_conductivity if charge == self.charge_ion else (1/np.inf)) * arrhenius_term(self.E_act_ion, T, self.T_ref_sigma_ion)

@dataclass
class PFSAIonomer(Ionomer):

    def o2_permeability(self, f_v, T=353.15):
        RT = ct.gas_constant * T
        return (6.74e-15 * np.exp(-21280e3/RT) + f_v * 50.5e-15 * np.exp(-20470e3/RT))

    def h2_permeability(self, T: float, f_v: float) -> float:
        RT = ct.gas_constant * T
        return (15.7e-15 * np.exp(-20280e3/RT) + f_v * 45e-15 * np.exp(-18930e3/RT))

    def calculate_electroosmotic_drag_coefficient(self, T, lmbd):
        return (0.02 * T - 3.86) / 22.5 * lmbd
