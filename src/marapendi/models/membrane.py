"""
Ionomer and membrane physics models for PEM/AEM fuel cells and electrolysers.

Classes
-------
IonomerModel
    Stateless strategy class for ionomer thermophysical calculations
    (water volume fraction, wet density, expansion, ionic conductivity).
MembraneModel
    Extends ``IonomerModel`` with membrane-specific transport equations
    (H₂ permeation, water diffusivity, sorption coefficient, water
    resistance).
PFSAModel
    Extends ``MembraneModel`` with PFSA-specific correlations for O₂
    and H₂ permeability and electroosmotic drag (Nafion / Aquivion).

Design note
-----------
Model classes are stateless strategy objects.  They accept component
dataclasses (:class:`~marapendi.components.ionomer.Ionomer`,
:class:`~marapendi.components.membrane.Membrane`) as explicit arguments
and return computed quantities without storing any state.  This keeps
component parameters separate from the equations that act on them.
"""

from dataclasses import dataclass
import numpy as np
import cantera as ct

from marapendi.models.electrochemistry import calculate_reversible_cell_voltage, STD_PRESSURE
from marapendi.components.electrolyte import ElectrolyteSolution
from marapendi.models.electrochemistry import enthalpy_condensation
from marapendi.tools.tools import arrhenius_term, polyval_vec
from marapendi.models.water import water_molecular_weight, water_molar_volume, water_density

@dataclass
class IonomerModel:
    """
    Stateless strategy class for ionomer thermophysical calculations.

    All methods accept ionomer parameters explicitly via a component
    argument so the same equations work for both membrane and
    catalyst-layer ionomers.
    """

    def water_vol_fraction(self, lmbd, V_w, V_ion):
        """Water volume fraction in the ionomer as a function of water content.

        Parameters
        ----------
        lmbd : float
            Water content [mol H₂O / mol SO₃⁻].
        V_w : float
            Molar volume of liquid water [m³/kmol].
        V_ion : float
            Molar volume of dry ionomer [m³/kmol].

        Returns
        -------
        float
            Water volume fraction [-].
        """
        lmbd_V_w = lmbd * V_w
        return lmbd_V_w / (V_ion + lmbd_V_w)

    def wet_density(self, lmbd, T, ionomer):
        """Density of the hydrated ionomer [kg/m³]."""
        water_mass = water_molecular_weight * lmbd
        return ionomer.EW_ion + water_mass / (ionomer.EW_ion / ionomer.bulk_density + water_mass / water_density(T))

    def heat_of_adsorption(self, T, ionomer):
        """Enthalpy of water adsorption into the ionomer [J/kmol].

        Approximated as the enthalpy of condensation at temperature *T*.
        """
        return enthalpy_condensation(T)

    def wet_expansion_factor(self, lmbd, T, ionomer):
        """Volumetric swelling factor of the ionomer due to water uptake [-]."""
        water_mass = water_molecular_weight * lmbd
        return 1 + ionomer.rho_dry_ion * water_mass / ionomer.EW_ion / water_density(T)

    def charge_conductivity(self, f_v, T, charge, ionomer):
        """Effective ionic conductivity of the ionomer [S/m].

        Uses a percolation model with an Arrhenius temperature correction.
        Returns zero (1/inf) for the non-native charge carrier.

        Parameters
        ----------
        f_v : float
            Water volume fraction in the ionomer [-].
        T : float
            Temperature [K].
        charge : str
            Charge carrier (``'proton'`` or ``'hydroxide'``).
        ionomer : Ionomer
            Ionomer parameter dataclass.
        """
        charge_conductivity = ionomer.sigma_ref_ion * np.maximum(0.01, f_v - ionomer.f_v_perc_ion) ** ionomer.n_sigma_ion
        return (charge_conductivity if charge == ionomer.charge_ion else (1/np.inf)) * arrhenius_term(ionomer.E_act_ion, T, ionomer.T_ref_sigma_ion)
    
@dataclass
class MembraneModel(IonomerModel):
    """
    Membrane transport model for PEM/AEM cells.

    Extends :class:`IonomerModel` with membrane-specific equations for
    water diffusivity, sorption isotherms, H₂ crossover, and the
    effective water-transport resistance.

    References
    ----------
    Wei, Z. et al. (2023) — water diffusivity and sorption correlations.
    """

    def calculate_h2_permeation_flux(self, T_memb, f_v_memb, p_h2, memb_thickness):
        """H₂ crossover flux through the membrane [kmol m⁻² s⁻¹].

        Parameters
        ----------
        T_memb : float
            Membrane temperature [K].
        f_v_memb : float
            Water volume fraction in the membrane [-].
        p_h2 : float
            H₂ partial pressure at the anode [Pa].
        memb_thickness : float
            Membrane thickness [m].
        """
        h2_permeability = self.h2_permeability(T_memb, f_v_memb)
        return h2_permeability * p_h2 / memb_thickness

    def diffusion_coefficient(self, lmbd, f_v, T, darken_num, darken_den, alpha_lmbd, E_act, T_ref=303.15):
        """Effective water diffusivity in the ionomer [m²/s].

        Applies the Darken correction (polynomial ratio) scaled by the
        water volume fraction and an Arrhenius temperature factor.
        See Wei et al. (2023).
        """
        return polyval_vec(darken_num[:,::-1], lmbd) / polyval_vec(darken_den[:,::-1], lmbd) * alpha_lmbd * f_v * arrhenius_term(E_act, T, T_ref)

    def sorption_coefficient(self, f_v, T, k_des, E_act, T_ref=303.15):
        """Surface desorption rate coefficient for water [m/s].

        See Wei et al. (2023).
        """
        return k_des * f_v * arrhenius_term(E_act, T, T_ref)

    def calculate_membrane_water_resistance(self, D_lmbd, thickness, eps_ion, c_ion, tort_ion):
        """Membrane water-transport resistance [s/m²·kmol].

        Parameters
        ----------
        D_lmbd : float
            Water diffusivity in the ionomer [m²/s].
        thickness : float
            Membrane thickness [m].
        eps_ion : float
            Ionomer volume fraction [-].
        c_ion : float
            Dry ionomer molar concentration [kmol/m³].
        tort_ion : float
            Ionomer tortuosity [-].
        """
        D_eff = D_lmbd * c_ion * eps_ion / tort_ion
        return thickness / D_eff

    def equilibrium_water_content(self, rh, sorption_coeffs):
        """Equilibrium water content from vapour-phase relative humidity.

        Polynomial isotherm evaluated at *rh*, clipped to [0, 1].
        See Wei et al. (2023).

        Parameters
        ----------
        rh : float
            Relative humidity [-].
        sorption_coeffs : np.ndarray
            Polynomial coefficients (shape: [n_nodes, degree+1]).

        Returns
        -------
        float
            Equilibrium water content λ [mol H₂O / mol SO₃⁻].
        """
        rh = np.clip(rh, 0, 1)
        return polyval_vec(sorption_coeffs[:,::-1], rh)

    def liquid_equilibrium_water_content(self, reference_liquid_water_content):
        """Equilibrium water content for a liquid-contacted interface.

        Returns the reference liquid saturation value unchanged.
        See Wei et al. (2023).
        """
        return reference_liquid_water_content
    
@dataclass
class PFSAModel(MembraneModel):
    """
    PFSA membrane model (Nafion / Aquivion) for PEM cells.

    Extends :class:`MembraneModel` with permeability correlations and
    an electroosmotic drag coefficient specific to perfluorosulfonic
    acid membranes.

    References
    ----------
    Goshtasbi, A. et al. (2020) — O₂ and H₂ permeability correlations.
    Ferrara, N. et al. (2018) — electroosmotic drag coefficient.
    """

    def o2_permeability(self, f_v, T=353.15):
        """O₂ permeability in the PFSA membrane [mol m⁻¹ s⁻¹ Pa⁻¹].

        Two-phase (dry + hydrated) Arrhenius correlation.
        See Goshtasbi et al. (2020).

        Parameters
        ----------
        f_v : float
            Water volume fraction in the membrane [-].
        T : float
            Temperature [K].
        """
        RT = ct.gas_constant * T
        return (6.74e-15 * np.exp(-21280e3/RT) + f_v * 50.5e-15 * np.exp(-20470e3/RT))

    def h2_permeability(self, T: float, f_v: float) -> float:
        """H₂ permeability in the PFSA membrane [mol m⁻¹ s⁻¹ Pa⁻¹].

        Two-phase (dry + hydrated) Arrhenius correlation.
        See Goshtasbi et al. (2020).

        Parameters
        ----------
        T : float
            Temperature [K].
        f_v : float
            Water volume fraction in the membrane [-].
        """
        RT = ct.gas_constant * T
        return (15.7e-15 * np.exp(-20280e3/RT) + f_v * 45e-15 * np.exp(-18930e3/RT))

    def calculate_electroosmotic_drag_coefficient(self, T, lmbd):
        """Electroosmotic drag coefficient [mol H₂O / mol H⁺].

        Linear correlation with temperature and water content.
        See Ferrara et al. (2018).

        Parameters
        ----------
        T : float
            Temperature [K].
        lmbd : float
            Water content [mol H₂O / mol SO₃⁻].
        """
        return (0.02 * T - 3.86) / 22.5 * lmbd
