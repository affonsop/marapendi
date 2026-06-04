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

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np
import cantera as ct

from marapendi.models.electrochemistry import calculate_reversible_cell_voltage, STD_PRESSURE
from marapendi.components.electrolyte import ElectrolyteSolution
from marapendi.models.electrochemistry import enthalpy_condensation
from marapendi.tools.tools import arrhenius_term, polyval_vec
from marapendi.models.water import water_molecular_weight, water_molar_volume, water_density

if TYPE_CHECKING:
    from marapendi.models.transient import TransientCellModel
    from marapendi.components.cell_state import CellState

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
        return (charge_conductivity if charge == ionomer.charge_ion else (1/np.inf)) * arrhenius_term(ionomer.E_act_cond_ion, T, ionomer.T_ref_sigma_ion)
    
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

    def diffusion_coefficient(self, lmbd, T, darken_num, darken_den, D_ref, E_act, T_ref=303.15):
        """Effective water diffusivity in the ionomer [m²/s].

        Evaluates the Darken-corrected diffusivity following Vetter & Schumacher
        (2019) Eq. 22 (fit to Mittelsteadt & Staser 2011 data for Nafion):

            Dλ = (num_poly(λ) / den_poly(λ)) · D_ref · exp(Ea/R·(1/T_ref − 1/T))

        The Bruggeman volume-fraction correction (ε_i) is applied downstream in
        ``calculate_membrane_water_resistance`` so it is **not** included here.

        Parameters
        ----------
        lmbd : ndarray
            Ionomer water content [mol/mol].
        T : ndarray
            Temperature [K].
        darken_num, darken_den : ndarray, shape (n_layers, n_coeffs)
            Polynomial coefficients in **ascending** degree order
            [c₀, c₁, c₂, …] (reversed internally before ``np.polyval``).
            For Vetter's Nafion fit: num = [0, 67.74, -32.03, 3.842],
            den = [103.37, -33.013, -2.115, 1.0].
        D_ref : ndarray
            Reference diffusivity [m²/s] (= 10⁻⁶ cm²/s = 10⁻¹⁰ m²/s for Nafion).
        E_act : ndarray
            Activation energy [J/kmol].
        T_ref : float
            Reference temperature [K] (default 303.15 K = 30 °C).
        """
        return (
            polyval_vec(darken_num[:, ::-1], lmbd)
            / polyval_vec(darken_den[:, ::-1], lmbd)
            * D_ref
            * arrhenius_term(E_act, T, T_ref)
        )

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

    def update_transport_matrices(self, state: CellState, cell, tm: TransientCellModel) -> None:
        """
        Fill state.R[:,i_lmbd], state.C[:,i_lmbd], state.S[:,i_lmbd] and
        state.S[:,i_cg[-1]]; compute and store state.J_des and state.lmbd_eq.
        """
        import numpy as np
        i_lmbd = tm.i_lmbd

        # Resistance: water diffusion through ionomer
        D_lmbd = self.diffusion_coefficient(
            state.lmbd, state.T,
            cell.darken_num_ion, cell.darken_den_ion,
            cell.D_lmbd_ref_ion, cell.E_act_ion,
        )
        state.R[:, i_lmbd, ...] = self.calculate_membrane_water_resistance(
            D_lmbd, cell.thickness, cell.eps_ion, cell.c_ion, cell.tort_ion,
        )

        # Capacity
        state.C[:, i_lmbd, ...] = cell.eps_ion * cell.c_ion
        state.C[~np.array(tm.ionomer_domain), i_lmbd, ...] = np.inf

        # Sorption flux J_des
        k_des = self.sorption_coefficient(
            state.f_v, state.T,
            cell.k_des_ref_ion, cell.E_act_ion, cell.T_ref_des_ion,
        )
        lmbd_eq = (
            (1 - state.s) * self.equilibrium_water_content(state.rh, cell.sorption_coeffs_ion)
            + state.s * self.liquid_equilibrium_water_content(cell.lmbd_liq_ref_ion)
        )
        state.lmbd_eq = lmbd_eq
        delta = state.lmbd - lmbd_eq
        rate_factor = np.where(delta > 0, 1.0, 3.53 / 14.2)
        J_des = k_des * delta * cell.c_ion * rate_factor
        state.J_des = J_des

        # Sources: ionomer water (absorption/desorption, electrochemical production)
        for side in (cell.an, cell.ca):
            L = side.cl.thickness
            state.S[side.cl.ix, i_lmbd, ...] -= J_des[side.cl.ix, ...] / L
            state.S[side.cl.ix, tm.i_cg[-1], ...] += J_des[side.cl.ix, ...] / L
        # Water produced in dissolved form at CCL (Vetter & Schumacher 2019, Eq. 6)
        state.S[cell.ca.cl.ix, i_lmbd, ...] += state.iF / 2 / cell.ca.cl.thickness

    def add_eod_flux(self, state: CellState, cell, tm: TransientCellModel) -> None:
        """Add electro-osmotic drag to the λ flux at both ionomer interfaces."""
        J_eod = (
            cell.memb.calculate_electroosmotic_drag_coefficient(state.T, state.lmbd)
            [tm.ionomer_domain, ...]
            * state.iF
        )
        state.J[[cell.an.cl.ix + 1, cell.memb.ix + 1], tm.i_lmbd, ...] += J_eod[[0, 2], ...]


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
        return 2.5 / 22.5 * lmbd

