"""
Catalyst-layer physics models for PEM/AEM fuel cells and electrolysers.

Classes
-------
CatalystLayerModel
    Stateless strategy class for generic catalyst-layer charge transport
    (ionomer sheet resistance, effective charge resistance, electrolyte
    sheet resistance).
PtCCatalystLayerModel
    Extends ``CatalystLayerModel`` with Pt/C-specific O₂ transport
    correlations (ionomer film resistance, water-film thickness).

Design note
-----------
Model classes are stateless strategy objects.  They accept component
dataclasses (:class:`~marapendi.components.catalyst_layers.CatalystLayer`,
:class:`~marapendi.components.catalyst_layers.PtCCatalystLayer`) as
explicit arguments and return computed quantities without storing state.

References
----------
Goshtasbi, A. et al. (2020); Neyerlin, K. C. et al. (2007);
Hao, L. et al. J. Electrochem. Soc. 162, F854 (2015).
"""

from dataclasses import dataclass
import numpy as np
import cantera as ct

from ..models.water import o2_water_diffusivity


@dataclass()
class CatalystLayerModel():
    """
    Stateless strategy class for catalyst-layer charge transport.

    Methods compute ionomer and electrolyte sheet resistances and the
    effective proton/hydroxide transport resistance through the catalyst
    layer, optionally with the Neyerlin correction for non-uniform
    current distribution.
    """

    def ionomer_sheet_charge_resistance(self, f_v, T, charge, ionomer_model, cl):
        """Ionomer sheet resistance for charge transport [Ω·m²].

        Parameters
        ----------
        f_v : float
            Water volume fraction in the ionomer [-].
        T : float
            Temperature [K].
        charge : str
            Charge carrier (``'proton'`` or ``'hydroxide'``).
        ionomer_model : IonomerModel
            Model providing ``charge_conductivity``.
        cl : CatalystLayer
            Catalyst-layer component dataclass.
        """
        sigma_ion = ionomer_model.charge_conductivity(f_v, T, charge, cl.ionomer)
        return cl.thickness * cl.tau_ion / (cl.eps_ion * sigma_ion)

    def effective_charge_resistance(self, i, f_v, T, electrolyte_saturation, charge, ionomer_model, cl, reaction, use_neyerlin_correction=False):
        """
        Effective charge resistance per Goshtasbi et al. (2020) / Neyerlin et al. (2007).

        Returns
        -------
        float
            Effective charge resistance [Ohm.m²].
        """
        self.sheet_resistance = 1. / (
            1. / self.ionomer_sheet_charge_resistance(f_v, T, charge, ionomer_model, cl)
            + 1. / self.electrolyte_sheet_resistance(T, electrolyte_saturation, cl)
        )

        nu = np.minimum(self.sheet_resistance * i / reaction.tafel_slope(T), 10)
        self.xi_neyerlin = nu * (-8.287e-3 * nu + 0.7184) - 2.072e-3 if use_neyerlin_correction else 0
        return self.sheet_resistance / (3 + self.xi_neyerlin)

    def electrolyte_sheet_resistance(self, T, electrolyte_saturation, cl):
        """Liquid electrolyte sheet resistance [Ω·m²].

        Uses a Bruggeman-type effective conductivity with the liquid-filled
        pore fraction raised to the 1.5 power.
        """
        sigma_el = cl.electrolyte.calculate_ionic_conductivity(T)
        return cl.thickness / ((np.maximum(electrolyte_saturation, 1e-12) * cl.eps_p) ** 1.5 * sigma_el)

@dataclass()
class PtCCatalystLayerModel(CatalystLayerModel):
    """
    Catalyst-layer model for Pt/C electrodes.

    Extends :class:`CatalystLayerModel` with O₂ transport equations for
    the ionomer and water films surrounding carbon agglomerates, following
    Hao et al. (2015).

    Attributes
    ----------
    k1_ion : float
        Pre-factor for gas/ionomer interface resistance [-] (default 8.5).
    k2_ion : float
        Pre-factor for Pt/ionomer interface resistance [-] (default 5.4).
    k3_ion : float
        Pre-factor for ionomer/water film resistance [-] (default 5.4).
    """
    k1_ion: float = 8.5
    k2_ion: float = 5.4
    k3_ion: float = 5.4

    def water_film_thickness(self, s, cl):
        """Liquid water film thickness on the ionomer surface [m].

        Derived from liquid water volume conservation in the pore space.

        Parameters
        ----------
        s : float
            Liquid water saturation in the catalyst layer [-].
        cl : PtCCatalystLayer
            Catalyst-layer component dataclass.
        """
        r_ion = cl.r_C + cl.t_ion_film
        return (s * cl.eps_p * cl.r_C ** 3 / cl.eps_C + r_ion ** 3) ** (1./3) - r_ion

    def o2_ionomer_film_bulk_resistance(self, lmbd, T, ionomer_model, ionomer_film_thickess):
        """Bulk O₂ diffusion resistance through the ionomer film [s·m/mol].

        Parameters
        ----------
        lmbd : float
            Water content of the ionomer film [mol H₂O / mol SO₃⁻].
        T : float
            Temperature [K].
        ionomer_model : PFSAModel
            Model providing ``o2_permeability``.
        ionomer_film_thickess : float
            Effective ionomer film thickness [m].
        """
        return ionomer_film_thickess / (ct.gas_constant * T * ionomer_model.o2_permeability(lmbd, T))

    def o2_ionomer_film_resistance(self, lmbd, T, cl, ionomer_model, ionomer_film_thickess, water_film_thickness, coverage_ratio=0):
        """Total O2 film resistance per Hao et al. (2015), neglecting water film."""
        R_bulk     = self.o2_ionomer_film_bulk_resistance(lmbd, T, ionomer_model, ionomer_film_thickess)
        R_pt_iface = (self.k2_ion + 1) / (1 - coverage_ratio) / (cl.L_Pt * cl.ecsa)
        R_gas_iface = self.k1_ion / (cl.a_ion * cl.thickness)
        R_water    = (self.k3_ion + 1) * water_film_thickness / o2_water_diffusivity(T) / (cl.a_ion * cl.thickness)
        return (R_gas_iface + R_pt_iface) * R_bulk + R_water