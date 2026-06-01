"""
Module providing classes to model catalyst layers in electrochemical cells.
"""
from dataclasses import dataclass, field, fields as dataclass_fields
import numpy as np 
import cantera as ct

from marapendi.models.electrochemistry import ElectrochemicalReaction 
from .ionomer import Ionomer
from .porous_layers import PorousLayer
from .water import o2_water_diffusivity 

@dataclass(eq=False)
class CatalystLayer(PorousLayer, Ionomer):
    """
    Represents a generic catalyst layer in a fuel cell or electrolyser, which includes
    an ionomer phase, catalyst particles, and electrochemical reaction sites.

    Attributes
    ----------
    ionomer : CatalystLayerIonomer
        Ionomer model associated with the catalyst layer.
    reaction : ElectrochemicalReaction
        Electrochemical reaction model.
    catalyst_loading : float
        Catalyst loading in g/cm² (default: 0.2 mg/cm²).
    ionomer_to_catalyst_ratio : float
        Ratio of ionomer mass to catalyst mass.
    catalyst_density : float
        Density of catalyst particles (kg/m³).
    ecsa : float
        Electrochemically active surface area (m²/g).
    eps_ion : float
        Volume fraction of ionomer in the layer.
    t_ion_film : float
        Thickness of the ionomer film around catalyst particles (m).
    theta_contact : float
        Contact angle (degrees).
    theta_catalyst : float
        Catalyst surface coverage (dimensionless).
    """
    ionomer: Ionomer = field(default=None)
    reaction: ElectrochemicalReaction = field(default_factory=ElectrochemicalReaction)
    catalyst_loading: float = 0.2e-6 * 1e4
    ionomer_to_catalyst_ratio: float = 0.75
    catalyst_density: float = 21450
    ecsa: float = 70e3
    eps_ion: float = 0.3
    t_ion_film: float = 2e-9
    theta_contact: float = 95.
    theta_catalyst: float = 0
    electrolyte_saturation: float = 0
    
    def __eq__(self, other):
        """Use identity comparison — layers are unique instances in the cell model.
        This avoids ambiguous truth-value errors from numpy array fields during
        membership tests like `layer in self.ionomer_layers`."""
        return self is other

    def __hash__(self):
        return id(self)

    def __post_init__(self):
        """Copy all ionomer fields onto self so ionomer properties are
        directly accessible (e.g. self.dry_density, self.equivalent_weight).
        Skipped if no ionomer was supplied."""
        if self.ionomer is not None:
            for f in dataclass_fields(self.ionomer):
                setattr(self, f.name, getattr(self.ionomer, f.name))
        super().__post_init__()

    def ionomer_sheet_charge_resistance(self, f_v, T, charge='proton'):
        sigma_ion = self.charge_conductivity(f_v, T, charge)
        return self.thickness * self.tau_ion / (self.eps_ion * sigma_ion)

    def effective_charge_resistance(self, i, f_v, T, charge='proton'):
        """
        Effective charge resistance per Goshtasbi et al. (2020) / Neyerlin et al. (2007).

        Returns
        -------
        float
            Effective charge resistance [Ohm.m²].
        """
        self.sheet_resistance = 1. / (
            1. / self.ionomer_sheet_charge_resistance(f_v, T, charge)
            + 1. / self.electrolyte_sheet_resistance(T)
        )
        nu = np.minimum(self.sheet_resistance * i / self.reaction.tafel_slope(T), 10)
        self.xi_neyerlin = nu * (-8.287e-3 * nu + 0.7184) - 2.072e-3
        return self.sheet_resistance / (3 + self.xi_neyerlin)

    def electrolyte_sheet_resistance(self, T):
        sigma_el = self.electrolyte.calculate_ionic_conductivity(T)
        return self.thickness / ((np.maximum(self.electrolyte_saturation, 1e-12) * self.eps_p) ** 1.5 * sigma_el)

    def activation_overpotential(self, i, activity):
        return self.reaction.tafel_overpotential(
            i / (self.ecsa * self.catalyst_loading),
            self.temperature,
            activity
        )
    
@dataclass(eq=False)

class PtCCatalystLayer(CatalystLayer):
    """
    Catalyst layer model with explicit platinum/carbon and ionomer structure.

    Attributes
    ----------
    L_Pt : float
        Pt loading [g/cm²].
    wt_Pt : float
        Pt weight percent in catalyst.
    ic_ratio : float
        Ionomer/carbon mass ratio.
    rho_Pt, rho_C : float
        Densities [kg/m³].
    r_C : float
        Radius of carbon agglomerates [m].
    """
    thickness: float = 10e-6
    eps_p: float = 0
    L_Pt: float = 0.2e-6 * 1e4
    wt_Pt: float = 0.5
    ic_ratio: float = 0.75
    rho_Pt: float = 21450
    rho_C: float = 1950
    a_Pt: float = 0
    r_C: float = 25e-9
    dry_eps_ion: float = 0
    eps_ion: float = 0
    t_ion_film: float = 0
    theta_contact: float = 95.
    omega_PtO: float = 3000e3
    k1_ion: float = 8.5
    k2_ion: float = 5.4
    k3_ion: float = 5.4

    def __post_init__(self):
        """
        Compute volume fractions and geometry based on catalyst composition.

        Equations come from Hao et al. (2015). 

        References
        ----------
        Hao, L. et al. J. Electrochem. Soc. 162, F854 (2015).
        """
        CatalystLayer.__post_init__(self)
        if self.a_Pt == 0:
            self.a_Pt = self.L_Pt * self.ecsa / self.thickness
        if self.eps_p == 0:
            self.carbon_loading = self.L_Pt * (1 / self.wt_Pt - 1)
            self.eps_C = self.carbon_loading / (self.thickness * self.rho_C)
            self.eps_Pt = self.L_Pt / (self.thickness * self.rho_Pt)
            self.eps_cat = self.eps_Pt + self.eps_C
            self.dry_eps_ion = (self.eps_C * self.rho_C *
                                             self.ic_ratio / self.rho_dry_ion)
        else:
            self.eps_cat = 1 - self.dry_eps_ion - self.eps_p

        self.a_agg = 4 * np.pi * self.r_C ** 2
        self.v_agg = self.a_agg * self.r_C / 3.
        self.N_agg = self.eps_C / self.v_agg
        self.set_ionomer_wet_properties(14, 300)
        self.set_water_film_thickness(0)
        PorousLayer.__post_init__(self)

    
    def set_ionomer_wet_properties(self, lmbd, T):
        """Update eps_ion, eps_p, t_ion_film and derived geometry for given water content."""
        self.ionomer_expansion_factor = self.wet_expansion_factor(np.clip(lmbd, 3, 20), T)
        self.eps_ion = self.dry_eps_ion * self.ionomer_expansion_factor
        self.eps_p   = 1 - self.eps_cat - self.eps_ion
        self.t_ion_film = self.r_C * ((self.eps_ion / self.eps_C + 1) ** (1/3) - 1)
        self.a_ion  = 4 * np.pi * (self.r_C + self.t_ion_film) ** 2 * self.N_agg
        self.ic_vol_ratio = (1 + self.t_ion_film / self.r_C) ** 3 - 1

    def set_water_film_thickness(self, s):
        r_ion = self.r_C + self.t_ion_film
        self.t_water = (s * self.eps_p * self.r_C ** 3 / self.eps_C + r_ion ** 3) ** (1./3) - r_ion

    def ionomer_sheet_charge_resistance(self, f_v, T, charge='proton'):
        """Ionomer film proton resistance per Hao et al. (2016)."""
        sigma_ion = self.charge_conductivity(f_v, T, charge)
        tort_ion  = self.eps_ion ** -0.5
        return self.thickness / (self.eps_ion / tort_ion * sigma_ion)

    def o2_ionomer_film_bulk_resistance(self, lmbd, T):
        return self.t_ion_film / (ct.gas_constant * T * self.ionomer.o2_permeability(lmbd, T))

    def o2_ionomer_film_resistance(self, lmbd, T):
        """Total O2 film resistance per Hao et al. (2015), neglecting water film."""
        R_bulk     = self.o2_ionomer_film_bulk_resistance(lmbd, T)
        R_pt_iface = (self.k2_ion + 1) / (1 - self.theta_catalyst) / (self.L_Pt * self.ecsa)
        R_gas_iface = self.k1_ion / (self.a_ion * self.thickness)
        R_water    = (self.k3_ion + 1) * self.t_water / o2_water_diffusivity(T) / (self.a_ion * self.thickness)
        return (R_gas_iface + R_pt_iface) * R_bulk + R_water

    def activation_overpotential(self, i, activity):
        return self.reaction.tafel_overpotential(
            i / (self.ecsa * self.L_Pt),
            self.temperature,
            activity
        )
