"""
Catalyst-layer component dataclasses.

These dataclasses hold geometric, structural, and material parameters for
catalyst layers.  Electrokinetic and transport computations (ionomer film
O₂ resistance, charge resistance, water-film thickness, etc.) are
implemented as stateless strategy methods in
:mod:`marapendi.models.catalyst_layer` (``CatalystLayerModel``,
``PtCCatalystLayerModel``).
"""
from dataclasses import dataclass, field, fields as dataclass_fields
import numpy as np 
import cantera as ct

from marapendi.models.electrochemistry import ElectrochemicalReaction 
from .ionomer import Ionomer
from .porous_layers import PorousLayer


@dataclass(eq=False)
class CatalystLayer(PorousLayer, Ionomer):
    """
    Base dataclass for a generic catalyst layer.

    Combines porous-layer geometry (:class:`~marapendi.components.porous_layers.PorousLayer`)
    with ionomer material parameters (:class:`~marapendi.components.ionomer.Ionomer`).

    Electrokinetic and transport computations (charge resistance, O₂ film
    resistance, water-film thickness) are implemented in
    :mod:`marapendi.models.catalyst_layer`.

    Attributes
    ----------
    ionomer : Ionomer, optional
        Ionomer parameter object whose fields are copied onto ``self``
        during ``__post_init__`` for direct attribute access.
    reaction : ElectrochemicalReaction
        Electrochemical reaction parameters.
    catalyst_loading : float
        Catalyst loading [kg/m²] (default: 0.2 mg/cm²).
    ionomer_to_catalyst_ratio : float
        Ionomer-to-catalyst mass ratio.
    catalyst_density : float
        Density of catalyst particles [kg/m³].
    ecsa : float
        Electrochemically active surface area [m²/kg].
    eps_ion : float
        Volume fraction of ionomer in the layer [-].
    t_ion_film : float
        Ionomer film thickness around catalyst particles [m].
    theta_contact : float
        Contact angle [degrees].
    theta_catalyst : float
        Catalyst surface coverage [-].
    electrolyte_saturation : float
        Fraction of pore volume filled by liquid electrolyte [-].
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
        self.update_ionomer_film_volume(1)
        PorousLayer.__post_init__(self)
    
    def update_ionomer_film_volume(self, ionomer_expansion_factor=1):
        """Update eps_ion, eps_p, t_ion_film and derived geometry for given water content.
        For now marapendi do not consider variations of the ionomer volume. This function is only called at 
        __post_init__."""
        self.ionomer_expansion_factor = ionomer_expansion_factor
        self.eps_ion = self.dry_eps_ion * ionomer_expansion_factor
        self.tort_ion = self.eps_ion ** -0.5
        self.eps_p   = 1 - self.eps_cat - self.eps_ion
        self.tort    = self.eps_p**(-0.5)
        self.t_ion_film = self.r_C * ((self.eps_ion / self.eps_C + 1) ** (1/3) - 1)
        self.a_ion  = 4 * np.pi * (self.r_C + self.t_ion_film) ** 2 * self.N_agg
        self.ic_vol_ratio = (1 + self.t_ion_film / self.r_C) ** 3 - 1
