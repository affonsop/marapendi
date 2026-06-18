"""
Membrane components: static physical properties and correlations.

A :class:`Membrane` (and specializations such as :class:`PFSA`) holds the
static physical properties of the membrane (equivalent weight, density,
thickness, transport-property correlations, ...). It inherits all the
generic ionomer correlations from :class:`~marapendi.ionomer.Ionomer`
and adds membrane-specific properties (dry thickness, hydrogen permeation,
sorption isotherms, proton conductivity).
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from marapendi.tools import arrhenius_term
from marapendi.thermo.constants import GAS_CONSTANT
from marapendi.thermo.water import water_molar_volume, water_molecular_weight, water_density
from marapendi.cell.water_balance import MembraneWaterBalanceModel
from marapendi.membrane.membrane_permeation_models import HydrogenPermeationModel
from .ionomer import Ionomer


@dataclass
class Membrane(Ionomer):
    """Static properties of a proton/anion exchange membrane.

    Inherits dry material properties and generic transport correlations from
    :class:`~marapendi.ionomer.Ionomer`.  Adds membrane-specific geometry
    (dry thickness, hydrogen permeation) and the water-balance model.

    Attributes
    ----------
    dry_thickness : float
        Membrane thickness (m).
    h2_permeation_model : HydrogenPermeationModel
        Correlation used for the hydrogen permeation flux.
    water_balance_model : MembraneWaterBalanceModel
        Model that solves the membrane water balance during a steady-state solve.
    water_content : float
        Current membrane water content (mol H2O / mol SO3-); updated by the water-balance model.
    """

    dry_thickness: float = 25e-6
    h2_permeation_model: HydrogenPermeationModel = field(default_factory=HydrogenPermeationModel)
    water_balance_model: MembraneWaterBalanceModel = field(default_factory=MembraneWaterBalanceModel)
    relaxation_time_constant: float = 0.067
    relaxation_time_activation_energy: float = 28e6
    uptake_relaxed_fraction_constant: float = 0.014
    reference_water_content: float = 15.01
    reference_liquid_water_content: float = 22.
    eta_adsorption: float = 0.4712
    characteristic_adsorption_energy: float = 1.047e6

    def __post_init__(self):
        super().__post_init__()   # Ionomer: sets dry_concentration, dry_molar_volume
        self.surface_concentration = self.dry_concentration * self.dry_thickness

    def hydrogen_permeation_flux(
        self,
        partial_pressure_h2: float,
        temperature: float,
        pressure_difference: float,
        water_vol_fraction: float,
    ) -> float:
        """Hydrogen permeation flux through the membrane (kmol/m^2/s)."""
        return self.h2_permeation_model.permeation_flux(
            self.dry_thickness, partial_pressure_h2, temperature, pressure_difference, water_vol_fraction,
        )

    def charge_conductivity(self, water_content, temperature, use_water_profile=True, charge='proton'):
        if charge == 'proton':
            return self.proton_conductivity(water_content, temperature, use_water_profile)
        elif charge == 'hydroxide':
            return self.hydroxide_conductivity(water_content, temperature)

    def charge_resistance(self, water_content, temperature, use_water_profile=True, charge='proton'):
        return self.dry_thickness / self.charge_conductivity(water_content, temperature, use_water_profile, charge)

    def equilibrium_water_content(self, rh, temperature, s_relax=None):
        """Equilibrium water content from the Dubinin–Astakhov (DA) model.

        References
        ----------
        Grimaldi et al. J. Power Sources (2023).
        """
        rh = np.clip(rh, .01, .99)
        A = -GAS_CONSTANT * temperature * np.log(rh)
        return np.exp(-(A / self.characteristic_adsorption_energy) ** self.eta_adsorption) * self.reference_water_content

    def equilibrium_water_content_derivative(self, rh, temperature, s_relax=None):
        rh = np.clip(rh, .01, .99)
        A = -GAS_CONSTANT * temperature * np.log(rh)
        K = (
            self.eta_adsorption
            * (A / self.characteristic_adsorption_energy) ** (self.eta_adsorption - 1)
            * (GAS_CONSTANT * temperature / rh / self.characteristic_adsorption_energy)
        )
        return K * np.exp(-(A / self.characteristic_adsorption_energy) ** self.eta_adsorption) * self.reference_water_content

    def liquid_equilibrium_water_content(self, temperature):
        return self.reference_liquid_water_content


@dataclass
class PFSA(Membrane):
    """Perfluorosulfonic-acid (PFSA, e.g. Nafion) membrane.

    Attributes
    ----------
    conductivity_correction : float
        Correction factor for the proton conductivity correlation
        (Vetter and Schumacher, 2020).
    conductivity_exp : float
        Exponent of the proton conductivity correlation.
    conductivity_activation_energy : float
        Activation energy for proton conductivity (J/kmol).
    phi : float
        Contribution of relaxation phenomena to the ionomer water uptake,
        according to Goshtasbi et al. (2019).

    References
    ----------
    Springer, T. E. et al. J. Electrochem. Soc. 138, 2334 (1991).
    Goshtasbi, A. et al. J. Electrochem. Soc. 167, 024518 (2020).
    """

    conductivity_correction: float = 1.
    conductivity_exp: float = 1.5
    conductivity_activation_energy: float = 15e6
    phi: float = 0.15

    def equilibrium_water_content(self, rh, temperature, s_relax=None):
        """Equilibrium water content from the Springer et al. (1991) polynomial isotherm.

        References
        ----------
        Springer, T. E. et al. J. Electrochem. Soc. 138, 2334 (1991).
        Goshtasbi et al. J. Electrochem. Soc. 2019, 166 (7), F3154.
        """
        rh = np.clip(rh, 0, 1)
        lmbd_eq_relaxed = ((36 * rh - 39.85) * rh + 17.18) * rh + 0.043
        return ((1 - self.phi) * lmbd_eq_relaxed + s_relax) if s_relax is not None else lmbd_eq_relaxed

    def equilibrium_water_content_derivative(self, rh, temperature, s_relax=None):
        rh = np.clip(rh, 0, 1)
        d_lmbd_eq_relaxed = (108 * rh - 79.70) * rh + 17.18
        return ((1 - self.phi) * d_lmbd_eq_relaxed + s_relax) if s_relax is not None else d_lmbd_eq_relaxed

    def liquid_equilibrium_water_content(self, temperature):
        """Equilibrium water content in contact with liquid water.

        References
        ----------
        Goshtasbi, A. et al. J. Electrochem. Soc. 167, 024518 (2020).
        """
        return 9.22 + 0.181 * (temperature - 273.15)

    def proton_conductivity(self, water_content_profile, temperature, use_water_profile=True):
        fv = self.water_vol_fraction(water_content_profile, water_molar_volume(temperature))
        return 1 / np.mean(
            1 / (
                self.conductivity_correction * 50
                * (np.maximum(fv, 0.11) - 0.1) ** self.conductivity_exp
                * arrhenius_term(self.conductivity_activation_energy, temperature, 298.15)
            ),
            axis=0,
        )

    def proton_resistance(self, temperature, water_saturation=0):
        """Through-plane proton resistance (Ω·m²).

        Weights liquid- and vapor-equilibrated conductivities by water saturation.
        """
        liquid_conductivity = self.proton_conductivity(self.liquid_eq_sat_water_profile, temperature)
        vapor_conductivity = self.proton_conductivity(self.vapor_eq_sat_water_profile, temperature)
        average_conductivity = (
            (1 - water_saturation) * vapor_conductivity + water_saturation * liquid_conductivity
        )
        return self.dry_thickness / average_conductivity


@dataclass
class AEM(Membrane):

    def proton_conductivity(self, water_content, temperature, use_water_profile=True):
        return 1e-6

    def calculate_electroosmotic_drag_coefficient(self, temperature, water_content):
        return -water_content / 14


@dataclass
class FAA3(AEM):
    """FAA3 anion-exchange membrane.

    References
    ----------
    Eon Chae, J. et al. J. Ind. Eng. Chem. 133, 255–262 (2024)
    Luo, X. et al. J. Memb. Sci. 598, 117680 (2020)
    Khalid, H. et al. Membranes (Basel) 12, 989 (2022).
    """

    dry_density: float = 1310.
    equivalent_weight: float = 1000 / 1.91

    def hydroxide_conductivity(self, water_content, temperature):
        return 3.1 * arrhenius_term(
            activation_energy=11.1e6, temperature=temperature, reference_temperature=298.15,
        )


@dataclass
class PAP85(AEM):
    """PAP85 anion-exchange membrane.

    References
    ----------
    Eon Chae, J. et al. J. Ind. Eng. Chem. 133, 255–262 (2024)
    Luo, X. et al. J. Memb. Sci. 598, 117680 (2020)
    Khalid, H. et al. Membranes (Basel) 12, 989 (2022).
    """

    dry_density: float = 1220.
    equivalent_weight: float = 1000 / 2.35
    reference_water_diffusivity: float = 4e-10
    reference_absorption_coefficient: float = 1e-6
    reference_temperature: float = 303.15
    reference_liquid_water_content: float = 14.
    water_diffusivity_activation_energy: float = 20e6 * 2.37
    water_absorption_activation_energy: float = 20e6 * 2.37

    def equilibrium_water_content(self, rh, temperature, s_relax=None):
        return ((14.41 * rh - 14.81) * rh + 13.13) * rh

    def hydroxide_conductivity(self, water_content, temperature):
        return 5.8 * arrhenius_term(
            activation_energy=22.5e6, temperature=temperature, reference_temperature=298.15,
        )


@dataclass
class SustainionX3750RT(AEM):
    """Sustainion X37-50 RT anion-exchange membrane."""

    dry_density: float = 1220.
    equivalent_weight: float = 1000 / 2.35
    dry_thickness: float = 50e-6
    ref_hydroxide_conductivity: float = 11.6
    conductivity_activation_energy: float = 10.7e6

    def hydroxide_conductivity(self, water_content, temperature):
        return self.ref_hydroxide_conductivity * arrhenius_term(
            activation_energy=self.conductivity_activation_energy,
            temperature=temperature,
            reference_temperature=333.15,
        )
