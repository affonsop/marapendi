"""
Membrane components: static physical properties and correlations.

A :class:`Membrane` (and specializations such as :class:`PFSA`) holds the
static physical properties of the membrane (equivalent weight, density,
thickness, transport-property correlations, ...). It inherits all the
generic ionomer correlations from :class:`~marapendi.ionomer.Ionomer`
and adds membrane-specific properties (dry thickness, hydrogen permeation,
sorption isotherms, proton conductivity).

Physical *variables* (water content, temperature, fluxes, ...) live in
:class:`marapendi.state.MembraneState` and are produced/consumed by
:class:`marapendi.water_balance.MembraneWaterBalanceModel`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from marapendi.tools.tools import arrhenius_term
from .membrane_permeation_models import HydrogenPermeationModel
from .water import water_molar_volume
from .ionomer import Ionomer


@dataclass
class Membrane(Ionomer):
    """Static properties of a proton/anion exchange membrane.

    Attributes
    ----------
    dry_thickness : float
        Membrane thickness (m).
    h2_permeation_model : HydrogenPermeationModel
        Correlation used for the hydrogen permeation flux.
    """

    dry_thickness: float = 25e-6
    h2_permeation_model: HydrogenPermeationModel = field(default_factory=HydrogenPermeationModel)

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
    relaxation_water_content_fraction : float
        Fraction of the equilibrium water content that relaxes towards
        ``s_relax`` rather than the Springer isotherm (Goshtasbi et al., 2019).

    References
    ----------
    Springer, T. E. et al. J. Electrochem. Soc. 138, 2334 (1991).
    Goshtasbi, A. et al. J. Electrochem. Soc. 167, 024518 (2020).
    """

    conductivity_correction: float = 1.
    conductivity_exp: float = 1.5
    conductivity_activation_energy: float = 15e6
    relaxation_water_content_fraction: float = 0.15

    def equilibrium_water_content(self, rh: float, temperature: float, s_relax: float | None = None) -> float:
        """Equilibrium water content (mol H2O / mol SO3-) from the Springer et al. (1991) isotherm.

        If ``s_relax`` is given (Goshtasbi et al. 2019 dynamic relaxation term), the isotherm
        value is scaled down by ``relaxation_water_content_fraction`` and ``s_relax`` is added
        back; otherwise the raw isotherm value is returned (steady-state).
        """
        rh = np.clip(rh, 0, 1)
        relaxed_water_content = np.polyval([36, -39.85, 17.18, 0.043], rh)
        if s_relax is None:
            return relaxed_water_content
        return (1 - self.relaxation_water_content_fraction) * relaxed_water_content + s_relax

    def equilibrium_water_content_derivative(self, rh: float, temperature: float, s_relax: float | None = None) -> float:
        """Derivative of :meth:`equilibrium_water_content` with respect to relative humidity."""
        rh = np.clip(rh, 0, 1)
        relaxed_water_content_derivative = np.polyval([108, -79.70, 17.18], rh)
        if s_relax is None:
            return relaxed_water_content_derivative
        return (1 - self.relaxation_water_content_fraction) * relaxed_water_content_derivative + s_relax

    def liquid_equilibrium_water_content(self, temperature: float) -> float:
        """Equilibrium water content (mol H2O / mol SO3-) in contact with liquid water.

        Reference
        ---------
        Goshtasbi, A. et al. J. Electrochem. Soc. 167, 024518 (2020).
        """
        return 9.22 + 0.181 * (temperature - 273.15)

    def proton_conductivity(self, water_content_profile: float, temperature: float) -> float:
        """Through-plane proton conductivity (S/m) from a membrane water content profile."""
        water_vol_fraction = self.water_vol_fraction(water_content_profile, water_molar_volume(temperature))
        local_conductivity = (
            self.conductivity_correction * 50 * (np.maximum(water_vol_fraction, 0.11) - 0.1) ** self.conductivity_exp
            * arrhenius_term(self.conductivity_activation_energy, temperature, 298.15)
        )
        return 1 / np.mean(1 / local_conductivity, axis=0)

    def proton_resistance(self, water_content_profile: float, temperature: float) -> float:
        """Through-plane proton resistance (Ohm.m^2) from a membrane water content profile."""
        return self.dry_thickness / self.proton_conductivity(water_content_profile, temperature)
