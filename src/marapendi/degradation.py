"""
Module providing classes describing platinum degradation mechanisms
in electrochemical cells (e.g. PEM fuel cells).

The implementation follows kinetic formulations proposed in:

Darling & Meyers,
"Kinetic Model of Platinum Dissolution in PEMFCs",
Journal of The Electrochemical Society, 150(11), A1523 (2003).

The module includes:
    - Platinum particle size distributions
    - Material thermodynamic properties
    - Platinum dissolution kinetics
    - Platinum oxide formation/dissolution
    - Coupled degradation rate evaluation
"""

from dataclasses import dataclass, field
import numpy as np
import cantera as ct
from .tools import potential_activation
from .catalyst_layers import CatalystLayer
from scipy.stats import lognorm, norm
import matplotlib.pyplot as plt



# ======================================================================
# Platinum particle size distribution
# ======================================================================


@dataclass
class PtSizeDistribution:
    """
    Represents a platinum nanoparticle size distribution
    inside the catalyst layer.

    The distribution is discretized into radius bins used
    for degradation calculations.

    Parameters
    ----------
    n_points : int
        Number of discretization points.
    r_mean : float [m]
        Mean particle radius.
    r_std : float [m]
        Standard deviation of particle radius.
    initial_platinum_loading : float [kg/m²]
        Platinum loading.
    initial_cl_thickness : float [m]
        Catalyst layer thickness.
    initial_ecsa : float [m²/kg]
        Initial electrochemical surface area.
    distribution_type : {"norm","lognorm"}
        Type of statistical radius distribution.
    """

    n_points: int = 32
    r_mean: float = 1.5e-9
    r_std: float = 1e-9
    initial_platinum_loading: float = 0.2e-2
    initial_cl_thickness: float = 10e-6
    initial_ecsa: float = 40e3
    distribution_type: str = "norm"

    def __post_init__(self):
        """Initialize statistical distribution and derived quantities."""

        self.platinum_loading = self.initial_platinum_loading
        self.cl_thickness = self.initial_cl_thickness

        # ---- Build particle size distribution
        if self.distribution_type == "lognorm":
            sigma = np.sqrt(
                np.log(1 + (self.r_std**2) / (self.r_mean**2))
            )
            mu = np.log(
                self.r_mean**2 /
                np.sqrt(self.r_mean**2 + self.r_std**2)
            )
            self.initial_dist = lognorm(s=sigma,
                                        scale=np.exp(mu))

        elif self.distribution_type == "norm":
            self.initial_dist = norm(
                loc=self.r_mean,
                scale=self.r_std
            )

        # Radius discretization
        self.r_array = np.linspace(
            *self.initial_dist.interval(0.999),
            self.n_points
        )

        # Normalized PDF
        self.normalized_distrib_array = \
            self.initial_dist.pdf(self.r_array)

        # Particle number density distribution
        self.number_density_distrib_array = (
            self.number_density()
            * self.normalized_distrib_array
        )

        # Conversion between geometrical surface and ECSA
        self.ecsa_to_particle_average_surface_ratio = (
            self.initial_ecsa /
            self.average_particle_surface()
        )

        self.initial_utilization_ratio = (
            self.initial_ecsa /
            (self.average_particle_surface()
             / self.average_particle_mass())
        )

    # ------------------------------------------------------------------
    def average_particle_volume(self):
        """Return average particle volume [m³]."""
        return (
            4 * np.pi/3 *
            np.trapezoid(
                self.normalized_distrib_array
                * self.r_array ** 3,
                self.r_array
            )
            / np.trapezoid(
                self.normalized_distrib_array,
                self.r_array
            )
        )

    def average_particle_mass(self):
        """Return average particle mass [kg]."""
        return platinum.density * self.average_particle_volume()

    def number_density(self):
        """
        Particle number density inside catalyst layer [1/m³].
        """
        return (
            self.platinum_loading
            / self.cl_thickness
            / self.average_particle_mass()
        )

    def average_particle_surface(self):
        """Return average particle surface area [m²]."""
        return (
            4 * np.pi *
            np.trapezoid(
                self.normalized_distrib_array
                * self.r_array**2,
                self.r_array
            )
            / np.trapezoid(
                self.normalized_distrib_array,
                self.r_array
            )
        )

    def geometrical_specific_surface_area(self):
        """Geometrical surface per unit mass [m²/kg]."""
        return (
            self.average_particle_surface()
            / self.average_particle_mass()
        )

    def ecsa(self):
        """
        Electrochemical surface area accounting
        for utilization losses.
        """
        return (
            self.average_particle_surface()
            * self.ecsa_to_particle_average_surface_ratio
        )

    def plot_distribution(self, ax=None,
                          label="Pt radius distribution"):
        """Plot particle radius probability density."""
        if ax is None:
            _, ax = plt.subplots()

        ax.plot(self.r_array,
                self.normalized_distrib_array,
                label=label)


# ======================================================================
# Platinum thermodynamic species
# ======================================================================

@dataclass
class PlatinumSpecies:
    """
    Thermodynamic properties of a platinum phase.

    Parameters
    ----------
    density : float [kg/m³]
    molecular_weight : float [kg/kmol]
    surface_tension : float [J/m²]
    reference_pontial : float [J/kmol]
        Reference Gibbs potential.
    """

    density: float
    molecular_weight: float
    surface_tension: float
    reference_pontial: float = 0

    def __post_init__(self):
        """Compute molar volume."""
        self.molar_volume = (
            self.molecular_weight / self.density
        )

    def surface_tension_potential_shift(
        self,
        particle_radius
    ):
        """
        Gibbs–Thomson potential correction due to
        nanoparticle curvature.
        """
        return (
            self.reference_pontial +
            self.surface_tension
            * self.molar_volume
            / particle_radius
        )


# Reference materials
platinum = PlatinumSpecies(
    density=21450.,
    molecular_weight=195.08,
    surface_tension=2.37
)

platinum_oxide = PlatinumSpecies(
    density=14100.,
    molecular_weight=211.08,
    surface_tension=1.00,
    reference_pontial=-42.3e6
)


# ======================================================================
# Platinum dissolution kinetics
# ======================================================================

@dataclass
class PlatinumDissoultion:
    """
    Platinum electrochemical dissolution reaction.

    Reference: 
    ---------- 
    Schneider et al. J. Electrochem. Soc. 2019, 166 (4), F322–F333.
    """

    rate_constant: float = 3.43e-12
    reference_potential: float = 1.188
    transfer_coeff_ca: float = 0.5
    transfer_coeff_an: float = 0.5
    relative_humidity_power: float = 1.7 # Value from Schneider et al. (2019)
    dissolved_platinum_reference_concentration: float = 1.

    def equilibrium_potential(self, particle_radius):
        """
        Size-dependent equilibrium potential including
        Gibbs–Thomson correction.
        """
        return (
            self.reference_potential -
            platinum.surface_tension_potential_shift(
                particle_radius
            ) / (2 * ct.faraday)
        )

    def rate_of_reaction(
        self,
        dissolved_platinum_concentration,
        platinum_oxide_coverage,
        potential,
        temperature,
        relative_humidity, 
        particle_radius,
    ):
        """
        Net platinum dissolution rate [kmol/m²/s].
        """

        potential_difference = (
            potential -
            self.equilibrium_potential(particle_radius)
        )

        concentration_ratio = (
            dissolved_platinum_concentration /
            self.dissolved_platinum_reference_concentration
        )

        return (
            self.rate_constant * 
            relative_humidity ** self.relative_humidity_power * 
            np.maximum(1 - platinum_oxide_coverage, 0)
            * (
                potential_activation(
                    self.transfer_coeff_an,
                    2,
                    temperature,
                    potential_difference,
                )
                -
                concentration_ratio
                * potential_activation(
                    self.transfer_coeff_ca,
                    2,
                    temperature,
                    -potential_difference,
                )
            )
        )
    
# ======================================================================
# Platinum oxide formation kinetics
# ======================================================================

@dataclass
class PlatinumOxideFormation:
    """
    Platinum oxide formation reaction kinetics.

    Model based on:
    Darling & Meyers (2003), J. Electrochem. Soc.

    Describes electrochemical oxidation of metallic platinum:

        Pt + H2O ⇌ PtO + 2H+ + 2e-

    Parameters
    ----------
    rate_constant : float [kmol/m²/s]
        Reaction kinetic prefactor.
    reference_potential : float [V]
        Standard equilibrium potential.
    transfer_coeff_ca : float
        Cathodic transfer coefficient.
    transfer_coeff_an : float
        Anodic transfer coefficient.
    omega_platinum_oxide_formation : float [J/kmol]
        Lateral interaction parameter between oxide species.
    proton_reference_concentration : float [kmol/m³]
        Reference proton concentration.
    """

    rate_constant: float = 1.36e-10
    reference_potential: float = 0.98
    transfer_coeff_ca: float = 0.15
    transfer_coeff_an: float = 0.35
    omega_platinum_oxide_formation: float = 30e6
    proton_reference_concentration: float = 1.

    def equilibrium_potential(self, particle_radius):
        """
        Size-dependent oxide formation equilibrium potential.

        Includes Gibbs–Thomson curvature correction between
        platinum and platinum oxide phases.
        """
        return (
            self.reference_potential +
            (
                platinum_oxide.surface_tension_potential_shift(
                    particle_radius
                )
                -
                platinum.surface_tension_potential_shift(
                    particle_radius
                )
            )
            / (2 * ct.faraday)
        )

    def rate_of_reaction(
        self,
        dissolved_platinum_concentration,
        platinum_oxide_coverage,
        proton_concentration,
        potential,
        temperature,
        particle_radius,
    ):
        """
        Net platinum oxide formation rate [kmol/m²/s].
        """

        potential_difference = (
            potential -
            self.equilibrium_potential(particle_radius)
        )

        proton_ratio = (
            proton_concentration /
            self.proton_reference_concentration
        )

        # Coverage-dependent interaction correction
        potential_correction = (
            self.omega_platinum_oxide_formation
            * platinum_oxide_coverage
            / (2 * ct.faraday * self.transfer_coeff_an)
        )

        return self.rate_constant * (
            potential_activation(
                self.transfer_coeff_an,
                2,
                temperature,
                potential_difference
                - potential_correction,
            )
            -
            platinum_oxide_coverage
            * potential_activation(
                self.transfer_coeff_ca,
                2,
                temperature,
                -potential_difference,
            )
            * proton_ratio**2
        )


# ======================================================================
# Platinum oxide dissolution
# ======================================================================

@dataclass
class PlatinumOxideDissolution:
    """
    Chemical dissolution of platinum oxide.

    Reaction couples oxide reduction and platinum ion
    concentration in the ionomer phase.

    Based on Darling & Meyers (2003).
    """

    rate_constant: float = 3.2e-23  # kmol/m²/s

    def rate_of_reaction(
        self,
        dissolved_platinum_concentration,
        platinum_oxide_coverage,
        proton_concentration,
        temperature,
        particle_radius,
        platinum_dissolution,
        platinum_oxide_formation,
    ):
        """
        Oxide dissolution reaction rate [kmol/m²/s].
        """

        # Equilibrium constant derived from reaction potentials
        K3 = potential_activation(
            1,
            2,
            temperature,
            platinum_dissolution.equilibrium_potential(
                particle_radius
            )
            -
            platinum_oxide_formation.equilibrium_potential(
                particle_radius
            ),
        )

        return self.rate_constant * (
            platinum_oxide_coverage
            * proton_concentration**2
            -
            dissolved_platinum_concentration / K3
        )


# ======================================================================
# Global platinum dissolution model
# ======================================================================

@dataclass
class PtDissolution:
    """
    Coupled platinum degradation model.

    Combines:
        - metallic platinum dissolution
        - oxide formation
        - oxide dissolution

    The model evolves:
        • particle radius distribution
        • dissolved platinum concentration

    using surface-integrated reaction rates.
    """

    platinum_dissolution: PlatinumDissoultion = field(
        default_factory=PlatinumDissoultion
    )

    platinum_oxide_formation: PlatinumOxideFormation = field(
        default_factory=PlatinumOxideFormation
    )

    platinum_oxide_dissolution: PlatinumOxideDissolution = field(
        default_factory=PlatinumOxideDissolution
    )

    catalyst_layer: CatalystLayer = field(
        default_factory=CatalystLayer
    )

    def time_derivatives(
        self,
        dissolved_platinum_concentration,
        platinum_oxide_coverage,
        proton_concentration,
        potential,
        temperature,
        relative_humidity,
    ):
        """
        Compute degradation time derivatives.

        Returns
        -------
        particle_radius_time_derivative : ndarray [m/s]
            Radius evolution for each particle size bin.

        dissolved_platinum_concentration_time_derivative :
            float [kmol/m³/s]
            Dissolved platinum accumulation rate.

        Reference: 
        ---------- 
        Schneider et al. J. Electrochem. Soc. 2019, 166 (4), F322–F333.
        Darling; Meyers. J. Electrochem. Soc. 2003, 150 (11), A1523.
        """

        r_particles = self.catalyst_layer.platinum_size_distribution.r_array[:,np.newaxis]
        N_particles = self.catalyst_layer.platinum_size_distribution.number_density_distrib_array[:,np.newaxis]
        # ---- Reaction rates
        r1 = self.platinum_dissolution.rate_of_reaction(
            dissolved_platinum_concentration,
            platinum_oxide_coverage,
            potential,
            temperature,
            relative_humidity,
            r_particles,
        )

        r2 = self.platinum_oxide_formation.rate_of_reaction(
            dissolved_platinum_concentration,
            platinum_oxide_coverage,
            proton_concentration,
            potential,
            temperature,
            r_particles,
        )

        r3 = self.platinum_oxide_dissolution.rate_of_reaction(
            dissolved_platinum_concentration,
            platinum_oxide_coverage,
            proton_concentration,
            temperature,
            r_particles,
            self.platinum_dissolution,
            self.platinum_oxide_formation,
        )

        # --------------------------------------------------------------
        # Particle shrinkage due to Pt dissolution and oxide formation 
        # --------------------------------------------------------------
        r_particles_time_derivative = (
            -platinum.molar_volume * (r1 + r2)
        )

        # --------------------------------------------------------------
        # Source term for dissolved platinum sink due to Pt band 
        # formation. Use a characteristic time approach since 
        # values for diffusivity and Pt band distance not available in 
        # Schneider et al. (2019). 
        # A characteristic time of ~3 is estimated based on results in 
        # figure 9 of that paper. 
        # --------------------------------------------------------------
        reference_temperature = 353.15
        diffusion_time = 3 * (temperature * relative_humidity 
                              / reference_temperature ) ** -2 
        platinum_band_sink = - (dissolved_platinum_concentration / 
                                diffusion_time)

        
        # --------------------------------------------------------------
        # Dissolved platinum accumulation in ionomer phase
        # Surface-integrated source term
        # --------------------------------------------------------------
        dissolved_platinum_concentration_time_derivative = (
            1 / self.catalyst_layer.ionomer_vol_fraction
            * np.trapezoid(
                4 * np.pi
                * r_particles**2
                * (r1 + r3)
                * N_particles,
                r_particles,
                axis=0
                ) - platinum_band_sink
        )

        # --------------------------------------------------------------
        # Oxide coverage 
        # ------------------------------------------------------------
        active_sites_per_platinum_area = 218e-6 # Assumes 210 uC/cm2 Pt in the H2 adsorption region, as in Schneider et al. (2019). 
        platinum_oxide_coverage_time_derivative = (
            (r2 - r3) / active_sites_per_platinum_area  + 
            - r_particles_time_derivative * 
            (2 * platinum_oxide_coverage / r_particles)
        ) 
        return (
            r_particles_time_derivative,
            dissolved_platinum_concentration_time_derivative,
            platinum_oxide_coverage_time_derivative
        )