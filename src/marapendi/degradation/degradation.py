"""
Platinum degradation kinetics for PEM fuel cells.

Implements particle-size-resolved dissolution, oxide formation, oxide
dissolution, oxide place-exchange, cathodic dissolution, and carbon corrosion
following Darling and Meyers (2003) and Schneider et al. (2019).

References
----------
Darling, R. M. & Meyers, J. P. J. Electrochem. Soc. 150, A1523 (2003).
Schneider, P. et al. J. Electrochem. Soc. 166, F322–F333 (2019).
"""

from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray

from ..models.thermo.constants import GAS_CONSTANT, FARADAY_CONSTANT
from ..tools import potential_activation
from ..porous_layers.catalyst_layers import CatalystLayer
from scipy.stats import lognorm, norm
from scipy.interpolate import interp1d 
from scipy.special import lambertw
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
    r_mean : float
        Mean particle radius (m).
    r_std : float
        Standard deviation of particle radius (m).
    initial_platinum_loading : float
        Platinum loading (kg/m²).
    initial_cl_thickness : float
        Catalyst layer thickness (m).
    initial_ecsa : float
        Initial electrochemical surface area (m²/kg).
    distribution_type : str
        Type of statistical radius distribution: ``'norm'`` or ``'lognorm'``.
    """
    number_density_array: NDArray[np.float64]
    r_array: NDArray[np.float64]
    n_points: int = 32
    r_mean: float = 0
    r_std: float = 1e-9
    initial_platinum_loading: float = 0.2e-2
    initial_cl_thickness: float = 10e-6
    initial_ecsa: float = 40e3
    distribution_type: str = "norm"


    def __post_init__(self):
        """Initialize statistical distribution and derived quantities."""

        self.initial_platinum_loading
        self.cl_thickness = self.initial_cl_thickness
        if self.r_mean > 0: 
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
            self.r_edges = np.linspace(
                *self.initial_dist.interval(0.999),
                self.n_points + 1
            )
            self.r_array = 0.5 * (self.r_edges[...,1:]
                                + self.r_edges[...,:-1])
            self.dr_array = np.diff(self.r_edges)
            
            # Normalized PDF
            self.normalized_distrib_array = \
                self.initial_dist.pdf(self.r_array)

            # Particle number density distribution
            self.number_density_distrib_array = (
                self.initial_number_density()
                * self.normalized_distrib_array
            )

            # Particle number density for each radius
            self.number_density_array = (
                self.number_density_distrib_array
                * self.dr_array 
            )
        else: 
            self.n_points = len(self.r_array)

        # Conversion between geometrical surface and ECSA
        self.ecsa_to_particle_average_surface_ratio = (
            self.initial_ecsa /
            self.average_particle_surface()
        )

        self.initial_utilization_ratio = (
            self.initial_ecsa /
            (self.average_particle_surface()
             / self.average_particle_mass(normalized=False))
        )
    
    # ------------------------------------------------------------------
    def average_particle_volume(self, normalized=False):
        """Return average particle volume [m³]."""
        if normalized: 
            n = self.normalized_distrib_array * self.dr_array 
        else: 
            n = self.number_density_array
        return (
            4 * np.pi/3 *
            np.sum(
                n * self.r_array ** 3, axis=0
            )
            / np.sum(
                n, axis=0
            )
        )

    def average_particle_mass(self, normalized=False):
        """Return average particle mass [kg]."""
        return platinum.density * self.average_particle_volume(normalized)

    def platinum_loading(self): 
        return (
            self.number_density()
            * self.cl_thickness
            * self.average_particle_mass()
        )
    
    def number_density(self): 
        return np.sum(self.number_density_array, axis=0)
    
    def initial_number_density(self):
        """
        Particle number density inside catalyst layer [1/m³].
        """
        return (
            self.initial_platinum_loading
            / self.cl_thickness
            / self.average_particle_mass(normalized=True)
        )

    def average_particle_surface(self,normalized=False):
        """Return average particle surface area [m²]."""
        if normalized: 
            n = self.normalized_distrib_array * self.dr_array 
        else: 
            n = self.number_density_array
        return (
            4 * np.pi *
            np.sum(
                self.number_density_array
                * self.r_array**2, axis=0
            )
            / np.sum(
                self.number_density_array, axis=0
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
            / self.average_particle_mass()
            * self.initial_utilization_ratio
        )

    def plot_distribution(self, ax=None,
                          label="Pt radius distribution"):
        """Plot particle radius probability density."""
        if ax is None:
            _, ax = plt.subplots()

        ax.plot(self.r_array,
                self.number_density_array,
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
    density : float
        Density (kg/m³).
    molecular_weight : float
        Molecular weight (kg/kmol).
    surface_tension : float
        Surface tension (J/m²).
    reference_pontential : float
        Reference Gibbs potential (J/kmol).
    """

    density: float
    molecular_weight: float
    surface_tension: float
    reference_pontential: float = 0

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
            self.reference_pontential +
            self.surface_tension
            * self.molar_volume
            / particle_radius
        )


# Reference materials
platinum = PlatinumSpecies(
    density=21450.,
    molecular_weight=195.08,
    surface_tension=2.37,
    reference_pontential=0.
)

platinum_oxide = PlatinumSpecies(
    density=14100.,
    molecular_weight=211.08,
    surface_tension=1.00,
    reference_pontential=-42.3e6
)


# ======================================================================
# Platinum dissolution kinetics
# ======================================================================

@dataclass
class PlatinumDissolution:
    """
    Platinum electrochemical dissolution reaction.

    References
    ----------
    Schneider, P. et al. J. Electrochem. Soc. 166, F322–F333 (2019).
    """

    rate_constant: float = 3.43e-23
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
            ) / (2 * FARADAY_CONSTANT)
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

    Electrochemical oxidation of metallic platinum to platinum oxide.

    References
    ----------
    Darling, R. M. & Meyers, J. P. J. Electrochem. Soc. 150, A1523 (2003).

    Parameters
    ----------
    rate_constant : float
        Reaction kinetic prefactor (kmol/m²/s).
    reference_potential : float
        Standard equilibrium potential (V).
    transfer_coeff_ca : float
        Cathodic charge-transfer coefficient (n.d.).
    transfer_coeff_an : float
        Anodic charge-transfer coefficient (n.d.).
    omega_platinum_oxide_formation : float
        Lateral interaction parameter between oxide species (J/kmol).
    proton_reference_concentration : float
        Reference proton concentration (kmol/m³).
    """

    rate_constant: float = 1.36e-10
    reference_potential: float = 0.98
    transfer_coeff_ca: float = 0.5
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
            / (2 * FARADAY_CONSTANT)
        )

    def rate_of_reaction(
        self,
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
            / (2 * FARADAY_CONSTANT * self.transfer_coeff_an)
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
# Oxide place-exchange kinetics
# ======================================================================

@dataclass
class OxidePlaceExchange:
    """
    Oxide place-exchange reaction kinetics.

    References
    ----------
    Schneider, P. et al. J. Electrochem. Soc. 166, F322–F333 (2019).
    """

    forward_rate_constant: float = 1.e-21
    backward_rate_constant: float = 1.e-19
    reference_potential: float = 1.2
    transfer_coeff: float = 0.5
    omega_forward: float = 10e6
    omega_backward: float = 90e6

    def forward_rate(
            self, 
            potential,
            platinum_oxide_coverage,
            temperature 
    ): 
        return (
            self.forward_rate_constant 
            * platinum_oxide_coverage 
            * potential_activation(
                self.transfer_coeff,
                1,
                temperature,
                potential
                - self.reference_potential
                + (
                    self.omega_forward
                    * platinum_oxide_coverage
                    / (FARADAY_CONSTANT * self.transfer_coeff)
                )
            )
        )
    def rate_of_reaction(
        self,
        platinum_oxide_coverage,
        place_exchanged_oxide_coverage, 
        potential,
        temperature,
    ):
        """
        Net place-exchanged oxide formation rate [kmol/m²/s].
        """
        RT = GAS_CONSTANT * temperature

        forward_rate = self.forward_rate(
            potential, 
            platinum_oxide_coverage, 
            temperature
        )
        backward_rate = (
            self.backward_rate_constant
            * place_exchanged_oxide_coverage
            * potential_activation(
                self.transfer_coeff,
                1,
                temperature,
                + (
                    self.omega_backward
                    * place_exchanged_oxide_coverage
                    / (FARADAY_CONSTANT * self.transfer_coeff)
                )
            )
        )

        return forward_rate - backward_rate
    
    def limiting_coverage(
        self, 
        potential, 
        platinum_oxide_coverage, 
        temperature
    ): 
        RT = GAS_CONSTANT * temperature 
        forward_rate = self.forward_rate(
            potential, 
            platinum_oxide_coverage, 
            temperature
        )
        return (
            RT / self.omega_backward 
            * np.real(lambertw(
                forward_rate 
                * self.omega_backward 
                / (self.backward_rate_constant * RT))
            )
        )
    
# ======================================================================
# Cathodic dissolution
# ======================================================================            

@dataclass
class CathodicDissolution:
    """
    Cathodic platinum dissolution kinetics.

    References
    ----------
    Schneider, P. et al. J. Electrochem. Soc. 166, F322–F333 (2019).
    """

    rate_constant: float = 1.4e-8

    def rate_of_reaction(
            self, 
            place_exchanged_oxide_coverage, 
            limiting_place_exchanged_coverage
    ):
        return self.rate_constant * np.maximum(
            place_exchanged_oxide_coverage 
            - limiting_place_exchanged_coverage, 
            0
        )
    
# ======================================================================
# Platinum oxide dissolution
# ======================================================================

@dataclass
class PlatinumOxideDissolution:
    """
    Chemical dissolution of platinum oxide.

    Couples oxide reduction with the dissolved platinum ion concentration
    in the ionomer phase.

    References
    ----------
    Darling, R. M. & Meyers, J. P. J. Electrochem. Soc. 150, A1523 (2003).
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
        K3 =  potential_activation(
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
# Carbon corrosion
# ======================================================================
@dataclass
class CarbonCorrosion:
    """
    Carbon support corrosion kinetics.

    Exponential potential dependence for the carbon oxidation reaction rate.

    Attributes
    ----------
    rate_constant : float
        Reaction rate prefactor (kmol/m²/s).
    potential_dependency : float
        Exponential sensitivity to potential (1/V).
    reference_potential : float
        Reference potential for the exponential term (V).
    """
    rate_constant: float = 1.5e-22
    potential_dependency: float = 19
    reference_potential: float = 0.2

    def reaction_rate(
            self, 
            potential 
    ): 
        return (
            self.rate_constant 
            * np.exp(
                self.potential_dependency
                * (potential - self.reference_potential) 
            )
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

    platinum_dissolution: PlatinumDissolution = field(
        default_factory=PlatinumDissolution
    )

    platinum_oxide_formation: PlatinumOxideFormation = field(
        default_factory=PlatinumOxideFormation
    )

    platinum_oxide_dissolution: PlatinumOxideDissolution = field(
        default_factory=PlatinumOxideDissolution
    )

    platinum_cathodic_dissolution: CathodicDissolution = field(
        default_factory=CathodicDissolution
    )
    platinum_oxide_place_exchange: OxidePlaceExchange = field(
        default_factory=OxidePlaceExchange
    )

    carbon_corrosion: CarbonCorrosion = field(
        default_factory=CarbonCorrosion
    )

    catalyst_layer: CatalystLayer = field(
        default_factory=CatalystLayer
    )

    def time_derivatives(
        self,
        dissolved_platinum_concentration,
        platinum_oxide_coverage,
        place_exchanged_oxide_coverage, 
        proton_concentration,
        potential,
        temperature,
        relative_humidity, 
        particle_radius,
        particle_number_density
    ):
        """
        Compute degradation time derivatives.

        Returns
        -------
        particle_radius_time_derivative : ndarray
            Radius evolution for each particle size bin (m/s).
        dissolved_platinum_concentration_time_derivative : float
            Dissolved platinum accumulation rate (kmol/m³/s).
        platinum_oxide_coverage_time_derivative : float
            Rate of change of oxide surface coverage (1/s).
        place_exchanged_oxide_coverage_time_derivative : float
            Rate of change of place-exchanged oxide coverage (1/s).
        number_density_time_derivative : ndarray
            Rate of change of particle number density (1/m³/s).

        References
        ----------
        Schneider, P. et al. J. Electrochem. Soc. 166, F322–F333 (2019).
        Darling, R. M. & Meyers, J. P. J. Electrochem. Soc. 150, A1523 (2003).
        """

        # ---- Reaction rates
        r_a_diss = self.platinum_dissolution.rate_of_reaction(
            dissolved_platinum_concentration,
            platinum_oxide_coverage,
            potential,
            temperature,
            relative_humidity,
            particle_radius,
        )

        r_ox = self.platinum_oxide_formation.rate_of_reaction(
            platinum_oxide_coverage,
            proton_concentration,
            potential,
            temperature,
            particle_radius,
        )

        r_chem = self.platinum_oxide_dissolution.rate_of_reaction(
            dissolved_platinum_concentration,
            platinum_oxide_coverage,
            proton_concentration,
            temperature,
            particle_radius,
            self.platinum_dissolution,
            self.platinum_oxide_formation,
        )

        r_pe = self.platinum_oxide_place_exchange.rate_of_reaction(
            platinum_oxide_coverage,
            place_exchanged_oxide_coverage,
            potential, 
            temperature
        )

        theta_OPt_lim = self.platinum_oxide_place_exchange.limiting_coverage(
            potential,
            platinum_oxide_coverage,
            temperature 
        )

        r_c_diss = self.platinum_cathodic_dissolution.rate_of_reaction(
            place_exchanged_oxide_coverage, 
            theta_OPt_lim
        )

        # --------------------------------------------------------------
        # Particle shrinkage due to Pt dissolution and oxide formation 
        # --------------------------------------------------------------
        r_particles_time_derivative = (
            -platinum.molar_volume * (r_a_diss + r_chem + r_c_diss)
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
        diffusion_time = 9.5/2.8 * (temperature * relative_humidity 
                              / reference_temperature ) ** -2 
        platinum_band_sink = (dissolved_platinum_concentration / 
                                diffusion_time)

        
        # --------------------------------------------------------------
        # Dissolved platinum accumulation in ionomer phase
        # Surface-integrated source term
        # --------------------------------------------------------------
        dissolved_platinum_concentration_time_derivative = (
            1. / self.catalyst_layer.ionomer_vol_fraction
            * (np.sum(
                4 * np.pi
                * particle_radius**2
                * (r_a_diss + r_chem + r_c_diss)
                * particle_number_density,
                axis=0
                ) - platinum_band_sink)
        )

        # --------------------------------------------------------------
        # Oxide coverage 
        # ------------------------------------------------------------
        active_sites_per_platinum_area = 2.18e-8 # Assumes 210 uC/cm2 Pt in the H2 adsorption region, as in Schneider et al. (2019). 
        platinum_oxide_coverage_time_derivative = (
            (r_ox - r_chem - r_pe) / active_sites_per_platinum_area  + 
            - r_particles_time_derivative * 
            (2 * platinum_oxide_coverage / particle_radius)
        ) 

        # --------------------------------------------------------------
        # Place-exchanged oxide coverage
        # ------------------------------------------------------------
        place_exchanged_oxide_coverage_time_derivative = (
            (r_pe - r_c_diss) / active_sites_per_platinum_area  + 
            - r_particles_time_derivative * 
            (2 * place_exchanged_oxide_coverage / particle_radius)
        ) 

        # --------------------------------------------------------------
        # Particle number density
        # ------------------------------------------------------------
        number_density_time_derivative = (
            - self.carbon_corrosion.reaction_rate(potential)
            * particle_number_density / particle_radius
        )

        return (
            r_particles_time_derivative,
            dissolved_platinum_concentration_time_derivative,
            platinum_oxide_coverage_time_derivative, 
            place_exchanged_oxide_coverage_time_derivative,
            number_density_time_derivative, 
        )