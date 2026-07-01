"""
Membrane hydrogen permeation models.
"""
from dataclasses import dataclass
from marapendi.tools import arrhenius_term


@dataclass
class HydrogenPermeationModel:
    """
    Hydrogen permeation model combining diffusion and convection contributions.

    Models the hydrogen crossover flux through a membrane as a function of
    temperature, pressure difference, and membrane water content, following
    Trinke et al. (2016). An interface resistance term is added in series to
    avoid overestimating crossover in thin membranes.

    Attributes
    ----------
    permeability_reference_temperature : float
        Reference temperature for the permeability coefficients (K).
    permeability_reference_water_vol_fraction : float
        Reference water volume fraction for the permeability coefficients (n.d.).
    reference_diffusion_permeability_coefficient : float
        Reference diffusion permeability coefficient (kmol/(m·s·Pa)).
    diffusion_permeability_activation_energy : float
        Activation energy for diffusion permeability (J/kmol).
    reference_convection_permeability_coefficient : float
        Reference convection permeability coefficient (kmol/(m·s·Pa²)).
    convection_permeability_activation_energy : float
        Activation energy for convection permeability (J/kmol).
    interface_resistance : float
        Interface resistance in series with the membrane ((m·s·Pa)/kmol).
    permeability_correction_factor : float
        Multiplicative correction factor applied to the permeation flux (n.d.).

    References
    ----------
    Trinke, P. et al. J. Electrochem. Soc. 163, F3164–F3170 (2016).
    Kang, Z., Pak, M. & Bender, G. Int. J. Hydrogen Energy 46, 15161–15167 (2021).
    """

    # Default values from table II in Trinke et al. (2016)
    permeability_reference_temperature: float = 333.15
    reference_diffusion_permeability_coefficient: float = 2.95e-17
    diffusion_permeability_activation_energy: float = 27e6
    reference_convection_permeability_coefficient: float = 9.01e-24
    convection_permeability_activation_energy: float = 2.7e6
    permeability_reference_water_vol_fraction: float = 0.37
    interface_resistance: float = 1.6e12
    permeability_correction_factor: float = 1.

    def calculate_permeability_coefficient(self, temperature, pressure_difference):
        """
        Effective hydrogen permeability coefficient at the given conditions.

        Sums diffusion and convection contributions, each with Arrhenius
        temperature dependence following Trinke et al. (2016).

        Parameters
        ----------
        temperature : float
            Temperature (K).
        pressure_difference : float
            Pressure difference between anode and cathode (Pa). Positive
            when the hydrogen side is at higher pressure.

        Returns
        -------
        float
            Effective hydrogen permeability coefficient (kmol/(m·s·Pa)).

        References
        ----------
        Trinke, P. et al. J. Electrochem. Soc. 163, F3164–F3170 (2016).
        """
        diffusion_permeability_coeff = (
            self.reference_diffusion_permeability_coefficient *
            arrhenius_term(
                self.diffusion_permeability_activation_energy,
                temperature,
                self.permeability_reference_temperature
            )
        )
        convection_permeability_coeff = (
            self.reference_convection_permeability_coefficient *
            arrhenius_term(
                self.convection_permeability_activation_energy,
                temperature,
                self.permeability_reference_temperature
            )
        )
        return diffusion_permeability_coeff + convection_permeability_coeff * pressure_difference

    def permeation_flux(self,
                        membrane_thickness: float,
                        partial_pressure_h2: float,
                        temperature: float,
                        pressure_difference: float,
                        water_vol_fraction: float
        ) -> float:
        """
        Hydrogen permeation flux through the membrane.

        Uses the Trinke et al. (2016) model with temperature dependence and
        an interface resistance in series to match experimental data for thin
        membranes (Kang et al., 2021).

        Parameters
        ----------
        membrane_thickness : float
            Membrane thickness (m).
        partial_pressure_h2 : float
            Hydrogen partial pressure on the high-pressure side (Pa).
        temperature : float
            Temperature (K).
        pressure_difference : float
            Pressure difference between anode and cathode (Pa). Positive
            when the hydrogen side is at higher pressure.
        water_vol_fraction : float
            Membrane water volume fraction (n.d.).

        Returns
        -------
        float
            Hydrogen permeation flux (kmol/(m²·s)).

        References
        ----------
        Trinke, P. et al. J. Electrochem. Soc. 163, F3164–F3170 (2016).
        Kang, Z., Pak, M. & Bender, G. Int. J. Hydrogen Energy 46, 15161–15167 (2021).
        """
        permeability_coefficient = self.calculate_permeability_coefficient(
            temperature,
            pressure_difference
        )

        # The interface resistance reproduces (within 10%) results from Kang et al. (2021).
        h2_permeability_resistance = (membrane_thickness / permeability_coefficient +
                                        self.interface_resistance)

        return (partial_pressure_h2 / h2_permeability_resistance  *
                    water_vol_fraction  / self.permeability_reference_water_vol_fraction *
                    self.permeability_correction_factor)
