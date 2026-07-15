"""
Module providing classes to model porous layers in electrochemical cells.
"""
from dataclasses import dataclass, field
import numpy as np
from ...models.thermo.constants import GAS_CONSTANT

from ...models.thermo.gas import GasModel
from ...simulation.state import GasState
from ...models.diffusion import PorousGasDiffusionModel
from ...models.darcy import DarcyTransportModel
from ...models.thermo.water import water_kinematic_viscosity, water_surface_tension, water_molecular_weight

@dataclass
class PorousLayer():
    """
    Represents a porous layer in a fuel cell or electrolyzer, defining its
    static geometry and material properties.

    Runtime temperature-dependent quantities (``RT``, ``breakthrough_pressure``,
    ``saturation_flow_resistance``) are computed by
    :meth:`update_state_at_temperature` and stored in the corresponding
    :class:`~marapendi.simulation.state.LayerState`.

    Attributes
    ----------
    thickness : float
        Thickness of the porous layer in meters (default is 1e-3 m).
    gas : GasState
        Initial gas composition.
    temperature : float
        Reference temperature used at construction time to pre-compute
        component-level capillary properties (K).
    pressure : float
        Reference pressure (Pa).
    porosity : float
        Porosity of the layer (0 < porosity < 1).
    tortuosity : float
        Tortuosity of the layer (tortuosity > 1).
    pore_diameter : float
        Average pore diameter in meters (default is large, so Knudsen diffusion negligible).
    transport_resistance_model : PorousGasDiffusionModel
        Model used to calculate gas transport resistance.
    two_phase_transport_model : DarcyTransportModel
        Model used to compute liquid water flow and capillarity.
    non_wetting_saturation : float
        Fraction of the pore volume occupied by non-wetting phase (0 to 1).
    thermal_conductivity : float
        Thermal conductivity in W/(m·K).
    volume_heat_capacity : float
        Volumetric heat capacity in J/(m3.K).
    absolute_permeability : float
        Absolute permeability in m².
    relative_permeability_exponent : float
        Exponent for relative permeability model.
    contact_angle : float
        Contact angle for the non-wetting phase in degrees (default is 120°).
    non_wetting_phase : str
        Non-wetting phase identifier ('water' or 'gas').
    breakthrough_pressure : float, optional
        Capillary entry-pressure scale σ·cos θ / √(K/ε) (Pa).  When ``None``
        (default), computed from geometry and ``temperature`` at construction
        time and recomputed at the actual operating temperature by
        :meth:`update_state_at_temperature`.  Provide an explicit value to
        override the geometry-based calculation.
    """

    thickness: float = 1e-3
    gas: GasState = field(default_factory=GasState)
    temperature: float = 300.
    pressure: float = 1e5
    porosity: float = 1
    tortuosity: float = 1
    pore_diameter: float = 1e12
    transport_resistance_model: PorousGasDiffusionModel = field(default_factory=PorousGasDiffusionModel)
    two_phase_transport_model: DarcyTransportModel = field(default_factory=DarcyTransportModel)
    non_wetting_saturation: float = 0
    thermal_conductivity: float = 1e12
    volume_heat_capacity: float = 1e6
    absolute_permeability: float = 1e6
    relative_permeability_exponent: float = 3
    contact_angle: float = 120.
    non_wetting_phase: str = 'water'
    wetting_phase: str = 'gas'
    breakthrough_pressure: float = None

    def __post_init__(self):
        self.sqrt_abs_permeability_porosity = np.sqrt(self.absolute_permeability * self.porosity)
        self.cosinus_contact_angle = np.abs(np.cos(np.pi / 180 * self.contact_angle))
        if self.contact_angle < 90:
            self.non_wetting_phase = 'gas'
            self.wetting_phase = 'water'
        self._bp_from_geometry = self.breakthrough_pressure is None
        if self._bp_from_geometry:
            self.breakthrough_pressure = self._compute_breakthrough_pressure(self.temperature)
        self.RT = GAS_CONSTANT * self.temperature
        self.saturation_pressure = None
        self.saturation_flow_resistance = self.calculate_saturation_flow_resistance(self.temperature)

    # ------------------------------------------------------------------
    # Temperature-dependent capillary helpers
    # ------------------------------------------------------------------

    def _compute_breakthrough_pressure(self, temperature: float) -> float:
        """Capillary entry-pressure scale (Pa) from layer geometry at *temperature*."""
        return (
            water_surface_tension(temperature) * self.cosinus_contact_angle
            / np.sqrt(self.absolute_permeability / self.porosity)
        )

    def update_state_at_temperature(self, layer_state, temperature: float) -> None:
        """Write temperature-dependent capillary quantities into *layer_state*.

        Called by :meth:`~marapendi.models.base.explicit_steady_state.ExplicitSteadyStateModel.set_initial_conditions`
        for every layer at stack temperature, and by
        :meth:`~marapendi.models.thermal.ThermalModel.set_mea_temperature` for
        catalyst layers at MEA temperature.  The component's own fields are
        not mutated.

        Parameters
        ----------
        layer_state : LayerState
            State object to populate.  Sets ``temperature``, ``RT``,
            ``diffusion_temp_and_pressure_correction``,
            ``breakthrough_pressure``, and ``saturation_flow_resistance``.
            Requires ``layer_state.pressure`` to already be set.
        temperature : float
            Temperature at which to evaluate all temperature-dependent
            quantities (K).
        """
        layer_state.temperature = temperature
        layer_state.RT = GAS_CONSTANT * temperature
        layer_state.diffusion_temp_and_pressure_correction = (
            GasModel.diffusion_temp_and_pressure_correction(temperature, layer_state.pressure)
        )
        layer_state.breakthrough_pressure = (
            self._compute_breakthrough_pressure(temperature)
            if self._bp_from_geometry
            else self.breakthrough_pressure
        )
        layer_state.saturation_flow_resistance = self.calculate_saturation_flow_resistance(
            temperature, getattr(self, 'electrolyte', None)
        )

    # ------------------------------------------------------------------
    # Static geometry / thermal
    # ------------------------------------------------------------------

    @property
    def thermal_resistance(self):
        """Thermal resistance of the layer (m²·K/W)."""
        return self.thickness / self.thermal_conductivity

    def calculate_saturation_flow_resistance(self, temperature, electrolyte=None):
        """Resistance to non-wetting phase flow at *temperature* (s·m²/mol).

        Parameters
        ----------
        temperature : float
            Temperature at which to evaluate fluid viscosity and surface
            tension (K).
        electrolyte : ElectrolyteSolution, optional
            Required only when ``non_wetting_phase == 'gas'`` and the wetting
            phase is a liquid electrolyte (provides ``surface_tension``).

        Returns
        -------
        float
            Saturation flow resistance in m²·s/kmol.
        """
        if self.non_wetting_phase == 'water':
            non_wetting_kinematic_viscosity = water_kinematic_viscosity(temperature)
            non_wetting_molecular_weight = water_molecular_weight
            non_wetting_surface_tension = water_surface_tension(temperature)
        elif self.non_wetting_phase == 'gas':
            non_wetting_kinematic_viscosity = GasModel.mixture_kinematic_viscosity(self)
            non_wetting_molecular_weight = GasModel.mixture_molecular_weight(self)
            non_wetting_surface_tension = (
                water_surface_tension(temperature) if self.wetting_phase == 'water'
                else electrolyte.surface_tension
            )
        return (
            (self.thickness * non_wetting_kinematic_viscosity * non_wetting_molecular_weight)
            / (self.sqrt_abs_permeability_porosity * self.cosinus_contact_angle * non_wetting_surface_tension)
        )

    def saturation_from_capillary_pressure(self, capillary_pressure):
        """Non-wetting saturation from capillary pressure via the two-phase transport model."""
        return self.two_phase_transport_model.saturation_from_capillary_pressure(self, capillary_pressure)

    def capillary_pressure_from_saturation(self, saturation):
        """Capillary pressure from non-wetting saturation via the two-phase transport model."""
        return self.two_phase_transport_model.capillary_pressure_from_saturation(self, saturation)

    def set_ionomer_wet_properties(self, ionomer_water_content, temperature):
        """Update ionomer transport properties at *ionomer_water_content* and *temperature*. Override in subclasses."""
        pass

    def set_water_film_thickness(self, water_saturation):
        """Set the liquid water film thickness from *water_saturation*. Override in subclasses."""
        pass


@dataclass
class GasDiffusionLayer(PorousLayer):
    """Gas diffusion layer: a :class:`PorousLayer` with typical GDL defaults."""

    thickness: float = 200e-6
    porosity: float = 0.6
    contact_angle: float = 120.
    absolute_permeability: float = 1e-12
    thermal_conductivity: float = 0.5
    volume_heat_capacity: float = 1.58e6

@dataclass
class MicroPorousLayer(PorousLayer):
    """Microporous layer (MPL): a :class:`PorousLayer` with typical MPL defaults."""

    thickness: float = 30e-6
    porosity: float = 0.4
    contact_angle: float = 130.
    absolute_permeability: float = 1e-13
    thermal_conductivity: float = 0.3
    volume_heat_capacity: float = 1.98e6