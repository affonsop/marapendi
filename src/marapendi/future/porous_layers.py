"""
Porous layer components: static physical properties and correlations.

A :class:`PorousLayer` (and its specializations :class:`GasDiffusionLayer`
and :class:`MicroPorousLayer`) holds only *static* physical properties
(geometry, permeability, wettability, ...) plus the correlation models used
to turn those properties, together with a :class:`~marapendi.future.state.LayerState`,
into transport quantities (gas diffusion resistance, capillary pressure /
saturation relations, thermal resistance).

No physical *variables* (temperature, pressure, gas composition, saturation,
...) are stored here -- those live in ``LayerState`` and are produced/consumed
by :class:`marapendi.future.model.CellModel`.

This module currently only defines the dataclasses and their static
properties. Correlation methods will be added component by component.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from ..legacy.transport_models import PorousGasResistanceModel, DarcyTransportModel
from ..legacy.water import water_molecular_weight
from .gas import GasModel, molecular_weights, species_indexes
from .water import WaterModel


class _StateLayerView:
    """Adapt a ``(layer, state)`` pair to the attribute-based interface expected by
    :class:`~marapendi.legacy.transport_models.DarcyTransportModel`."""

    def __init__(self, layer: 'PorousLayer', state):
        self.relative_permeability_exponent = layer.relative_permeability_exponent
        self.capillary_pressure_J_ratio = layer.capillary_pressure_J_ratio(state)
        self.saturation_flow_resistance = layer.saturation_flow_resistance(state)


@dataclass
class PorousLayer:
    """Static properties of a porous layer (GDL, MPL or catalyst layer).

    Attributes
    ----------
    thickness : float
        Layer thickness (m).
    porosity : float
        Layer porosity (-).
    effective_gas_diffusion_ratio : float
        Ratio of the effective to bulk gas diffusivity (-).
    pore_diameter : float
        Mean pore diameter (m), used for Knudsen diffusion.
    absolute_permeability : float
        Absolute (intrinsic) permeability (m^2).
    relative_permeability_exponent : float
        Exponent of the relative permeability law (-).
    contact_angle : float
        Contact angle of the wetting phase (degrees).
    thermal_conductivity : float
        Thermal conductivity (W/m/K).
    non_wetting_phase, wetting_phase : str
        Identity of the two fluid phases ('water' or 'gas').
    transport_resistance_model : PorousGasResistanceModel
        Correlation used for gas diffusion resistance.
    two_phase_transport_model : DarcyTransportModel
        Correlation used for the capillary-pressure / saturation relation.
    """

    thickness: float = 1e-3
    porosity: float = 1.
    effective_gas_diffusion_ratio: float = 1.
    pore_diameter: float = 1e12
    absolute_permeability: float = 1e6
    relative_permeability_exponent: float = 3.
    contact_angle: float = 120.
    thermal_conductivity: float = 1e12
    non_wetting_phase: str = 'water'
    wetting_phase: str = 'gas'
    transport_resistance_model: PorousGasResistanceModel = field(default_factory=PorousGasResistanceModel)
    two_phase_transport_model: DarcyTransportModel = field(default_factory=DarcyTransportModel)

    def __post_init__(self):
        self.sqrt_abs_permeability_porosity = np.sqrt(self.absolute_permeability * self.porosity)
        self.cosinus_contact_angle = np.abs(np.cos(np.pi / 180 * self.contact_angle))

    def capillary_pressure_J_ratio(self, state) -> float:
        """Capillary-pressure scale of the layer's J-function, given the layer's ``state``."""
        return (
            WaterModel.surface_tension(state) * self.cosinus_contact_angle
            / np.sqrt(self.absolute_permeability / self.porosity)
        )

    def saturation_flow_resistance(self, state) -> float:
        """Resistance to non-wetting phase flow due to a saturation gradient (s.m^2/mol)."""
        if self.non_wetting_phase == 'water':
            non_wetting_kinematic_viscosity = WaterModel.kinematic_viscosity(state)
            non_wetting_molecular_weight = water_molecular_weight
        else:
            non_wetting_kinematic_viscosity = GasModel.mixture_kinematic_viscosity(state)
            non_wetting_molecular_weight = GasModel.mixture_molecular_weight(state)
        non_wetting_surface_tension = WaterModel.surface_tension(state)
        return (
            self.thickness * non_wetting_kinematic_viscosity * non_wetting_molecular_weight
            / (self.sqrt_abs_permeability_porosity * self.cosinus_contact_angle * non_wetting_surface_tension)
        )

    def saturation_from_capillary_pressure(self, state, capillary_pressure: float) -> float:
        """Non-wetting phase saturation from ``capillary_pressure``, given the layer's ``state``."""
        return self.two_phase_transport_model.saturation_from_capillary_pressure(
            _StateLayerView(self, state), capillary_pressure,
        )

    def capillary_pressure_from_saturation(self, state, non_wetting_saturation: float) -> float:
        """Capillary pressure from ``non_wetting_saturation``, given the layer's ``state``."""
        return self.two_phase_transport_model.capillary_pressure_from_saturation(
            _StateLayerView(self, state), non_wetting_saturation,
        )

    def non_wetting_saturation_from_flux(
        self, state, non_wetting_flux: float, upstream_capillary_pressure: float = 0.,
    ) -> tuple[float, float]:
        """Non-wetting (liquid water) saturation and downstream capillary pressure.

        Vectorized, unmasked port of
        :meth:`marapendi.legacy.transport_models.DarcyTransportModel.calculate_non_wetting_saturation`,
        from the non-wetting phase flux through the layer and the capillary
        pressure at its upstream boundary.

        Returns
        -------
        average_saturation : float
            Saturation averaged across the layer.
        downstream_capillary_pressure : float
            Capillary pressure at the downstream boundary, for chaining into
            the next layer's ``upstream_capillary_pressure``.
        """
        q = self.relative_permeability_exponent
        n = self.two_phase_transport_model.J_function_exponent
        exponent = 1. / (q + n)

        upstream_saturation = self.saturation_from_capillary_pressure(state, upstream_capillary_pressure)
        flux = np.maximum(0., non_wetting_flux)
        downstream_saturation = np.clip(
            upstream_saturation + (self.saturation_flow_resistance(state) * flux * (q + n) / n) ** exponent,
            0., 0.9,
        )
        average_saturation = np.clip(
            (downstream_saturation - upstream_saturation) * (q + n) / (q + n + 1) + upstream_saturation,
            0., 0.9,
        )
        downstream_capillary_pressure = self.capillary_pressure_from_saturation(state, downstream_saturation)
        return average_saturation, downstream_capillary_pressure

    def thermal_resistance(self) -> float:
        """Thermal resistance of the layer (m^2 K / W)."""
        return self.thickness / self.thermal_conductivity

    def gas_transport_resistance(self, state, species: str = 'o2') -> float:
        """Gas diffusion resistance for ``species`` (s/m), given the layer's ``state``."""
        return self.transport_resistance_model.total_diffusion_resistance(
            self,
            state.temperature,
            GasModel.species_diffusion_coefficient(state, species),
            molecular_weights[species_indexes[species]],
            state.non_wetting_saturation,
        )


@dataclass
class GasDiffusionLayer(PorousLayer):
    """Gas diffusion layer: a :class:`PorousLayer` with typical GDL defaults."""

    thickness: float = 200e-6
    porosity: float = 0.6
    contact_angle: float = 120.
    absolute_permeability: float = 1e-12
    thermal_conductivity: float = 0.5


@dataclass
class MicroPorousLayer(PorousLayer):
    """Microporous layer (MPL): a :class:`PorousLayer` with typical MPL defaults."""

    thickness: float = 30e-6
    porosity: float = 0.4
    contact_angle: float = 130.
    absolute_permeability: float = 1e-13
    thermal_conductivity: float = 0.3
