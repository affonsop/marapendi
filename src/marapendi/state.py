"""
State containers for the reference cell model.

Mirrors the *shape* of the cell (``ca``/``an`` sides, each with a catalyst
layer, GDL/MPL and flow channel, plus a membrane) so that state objects line
up 1:1 with the components that own the corresponding static parameters.
This is a pure data layer: dataclasses only, no physics.

Iteration helpers (``layers``, ``sides``, ``porous_layers``) let model code
loop over all layers/sides without hardcoding ``ca``/``an``/``cl``/``gdl``
everywhere.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .gas import GasState


@dataclass
class LayerState:
    """State of a single porous layer (GDL, MPL, catalyst layer, ...)."""

    gas: GasState = field(default_factory=GasState)
    temperature: float = None
    pressure: float = None
    saturation_pressure: float = None
    surface_tension: float = None
    kinematic_viscosity: float = None
    density: float = None
    relative_humidity: float = None
    liquid_saturation: float = 0.
    non_wetting_saturation: float = 0.
    capillary_pressure: float = None
    # Two-phase transport state (set by water saturation model)
    non_wetting_flux: float = None
    downstream_saturation: float = None
    upstream_saturation: float = None
    downstream_capillary_pressure: float = None
    electrolyte_saturation: float = None


@dataclass
class CatalystLayerState(LayerState):
    """State of a catalyst layer, in addition to the generic layer state."""

    ionomer_water_content: float = None
    overpotential: float = None
    water_film_thickness: float = None
    proton_resistance: float = None
    theta_catalyst: float = None

    eq_water_content: float = None
    membrane_interface_water_content: float = None


@dataclass
class MembraneState:
    """State of the membrane."""

    temperature: float = None
    saturation_pressure: float = None
    density: float = None
    water_content: float = None
    water_flux: float = None
    h2_permeation_flux: float = None
    proton_resistance: float = None

    peclet_number: float = None
    ePe: object = None
    ePexi: object = None
    xi: object = None
    water_diffusion_resistance: float = None
    water_content_profile: float = None
    water_content_derivative_profile: float = None


@dataclass
class FlowChannelState:
    """State of a flow channel (anode or cathode)."""

    gas: GasState = field(default_factory=GasState)
    temperature: float = None
    pressure: float = None
    saturation_pressure: float = None
    inlet_gas_flow_rate: float = 1e-12
    inlet_liquid_flow_rate: float = None
    inlet_liquid_saturation: float = None
    inlet_stoichiometry: float = None


@dataclass
class CellSideState:
    """State of one side (anode or cathode) of the cell."""

    cl: CatalystLayerState = field(default_factory=CatalystLayerState)
    gdl: LayerState | None = field(default_factory=LayerState)
    mpl: LayerState | None = None
    ch: FlowChannelState = field(default_factory=FlowChannelState)

    h2ov_transport_resistance: float = None
    reactant_transport_resistance: float = None
    reactant_consumption: float = None

    h2o_production: float = 0.
    s_relax: float | None = None
    membrane_water_flux: float = None
    water_flux: float = None
    liquid_flux: float = None
    vapor_flux: float = None
    gas_flux: float = None

    # Water balance intermediate state
    rh_at_cl_without_crossover: float = None
    estimated_water_content: float = None
    estimated_water_content_derivative: float = None
    liquid_eq_water_content: float = None
    vapor_eq_water_content: float = None

    # Non-dimensional water transport parameters (set by update_non_dimensional_parameters)
    is_liquid_equilibrated: bool = False
    alpha: float = None
    peclet_over_modified_biot: float = None
    biot_number: float = None

    @property
    def porous_layers(self) -> list[LayerState]:
        """All layer states on this side, GDL-to-CL order, including MPL if present."""
        return [layer for layer in (self.gdl, self.mpl, self.cl) if layer is not None]

    @property
    def layers(self) -> list[LayerState]:
        """All layer states, channel to CL order, including MPL if present."""
        return [layer for layer in (self.ch, self.gdl, self.mpl, self.cl) if layer is not None]


@dataclass
class CellState:
    """Full state of the cell at one operating point."""

    ca: CellSideState = field(default_factory=CellSideState)
    an: CellSideState = field(default_factory=CellSideState)
    membrane: MembraneState = field(default_factory=MembraneState)

    current_density: float = None
    temperature: float = None
    cell_voltage: float = None
    E_rev: float = None
    eta_act: float = None
    eta_ohm: float = None

    # Thermal state
    thermal_resistance: float = None
    mea_temperature: float = None
    mea_temperature_increase: float = None
    mea_water_molar_volume: float = None

    # Crossover
    crossover_current: float = None

    @property
    def sides(self) -> tuple[CellSideState, CellSideState]:
        """``(ca, an)`` side states, for ``for side in state.sides: ...`` loops."""
        return (self.ca, self.an)

    @property
    def side_layers(self) -> list[LayerState]:
        """All layer states across both sides, GDL-to-CL order."""
        return [layer for side in self.sides for layer in side.layers]

    @property
    def layers(self) -> list[LayerState]:
        """All layer states across both sides plus the membrane."""
        return self.side_layers + [self.membrane]
