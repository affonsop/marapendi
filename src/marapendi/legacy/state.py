"""
State containers for the vendored ``legacy`` fuel-cell model.

Mirrors the *shape* of :class:`marapendi.legacy.fuelcell.FuelCell`
(``ca``/``an`` sides, each with a catalyst layer, GDL and flow channel, plus
a membrane) so that state objects line up 1:1 with the components that own
the corresponding static parameters. This is a pure data layer: dataclasses
only, no physics. Component methods are expected to evolve from
``self.attr = ...`` mutation towards ``state = model(component, state)``.

Iteration helpers (``layers``, ``sides``) let model code loop over all
layers/sides without hardcoding ``ca``/``an``/``cl``/``gdl`` everywhere.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
import numpy as np


@dataclass
class LayerState:
    """State of a single porous layer (GDL, catalyst layer, ...)."""
    temperature: float = None
    pressure: float = None
    dry_o2_mole_fraction: float = None
    dry_h2_mole_fraction: float = None
    relative_humidity: float = None
    liquid_saturation: float = 0.
    non_wetting_saturation: float = 0.
    non_wetting_flux: float = None
    capillary_pressure: float = None
    upstream_saturation: float = None
    downstream_saturation: float = None
    downstream_capillary_pressure: float = None
    flow_resistance_with_rel_permeability: float = None
    equivalent_flow_resistance: float = None
    liquid_balance: float = None


@dataclass
class CatalystLayerState(LayerState):
    """State of a catalyst layer, in addition to the generic layer state."""
    ionomer_water_content: float = None
    overpotential: float = None
    crossover_current: float = None
    water_film_thickness: float = None
    proton_resistance: float = None
    theta_catalyst: float = None


@dataclass
class MembraneState:
    """State of the membrane."""
    temperature: float = None
    water_content: float = None
    water_flux: float = None
    h2_permeation_flux: float = None
    proton_resistance: float = None


@dataclass
class FlowChannelState:
    """State of a flow channel (anode or cathode)."""
    inlet_gas_flow_rate: float = None
    inlet_liquid_flow_rate: float = None
    inlet_liquid_saturation: float = None
    inlet_stoichiometry: float = None


@dataclass
class FuelCellSideState:
    """State of one side (anode or cathode) of the fuel cell."""
    cl: CatalystLayerState = field(default_factory=CatalystLayerState)
    gdl: LayerState = field(default_factory=LayerState)
    mpl: LayerState | None = None
    ch: FlowChannelState = field(default_factory=FlowChannelState)

    reactant_consumption: float = None
    h2o_production: float = None
    reactant_transport_resistance: float = None
    h2ov_transport_resistance: float = None
    liquid_flux: float = None
    cl_to_gdl_liquid_flux: float = None
    cl_to_mpl_liquid_flux: float = None
    mpl_to_gdl_liquid_flux: float = None
    gdl_to_ch_liquid_flux: float = None
    is_liquid_equilibrated: bool = False
    s_relax: float = None
    t_relax: float = None
    membrane_water_flux: float = None
    water_flux: float = None

    @property
    def layers(self) -> list[LayerState]:
        """All layer states on this side, GDL-to-CL order, including MPL if present."""
        return [layer for layer in (self.gdl, self.mpl, self.cl) if layer is not None]


@dataclass
class FuelCellState:
    """Full state of the fuel cell at one operating point."""
    ca: FuelCellSideState = field(default_factory=FuelCellSideState)
    an: FuelCellSideState = field(default_factory=FuelCellSideState)
    membrane: MembraneState = field(default_factory=MembraneState)

    current_density: float = None
    temperature: float = None
    mea_temperature: float = None
    cell_voltage: float = None
    heat_release_rate: float = None
    thermal_resistance: float = None
    E_rev: float = None
    eta_act: float = None
    eta_ohm: float = None

    # Reference to the ``FuelCell`` this state was snapshotted from. Used by
    # models (e.g. ``VoltageModel``) that still need access to component
    # objects/methods not yet captured as plain data in this state.
    fuel_cell: object = field(default=None, repr=False, compare=False)

    @property
    def sides(self) -> tuple[FuelCellSideState, FuelCellSideState]:
        """``(ca, an)`` side states, for ``for side in state.sides: ...`` loops."""
        return (self.ca, self.an)

    @property
    def layers(self) -> list[LayerState]:
        """All layer states across both sides (GDL/MPL/CL), GDL-to-CL order."""
        return [layer for side in self.sides for layer in side.layers]
