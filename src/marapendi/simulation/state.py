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
from types import SimpleNamespace as _SimpleNamespace

import numpy as np
from .conditions import SideConditions
from ..models.thermo.constants import GAS_CONSTANT
from ..models.thermo.gas import *
from ..models.thermo.water import water_molar_volume

@dataclass
class GasState:
    """Composition of the gas mixture.

    Attributes
    ----------
    X : np.ndarray
        Mole fractions of (O2, N2, H2, H2O), in the order given by
        :data:`~marapendi.thermo.gas.species_indexes`.
    """

    X: np.ndarray = field(default_factory=lambda: np.array([1., 0., 0., 0.]))


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
    # Temperature-derived capillary quantities (set by PorousLayer.update_state_at_temperature)
    RT: float = None
    diffusion_temp_and_pressure_correction: float = None
    breakthrough_pressure: float = None
    saturation_flow_resistance: float = None
    # Two-phase transport state (set by water saturation model)
    non_wetting_flux: float = None
    downstream_saturation: float = None
    upstream_saturation: float = None
    downstream_capillary_pressure: float = None
    electrolyte_saturation: float = None
    gas_transport_resistance: dict = field(default_factory=dict)

@dataclass
class CatalystLayerState(LayerState):
    """State of a catalyst layer, in addition to the generic layer state."""

    ionomer_water_content: float = None
    overpotential: float = None
    water_film_thickness: float = None
    proton_resistance: float = None
    theta_catalyst: float = None
    local_o2_resistance: float = None
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

    # Transport properties (set by MembraneWaterBalanceModel.calculate_membrane_transport_properties)
    eod_speed: float = None
    absorption_coefficient: float = None
    water_diffusivity: float = None
    water_diffusion_resistance: float = None
    vapor_equilibrium_saturation_water_content: float = None

    # Non-dimensional profile quantities (set by update_non_dimensional_parameters)
    peclet_number: float = None
    ePe: object = None
    ePexi: object = None
    xi: object = None

    # Profile (set by update_water_profile / _initialize_interface_water_contents)
    water_content_profile: object = None
    water_content_derivative_profile: object = None

    # Internal flux diagnostics (set by update_internal_water_fluxes)
    diffusion_flux: object = None
    eod_flux: object = None
    water_net_flux: object = None

@dataclass
class GasFlowState:
    """Gas + liquid flow state at one point (inlet or outlet) of a cell side.

    Bundles per-species molar flow rates, temperature and pressure — the
    natural state variables for a system-level mass balance around a fuel
    cell side — and converts to :class:`~marapendi.simulation.conditions.SideConditions`
    (dry composition + relative humidity, with ``stoichiometry=0`` since the
    flow is already fully specified) so it can drive
    :class:`~marapendi.models.base.explicit_steady_state.ExplicitSteadyStateModel`
    directly. Exposing ``gas``/``temperature``/``pressure``/``RT`` lets a
    :class:`GasFlowState` be passed directly to :class:`~marapendi.models.thermo.gas.GasModel`
    methods, the same way a :class:`LayerState` or :class:`FlowChannelState` is.

    A system model can specify only inlet ``GasFlowState`` objects, solve the
    cell, and derive the outlet ``GasFlowState`` from a mass balance
    (inlet flow − reactant consumption + water production) without ever
    building a :class:`~marapendi.simulation.conditions.SideConditions` by hand.

    Attributes
    ----------
    temperature : float
        Gas temperature (K).
    pressure : float
        Gas pressure (Pa).
    gas_species_molar_flow_rates : np.ndarray, shape (4,)
        Molar flow rate of each species (O2, N2, H2, H2O), in the order
        given by :data:`~marapendi.models.thermo.gas.species_indexes` (kmol/s).
    liquid_molar_flow_rate : float
        Molar flow rate of liquid water (kmol/s).
    """

    temperature: float
    pressure: float
    gas_species_molar_flow_rates: np.ndarray = field(default_factory=lambda: np.array([1., 0., 0., 0.]))
    liquid_molar_flow_rate: float = 0.
    saturation_pressure: float = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.RT = GAS_CONSTANT * self.temperature

    @property
    def gas_molar_flow_rate(self) -> float:
        """Total gas-phase molar flow rate (kmol/s)."""
        return float(np.sum(self.gas_species_molar_flow_rates))

    @property
    def gas(self) -> GasState:
        """Gas composition as mole fractions."""
        return GasState(X=self.gas_species_molar_flow_rates / self.gas_molar_flow_rate)

    def to_side_conditions(self) -> SideConditions:
        """Equivalent :class:`~marapendi.simulation.conditions.SideConditions`.

        ``stoichiometry`` is set to 0 and the full flow carried on
        ``inlet_gas_flow_rate`` (a volumetric flow, m^3/s), so the solver
        does not also add a stoichiometric term on top of it.
        """
        X = self.gas.X
        dry_fraction = 1 - X[index_h2ov]
        concentration = GasModel.concentration(self)
        return SideConditions(
            inlet_temperature=self.temperature,
            inlet_pressure=self.pressure,
            outlet_pressure=self.pressure,
            dry_o2_mole_fraction=X[index_o2] / dry_fraction,
            dry_h2_mole_fraction=X[index_h2] / dry_fraction,
            inlet_relative_humidity=GasModel.relative_humidity(self),
            stoichiometry=0.,
            inlet_gas_flow_rate=self.gas_molar_flow_rate / concentration,
            inlet_liquid_flow_rate=self.liquid_molar_flow_rate * water_molar_volume(self.temperature),
        )

    @classmethod
    def from_side_conditions(cls, side_conditions: SideConditions, stack_temperature: float,
                              reactant: str, reactant_consumption: float,
                              minimal_reactant_consumption: float, area: float) -> 'GasFlowState':
        """Inlet :class:`GasFlowState` implied by *side_conditions* — the
        inverse of :meth:`to_side_conditions`, reproducing the same
        stoichiometry formula
        :meth:`~marapendi.models.base.explicit_steady_state.ExplicitSteadyStateModel._set_flow_rates`
        applies internally, so a solve driven by *side_conditions* and one
        driven by ``from_side_conditions(...).to_side_conditions()`` agree.

        Composition (dry mole fractions + relative humidity) is evaluated at
        ``side_conditions.inlet_temperature``/``inlet_pressure`` as usual, but
        — matching the solver's own convention — the volumetric-to-molar
        flow conversion uses the *average* pressure
        (``side_conditions.average_pressure``) and *stack* temperature, not
        the literal inlet values, since that is the concentration basis
        ``inlet_gas_flow_rate`` is defined against internally. The returned
        object's ``pressure``/``temperature`` follow that same average/stack
        convention, so a round trip through :meth:`to_side_conditions` is
        exact when ``inlet_temperature == stack_temperature`` (true in most
        operating points, where inlet and outlet gas are close to the
        stack temperature).

        Parameters
        ----------
        side_conditions : SideConditions
        stack_temperature : float
            Cell stack temperature (K), i.e. ``cell_conditions.cell_temperature``.
        reactant : str
            Which species stoichiometry is computed against ('o2' or 'h2'),
            i.e. ``cell_side.ch.reactant``.
        reactant_consumption : float
            Reactant consumption rate for this side (kmol/(m^2 s)), i.e.
            ``side_state.reactant_consumption``.
        minimal_reactant_consumption : float
            Floor on *reactant_consumption* used for the stoichiometric flow
            (``side_conditions.minimum_current_density_for_stoich`` converted
            to a consumption rate).
        area : float
            Active cell area (m^2), i.e. ``cell.area``.
        """
        composition_state = _SimpleNamespace(gas=GasState())
        GasModel.set_composition(
            composition_state,
            side_conditions.dry_o2_mole_fraction, side_conditions.dry_h2_mole_fraction,
            side_conditions.inlet_relative_humidity,
            side_conditions.inlet_pressure, side_conditions.inlet_temperature,
        )
        X = composition_state.gas.X

        concentration = side_conditions.average_pressure / (GAS_CONSTANT * stack_temperature)
        total_volumetric_flow = (
            side_conditions.stoichiometry
            * max(reactant_consumption, minimal_reactant_consumption)
            * area / (X[species_indexes[reactant]] * concentration)
            + side_conditions.inlet_gas_flow_rate
        )
        total_molar_flow = total_volumetric_flow * concentration

        return cls(
            temperature=stack_temperature,
            pressure=side_conditions.average_pressure,
            gas_species_molar_flow_rates=X * total_molar_flow,
            liquid_molar_flow_rate=side_conditions.inlet_liquid_flow_rate / water_molar_volume(stack_temperature),
        )

    def consume(self, reactant: str, reactant_consumption: float,
                vapor_flux: float, liquid_flux: float, area: float) -> 'GasFlowState':
        """Outlet :class:`GasFlowState` from a mass balance across a cell side.

        Subtracts *reactant_consumption* (kmol/(m^2 s)) of *reactant* from
        the gas stream and adds product water — *vapor_flux* to the gas
        phase, *liquid_flux* to the liquid phase (both kmol/(m^2 s), as set
        by :meth:`~marapendi.models.water_balance.water_balance.WaterBalanceModel.update_cell_side_water_fluxes`)
        — over *area* (m^2), i.e. ``cell.area``. Reactant crossover through
        the membrane is not accounted for, matching the same simplification
        :meth:`~marapendi.models.base.explicit_steady_state.ExplicitSteadyStateModel._set_consumption_production`
        already makes.

        Temperature and pressure are carried over unchanged from the inlet —
        this model does not track a separate gas outlet temperature/pressure.
        """
        species = self.gas_species_molar_flow_rates.copy()
        species[species_indexes[reactant]] = species[species_indexes[reactant]] - reactant_consumption * area
        species[index_h2ov] = species[index_h2ov] + vapor_flux * area
        return GasFlowState(
            temperature=self.temperature,
            pressure=self.pressure,
            gas_species_molar_flow_rates=species,
            liquid_molar_flow_rate=self.liquid_molar_flow_rate + liquid_flux * area,
        )

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
    gas_transport_resistance: dict = field(default_factory=dict)

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
    gas_transport_resistance: dict = field(default_factory=dict)

    h2o_production: float = 0.
    s_relax: float | None = None
    membrane_water_flux: float = None
    water_flux: float = None
    liquid_flux: float = None
    vapor_flux: float = None
    gas_flux: float = 0.

    # Inlet/outlet flow bookkeeping (set by set_gas_flow_states, once
    # reactant_consumption/h2o_production and vapor_flux/liquid_flux are
    # available — see ExplicitSteadyStateModel.solve and TransientModel)
    inlet_gas_flow_state: GasFlowState | None = None
    outlet_gas_flow_state: GasFlowState | None = None

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
    heat_release: float = None

    # Crossover
    crossover_current: float = None

    # Derived diagnostics (set by evaluate)
    hfr: float = None

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
