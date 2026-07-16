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

from dataclasses import dataclass, field, InitVar

import numpy as np
from .conditions import SideConditions
from ..models.thermo.constants import GAS_CONSTANT
from ..models.thermo.gas import *
from ..models.thermo.gas import _horner5, _viscosity_polynomials, _FICK_PT_REFERENCE_FACTOR, species_list
from ..models.thermo.water import water_molar_volume, water_saturation_pressure

@dataclass
class GasState:
    """Composition and thermodynamic state of a gas mixture, with the
    correlations that depend on them (relative humidity, vapor pressure,
    diffusion coefficients, kinematic viscosity, ...) attached as methods.

    ``temperature``/``pressure`` are the gas's own — in principle distinct
    from (though today always equal to) the temperature/pressure of the
    surrounding :class:`LayerState`/:class:`FlowChannelState`, which expose
    them as pass-through ``@property`` accessors onto ``self.gas``.

    Attributes
    ----------
    X : np.ndarray
        Mole fractions of (O2, N2, H2, H2O), in the order given by
        :data:`~marapendi.models.thermo.gas.species_indexes`.
    temperature : float
        Gas temperature (K).
    pressure : float
        Gas pressure (Pa).
    """

    X: np.ndarray = field(default_factory=lambda: np.array([1., 0., 0., 0.]))
    temperature: float = None
    pressure: float = None
    _saturation_pressure: float = field(default=None, init=False, repr=False)
    _diffusion_temp_and_pressure_correction: float = field(default=None, init=False, repr=False)

    @property
    def RT(self) -> float:
        """``GAS_CONSTANT * temperature``."""
        return GAS_CONSTANT * self.temperature

    @property
    def saturation_pressure(self) -> float:
        """Saturation pressure of water at ``temperature`` (Pa), cached on first access."""
        if self._saturation_pressure is None:
            self._saturation_pressure = water_saturation_pressure(self.temperature)
        return self._saturation_pressure

    @saturation_pressure.setter
    def saturation_pressure(self, value) -> None:
        self._saturation_pressure = value

    @property
    def diffusion_temp_and_pressure_correction(self) -> float:
        """Species-independent Fick's law adjustment ``T^1.5 / P`` for
        :meth:`species_diffusion_coefficient`, cached on first access since
        up to 3 species share the same ``temperature``/``pressure``.
        """
        if self._diffusion_temp_and_pressure_correction is None:
            # T**1.5 written as T * sqrt(T): np.sqrt dispatches faster than np.power
            # with a non-integer exponent, and this is on the transient ODE hot path.
            self._diffusion_temp_and_pressure_correction = (
                _FICK_PT_REFERENCE_FACTOR * self.temperature * np.sqrt(self.temperature) / self.pressure
            )
        return self._diffusion_temp_and_pressure_correction

    @diffusion_temp_and_pressure_correction.setter
    def diffusion_temp_and_pressure_correction(self, value) -> None:
        self._diffusion_temp_and_pressure_correction = value

    def set_composition(self, dry_o2_mole_fraction: float, dry_h2_mole_fraction: float,
                         relative_humidity: float, inlet_pressure: float, inlet_temperature: float) -> None:
        """Set ``self.X`` from a dry composition and relative humidity.

        The water vapor mole fraction is fixed by the relative humidity at
        the inlet conditions (``inlet_pressure``, ``inlet_temperature``),
        which may differ from ``self.pressure``/``self.temperature``
        (typically the average of the inlet and outlet conditions).
        """
        dry_mole_fractions = np.zeros_like(self.X)
        dry_mole_fractions[..., index_o2] = dry_o2_mole_fraction
        dry_mole_fractions[..., index_h2] = dry_h2_mole_fraction
        dry_mole_fractions[..., index_n2] = 1 - dry_o2_mole_fraction - dry_h2_mole_fraction

        inlet_saturation_pressure = water_saturation_pressure(inlet_temperature)
        h2o_mole_fraction = relative_humidity * inlet_saturation_pressure / inlet_pressure
        vapor_mole_fractions = np.zeros_like(self.X)
        vapor_mole_fractions[..., index_h2ov] = h2o_mole_fraction

        self.X = (
            dry_mole_fractions * (1 - vapor_mole_fractions[..., index_h2ov, np.newaxis])
            + vapor_mole_fractions
        )

    def species_mole_fraction(self, species: str) -> float:
        """Mole fraction of ``species`` in the gas mixture."""
        return self.X[..., species_indexes[species]]

    def species_partial_pressure(self, species: str) -> float:
        """Partial pressure of ``species`` (Pa)."""
        return self.species_mole_fraction(species) * self.pressure

    def species_concentration(self, species: str) -> float:
        """Concentration of ``species`` (kmol/m^3)."""
        return self.species_partial_pressure(species) / self.RT

    @property
    def vapor_pressure(self) -> float:
        """Partial pressure of water vapor (Pa)."""
        return self.species_partial_pressure('h2o')

    @property
    def vapor_concentration(self) -> float:
        """Concentration of water vapor (kmol/m^3)."""
        return self.species_concentration('h2o')

    @property
    def saturation_concentration(self) -> float:
        """Saturation concentration of water vapor (kmol/m^3)."""
        return self.saturation_pressure / self.RT

    @property
    def relative_humidity(self) -> float:
        """Relative humidity (0 to 1)."""
        return self.vapor_pressure / self.saturation_pressure

    @property
    def mixture_molecular_weight(self) -> float:
        """Mean molecular weight of the gas mixture (kg/kmol)."""
        return np.sum(molecular_weights * self.X, axis=-1)

    @property
    def concentration(self) -> float:
        """Total molar concentration of the gas mixture (kmol/m^3)."""
        return self.pressure / self.RT

    @property
    def density(self) -> float:
        """Mass density of the gas mixture (kg/m^3)."""
        return self.concentration * self.mixture_molecular_weight

    def species_kinematic_viscosity(self, species: str) -> float:
        """Kinematic viscosity of pure ``species`` at ``self.temperature`` (m^2/s)."""
        log_temperature = np.log(self.temperature)
        v = _horner5(_viscosity_polynomials[species], log_temperature)
        return v * v * np.sqrt(self.temperature)

    @property
    def mixture_kinematic_viscosity(self) -> float:
        """Kinematic viscosity of the gas mixture (m^2/s), as a mole-weighted average."""
        species_kinematic_viscosities = np.array(
            [self.species_kinematic_viscosity(species) for species in species_list],
        ).transpose()
        return (
            np.sum(self.X * species_kinematic_viscosities * molecular_weights, axis=-1)
            / self.mixture_molecular_weight
        )

    def species_diffusion_coefficient(self, species: str) -> float:
        """Binary diffusion coefficient of ``species`` in the gas mixture (m^2/s).

        Uses empirical correlations based on reference values adjusted for
        temperature and pressure. Data from Vetter and Schumacher (2019).

        References
        ----------
        Vetter, R. & Schumacher, J. O. Comput. Phys. Commun. 234, 223-234 (2019).
        """
        if species == 'o2':
            reference_diffusion_coefficient = 0.28e-4
        elif species == 'h2':
            reference_diffusion_coefficient = 1.24e-4
        elif species == 'h2o':
            # If H2 is present, assume H2-H2O; else O2-H2O
            if np.max(self.species_mole_fraction('h2')) > 0:
                reference_diffusion_coefficient = 1.24e-4
            else:
                reference_diffusion_coefficient = 0.36e-4

        return reference_diffusion_coefficient * self.diffusion_temp_and_pressure_correction


@dataclass
class LayerState:
    """State of a single porous layer (GDL, MPL, catalyst layer, ...).

    ``temperature``/``pressure``/``RT``/``saturation_pressure``/
    ``diffusion_temp_and_pressure_correction`` are pass-through properties
    onto :attr:`gas` (see :class:`GasState`) rather than independent fields,
    so the gas state stays the single source of truth. ``temperature``/
    ``pressure`` remain accepted as constructor keywords for convenience
    (``LayerState(temperature=..., pressure=...)``).
    """

    gas: GasState = field(default_factory=GasState)
    temperature: InitVar[float] = None
    pressure: InitVar[float] = None
    liquid_saturation: float = 0.
    non_wetting_saturation: float = 0.
    capillary_pressure: float = None
    # Temperature-derived capillary quantities (set by PorousLayer.update_state_at_temperature)
    breakthrough_pressure: float = None
    saturation_flow_resistance: float = None
    # Two-phase transport state (set by water saturation model)
    non_wetting_flux: float = None
    downstream_saturation: float = None
    upstream_saturation: float = None
    downstream_capillary_pressure: float = None
    electrolyte_saturation: float = None
    gas_transport_resistance: dict = field(default_factory=dict)

    def __post_init__(self, temperature, pressure):
        if temperature is not None:
            self.gas.temperature = temperature
        if pressure is not None:
            self.gas.pressure = pressure

    @property
    def RT(self) -> float:
        return self.gas.RT

    @property
    def saturation_pressure(self) -> float:
        return self.gas.saturation_pressure

    @saturation_pressure.setter
    def saturation_pressure(self, value) -> None:
        self.gas.saturation_pressure = value

    @property
    def diffusion_temp_and_pressure_correction(self) -> float:
        return self.gas.diffusion_temp_and_pressure_correction

    @diffusion_temp_and_pressure_correction.setter
    def diffusion_temp_and_pressure_correction(self, value) -> None:
        self.gas.diffusion_temp_and_pressure_correction = value


# `temperature`/`pressure` are declared as InitVar above (not real dataclass
# fields) purely so LayerState(temperature=..., pressure=...) keeps working;
# the actual get/set always goes through self.gas. Defined as properties here,
# after the class body, because a same-named @property *inside* the class body
# would shadow the InitVar default value that __init__ relies on.
LayerState.temperature = property(
    lambda self: self.gas.temperature,
    lambda self, value: setattr(self.gas, 'temperature', value),
)
LayerState.pressure = property(
    lambda self: self.gas.pressure,
    lambda self, value: setattr(self.gas, 'pressure', value),
)

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
    directly. :attr:`gas` synthesizes a :class:`GasState` (composition, carrying
    this object's own ``temperature``/``pressure``) on every access, so
    correlations are available as ``flow_state.gas.some_method(...)``, the
    same way as for a :class:`LayerState` or :class:`FlowChannelState`.

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

    @property
    def RT(self) -> float:
        return GAS_CONSTANT * self.temperature

    @property
    def gas_molar_flow_rate(self) -> float:
        """Total gas-phase molar flow rate (kmol/s)."""
        return float(np.sum(self.gas_species_molar_flow_rates))

    @property
    def gas(self) -> GasState:
        """Gas composition, carrying this object's own temperature/pressure."""
        return GasState(
            X=self.gas_species_molar_flow_rates / self.gas_molar_flow_rate,
            temperature=self.temperature, pressure=self.pressure,
        )

    def to_side_conditions(self) -> SideConditions:
        """Equivalent :class:`~marapendi.simulation.conditions.SideConditions`.

        ``stoichiometry`` is set to 0 and the full flow carried on
        ``inlet_gas_flow_rate`` (a volumetric flow, m^3/s), so the solver
        does not also add a stoichiometric term on top of it.
        """
        gas = self.gas
        X = gas.X
        dry_fraction = 1 - X[index_h2ov]
        concentration = gas.concentration
        return SideConditions(
            inlet_temperature=self.temperature,
            inlet_pressure=self.pressure,
            outlet_pressure=self.pressure,
            dry_o2_mole_fraction=X[index_o2] / dry_fraction,
            dry_h2_mole_fraction=X[index_h2] / dry_fraction,
            inlet_relative_humidity=gas.relative_humidity,
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
        gas = GasState()
        gas.set_composition(
            side_conditions.dry_o2_mole_fraction, side_conditions.dry_h2_mole_fraction,
            side_conditions.inlet_relative_humidity,
            side_conditions.inlet_pressure, side_conditions.inlet_temperature,
        )
        X = gas.X

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
    """State of a flow channel (anode or cathode).

    See :class:`LayerState` for why ``temperature``/``pressure``/``RT``/
    ``saturation_pressure``/``diffusion_temp_and_pressure_correction`` are
    pass-through properties onto :attr:`gas` rather than independent fields.
    """

    gas: GasState = field(default_factory=GasState)
    temperature: InitVar[float] = None
    pressure: InitVar[float] = None
    inlet_gas_flow_rate: float = 1e-12
    inlet_liquid_flow_rate: float = None
    inlet_liquid_saturation: float = None
    inlet_stoichiometry: float = None
    gas_transport_resistance: dict = field(default_factory=dict)

    def __post_init__(self, temperature, pressure):
        if temperature is not None:
            self.gas.temperature = temperature
        if pressure is not None:
            self.gas.pressure = pressure

    @property
    def RT(self) -> float:
        return self.gas.RT

    @property
    def saturation_pressure(self) -> float:
        return self.gas.saturation_pressure

    @saturation_pressure.setter
    def saturation_pressure(self, value) -> None:
        self.gas.saturation_pressure = value

    @property
    def diffusion_temp_and_pressure_correction(self) -> float:
        return self.gas.diffusion_temp_and_pressure_correction

    @diffusion_temp_and_pressure_correction.setter
    def diffusion_temp_and_pressure_correction(self, value) -> None:
        self.gas.diffusion_temp_and_pressure_correction = value


FlowChannelState.temperature = property(
    lambda self: self.gas.temperature,
    lambda self, value: setattr(self.gas, 'temperature', value),
)
FlowChannelState.pressure = property(
    lambda self: self.gas.pressure,
    lambda self, value: setattr(self.gas, 'pressure', value),
)

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

    # Raw scipy.integrate.OdeResult from TransientModel.solve() (None for
    # steady-state solves, or when compute_diagnostics=False).
    ode_solution: object = None

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
