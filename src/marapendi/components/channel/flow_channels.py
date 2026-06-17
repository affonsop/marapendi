"""
Flow channel components: static geometry and runtime state.

:class:`FlowChannel` holds the static geometry of a fuel-cell flow channel
(dimensions, number of parallel channels) together with the runtime state
that must live on the component for the current explicit-state architecture
(inlet flow rates, stoichiometry, gas composition).  Pure computations that
derive quantities from that state are collected in :class:`FlowChannelModel`
so they can be reused independently of the component.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from ...models.constants import GAS_CONSTANT

from ...models.gas import GasModel, species_indexes
from ..porous.porous_layers import PorousLayer
from ...models.channel.channel import ChannelGasResistanceModel
from ...models.water import water_kinematic_viscosity, water_molar_volume


@dataclass
class ChannelConditions:
    """Operating conditions for a flow channel.

    Attributes
    ----------
    temperature : float
        Temperature (K).
    rh : float
        Relative humidity (0–1).
    pressure : float
        Total pressure (Pa).
    dry_o2_mole_fraction : float
        O2 mole fraction in the dry gas.
    dry_h2_mole_fraction : float
        H2 mole fraction in the dry gas.
    stoichiometry : float
        Stoichiometric ratio (actual flow / required flow).
    """

    temperature: float
    rh: float
    pressure: float
    dry_o2_mole_fraction: float
    dry_h2_mole_fraction: float
    stoichiometry: float

    def __post_init__(self):
        pass

    def set_conditions(self, temperature=None, rh=None, pressure=None,
                       dry_o2_mole_fraction=None, dry_h2_mole_fraction=None,
                       stoichiometry=None):
        """Update one or more channel operating conditions."""
        if temperature is not None:
            self.temperature = temperature
        if rh is not None:
            self.rh = rh
        if pressure is not None:
            self.pressure = pressure
        if dry_o2_mole_fraction is not None:
            self.dry_o2_mole_fraction = dry_o2_mole_fraction
        if dry_h2_mole_fraction is not None:
            self.dry_h2_mole_fraction = dry_h2_mole_fraction
        if stoichiometry is not None:
            self.stoichiometry = stoichiometry
        self.__post_init__()


@dataclass
class FlowChannel(PorousLayer):
    """Flow channel geometry and state of a single fuel-cell side.

    Inherits porous-layer geometry from :class:`~marapendi.porous_layers.PorousLayer`
    so that it can be used in the same gas-transport pipeline.

    Attributes
    ----------
    reactant : str
        Primary reactant species ('o2' or 'h2').
    inlet_stoichiometry : float
        Inlet stoichiometric ratio (actual / required flow).
    inlet_gas_flow_rate : float
        Volumetric gas flow rate at the channel inlet (m³/s).
    inlet_liquid_flow_rate : float
        Liquid flow rate at the channel inlet (m³/s).
    inlet_liquid_saturation : float
        Liquid saturation fraction at the inlet.
    width, height, length : float
        Channel cross-section width, height, and length (m).
    channel_land_ratio : float
        Ratio of channel width to land width.
    n_parallel : int
        Number of parallel channels.
    transport_resistance_model : ChannelGasResistanceModel
        Correlation for the channel gas transport resistance.
    """

    reactant: str = 'o2'
    inlet_stoichiometry: float = 0
    inlet_gas_flow_rate: float = 1e-12
    inlet_liquid_flow_rate: float = 0
    inlet_liquid_saturation: float = 0
    width: float = 1e-3
    height: float = 1e-3
    length: float = 100e-3
    channel_land_ratio: float = 1.
    n_parallel: int = 14
    transport_resistance_model: ChannelGasResistanceModel = field(default_factory=ChannelGasResistanceModel)

    def __post_init__(self):
        self.RT = GAS_CONSTANT * self.temperature
        self.hydraulic_diameter = 2 * self.width * self.height / (self.width + self.height)
        self.channel_flow_section = self.width * self.height
        self.half_width = 0.5 * self.width
        self.total_flow_section = self.n_parallel * self.channel_flow_section
        PorousLayer.__post_init__(self)

    def gas_transport_resistance(self, state, species=None, volume_flow_rate=None):
        """Gas transport resistance for ``species`` in the channel (s/m).

        Parameters
        ----------
        state : object
            State object with ``temperature``, ``pressure``, and ``gas.X``.
            Pass the channel itself when no separate state is available.
        species : str, optional
            Species identifier ('o2', 'h2', 'h2o').
        volume_flow_rate : float, optional
            Volumetric flow rate (m³/s); overrides the channel's inlet flow rate.
        """
        diffusion_coeff = GasModel.species_diffusion_coefficient(state, species)
        return self.transport_resistance_model.total_resistance(
            self, diffusion_coeff, volume_flow_rate)


@dataclass
class FlowChannelModel:
    """Pure computations on a :class:`FlowChannel` and its state.

    All methods take the channel (or a compatible state object) as their first
    argument so they can be called independently of the component class.
    """

    def reactant_mole_fraction(self, ch) -> float:
        """Mole fraction of the primary reactant in ``ch``."""
        return ch.gas.X[..., species_indexes[ch.reactant]]

    def calculate_inlet_gas_flow_rate(self, ch, reactant_consumption: float) -> float:
        """Volumetric inlet gas flow rate required to meet the stoichiometry (m³/s).

        Parameters
        ----------
        ch : FlowChannel
            Channel whose ``inlet_stoichiometry`` and gas state are used.
        reactant_consumption : float
            Reactant consumption rate (kmol/s).
        """
        return (
            ch.inlet_stoichiometry * reactant_consumption
            / (self.reactant_mole_fraction(ch) * ch.gas.concentration())
        )

    def calculate_inlet_stoichiometry(self, ch, reactant_consumption: float) -> float:
        """Effective stoichiometry computed from the current inlet flow rate.

        Parameters
        ----------
        ch : FlowChannel
            Channel whose ``inlet_gas_flow_rate`` and gas state are used.
        reactant_consumption : float
            Reactant consumption rate (kmol/s).
        """
        return (
            ch.inlet_gas_flow_rate * self.reactant_mole_fraction(ch)
            * ch.gas.concentration() / reactant_consumption
        )

    def inlet_molar_flow_rates(self, ch) -> np.ndarray:
        """Molar flow rates for all species at the channel inlet (kmol/s).

        Returns
        -------
        np.ndarray
            One value per species (O₂, H₂, H₂O, N₂).
        """
        total_inlet_molar_flow = ch.inlet_gas_flow_rate * ch.gas.concentration()
        return total_inlet_molar_flow * ch.gas.X

    def gas_superficial_speed(self, ch, volumetric_flow_rate: float = None) -> float:
        """Gas superficial speed in the channel (m/s).

        Parameters
        ----------
        ch : FlowChannel
        volumetric_flow_rate : float, optional
            If given, overrides ``ch.inlet_gas_flow_rate``.
        """
        vol_flow = volumetric_flow_rate if volumetric_flow_rate is not None else ch.inlet_gas_flow_rate
        return vol_flow / ch.total_flow_section

    def liquid_to_gas_velocity_ratio(self, ch) -> float:
        """Liquid-to-gas velocity ratio in the channel (dimensionless).

        References
        ----------
        Zhang et al. Energy Environ. Sci. 2026.
        https://doi.org/10.1039/d5ee04599a
        """
        liquid_saturation = ch.non_wetting_saturation if ch.wetting_phase == 'gas' else (1 - ch.non_wetting_saturation)
        return (
            (liquid_saturation / (1 - liquid_saturation)) ** 3
            * (ch.gas.mixture_kinematic_viscosity / water_kinematic_viscosity(ch.gas.temperature))
        )

    def outlet_liquid_speed(self, ch) -> float:
        """Liquid superficial speed at the channel outlet (m/s)."""
        return self.liquid_to_gas_velocity_ratio(ch) * self.gas_superficial_speed(ch)

    def outlet_liquid_flow_rate(self, ch) -> float:
        """Liquid volumetric flow rate at the channel outlet (m³/s)."""
        return self.outlet_liquid_speed(ch) * ch.total_flow_section

    def outlet_liquid_molar_flow_rate(self, ch) -> float:
        """Liquid molar flow rate at the channel outlet (kmol/s)."""
        return self.outlet_liquid_flow_rate(ch) / water_molar_volume(ch.gas.temperature)
