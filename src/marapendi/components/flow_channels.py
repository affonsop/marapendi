"""
Module providing classes to model flow channels and their operating conditions 
in a PEM fuel cell. This includes representations for channel dimensions, 
flow rates, stoichiometries, and methods to compute gas transport resistances.

Classes
-------
ChannelConditions
    Encapsulates the conditions (temperature, RH, pressure, gas composition) 
    for a flow channel.

FlowChannel
    Represents a flow channel geometry and state in a PEM fuel cell, 
    providing methods to calculate flow rates and gas transport resistances.
"""

from dataclasses import dataclass, field
import numpy as np 
import cantera as ct

from .porous_layers import PorousLayer
from marapendi.models.transport import ChannelGasResistanceModel
from ..models.water import water_kinematic_viscosity, water_molar_volume

@dataclass(eq=False) 
class ChannelConditions:  
    """
    Class to encapsulate the operating conditions of the gas in a flow channels.

    Attributes
    ----------
    temperature : float
        Temperature in Kelvin.
    rh : float
        Relative humidity (0-1).
    pressure : float
        Total pressure in Pa.
    dry_o2_mole_fraction : float
        Mole fraction of O2 in the dry gas mixture.
    dry_h2_mole_fraction : float
        Mole fraction of H2 in the dry gas mixture.
    stoichiometry : float
        Stoichiometric ratio (actual flow / required flow).

    Methods
    -------
    set_conditions(...)
        Update one or more of the channel operating conditions.
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
        """
        Update the channel conditions.

        Parameters
        ----------
        temperature : float, optional
            New temperature in K.
        rh : float, optional
            New relative humidity (0-1).
        pressure : float, optional
            New total pressure in Pa.
        dry_o2_mole_fraction : float, optional
            New dry O2 mole fraction.
        dry_h2_mole_fraction : float, optional
            New dry H2 mole fraction.
        stoichiometry : float, optional
            New stoichiometry value.

        Notes
        -----
        Only the provided arguments will update; others remain unchanged.
        """
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


@dataclass(eq=False)
class FlowChannel(PorousLayer):
    """
    Class to represent a fuel cell flow channels, inheriting from PorousLayer 
    to allow similar interface for transport calculations. Corresponds to channels of
    a single cell. 

    Attributes
    ----------
    reactant : str
        The primary reactant species ('o2' or 'h2').
    inlet_stoichiometry : float
        Inlet stoichiometric ratio.
    inlet_gas_flow_rate : float
        Gas flow rate entering the channel (m3/s).
    inlet_liquid_flow_rate : float
        Liquid flow rate entering the channel (m3/s).
    inlet_liquid_saturation : float
        Saturation fraction of the inlet stream.
    width : float
        Channel width (m).
    height : float
        Channel height (m).
    length : float
        Channel length (m).
    channel_land_ratio : float
        Ratio of channel width to land width.
    n_parallel : int
        Number of parallel channels.
    transport_resistance_model : ChannelGasResistanceModel
        Model to calculate total gas transport resistance.

    Derived Attributes
    ------------------
    hydraulic_diameter : float
        Hydraulic diameter of the channel (m).
    channel_flow_section : float
        Cross-sectional area of a single channel (m2).
    total_flow_section : float
        Total flow area accounting for all channels (m2).
    RT : float
        Product of universal gas constant and temperature (J/kmolK).

    Methods
    -------
    set_inlet_stoichiometry(stoichiometry)
        Set the inlet stoichiometry.
    reactant_mole_fraction()
        Return the mole fraction of the primary reactant.
    set_fixed_inlet_gas_flow_rate(inlet_gas_flow_rate)
        Fix the inlet gas flow rate and update liquid saturation.
    set_fixed_inlet_liquid_flow_rate(inlet_liquid_flow_rate)
        Fix the inlet liquid flow rate and update liquid saturation.
    set_inlet_gas_flow_rate_from_stoichiometry(reactant_consumption, stoichiometry, fixed_inlet_gas_flow_rate)
        Compute inlet gas flow from stoichiometry and consumption.
    calculate_inlet_gas_flow_rate(reactant_consumption)
        Calculate gas flow rate based on stoichiometry and reactant consumption.
    calculate_inlet_stochiometry(reactant_consumption)
        Compute effective stoichiometry from flow rate and consumption.
    gas_transport_resistance(species, volume_flow_rate=None)
        Compute gas transport resistance in the channel.
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
    eps_p: float = 1
    tort: float = 1.
    sherwood: float = 4.
    transport_resistance_model: ChannelGasResistanceModel = field(default_factory=ChannelGasResistanceModel)

    def __post_init__(self):
        """
        Initialize geometric and thermodynamic properties after instantiation.
        """
        self.hydraulic_diameter = 2 * self.width * self.height / (self.width + self.height)
        self.channel_flow_section = self.width * self.height
        self.half_width = 0.5 * self.width
        self.aspect_ratio = min(self.width/self.height, self.height/self.width)
        self.total_flow_section = self.n_parallel * self.channel_flow_section
        self.total_volume = self.total_flow_section * self.length
        self.thickness = self.height 
        self.fRe = 24 * np.polyval([-.2537, 0.9564, -1.7012, 1.9467, - 1.3553, 1], self.aspect_ratio)


        PorousLayer.__post_init__(self)

    def superficial_velocity(self, volumetric_flow_rate):
        """Return gas superficial speed [m/s] for a given volumetric flow rate [m³/s]."""
        return volumetric_flow_rate / self.total_flow_section

    def set_inlet_stoichiometry(self, stoichiometry):
        """
        Set the inlet stoichiometric ratio.

        Parameters
        ----------
        stoichiometry : float
            New stoichiometric value.
        """
        self.inlet_stoichiometry = stoichiometry

    
    def set_fixed_inlet_gas_flow_rate(self, inlet_gas_flow_rate): 
        """
        Fix the inlet gas flow rate and update liquid saturation.

        Parameters
        ----------
        inlet_gas_flow_rate : float
            New gas flow rate (mol/s).
        """
        self.inlet_gas_flow_rate = inlet_gas_flow_rate 
        self.inlet_liquid_saturation = self.inlet_liquid_flow_rate / np.maximum(
            self.inlet_liquid_flow_rate + self.inlet_gas_flow_rate, 1e-12)
        
    def set_fixed_inlet_liquid_flow_rate(self, inlet_liquid_flow_rate): 
        """
        Fix the inlet liquid flow rate and update liquid saturation.

        Parameters
        ----------
        inlet_liquid_flow_rate : float
            New liquid flow rate (mol/s).
        """
        self.inlet_liquid_flow_rate = inlet_liquid_flow_rate
        self.inlet_liquid_saturation = self.inlet_liquid_flow_rate / np.maximum(
            self.inlet_liquid_flow_rate + self.inlet_gas_flow_rate, 1e-12)

    def set_inlet_gas_flow_rate_from_stoichiometry(self, reactant_consumption, stoichiometry=0, fixed_inlet_gas_flow_rate=0):
        """
        Set the inlet gas flow rate from stoichiometry and reactant consumption.

        Parameters
        ----------
        reactant_consumption : float
            Consumption rate of the reactant (kmol/s).
        stoichiometry : float, optional
            Stoichiometric ratio to override (default 0, uses existing).
        fixed_inlet_gas_flow_rate : float, optional
            Additional fixed flow rate to add (default 0).
        """
        try: 
            if stoichiometry > 0: 
                self.inlet_stoichiometry = stoichiometry 
        except ValueError: 
            self.inlet_stoichiometry = stoichiometry 
        inlet_gas_flow_rate = self.calculate_inlet_gas_flow_rate(reactant_consumption) + fixed_inlet_gas_flow_rate
        self.set_fixed_inlet_gas_flow_rate(inlet_gas_flow_rate)

    def inlet_gas_flow_rate(self, reactant_consumption, reactant_mole_fraction, gas_concentration): 
        """
        Calculate gas flow rate required by stoichiometry.

        Parameters
        ----------
        reactant_consumption : float
            Reactant consumption rate (kmol/s).

        Returns
        -------
        float
            Required inlet gas flow rate (m3/s).
        """
        return self.inlet_stoichiometry * reactant_consumption / \
            reactant_mole_fraction / gas_concentration
    
    def calculate_inlet_stochiometry(self, reactant_consumption, reactant_mole_fraction, gas_concentration): 
        """
        Compute stoichiometry from known flow rate and consumption.

        Parameters
        ----------
        reactant_consumption : float
            Reactant consumption rate (kmol/s).

        Returns
        -------
        float
            Effective stoichiometric ratio.
        """
        return self.inlet_gas_flow_rate * reactant_mole_fraction * \
            gas_concentration / reactant_consumption

    def gas_transport_resistance(self, species, volume_flow_rate=None): 
        """
        Calculate total gas transport resistance for a species.

        Parameters
        ----------
        species : str
            Species name (e.g., 'o2', 'h2o').
        volume_flow_rate : float, optional
            Specific volume flow rate (m3/s), overrides default inlet gas flow.

        Returns
        -------
        float
            Total transport resistance (s/m).
        """
        diffusion_coeff = self.species_diffusion_coefficient(species)
        return self.transport_resistance_model.total_resistance(
            self, diffusion_coeff, volume_flow_rate if volume_flow_rate else self.inlet_gas_flow_rate)

    
    def gas_superficial_speed(self, volumetric_flow_rate: float = None) -> float:
        """
        Calculate the gas superficial speed in the channel.

        Parameters
        ----------
        volumetric_flow_rate : float, optional
            Volumetric flow rate of the gas (m³/s). If None, uses the inlet gas flow rate.

        Returns
        -------
        float
            Gas superficial speed in the channel (m/s).
        """
        return volumetric_flow_rate  / self.total_flow_section
    
    def liquid_to_gas_velocity_ratio(self, liquid_saturation, gas_viscosity, liquid_viscosity): 
        """
        Calculate the liquid-to-gas velocity ratio in the channel. Uses the eq. 10 of Zhang et al. (2026).

        Note: Supposes kinematic viscosity of water for the inlet liquid.

        Returns
        -------
        float
            Ratio between liquid and gas velocity in the channel (n.d.).

        Reference
        ---------
        Zhang et al. Advancing Next-Generation Proton Exchange Membrane Fuel Cell Design through Multi-Physics and AI Modeling. 
        Energy Environ. Sci. 2026. https://doi.org/10.1039/d5ee04599a.

        """
        return (liquid_saturation / (1-liquid_saturation)) ** 3 * (gas_viscosity / liquid_viscosity)
    
    def outlet_liquid_speed(self): 
        return self.liquid_to_gas_velocity_ratio() * self.gas_superficial_speed()
    
    def outlet_liquid_flow_rate(self): 
        return self.outlet_liquid_speed() * self.total_flow_section
    
    def outlet_liquid_molar_flow_rate(self): 
        return self.outlet_liquid_flow_rate() /  water_molar_volume(self.gas.temperature)