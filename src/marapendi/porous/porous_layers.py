"""
Module providing classes to model porous layers in electrochemical cells.
"""
from dataclasses import dataclass, field
import numpy as np
from ..thermo.constants import GAS_CONSTANT

from ..thermo.gas import GasModel, GasState
from ..porous_layers.diffusion import PorousGasResistanceModel
from ..porous_layers.darcy import DarcyTransportModel
from ..thermo.water import water_kinematic_viscosity, water_surface_tension, water_molecular_weight

@dataclass
class PorousLayer():
    """
    Represents a porous layer in a fuel cell or electrolyzer, defining its
    properties related to gas transport, permeability, capillarity, and thermal behavior.

    Attributes
    ----------
    thickness : float
        Thickness of the porous layer in meters (default is 1e-3 m).
    gas : GasComposition
        Gas composition object representing the gas properties in the layer.
    porosity : float
        Porosity of the layer (0 < porosity < 1).
    effective_gas_diffusion_ratio : float
        Ratio accounting for effective gas diffusion through the porous medium (default is 1).
    pore_diameter : float
        Average pore diameter in meters (default is large, so Knudsen diffusion negligible).
    transport_resistance_model : PorousGasResistanceModel
        Model used to calculate gas transport resistance.
    two_phase_transport_model : DarcyTransportModel
        Model used to compute liquid water flow and capillarity.
    non_wetting_saturation : float
        Fraction of the pore volume occupied by non-wetting phase (0 to 1).
    thermal_conductivity : float
        Thermal conductivity in W/(m·K).
    absolute_permeability : float
        Absolute permeability in m².
    relative_permeability_exponent : float
        Exponent for relative permeability model (default is 3, often used for cubic relationship).
    contact_angle : float
        Contact angle for liquid water in degrees (default is 120°).
    non_wetting_phase : str
        Non-wetting phase (liquid or gas).

    Notes
    -----
    This class internally computes quantities like capillary pressure scaling
    and liquid flow resistance based on current temperature, saturation, and geometry.
    """

    thickness: float = 1e-3
    gas: GasState = field(default_factory=GasState)
    temperature: float = 300.
    pressure: float = 1e5
    porosity: float = 1
    effective_gas_diffusion_ratio: float = 1
    pore_diameter: float=1e12
    transport_resistance_model: PorousGasResistanceModel = field(default_factory=PorousGasResistanceModel)
    two_phase_transport_model: DarcyTransportModel = field(default_factory=DarcyTransportModel)
    non_wetting_saturation: float = 0
    thermal_conductivity: float = 1e12 
    absolute_permeability: float = 1e6
    relative_permeability_exponent: float = 3
    contact_angle: float = 120. 
    non_wetting_phase: str = 'water'
    wetting_phase: str = 'gas'

    def __post_init__(self):
        self.sqrt_abs_permeability_porosity = np.sqrt(self.absolute_permeability * self.porosity)
        self.cosinus_contact_angle = np.abs(np.cos(np.pi / 180 * self.contact_angle))
        if self.contact_angle < 90:
            self.non_wetting_phase = 'gas'
            self.wetting_phase = 'water'
        self.RT = GAS_CONSTANT * self.temperature
        self.saturation_pressure = None

    def set_temperature(self, temperature: float) -> None:
        self.temperature = temperature
        self.RT = GAS_CONSTANT * temperature
        self.saturation_pressure = None  # invalidated; recomputed by GasModel on demand

    def set_temperature_and_pressure(self, temperature: float, pressure: float) -> None:
        self.set_temperature(temperature)
        self.pressure = pressure

    @property
    def capillary_pressure_J_ratio(self) -> float:
        return (
            water_surface_tension(self.temperature) * self.cosinus_contact_angle
            / np.sqrt(self.absolute_permeability / self.porosity)
        )

    @property
    def saturation_flow_resistance(self) -> float:
        return self.calculate_saturation_flow_resistance()

    @property
    def thermal_resistance(self):
        """
        Computes the thermal resistance of the layer.

        Returns
        -------
        float
            Thermal resistance in m²·K/W.
        """
        return self.thickness / self.thermal_conductivity

    # def calculate_darcy_flow_resistance(self):
    #     self.darcy_flow_resistance = {
    #         'water': (
    #             (self.thickness * water_kinematic_viscosity(self.temperature) * water_molecular_weight) /
    #             self.absolute_permeability
    #         ),
    #         'gas': (
    #             (self.thickness * GasModel.mixture_kinematic_viscosity(self) * GasModel.mixture_molecular_weight(self)) /
    #             self.absolute_permeability
    #         )
    #     }
        
    def calculate_saturation_flow_resistance(self, electrolyte=None):
        """
        Computes the resistance to non-wetting phase flow of the layer. 
        Based on a water saturation gradient.

        Returns
        -------
        float
            Saturation flow resistance in s.m²/mol.
        """
        if self.non_wetting_phase == 'water': 
            non_wetting_kinematic_viscosity = water_kinematic_viscosity(self.temperature)
            non_wetting_molecular_weight = water_molecular_weight
            non_wetting_surface_tension = water_surface_tension(self.temperature)
        elif self.non_wetting_phase == 'gas':
            non_wetting_kinematic_viscosity = GasModel.mixture_kinematic_viscosity(self)
            non_wetting_molecular_weight = GasModel.mixture_molecular_weight(self)
            non_wetting_surface_tension = water_surface_tension(self.temperature) if self.wetting_phase == 'water' else electrolyte.surface_tension
            
        return ((self.thickness * non_wetting_kinematic_viscosity * non_wetting_molecular_weight) / 
                    (self.sqrt_abs_permeability_porosity * self.cosinus_contact_angle * non_wetting_surface_tension))
    

    def saturation_from_capillary_pressure(self, capillary_pressure):
        """
        Computes the non-wetting phase saturation given a capillary pressure
        using the layer's two-phase transport model.

        Parameters
        ----------
        capillary_pressure : float
            Capillary pressure in Pascals (Pa).

        Returns
        -------
        float
            Non-wetting phase saturation (0 to 1).
        """
        return self.two_phase_transport_model.saturation_from_capillary_pressure(self, capillary_pressure) 

    def capillary_pressure_from_saturation(self, saturation):
        """
        Computes the capillary pressure given a liquid saturation
        using the layer's liquid transport model.

        Parameters
        ----------
        saturation : float
            Liquid saturation (0 to 1).

        Returns
        -------
        float
            Capillary pressure in Pascals (Pa).
        """
        return self.two_phase_transport_model.capillary_pressure_from_saturation(self, saturation)

    def set_ionomer_wet_properties(self, ionomer_water_content, temperature):
        pass

    def set_water_film_thickness(self, water_saturation):
        pass


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