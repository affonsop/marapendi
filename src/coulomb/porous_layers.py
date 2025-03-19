"""
Module providing a classes to model porous layers in electrochemical cells. 
"""
from dataclasses import dataclass, field
import numpy as np 
import cantera as ct

from .gas_composition import GasComposition, index_h2, index_o2, index_h2ov, species_indexes
from .tools import calculate_arrhenius_term
from .electrochemistry import ElectrochemicalReaction 
from .transport import PorousGasResistanceModel
from .water import water_kinematic_viscosity, water_surface_tension, water_molecular_weight, water_molar_volume
from .membrane import Membrane

@dataclass
class PorousLayer(): 
    """
    Represents a porous layer in a fuel cell or electrolyzer, defining its properties related 
    to gas transport, permeability, and thermal characteristics.

    Attributes
    ----------
    thickness : float
        Thickness of the porous layer in meters (default is 1 mm).
    gas : GasComposition
        Gas composition object representing the gas properties in the layer.
    porosity : float
        Porosity of the layer (default is 1).
    effective_gas_diffusion_ratio : float
        Ratio accounting for gas diffusion efficiency through the porous layer (default is 1).
    pore_diameter : float
        Average pore diameter in meters (default is 1e12, with negligible Knudsen diffusion).
    transport_resistance_model : PorousGasResistanceModel
        Model used to calculate gas transport resistance.
    water_saturation : float
        Fraction of the pore volume occupied by liquid water (default is 0).
    thermal_conductivity : float
        Thermal conductivity in W/(m·K) (default is 1e12, high conductivity).
    absolute_permeability : float
        Absolute permeability of the layer in m² (default is 10000).
    contact_angle : float
        Contact angle for liquid water in degrees (default is 120°).
    """

    thickness: float = 1e-3
    gas: GasComposition = field(default_factory=GasComposition)
    porosity: float = 1
    effective_gas_diffusion_ratio: float = 1
    pore_diameter: float=1e12
    transport_resistance_model: PorousGasResistanceModel = field(default_factory=PorousGasResistanceModel)
    water_saturation: float = 0
    thermal_conductivity: float = 1e12 
    absolute_permeability: float = 10000
    contact_angle: float = 120. 

    def __post_init__(self): 
        self.temperature = self.gas.temperature
        self.RT = ct.gas_constant * self.temperature
        self.pressure = self.gas.pressure 
        self.sqrt_abs_permeability_porosity = np.sqrt(self.absolute_permeability * self.porosity)
        self.cosinus_contact_angle = np.abs(np.cos(np.pi / 180 * self.contact_angle))

    def o2_mole_fraction(self):
        """
        Returns the mole fraction of oxygen (O₂) in the gas phase.

        Returns
        -------
        float
            Oxygen mole fraction.
        """
        return self.gas.X[..., index_o2]

    def h2_mole_fraction(self):
        """
        Returns the mole fraction of hydrogen (H₂) in the gas phase.

        Returns
        -------
        float
            Hydrogen mole fraction.
        """
        return self.gas.X[..., index_h2]

    def species_mole_fraction(self, species):
        """
        Returns the mole fraction of a given species in the gas phase.

        Parameters
        ----------
        species : str
            The chemical species (e.g., 'o2', 'h2', 'h2o', 'n2').

        Returns
        -------
        float
            Mole fraction of the specified species.
        """
        return self.gas.X[..., species_indexes[species]]

    def species_diffusion_coefficient(self, species):
        """
        Returns the diffusion coefficient of a given species in the gas phase.

        Parameters
        ----------
        species : str
            The chemical species for which the diffusion coefficient is needed.

        Returns
        -------
        float
            Diffusion coefficient of the specified species (m²/s).
        """
        return self.gas.species_diffusion_coefficient(species)

    def species_molecular_weight(self, species):
        """
        Returns the molecular weight of a given species.

        Parameters
        ----------
        species : str
            The chemical species for which the molecular weight is needed.

        Returns
        -------
        float
            Molecular weight of the specified species (g/mol).
        """
        return self.gas.molecular_weights[species_indexes[species]]

    def gas_temperature(self):
        """
        Returns the temperature of the gas phase.

        Returns
        -------
        float
            Gas temperature in Kelvin (K).
        """
        return self.temperature

    def gas_pressure(self):
        """
        Returns the pressure of the gas phase.

        Returns
        -------
        float
            Gas pressure in Pascals (Pa).
        """
        return self.pressure
    
    def set_gas_composition(self, dry_o2_mole_fraction, dry_h2_mole_fraction, relative_humidity):
        """
        Sets the gas composition in terms of oxygen, hydrogen, and humidity.

        Parameters
        ----------
        dry_o2_mole_fraction : float
            Mole fraction of oxygen in dry gas.
        dry_h2_mole_fraction : float
            Mole fraction of hydrogen in dry gas.
        relative_humidity : float
            Relative humidity (0 to 1).
        """
        self.gas.set_composition(dry_o2_mole_fraction, dry_h2_mole_fraction, relative_humidity)

    def set_gas_pressure(self, pressure):
        """
        Sets the gas pressure.

        Parameters
        ----------
        pressure : float
            Gas pressure in Pascals (Pa).
        """
        self.pressure = pressure
        self.gas.set_pressure(pressure)

    def set_gas_temperature(self, temperature):
        """
        Sets the gas temperature.

        Parameters
        ----------
        temperature : float
            Gas temperature in Kelvin (K).
        """
        self.temperature = temperature
        self.RT = ct.gas_constant * self.temperature
        self.gas.set_temperature(temperature)

    def set_gas_temperature_and_pressure(self, temperature, pressure):
        """
        Sets both the gas temperature and pressure.

        Parameters
        ----------
        temperature : float
            Gas temperature in Kelvin (K).
        pressure : float
            Gas pressure in Pascals (Pa).
        """
        self.temperature = temperature
        self.RT = ct.gas_constant * self.temperature
        self.pressure = pressure
        self.gas.set_temperature_and_pressure(temperature, pressure)

    def species_concentration(self, species):
        """
        Computes the concentration of a given species.

        Parameters
        ----------
        species : str
            The chemical species (e.g., 'o2', 'h2', 'h2o', 'n2').

        Returns
        -------
        float
            Species concentration in mol/m³.
        """
        return self.gas.X[..., species_indexes[species]] * self.gas.pressure / self.RT

    def species_partial_pressure(self, species):
        """
        Computes the partial pressure of a given species.

        Parameters
        ----------
        species : str
            The chemical species (e.g., 'o2', 'h2', 'h2o', 'n2').

        Returns
        -------
        float
            Partial pressure in Pascals (Pa).
        """
        return self.gas.X[..., species_indexes[species]] * self.pressure

    def vapor_pressure(self):
        """
        Computes the vapor pressure of water in the gas mixture.

        Returns
        -------
        float
            Vapor pressure in Pascals (Pa).
        """
        return self.gas.X[..., index_h2ov] * self.pressure

    def relative_humidity(self):
        """
        Returns the relative humidity of the gas mixture.

        Returns
        -------
        float
            Relative humidity (0 to 1).
        """
        return self.gas.relative_humidity

    def saturation_pressure(self):
        """
        Returns the saturation pressure of water in the gas mixture.

        Returns
        -------
        float
            Saturation pressure in Pascals (Pa).
        """
        return self.gas.saturation_pressure

    def saturation_concentration(self):
        """
        Computes the saturation concentration of water vapor.

        Returns
        -------
        float
            Saturation concentration in mol/m³.
        """
        return self.saturation_pressure() / self.RT

    def vapor_concentration(self):
        """
        Computes the water vapor concentration.

        Returns
        -------
        float
            Vapor concentration in mol/m³.
        """
        return self.vapor_pressure() / self.RT
    
    def gas_transport_resistance(self, species='o2'):
        """
        Computes the gas transport resistance for a given species.

        Parameters
        ----------
        species : str, optional
            The gas species for which the transport resistance is computed 
            (default is 'o2').

        Returns
        -------
        float
            Gas transport resistance in s/m.
        """
        return self.transport_resistance_model.total_diffusion_resistance(
            self, 
            self.temperature, 
            self.species_diffusion_coefficient(species), 
            self.species_molecular_weight(species), 
            self.water_saturation
        )

    def thermal_resistance(self):
        """
        Computes the thermal resistance of the layer.

        Returns
        -------
        float
            Thermal resistance in m²·K/W.
        """
        return self.thickness / self.thermal_conductivity

    def saturation_flow_resistance(self):
        """
        Computes the resistance to liquid water flow of the layer. 
        Based on a water saturation gradient.

        Returns
        -------
        float
            Saturation flow resistance in s.m²/mol.
        """
        return ((self.thickness / self.sqrt_abs_permeability_porosity) *
                water_kinematic_viscosity(self.temperature) * water_molecular_weight /
                (self.cosinus_contact_angle * water_surface_tension(self.temperature)))

        

@dataclass 
class CatalystLayerIonomerModel(Membrane): 
    dry_density: float = 2004 
    equivalent_weight: float = 952. 
    
    hydrated_proton_conductivity: float = 11 # S/m
    proton_conductivity_water_content_exponent: float = 0
    proton_conductivity_rh_exponent: float = 2.7
    proton_conductivity_activation_energy: float = 11e6
    hydrated_o2_diffusion: float = 1.14698e-10*14**0.708
    o2_diffusion_exponent: float = 0.708
    o2_diffusion_activation_energy: float = 24e6

    def __post_init__(self): 
        Membrane.__post_init__(self)

    def o2_film_diffusion_coefficient(self, water_content, temperature= 353.15):
        # Linear regression of data from Jinnouchi et al. (2021), neglecting bulk diffusion.
        # Activation energy obtained by Kudo et al. (2006).
        return (self.hydrated_o2_diffusion * 
                (water_content/14) ** self.o2_diffusion_exponent *
                calculate_arrhenius_term(self.o2_diffusion_activation_energy, temperature, 353.15)) 

    def o2_permeability(self, water_content, temperature= 353.15):
        fv = self.water_vol_fraction(water_content, water_molar_volume(temperature))
        RT = ct.gas_constant * temperature
        return (6.74e-12 * np.exp(-21280e3/RT) + fv * 50.5e-12 * np.exp(-20470e3/RT)) * 1e-3

    def proton_conductivity(self, relative_humidity, water_content, temperature):
        return (self.hydrated_proton_conductivity *
                (water_content/14.) ** (self.proton_conductivity_water_content_exponent) *
                relative_humidity ** self.proton_conductivity_rh_exponent *
                calculate_arrhenius_term(self.proton_conductivity_activation_energy, temperature, 353.15)) # Following measurements of Hutapea et al. (2023)

NafionD2020 = CatalystLayerIonomerModel(dry_density=2004., equivalent_weight=952.)


from dataclasses import dataclass, field
import numpy as np

@dataclass
class CatalystLayer(PorousLayer):
    """
    Represents the catalyst layer in a fuel cell, containing platinum, ionomer, and carbon structure.
    Catalyst properties are calculated based on the work of Hao et al. (2015)

    Attributes
    ----------
    thickness : float
        Thickness of the catalyst layer (default: 10 µm).
    porosity : float
        Initial porosity of the layer (default: 0, will be recalculated).
    ionomer : CatalystLayerIonomerModel
        Ionomer model associated with the layer.
    reaction : ElectrochemicalReaction
        Electrochemical reaction model.
    platinum_loading : float
        Platinum loading in g/cm² (default: 0.2 mg/cm²).
    catalyst_platinum_weight_percent : float
        Percentage of platinum in the catalyst (default: 50%).
    ionomer_to_carbon_ratio : float
        Ratio of ionomer to carbon content.
    platinum_density : float
        Density of platinum (21450 kg/m³).
    carbon_density : float
        Density of carbon (1950 kg/m³).
    ecsa : float
        Electrochemically active surface area (70 m²/g).
    platinum_vol_surface_area : float
        Volume-specific platinum surface area (calculated if not provided).
    carbon_agglomerate_radius : float
        Radius of carbon agglomerates (default: 20 nm).
    ionomer_vol_fraction : float
        Volume fraction of ionomer (calculated).
    ionomer_film_thickness : float
        Thickness of ionomer film around carbon agglomerates.
    contact_angle : float
        Contact angle for water (default: 95°).
    omega_PtO : float
        Oxide coverage parameter for Pt (default: 3000 kJ/mol).
    theta_PtO : float
        PtO surface coverage.
    """
    thickness: float = 10e-6 
    porosity: float = 0  # Will be recalculated
    ionomer: CatalystLayerIonomerModel = field(default_factory=CatalystLayerIonomerModel)
    reaction: ElectrochemicalReaction = field(default_factory=ElectrochemicalReaction)
    platinum_loading: float = 0.2e-6 * 1e4  # 0.2 mg/cm²
    catalyst_platinum_weight_percent: float = 0.5
    ionomer_to_carbon_ratio: float = 0.75
    platinum_density: float = 21450  # kg/m³
    carbon_density: float = 1950  # kg/m³
    ecsa: float = 70e3  # m²/g
    platinum_vol_surface_area: float = 0  # Will be calculated if zero
    carbon_agglomerate_radius: float = 25e-9 # Vulcan XC-72 radius according to Hao et al. (2015)
    ionomer_vol_fraction: float = 0  # Will be calculated
    ionomer_film_thickness: float = 0  # Will be calculated
    contact_angle: float = 95.  # degrees
    omega_PtO: float = 3000e3  # kJ/mol
    theta_PtO: float = 0 

    def __post_init__(self):
        if self.platinum_vol_surface_area == 0:
            self.platinum_vol_surface_area = self.platinum_loading * self.ecsa / self.thickness

        if self.porosity == 0:
            self.carbon_loading = self.platinum_loading * (1 / self.catalyst_platinum_weight_percent - 1)
            self.carbon_vol_fraction = self.carbon_loading / (self.thickness * self.carbon_density)
            self.platinum_vol_fraction = self.platinum_loading / (self.thickness * self.platinum_density)
            self.catalyst_vol_fraction = self.platinum_vol_fraction + self.carbon_vol_fraction
            self.ionomer_vol_fraction = (self.carbon_vol_fraction * self.carbon_density *
                                         self.ionomer_to_carbon_ratio / self.ionomer.dry_density)
            self.porosity = 1 - self.catalyst_vol_fraction - self.ionomer_vol_fraction
        else:
            self.catalyst_vol_fraction = 1 - self.ionomer_vol_fraction - self.porosity

        # Carbon agglomerate properties
        self.carbon_agglomerate_surface = 4 * np.pi * self.carbon_agglomerate_radius ** 2
        self.carbon_agglomerate_volume = self.carbon_agglomerate_surface * self.carbon_agglomerate_radius / 3.
        self.carbon_agglomerate_number_density = self.catalyst_vol_fraction / self.carbon_agglomerate_volume

        # Ionomer properties 
        if self.ionomer_film_thickness == 0:
            self.ionomer_film_thickness = (self.ionomer_vol_fraction /
                                           (self.carbon_agglomerate_number_density * self.carbon_agglomerate_surface))
        self.ionomer_vol_surface_area = (4 * np.pi *
                                         (self.carbon_agglomerate_radius + self.ionomer_film_thickness) ** 2 *
                                         self.carbon_agglomerate_number_density)

        self.effective_gas_diffusion_ratio = self.porosity ** 1.5
        PorousLayer.__post_init__(self)

    def o2_ionomer_film_bulk_resistance(self, ionomer_water_content, temperature): 
        """
        Calculate the oxygen bulk resistance in the ionomer film.

        Parameters
        ----------
        ionomer_water_content : float
            Water content in the ionomer film.
        temperature : float
            Operating temperature in Kelvin.

        Returns
        -------
        float
            Oxygen film resistance [s/m].
        """

       
        return  (self.ionomer_film_thickness / 
                 (ct.gas_constant * temperature * self.ionomer.o2_permeability(ionomer_water_content, temperature)))
    
    def o2_ionomer_film_resistance(self, ionomer_water_content, temperature): 
        """
        Calculate the oxygen film resistance in the ionomer film.
        Uses the formulation from Hao et al. (2015).

        Parameters
        ----------
        ionomer_water_content : float
            Water content in the ionomer film.
        temperature : float
            Operating temperature in Kelvin.

        Returns
        -------
        float
            Oxygen film resistance [s/m].
        """

        # k1 and k2 values from Hao et al. (2015)
        k1 = 8.5
        k2 = 5.4
        ionomer_pt_interface_term = (k2 + 1) / (1 - self.theta_PtO) / (self.platinum_loading * self.ecsa)
        ionomer_gas_interface_term = k1 / (self.ionomer_vol_surface_area * self.thickness)
        return  (ionomer_gas_interface_term + ionomer_pt_interface_term) * self.o2_ionomer_film_bulk_resistance(ionomer_water_content, temperature)
    
    def ionomer_sheet_proton_resistance(self, relative_humidity, ionomer_water_content, temperature): 
        """
        Calculate the proton resistance of the ionomer film.

        Parameters
        ----------
        relative_humidity : float
            Relative humidity in the catalyst layer.
        ionomer_water_content : float
            Water content in the ionomer.
        temperature : float
            Operating temperature in Kelvin.

        Returns
        -------
        float
            Ionomer film proton resistance [Ohm.m2].
        """
        # Consider ionomer_tortuosity = 1
        ionomer_proton_conductivity = self.ionomer.proton_conductivity(relative_humidity, ionomer_water_content, temperature)
        return self.thickness / (self.ionomer_vol_fraction * ionomer_proton_conductivity)
    
    def effective_proton_resistance(self,current_density, relative_humidity, ionomer_water_content, temperature): 
        """
        Calculate the effective proton resistance in the catalyst layer 
        based on Goshtasbi et al. (2020) and Neyerlin et al. (2007).

        Parameters
        ----------
        current_density : float
            Current density [A/m²].
        relative_humidity : float
            Relative humidity in the catalyst layer.
        ionomer_water_content : float
            Water content in the ionomer.
        temperature : float
            Operating temperature in Kelvin.

        Returns
        -------
        float
            Effective catalyst layer proton resistance [Ohm.m2].
        """
        # Based on the method proposed by Goshtasbi et al. (2020), based on Neyerlin et al. (2007)
        self.ionomer_sheet_resistance = self.ionomer_sheet_proton_resistance(relative_humidity, ionomer_water_content, temperature)
        nu = np.minimum(self.ionomer_sheet_resistance * current_density / self.reaction.tafel_slope(temperature), 10) # Goshtasbi parametrization valid for nu < 10
        self.xi_neyerlin = nu * (-8.287e-3 * nu + 0.7184) - 2.072e-3
        return self.ionomer_sheet_resistance / (3 + self.xi_neyerlin)