"""
Module providing a fuel cell class intended to be the base class for different fuel cell models. 
"""
from dataclasses import dataclass, field
from scipy.optimize import root
import numpy as np 
import cantera as ct

from .electrochemistry import calculate_reversible_cell_voltage, h2_lhv
from .porous_layers import PorousLayer, CatalystLayer
from .flow_channels import GasFlowChannel
from .membrane import Membrane
from .gas_composition import species_indexes 
from .transport import PorousLiquidTransportModel
from .water import water_molar_volume

@dataclass
class FuelCellSide:
    cl: PorousLayer = field(default_factory=CatalystLayer) 
    gdl: PorousLayer = field(default_factory=PorousLayer)
    mpl: PorousLayer = field(default_factory=PorousLayer)
    ch: GasFlowChannel = field(default_factory=GasFlowChannel)
    has_mpl: bool = False
    liq_transport_model: PorousLiquidTransportModel = field(default_factory=PorousLiquidTransportModel)
    membrane_surface_water_content: float = 0 
    thermal_contact_resistance: float = 0 

    def __post_init__(self): 
        self.porous_layers = [self.cl, self.mpl, self.gdl] if self.has_mpl else [self.cl, self.gdl]
        self.components = self.porous_layers + [self.ch]
        self.o2_transport_resistance = 0
        self.h2ov_transport_resistance = 0
        self.h2_transport_resistance = 0   

    def set_catalyst_layer(self,cl): 
        self.cl = cl 
        self.__post_init__()
    
    def set_gas_diffusion_layer(self, gdl): 
        self.gdl = gdl
        self.__post_init__()
    
    def set_channel(self, ch): 
        self.ch = ch 
        self.__post_init__()

    def calculate_gas_transport_resistance(self, species, ionomer_water_content=11):
        return (sum(layer.calculate_gas_transport_resistance(species) for layer in self.porous_layers) +
                self.ch.calculate_gas_transport_resistance(species) + 
                (self.cl.calculate_o2_film_resistance(ionomer_water_content, self.cl.temperature) if species == 'o2' else 0))

    def calculate_heat_transfer_resistance(self): 
        return sum(layer.calculate_heat_transfer_resistance() for layer in self.porous_layers) + self.thermal_contact_resistance
                
    def calculate_water_saturation(self): 
        return self.liq_transport_model.calculate_water_saturation(self)
    
    def calculate_equivalent_flow_resistance(self): 
            return sum(layer.calculate_saturation_absolute_flow_resistance() for layer in self.porous_layers)
    
    def max_water_vapor_removal(self): 
        return (self.cl.get_saturation_concentration() - self.ch.get_vapor_concentration()) / self.calculate_gas_transport_resistance('h2o')
    

@dataclass
class FuelCell: 
    cell_area: float
    cell_number: int
    an: FuelCellSide = field(default_factory=FuelCellSide)
    ca: FuelCellSide = field(default_factory=FuelCellSide)
    membrane: Membrane = field(default_factory=Membrane)
    electrical_resistance: float = 0
    h2_permeation_flux: float = 0 
    crossover_current: float = 0
    thermal_resistance: float = 0
    heat_release_rate: float = 0
    mea_temperature_increase: float = 0
    mea_temperature: float = 0

    def reversible_cell_voltage(self): 
        return calculate_reversible_cell_voltage(
            self.ca.cl.temperature,
            self.ca.cl.get_species_partial_pressure('o2'),
            self.an.cl.get_species_partial_pressure('h2')
        )
    
    def activation_overpotential(self, theta_PtO=0): 
        self.h2_permeation_flux = self.membrane.hydrogen_permeation_flux(self.an.cl.get_species_partial_pressure('h2'), 
                                                                        self.membrane.temperature, 
                                                                        self.an.cl.pressure - self.ca.cl.pressure,
                                                                        self.membrane.water_vol_fraction(
                                                                            self.membrane.water_content, 
                                                                            water_molar_volume(self.membrane.temperature)
                                                                            )
                                                                        )
        self.crossover_current = self.h2_permeation_flux * (2 * ct.faraday)

        return self.ca.cl.reaction.tafel_overpotential(
            (self.current_density + self.crossover_current) / (self.ca.cl.ecsa * self.ca.cl.platinum_loading * (1-theta_PtO)),
            self.ca.cl.temperature,
            self.ca.cl.get_species_partial_pressure('o2')
        )  + self.ca.cl.omega_PtO * theta_PtO / (self.ca.cl.reaction.number_of_electrons * self.ca.cl.reaction.charge_transfer_coeff * ct.faraday)
    
    def high_frequency_resistance(self): 
        return self.membrane.proton_resistance(self.membrane.temperature, 0, self.membrane.water_content) + self.electrical_resistance
    
    def ohmic_overpotential(self): 
        self.ca.cl.proton_resistance = self.ca.cl.calculate_effective_proton_resistance(self.current_density, 
                                                                         self.ca.cl.get_relative_humidity(), 
                                                                         self.ca.cl.water_content, 
                                                                         self.ca.cl.temperature)
        return self.current_density * (self.ca.cl.proton_resistance + self.high_frequency_resistance())

    def reversible_voltage_vs_RHE(self): 
        return calculate_reversible_cell_voltage(
            self.ca.cl.temperature,
            self.ca.cl.get_species_partial_pressure('o2'),
            1e5,
        )

    def cell_voltage(self):
        E_rev_vs_RHE = self.reversible_voltage_vs_RHE()
      
        theta_PtO = 0
        if self.ca.cl.omega_PtO > 0: 
            eps_max = 10
            eta_act = 0
            while  eps_max > 0.001:
                eta_act_old = eta_act
                eta_act = self.activation_overpotential(theta_PtO) 
                theta_PtO = 0.5 * theta_PtO +0.5 / (1 + np.exp(22.4 * (0.818 - E_rev_vs_RHE + eta_act)))
            
                eps_max = np.mean(np.abs(eta_act - eta_act_old))
        else:
            eta_act = self.activation_overpotential(theta_PtO=0) 
        return np.maximum(0,self.reversible_cell_voltage() - eta_act - self.ohmic_overpotential())
    
    def set_mea_temperature(self, mea_temperature): 
        self.mea_temperature = mea_temperature
        self.ca.cl.set_gas_temperature(mea_temperature)
        self.an.cl.set_gas_temperature(mea_temperature)

        # self.ca.gdl.gas.set_temperature(mea_temperature)
        self.membrane.temperature = mea_temperature
        self.mea_temperature_increase = self.mea_temperature - self.temperature
    
    def calculate_water_transport(self): 
        self.ca.h2ov_transport_resistance = self.ca.calculate_gas_transport_resistance('h2o')
        self.an.h2ov_transport_resistance = self.an.calculate_gas_transport_resistance('h2o')
        self.membrane.water_balance_model.water_balance(self)
        self.ca.gdl.water_saturation = self.ca.calculate_water_saturation()
        self.an.gdl.water_saturation = self.ca.calculate_water_saturation()
        self.ca.h2ov_transport_resistance = self.ca.calculate_gas_transport_resistance('h2o')
        self.an.h2ov_transport_resistance = self.an.calculate_gas_transport_resistance('h2o')
    
    def calculate_reactant_concentration_at_cl(self): 
        ca = self.ca
        ca.o2_transport_resistance = ca.calculate_gas_transport_resistance('o2', self.ca.cl.water_content)
        ca.cl.gas.X[...,species_indexes['o2']] = np.maximum(1e-12, 
            ca.ch.gas.X[...,species_indexes['o2']] * ca.ch.gas.concentration() - 
            self.o2_consumption * ca.o2_transport_resistance) / ca.cl.gas.concentration()
        
        an = self.an
        an.h2_transport_resistance = an.calculate_gas_transport_resistance('h2')
        an.cl.gas.X[...,species_indexes['h2']] = np.maximum(1e-12, 
            an.ch.gas.X[...,species_indexes['h2']] * an.ch.gas.concentration() - 
            self.h2_consumption * an.h2_transport_resistance) / an.cl.gas.concentration()
        
        for side in (self.ca, self.an):
            side.cl.gas.X[...,species_indexes['h2o']] = np.maximum(1e-12,
                side.ch.gas.X[...,species_indexes['h2o']] * side.ch.gas.concentration() +
                side.vapor_flux * side.h2ov_transport_resistance) / side.cl.gas.concentration() 
            side.cl.gas.calculate_relative_humidity()
            side.cl.gas.X[...,species_indexes['n2']] = (1 - side.cl.gas.X[...,species_indexes['o2']] -
                                                        side.cl.gas.X[...,species_indexes['h2o']] -
                                                        side.cl.gas.X[...,species_indexes['h2']])
    
    def calculate_heat_transfer_resistance(self): 
        self.thermal_resistance = 1/sum(1./side.calculate_heat_transfer_resistance() for side in (self.ca, self.an))

    def calculate_heat_transport(self): 
        self.calculate_heat_transfer_resistance()
        self.heat_release_rate = (-h2_lhv(self.temperature) / (2 * ct.faraday) - self.cell_voltage()) * self.current_density
        self.mea_temperature_increase = self.heat_release_rate * self.thermal_resistance

    def solve_transport(self):
        def f(dT): 
            mea_temperature = np.minimum(self.temperature + np.maximum(dT, 0), 373.15) 
            self.set_mea_temperature(mea_temperature)
            self.calculate_water_transport()
            self.calculate_reactant_concentration_at_cl()
            self.calculate_heat_transport()
            return dT - self.mea_temperature_increase 
        
        self.calculate_heat_transfer_resistance()
        res = root(f, 1.4 * self.current_density * self.thermal_resistance, method='broyden1', options={'fatol':1e-2, 'maxiter':5})
        # self.calculate_water_transport()
        # self.calculate_reactant_concentration_at_cl()
        # self.calculate_heat_transport()

    def set_conditions(self, stack_temperature, current_density, cathode_conditions, anode_conditions): 
        self.current_density = current_density
        self.o2_consumption = current_density / (4 * ct.faraday)
        self.h2_consumption = 2 * self.o2_consumption
        self.h2o_production = self.h2_consumption
        self.temperature = stack_temperature
        self.membrane.temperature = stack_temperature
        for side in (self.ca, self.an): 
            for layer in side.components: 
                layer.water_saturation = 0 

        for cell_side, conditions in zip((self.ca, self.an), (cathode_conditions, anode_conditions)): 
            
            for component in cell_side.components: 
                component.water_saturation = 0

                component.gas.X = np.zeros_like(self.current_density[...,np.newaxis]) * np.array([0,0,0,0])
                component.set_gas_temperature_and_pressure(conditions.inlet_temperature, conditions.inlet_pressure)
                component.set_gas_composition(conditions.dry_o2_mole_fraction, 
                                              conditions.dry_h2_mole_fraction,
                                              conditions.inlet_relative_humidity)
                
                # component.gas.set_temperature_and_pressure(conditions.inlet_temperature, conditions.inlet_pressure)
                component.set_gas_temperature_and_pressure(stack_temperature, conditions.inlet_pressure)
                
            cell_side.ch.set_inlet_gas_flow_rate_from_stoichiometry(
                (self.o2_consumption if cell_side == self.ca else self.h2_consumption) * self.cell_area, conditions.stoichiometry
            )
        
class OperatingConditions():
    
    def __init__(self, 
                 inlet_temperature=353.15, 
                 inlet_pressure=False, 
                 outlet_pressure=False, 
                 dry_o2_mole_fraction=0.2, 
                 dry_h2_mole_fraction=0., 
                 inlet_relative_humidity=0.5, 
                 stoichiometry=2):
        self.inlet_temperature = inlet_temperature
        self.inlet_relative_humidity = inlet_relative_humidity
        try: 
            self.inlet_pressure = inlet_pressure if inlet_pressure else outlet_pressure
        except ValueError: # Raised if a np.array is passed 
            self.inlet_pressure = inlet_pressure 
        try: 
            self.outlet_pressure = outlet_pressure if outlet_pressure else inlet_pressure
        except ValueError: 
            self.outlet_pressure = outlet_pressure
        self.average_pressure = 0.5 * (self.inlet_pressure + self.outlet_pressure) 
        self.dry_o2_mole_fraction = dry_o2_mole_fraction
        self.dry_h2_mole_fraction = dry_h2_mole_fraction
        self.stoichiometry = stoichiometry


