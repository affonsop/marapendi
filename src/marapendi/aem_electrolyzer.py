"""
Module providing a AEM water electrolyzer class. 
"""
from dataclasses import dataclass, field
from scipy.optimize import root
import numpy as np 
import cantera as ct

from .fuelcell import FuelCell, FuelCellSide
from .electrochemistry import calculate_reversible_cell_voltage, h2_lhv, STD_PRESSURE
from .porous_layers import PorousLayer, PtCCatalystLayer
from .flow_channels import FlowChannel
from .membrane import Membrane
from .gas_composition import species_indexes 
from .transport import DarcyLiquidTransportModel
from .water import water_molar_volume

@dataclass
class ElectrolyzerCellSide(FuelCellSide):
    has_gdl: bool = False

    def __post_init__(self): 
        self.porous_layers = [self.cl] 
        if self.has_mpl: 
            self.porous_layers += [self.mpl]
        if self.has_gdl: 
            self.porous_layers += [self.gdl]
        self.components = self.porous_layers + [self.ch]
        self.o2_transport_resistance = 0
        self.h2ov_transport_resistance = 0
        self.h2_transport_resistance = 0   

    def calculate_dry_gas_pressure(self): 
        solution_saturation_pressure = self.electrolyte.solution_sat_pressure
        return np.where(self.cl.liquid_saturation > 0, 
                 self.cl.pressure - solution_saturation_pressure,
                 self.cl.pressure - self.cl.vapor_pressure())
         
class ElectrolyzerCell(FuelCell): 

    def reversible_cell_voltage(self): 
        """
        Calculate the reversible cell voltage based on the Nernst equation.

        Returns
        -------
        float
            The reversible cell voltage (also known as the Nernst potential) in volts.

        Notes
        -----
        The calculation considers the temperature of the catalyst layer and the partial pressures
        of oxygen and hydrogen at the cathode and anode, respectively.
        """
        h2_activity = self.ca.calculate_dry_gas_pressure()/STD_PRESSURE
        o2_activity = self.an.calculate_dry_gas_pressure()/STD_PRESSURE
        h2o_activity = self.ca.electrolyte.solution_sat_pressure / self.ca.cl.saturation_pressure()
        activities_ratio = h2o_activity / (h2_activity * o2_activity ** 0.5)
        
        return calculate_reversible_cell_voltage(
            self.mea_temperature,
            activities_ratio, 
        )
    
    def ohmic_overpotential(self): 
        """
        Compute the ohmic overpotential of the electrolyzer.

        Returns
        -------
        float
            The ohmic overpotential in volts.

        Notes
        -----
        The ohmic overpotential arises from the resistance to proton conduction in the 
        catalyst layer and the membrane, as well as the electronic resistance of the 
        cell components. It is calculated as the product of the current density and 
        the total internal resistance.
        """
        self.ca.cl.charge_resistance = self.ca.cl.effective_charge_resistance(
            self.current_density, 
            self.ca.cl.ionomer_water_content, 
            self.ca.cl.temperature, 
            charge='hydroxide'
        )
        self.an.cl.charge_resistance = self.an.cl.effective_charge_resistance(
            self.current_density, 
            self.an.cl.ionomer_water_content, 
            self.an.cl.temperature,
            charge='hydroxide'
        )
        return self.current_density * (self.ca.cl.charge_resistance + 
                                       self.high_frequency_resistance() + 
                                       self.an.cl.charge_resistance)
    
    def high_frequency_resistance(self): 
        """
        Compute the high-frequency resistance (HFR) of the fuel cell.

        Returns
        -------
        float
            The high-frequency resistance in ohms.

        Notes
        -----
        The high-frequency resistance is mainly due to the proton resistance of the membrane 
        and the electrical resistance of the cell components. It is an important parameter in 
        electrochemical impedance spectroscopy (EIS) measurements.
        """
        return self.membrane.charge_resistance(20, self.membrane.temperature, 
                                               use_water_profile=False, charge='hydroxide') + self.electrical_resistance
    
    def cell_voltage(self):
        """
        Compute the cell voltage of the electrolyzer.

        Returns
        -------
        float
            The electrolyzer voltage in volts.

        Notes
        -----
        The cell voltage is calculated as the sum of the reversible cell voltage 
        and the activation and ohmic overpotentials.
        """ 
        
        E_rev = self.reversible_cell_voltage()
        eta_ohm = self.ohmic_overpotential()
        eta_act = self.activation_overpotential()
        return np.maximum(0, E_rev + eta_act + eta_ohm)

    def activation_overpotential(self, theta_PtO=0): 
        """
        Compute the activation overpotential of the electrolyzer cell.

        Parameters
        ----------
        theta_PtO : float, optional
            The coverage fraction of PtO species on the catalyst surface. Default is 0.

        Returns
        -------
        float
            The activation overpotential in volts.

        Notes
        -----
        The activation overpotential is calculated using the Tafel equation, considering 
        the hydrogen crossover current, oxygen partial pressure, and platinum surface coverage.
        It accounts for the voltage drop due to the PtO coverage effect.
        """
        self.h2_permeation_flux = self.membrane.hydrogen_permeation_flux(self.an.cl.species_partial_pressure('h2'), 
                                                                        self.membrane.temperature, 
                                                                        self.an.cl.pressure - self.ca.cl.pressure,
                                                                        self.membrane.water_vol_fraction(
                                                                            self.membrane.water_content, 
                                                                            water_molar_volume(self.membrane.temperature)
                                                                            )
                                                                        )
        self.crossover_current = self.h2_permeation_flux * (2 * ct.faraday)
  
        tafel_overpotential_ca = self.ca.cl.activation_overpotential(self.current_density, 1.)
        tafel_overpotential_an = self.an.cl.activation_overpotential(self.current_density, 1.)
        return tafel_overpotential_ca + tafel_overpotential_an
    
    def set_conditions(self, stack_temperature, current_density, cathode_conditions, anode_conditions):  
        """
        Set the operating conditions of the electrolyzer stack.

        This method initializes key operating parameters such as current density, 
        water consumption rates, O2 and H2 production, and temperature. It also 
        updates the gas composition and electrolyte flow conditions for the cathode and anode.

        Parameters
        ----------
        stack_temperature : float
            The operating temperature of the electrolyzer stack in Kelvin.
        current_density : float
            The current density of the cell in A/m².
        cathode_conditions : OperatingConditions
            The inlet conditions at the cathode side, including temperature, 
            pressure, oxygen mole fraction, relative humidity and electrolyte saturation.
        anode_conditions : OperatingConditions
            The inlet conditions at the anode side, including temperature, 
            pressure, hydrogen mole fraction, relative humidity and electrolyte saturation.
        """
        self.current_density = current_density
        self.o2_production = current_density / (4 * ct.faraday)
        self.h2_production = 2 * self.o2_production
        self.ca.h2o_consumption = 2 * self.h2_production
        self.an.h2o_production = self.h2_production
        self.temperature = stack_temperature
        self.membrane.temperature = stack_temperature
        self.mea_temperature = stack_temperature

        for cell_side, conditions in zip((self.ca, self.an), (cathode_conditions, anode_conditions)): 
            cell_side.electrolyte = conditions.inlet_liquid
            cell_side.electrolyte.set_temperature(self.membrane.temperature)
            for component in cell_side.components: 
                try: 
                    component.gas.X = np.zeros_like(self.current_density[...,np.newaxis]) * np.array([0,0,0,0])
                except TypeError: 
                    component.gas.X = self.current_density * np.array([0,0,0,0])
                component.set_gas_temperature_and_pressure(conditions.inlet_temperature, conditions.inlet_pressure)
                component.set_gas_composition(conditions.dry_o2_mole_fraction, 
                                              conditions.dry_h2_mole_fraction,
                                              conditions.inlet_relative_humidity)
                component.set_gas_temperature_and_pressure(stack_temperature, conditions.inlet_pressure)
            cell_side.ch.set_fixed_inlet_liquid_flow_rate(conditions.inlet_liquid_flow_rate)
            cell_side.ch.set_fixed_inlet_gas_flow_rate(conditions.inlet_gas_flow_rate)
            for component in cell_side.components: 
                component.electrolyte = cell_side.electrolyte
                component.liquid_saturation = cell_side.ch.inlet_liquid_saturation