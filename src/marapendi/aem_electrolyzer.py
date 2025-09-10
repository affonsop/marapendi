"""
Module providing an AEM water electrolyzer class.
"""

from dataclasses import dataclass, field
from scipy.optimize import root
import numpy as np 
import cantera as ct

from .fuelcell import FuelCell, FuelCellSide
from .electrochemistry import calculate_reversible_cell_voltage, STD_PRESSURE
from .water import water_molar_volume

@dataclass
class ElectrolyzerCellSide(FuelCellSide):
    """
    Represents one side (anode or cathode) of an electrolyzer cell.

    Attributes
    ----------
    has_gdl : bool
        Indicates if a gas diffusion layer (GDL) is present.
    porous_layers : list
        List of porous layers in the cell side.
    components : list
        All components (porous layers + channel) in the cell side.
    """
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
        """
        Calculate the partial pressure of the dry gas phase.

        Returns
        -------
        float or np.ndarray
            Dry gas pressure in Pa.
        """
        solution_saturation_pressure = self.electrolyte.solution_sat_pressure
        return np.where(self.cl.non_wetting_saturation > 0,
                        self.cl.pressure - solution_saturation_pressure,
                        self.cl.pressure - self.cl.vapor_pressure())

@dataclass
class ElectrolyzerCell(FuelCell):
    """
    Class representing an AEM water electrolyzer cell.
    """

    def reversible_cell_voltage(self):
        """
        Calculate the reversible cell voltage based on the Nernst equation.
        Follows eq. 8 in Lawand et al. (2024)

        Returns
        -------
        float
            The reversible cell voltage (also known as the Nernst potential) in volts.

        Reference 
        --------- 
        Lawand, K. et al. J. Power Sources 595, 234047 (2024).

        Notes
        -----
        The water activity is defined as the ratio between the solution saturation pressure
        and the pure water saturation pressure. 
        """
        h2_activity = self.ca.calculate_dry_gas_pressure() / STD_PRESSURE
        o2_activity = self.an.calculate_dry_gas_pressure() / STD_PRESSURE
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
            Ohmic overpotential in volts.
        """
        self.ca.cl.charge_resistance = self.ca.cl.effective_charge_resistance(
            self.current_density, self.ca.cl.ionomer_water_content,
            self.ca.cl.temperature, charge='hydroxide')

        self.an.cl.charge_resistance = self.an.cl.effective_charge_resistance(
            self.current_density, self.an.cl.ionomer_water_content,
            self.an.cl.temperature, charge='hydroxide')

        return self.current_density * (
            self.ca.cl.charge_resistance +
            self.high_frequency_resistance() +
            self.an.cl.charge_resistance)

    def high_frequency_resistance(self):
        """
        Compute the high-frequency resistance (HFR) of the electrolyzer.

        Returns
        -------
        float
            High-frequency resistance in ohm·m².
        """
        liquid_eq_water_content = 20.0 # Dummy value since not used for AEM membranes so far. 
        return self.membrane.charge_resistance(liquid_eq_water_content, self.membrane.temperature,
                                               use_water_profile=False, charge='hydroxide') + self.electrical_resistance

    def cell_voltage(self):
        """
        Compute the overall cell voltage of the electrolyzer.

        Returns
        -------
        float
            Cell voltage in volts.
        """
        E_rev = self.reversible_cell_voltage()
        eta_ohm = self.ohmic_overpotential()
        eta_act = self.activation_overpotential()
        return np.maximum(0, E_rev + eta_act + eta_ohm)

    def activation_overpotential(self):
        """
        Compute the activation overpotential of the electrolyzer.

        Returns
        -------
        float
            Activation overpotential in volts.
        """
        self.h2_permeation_flux = self.membrane.hydrogen_permeation_flux(
            self.an.cl.species_partial_pressure('h2'),
            self.membrane.temperature,
            self.an.cl.pressure - self.ca.cl.pressure,
            self.membrane.water_vol_fraction(
                self.membrane.water_content,
                water_molar_volume(self.membrane.temperature)))

        self.crossover_current = self.h2_permeation_flux * (2 * ct.faraday)

        unity_activity = 1.0
        tafel_overpotential_ca = self.ca.cl.activation_overpotential(self.current_density, unity_activity)
        tafel_overpotential_an = self.an.cl.activation_overpotential(self.current_density, unity_activity)

        return tafel_overpotential_ca + tafel_overpotential_an


    def set_conditions(self, stack_temperature, current_density, cathode_conditions, anode_conditions):
        """
        Set the operating conditions of the electrolyzer stack.

        Parameters
        ----------
        stack_temperature : float
            Operating temperature in K.
        current_density : float
            Current density in A/m².
        cathode_conditions : OperatingConditions
            Inlet conditions at the cathode.
        anode_conditions : OperatingConditions
            Inlet conditions at the anode.
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
                    component.gas.X = np.ones_like(self.current_density[..., np.newaxis]) * np.array([1, 0, 0, 0])
                except TypeError:
                    component.gas.X = np.array([1, 0, 0, 0])
                component.set_gas_composition(
                    conditions.dry_o2_mole_fraction,
                    conditions.dry_h2_mole_fraction,
                    conditions.inlet_relative_humidity)
                component.set_gas_temperature_and_pressure(conditions.inlet_temperature, conditions.inlet_pressure)

                component.set_gas_temperature_and_pressure(stack_temperature, conditions.inlet_pressure)

            cell_side.ch.set_fixed_inlet_liquid_flow_rate(conditions.inlet_liquid_flow_rate)
            cell_side.ch.set_fixed_inlet_gas_flow_rate(conditions.inlet_gas_flow_rate)

            for component in cell_side.components:
                component.electrolyte = cell_side.electrolyte
                component.non_wetting_saturation = cell_side.ch.inlet_liquid_saturation
