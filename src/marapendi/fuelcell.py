"""
Module providing a fuel cell class intended to be the base class for different fuel cell models. 
"""
from dataclasses import dataclass, field
from scipy.optimize import root
import numpy as np 
import cantera as ct

from .electrochemistry import calculate_reversible_cell_voltage, h2_lhv, STD_PRESSURE
from .porous_layers import PorousLayer
from .catalyst_layers import PtCCatalystLayer
from .flow_channels import FlowChannel
from .membrane import Membrane
from .gas_composition import species_indexes 
from .water import water_molar_volume


from dataclasses import dataclass, field

@dataclass
class FuelCellSide:
    """
    A class representing one side (anode or cathode) of a PEM fuel cell, 
    composed of porous layers (catalyst layer, microporous layer, gas diffusion layer)
    and flow channels. This class provides methods to calculate transport and 
    thermal resistances, as well as water management parameters.

    Attributes
    ----------
    cl : CatalystLayer
        Catalyst layer object (defaults to PtCCatalystLayer).
    gdl : PorousLayer
        Gas diffusion layer object.
    mpl : PorousLayer, optional
        Microporous layer object.
    ch : FlowChannel
        Flow channel object for gas transport.
    has_mpl : bool
        Whether this side includes a microporous layer.
    membrane_surface_water_content : float
        Water content at the membrane surface (lambda).
    thermal_contact_resistance : float
        Additional thermal contact resistance at interfaces.

    Derived Attributes
    ------------------
    porous_layers : list of PorousLayer
        Ordered list of porous layers (includes mpl if has_mpl is True).
    components : list
        List of all components including porous layers and channel.
    o2_transport_resistance : float
        Cached value for O2 transport resistance (initialized as 0).
    h2ov_transport_resistance : float
        Cached value for H2O vapor transport resistance (initialized as 0).
    h2_transport_resistance : float
        Cached value for H2 transport resistance (initialized as 0).
    liquid_water_flux : float 
        Cached value for liquid water flux from the catalyst layer to the channel (initialized as 0).
    """

    cl: PorousLayer = field(default_factory=PtCCatalystLayer) 
    gdl: PorousLayer = field(default_factory=PorousLayer)
    mpl: PorousLayer = field(default_factory=PorousLayer)
    ch: FlowChannel = field(default_factory=FlowChannel)
    has_mpl: bool = False
    membrane_surface_water_content: float = 0 
    thermal_contact_resistance: float = 0 

    def __post_init__(self): 
        """
        Post-initialization to update the ordered list of porous layers and all components.
        Sets up porous_layers and components lists based on whether MPL is present.
        """
        self.porous_layers = [self.cl, self.mpl, self.gdl] if self.has_mpl else [self.cl, self.gdl]
        self.components = self.porous_layers + [self.ch]
        self.o2_transport_resistance = 0
        self.h2ov_transport_resistance = 0
        self.h2_transport_resistance = 0   
        self.liquid_water_flux = 0

    def set_catalyst_layer(self, cl): 
        """
        Set a new catalyst layer and update internal component structure.

        Parameters
        ----------
        cl : PorousLayer
            The new catalyst layer object.
        """
        self.cl = cl 
        self.__post_init__()
    
    def set_gas_diffusion_layer(self, gdl): 
        """
        Set a new gas diffusion layer and update internal component structure.

        Parameters
        ----------
        gdl : PorousLayer
            The new gas diffusion layer object.
        """
        self.gdl = gdl
        self.__post_init__()
    
    def set_channel(self, ch): 
        """
        Set a new flow channel and update internal component structure.

        Parameters
        ----------
        ch : FlowChannel
            The new flow channel object.
        """
        self.ch = ch 
        self.__post_init__()

    def gas_transport_resistance(self, species, ionomer_water_content=11):
        """
        Calculate the gas transport resistance for a given species.

        Parameters
        ----------
        species : str
            The gas species ('o2', 'h2o', etc.) for which to compute resistance.
        ionomer_water_content : float, optional
            Water content in the ionomer (lambda), relevant for O2 film resistance (default is 11).

        Returns
        -------
        float
            The total gas transport resistance, including porous layers,
            channel, and O2 ionomer film resistance if applicable.
        """
        return (sum(layer.gas_transport_resistance(species) for layer in self.porous_layers) +
                self.ch.gas_transport_resistance(species) + 
                (self.cl.o2_ionomer_film_resistance(ionomer_water_content, self.cl.temperature) if species == 'o2' else 0))

    def heat_transfer_resistance(self): 
        """
        Calculate the total heat transfer resistance across this side.

        Returns
        -------
        float
            Total heat transfer resistance, summing all porous layers and 
            the thermal contact resistance.
        """
        return sum(layer.thermal_resistance() for layer in self.porous_layers if layer != self.cl) + self.thermal_contact_resistance
                
    def calculate_water_saturation(self): 
        """
        Compute and update the water saturation in each porous layer.

        Notes
        -----
        This method updates each layer's `water_saturation` attribute based on 
        the liquid flux and upstream capillary pressures. It assumes that 
        `self.liquid_flux` has been set externally.
        """
        self.gdl.liq_transport_model.calculate_water_saturation(self.gdl, self.liquid_flux, upstream_capillary_pressure=0)
        if self.has_mpl: 
             self.mpl.liq_transport_model.calculate_water_saturation(self.gdl, self.liquid_flux, upstream_capillary_pressure=self.gdl.downstream_capillary_pressure)
             self.cl.liq_transport_model.calculate_water_saturation(self.cl, self.liquid_flux, upstream_capillary_pressure=self.mpl.downstream_capillary_pressure)
        else:
            self.cl.liq_transport_model.calculate_water_saturation(self.cl, self.liquid_flux, upstream_capillary_pressure=self.gdl.downstream_capillary_pressure)

    def calculate_equivalent_flow_resistance(self): 
        """
        Compute and update equivalent flow resistances for each porous layer.

        Returns
        -------
        float
            The total equivalent flow resistance across all porous layers.

        Notes
        -----
        Updates each layer's `equivalent_flow_resistance` attribute 
        based on its saturation-dependent flow resistance.
        """
        total_equivalent_flow_resistance = 0 
        for k, layer in enumerate(self.porous_layers[::-1]): 
            layer.equivalent_flow_resistance = layer.saturation_flow_resistance
            total_equivalent_flow_resistance += layer.equivalent_flow_resistance
        return total_equivalent_flow_resistance

    def max_water_vapor_removal(self):
        """
        Calculate the maximum removable water vapor concentration.

        Returns
        -------
        float
            The maximum amount of water vapor that can be removed, 
            computed from the difference between the saturation 
            concentration at the CL and the vapor concentration 
            in the gas flow channel, divided by the vapor transport resistance.
        """
        return ((self.cl.saturation_concentration() - self.ch.vapor_concentration()) 
                / self.gas_transport_resistance('h2o'))

@dataclass
class FuelCell: 
    """
    Represents a proton exchange membrane fuel cell (PEMFC).

    This class defines the key components and parameters of a fuel cell, 
    including the anode, cathode, membrane, and various electrical and thermal 
    properties.

    Attributes
    ----------
    cell_area : float
        Active area of a single fuel cell in m².
    cell_number : int
        Number of cells in the fuel cell stack.
    an : FuelCellSide
        Anode side of the fuel cell.
    ca : FuelCellSide
        Cathode side of the fuel cell.
    membrane : Membrane
        Membrane component responsible for proton conduction.
    electrical_resistance : float, optional
        Electrical resistance of the cell in ohms (default is 0).
    h2_permeation_flux : float, optional
        Hydrogen permeation flux through the membrane (default is 0).
    crossover_current : float, optional
        Parasitic current due to hydrogen crossover (default is 0).
    thermal_resistance : float, optional
        Thermal resistance of the membrane electrode assembly (default is 0).
    heat_release_rate : float, optional
        Heat generated by the cell in W/m² (default is 0).
    mea_temperature_increase : float, optional
        Increase in membrane electrode assembly (MEA) temperature (default is 0).
    mea_temperature : float, optional
        Operating temperature of the MEA (default is 0 K).
    use_eq_water_content_for_ionomer : bool, optional
        Flag indicating if ionomer water content is equal to equilibrium water content at 
        the CL. If not, use values of water content at the membrane interface
    """
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
    use_eq_water_content_for_ionomer: bool = True
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
        activity_o2 = self.ca.cl.species_partial_pressure('o2') / STD_PRESSURE
        activity_h2 = self.an.cl.species_partial_pressure('h2') / STD_PRESSURE
        return calculate_reversible_cell_voltage(
            self.ca.cl.temperature,
            activity_o2 ** 0.5 * activity_h2
        )
    
    def activation_overpotential(self, theta_PtO=0): 
        """
        Compute the activation overpotential of the fuel cell.

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
        omega_PtO_voltage_drop = self.ca.cl.omega_PtO * theta_PtO / (self.ca.cl.reaction.number_of_electrons *
                                                                      self.ca.cl.reaction.charge_transfer_coeff * ct.faraday)
        tafel_overpotential = self.ca.cl.reaction.tafel_overpotential(
            (self.current_density + self.crossover_current) / (self.ca.cl.ecsa * self.ca.cl.platinum_loading * (1-theta_PtO)),
            self.ca.cl.temperature,
            self.ca.cl.species_partial_pressure('o2')
        )
        return tafel_overpotential + omega_PtO_voltage_drop
    
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
        return self.membrane.proton_resistance(self.membrane.water_content, self.membrane.temperature, water_saturation=self.ca.cl.liquid_saturation) + self.electrical_resistance

    def ohmic_overpotential(self): 
        """
        Compute the ohmic overpotential of the fuel cell.

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
        self.ca.cl.proton_resistance = self.ca.cl.effective_charge_resistance(
            self.current_density, 
            self.ca.cl.ionomer_water_content, 
            self.ca.cl.temperature
        )
        return self.current_density * (self.ca.cl.proton_resistance + self.high_frequency_resistance())

    def reversible_voltage_vs_RHE(self): 
        """
        Compute the reversible cell voltage referenced to the reversible hydrogen electrode (RHE).

        Returns
        -------
        float
            The reversible voltage in volts.

        Notes
        -----
        This function calculates the theoretical reversible voltage assuming a reference 
        hydrogen pressure of 1 bar (100 kPa). It provides a useful reference for evaluating 
        fuel cell performance.
        """
        return calculate_reversible_cell_voltage(
            self.ca.cl.temperature,
            self.ca.cl.species_partial_pressure('o2') / STD_PRESSURE
        )
    def calculate_theta_PtO(self): 
        """
        Calculate the coverage of platinum oxide (θ_PtO) on the cathode catalyst layer.

        Returns
        -------
        float
            The coverage of platinum oxide (θ_PtO).

        Notes
        -----
        This function determines the fraction of the platinum catalyst surface covered 
        by oxygen species, which affects the activation overpotential. The calculation 
        iteratively solves for θ_PtO using a convergence criterion of 0.001 V for the 
        activation overpotential.
        """

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
            
        self.ca.cl.theta_catalyst = theta_PtO

    def cell_voltage(self):
        """
        Compute the cell voltage of the fuel cell.

        Returns
        -------
        float
            The fuel cell voltage in volts.

        Notes
        -----
        The cell voltage is calculated as the difference between the reversible cell voltage 
        and the sum of the activation and ohmic overpotentials. The function first calculates 
        the platinum oxide coverage (θ_PtO), which is used in the activation overpotential 
        calculation.
        """ 
        self.calculate_theta_PtO()
        E_rev = self.reversible_cell_voltage()
        eta_ohm = self.ohmic_overpotential()
        eta_act = self.activation_overpotential(self.ca.cl.theta_catalyst)
        return np.maximum(0, E_rev - eta_act - eta_ohm)
    
    def set_mea_temperature(self, mea_temperature): 
        """
        Set the membrane electrode assembly (MEA) temperature and update associated components.
        The underlying hypotheses is that the membrane and the CLs are at the
        same temperature.

        Parameters
        ----------
        mea_temperature : float
            The new temperature of the MEA in Kelvin.

        Notes
        -----
        This method updates the temperature of various fuel cell components, including:
        - Cathode catalyst layer (CL)
        - Anode catalyst layer (CL)
        - Membrane

        It also calculates the temperature increase of the MEA relative to the initial temperature.
        """
        self.mea_temperature = mea_temperature
        self.ca.cl.set_gas_temperature(mea_temperature)
        self.an.cl.set_gas_temperature(mea_temperature)

        # self.ca.gdl.gas.set_temperature(mea_temperature)
        self.membrane.temperature = mea_temperature
        self.mea_temperature_increase = self.mea_temperature - self.temperature
    
    def calculate_water_transport(self): 
        """
        Calculate the water balance across the fuel cell components.

        This method updates water vapor transport resistance for both the cathode 
        and anode, computes the membrane water balance, and recalculates the 
        equivalent flow resistance and water saturation in the cathode.

        Notes
        -----
        - The water transport resistance for water vapor (`h2o`) is computed for both 
        the cathode (`ca`) and anode (`an`).
        - The membrane water balance is updated using the defined water balance model.
        - The equivalent flow resistance and water saturation in the cathode are recalculated.
        - Finally, the water vapor transport resistance values are updated again for consistency.
        """
        
        self.ca.h2ov_transport_resistance = self.ca.gas_transport_resistance('h2o')
        self.an.h2ov_transport_resistance = self.an.gas_transport_resistance('h2o')
        self.membrane.water_balance_model.solve_water_balance(self)
        for cl in (self.ca.cl, self.an.cl): 
            if self.use_eq_water_content_for_ionomer: 
                cl.ionomer_water_content = cl.eq_water_content
            else: 
                cl.ionomer_water_content = cl.memb_interface_water_content
            cl.set_ionomer_wet_properties(cl.ionomer_water_content, cl.temperature)

        self.ca.calculate_equivalent_flow_resistance()
        self.ca.calculate_water_saturation()
        self.ca.cl.set_water_film_thickness(self.ca.cl.liquid_saturation)
        self.ca.h2ov_transport_resistance = self.ca.gas_transport_resistance('h2o')
        self.an.h2ov_transport_resistance = self.an.gas_transport_resistance('h2o')
    
    def calculate_gas_concentrations_at_cl(self): 
        """
        Calculate gas concentrations at the catalyst layer (CL) for both cathode and anode.

        This method updates the gas concentrations of reactants (O2, H2), water (H2O), 
        and nitrogen (N2) in the catalyst layer based on transport resistances and consumption.

        The calculation is performed for both the cathode (ca) and anode (an) sides.

        Updates:
            - Reactant concentrations (O2, H2) considering gas transport resistance.
            - Water concentration using vapor flux and transport resistance.
            - Nitrogen concentration from species balance.
            - Computes the relative humidity at the catalyst layer.

        Notes:
            - Uses `side.gas_transport_resistance()` to compute reactant transport resistance.
            - Ensures gas mole fractions remain non-negative using `np.maximum(1e-12, value)`.
            - `species_indexes` is assumed to be a predefined mapping of species names to indices.
        """
        
        for side, reactant in [(self.ca, 'o2'), (self.an, 'h2')]:
            # reactant concentration
            side.reactant_transport_resistance = side.gas_transport_resistance(reactant, self.ca.cl.ionomer_water_content)
            gas_concentration = side.cl.gas.concentration()

            side.cl.gas.X[...,species_indexes[reactant]] = np.maximum(1e-12, 
                side.ch.species_concentration(reactant) - 
                side.reactant_consumption * side.reactant_transport_resistance) / gas_concentration
            
            # water concentration
            side.cl.gas.X[...,species_indexes['h2o']] = np.maximum(1e-12,
                side.ch.species_concentration('h2o') +
                side.vapor_flux * side.h2ov_transport_resistance) / gas_concentration
            side.cl.gas.calculate_relative_humidity()

            # n2 concentration (from species balance)
            side.cl.gas.X[...,species_indexes['n2']] = (1 - side.cl.gas.X[...,species_indexes['o2']] -
                                                        side.cl.gas.X[...,species_indexes['h2o']] -
                                                        side.cl.gas.X[...,species_indexes['h2']])
    
    def calculate_heat_transfer_resistance(self): 
        """
        Calculate the overall heat transfer resistance of the membrane electrode assembly (MEA).

        Notes
        -----
        The heat transfer resistance is determined as the inverse of the sum of the 
        reciprocals of the individual heat transfer resistances of the cathode and anode sides.
        This calculation assumes a thermal resistance network in parallel.
        """
        self.thermal_resistance = 1/sum(1./side.heat_transfer_resistance() for side in (self.ca, self.an))

    def calculate_heat_transport(self): 
        """
        Compute the heat transport parameters in the fuel cell.

        Notes
        -----
        This function first calculates the total heat transfer resistance and then determines 
        the rate of heat release due to electrochemical reactions and irreversibilities. 
        The MEA temperature increase is then estimated based on the thermal resistance 
        and heat release rate.

        The heat release rate is computed using the lower heating value (LHV) of hydrogen 
        and the cell voltage, considering the electrochemical reaction energy balance.
        """ 
        self.calculate_heat_transfer_resistance()
        self.heat_release_rate = (-h2_lhv(self.temperature) / (2 * ct.faraday) - self.cell_voltage()) * self.current_density
        self.mea_temperature_increase = self.heat_release_rate * self.thermal_resistance

    def solve_transport(self):
        def f(dT): 
            mea_temperature = np.minimum(self.temperature + np.maximum(dT, 0), 373.15) 
            self.set_mea_temperature(mea_temperature)
            self.calculate_water_transport()
            self.calculate_gas_concentrations_at_cl()
            self.calculate_heat_transport()
            return dT - self.mea_temperature_increase 
        
        self.calculate_heat_transfer_resistance()
        res = root(f, 1.4 * self.current_density * self.thermal_resistance, method='broyden1', options={'fatol':1e-1, 'maxiter':5})
    
    def compute_ui_curve(self, current_density, stack_temperature, cathode_conditions, anode_conditions, model='explicit_steady_state'):
        """
        Calculation of polarization curve for given operating conditions.

        Parameters
        ----------
        current_density : float
            The current density of the cell in A/m².
        stack_temperature : float
            The operating temperature of the fuel cell stack in Kelvin.
        cathode_conditions : OperatingConditions
            The inlet conditions at the cathode side, including temperature, 
            pressure, oxygen mole fraction, and relative humidity.
        anode_conditions : OperatingConditions
            The inlet conditions at the anode side, including temperature, 
            pressure, hydrogen mole fraction, and relative humidity.
        model : string
            The model to be used. Only 'explicit_steady_state' supported for now.

        Returns
        -------
        float
            The fuel cell voltage in volts.
        """
        self.set_conditions(stack_temperature, current_density,cathode_conditions, anode_conditions)
        if model == 'explicit_steady_state': 
            return self.explicit_steady_state_model()
        
    def explicit_steady_state_model(self): 
        """
        Simplified steady-state model where all calculations are explicit. 
        
        The main hypotheses are: 
        - The fuel cell HHV efficiency is constant; 
        (MEA temperature increases linearly with current density)
        - The membrane water transport is calculated considering dry conditions 
        (Gas transport resistances or membrane water sorption/balance are not impacted by 
        the presence of liquid water)
        - Water saturation is given by a power law of the liquid water flux

        Returns
        -------
        float
            The fuel cell voltage in volts.
        """
        self.calculate_heat_transfer_resistance()
        mea_temperature = self.temperature + (self.current_density * 0.7) * self.thermal_resistance
        self.set_mea_temperature(mea_temperature)
        self.calculate_water_transport()
        self.calculate_gas_concentrations_at_cl()
        return self.cell_voltage()
    
    def set_conditions(self, stack_temperature, current_density, cathode_conditions, anode_conditions):  
        """
        Set the operating conditions of the fuel cell stack.

        This method initializes key operating parameters such as current density, 
        reactant consumption rates, water production, and temperature. It also 
        updates the gas composition and flow conditions for the cathode and anode.

        Parameters
        ----------
        stack_temperature : float
            The operating temperature of the fuel cell stack in Kelvin.
        current_density : float
            The current density of the cell in A/m².
        cathode_conditions : OperatingConditions
            The inlet conditions at the cathode side, including temperature, 
            pressure, oxygen mole fraction, and relative humidity.
        anode_conditions : OperatingConditions
            The inlet conditions at the anode side, including temperature, 
            pressure, hydrogen mole fraction, and relative humidity.

        Notes
        -----
        - The oxygen and hydrogen consumption rates are calculated based on 
        the current density.
        - The membrane and stack temperatures are updated.
        - The gas composition and saturation levels for cathode and anode 
        components are initialized.
        - The gas flow rate at the channel inlet is determined based on the 
        stoichiometric ratio and reactant consumption.
        """
        self.current_density = current_density
        self.o2_consumption = current_density / (4 * ct.faraday)
        self.h2_consumption = 2 * self.o2_consumption
        self.ca.reactant_consumption = self.o2_consumption
        self.an.reactant_consumption = self.h2_consumption
        self.h2o_production = self.h2_consumption
        self.temperature = stack_temperature
        self.membrane.temperature = stack_temperature

        for side in (self.ca, self.an): 
            for layer in side.components: 
                layer.liquid_saturation = 0 
            side.cl.set_water_film_thickness(0)
            
        for cell_side, conditions in zip((self.ca, self.an), (cathode_conditions, anode_conditions)): 
            cell_side.cl.ionomer_water_content = 10
            cell_side.cl.set_ionomer_wet_properties(cell_side.cl.ionomer_water_content, self.temperature)
            for component in cell_side.components: 
                cell_side.electrolyte = conditions.inlet_liquid
                cell_side.electrolyte.set_temperature(self.membrane.temperature)
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
            
            cell_side.ch.set_inlet_gas_flow_rate_from_stoichiometry(
                (self.o2_consumption if cell_side == self.ca else self.h2_consumption) * self.cell_area, conditions.stoichiometry, 
                fixed_inlet_gas_flow_rate=conditions.inlet_gas_flow_rate
            )
            for component in cell_side.components: 
                component.electrolyte = cell_side.electrolyte
                component.liquid_saturation = cell_side.ch.inlet_liquid_saturation
            

from .electrolyte import ElectrolyteSolution

@dataclass
class OperatingConditions:
    """
    Represents the operating conditions of a fuel cell system side (anode/cathode).

    Attributes
    ----------
    inlet_temperature : float
        The temperature of the gas at the inlet (K).
    inlet_pressure : float
        The pressure at the inlet (Pa). Defaults to outlet_pressure if not provided.
    outlet_pressure : float
        The pressure at the outlet (Pa). Defaults to inlet_pressure if not provided.
    dry_o2_mole_fraction : float
        The mole fraction of oxygen in the dry gas mixture.
    dry_h2_mole_fraction : float
        The mole fraction of hydrogen in the dry gas mixture.
    inlet_relative_humidity : float
        The relative humidity of the gas at the inlet (0 to 1).
    stoichiometry : float
        The stoichiometric ratio of reactant.
    average_pressure : float
        The average pressure between inlet and outlet (Pa).
    inlet_liquid_saturation : float 
        The volume fraction of the liquid phase at the inlet. Defaults to zero. 
    inlet_liquid : ElectrolyteSolution
        The nature of the liquid phase.
    inlet_liquid_volume_flow_rate : float 
        The inlet liquid flow rate (m3/s).
    inlet_liquid_volume_flow_rate : float 
        The inlet liquid flow rate (m3/s).
    inlet_gas_volume_flow_rate : float 
        The inlet liquid flow rate (m3/s).
    """
    inlet_temperature: float = 353.15
    inlet_pressure: float = None
    outlet_pressure: float = None
    dry_o2_mole_fraction: float = 0.2
    dry_h2_mole_fraction: float = 0.0
    inlet_relative_humidity: float = 0.5
    stoichiometry: float = 2
    inlet_liquid_saturation: float = 0
    inlet_liquid: ElectrolyteSolution = field(default_factory=ElectrolyteSolution)
    inlet_liquid_flow_rate: float = 0
    inlet_gas_flow_rate: float = 0

    def __post_init__(self):
        if self.inlet_pressure is None:
            self.inlet_pressure = self.outlet_pressure if self.outlet_pressure is not None else 101325.0
        if self.outlet_pressure is None:
            self.outlet_pressure = self.inlet_pressure
        
        self.average_pressure = 0.5 * (self.inlet_pressure + self.outlet_pressure)


