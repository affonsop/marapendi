"""
Module providing a fuel cell class intended to be the base class for different fuel cell models.
"""
from dataclasses import dataclass, field
from scipy.optimize import root, least_squares
import numpy as np
import cantera as ct

from .electrochemistry import calculate_reversible_cell_voltage, h2_lhv, STD_PRESSURE
from .porous_layers import PorousLayer
from .catalyst_layers import PtCCatalystLayer
from .flow_channels import FlowChannel
from .membrane import Membrane
from .gas_composition import species_indexes
from .cell import Cell
from .voltage import VoltageModel
from .thermal import ThermalModel
from .model import ExplicitSteadyStateModel

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
    is_wet: bool 
        Whether this side is flooded with water (used for water electrolysis). 
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
    liquid_flux : float 
        Cached value for liquid water flux from the catalyst layer to the channel (initialized as 0).
    """

    cl: PorousLayer = field(default_factory=PtCCatalystLayer) 
    gdl: PorousLayer = field(default_factory=PorousLayer)
    mpl: PorousLayer = field(default_factory=PorousLayer)
    ch: FlowChannel = field(default_factory=FlowChannel)
    has_mpl: bool = False
    has_gdl: bool = True
    is_wet: bool = False
    membrane_surface_water_content: float = 0 
    thermal_contact_resistance: float = 0 
    is_liquid_equilibrated: bool = False

    def __post_init__(self): 
        """
        Post-initialization to update the ordered list of porous layers and all components.
        Sets up porous_layers and components lists based on whether MPL is present.
        """
        self.porous_layers = ([self.cl, self.mpl] if self.has_mpl else [self.cl]) + ([self.gdl] if self.has_gdl else [])
        self.components = self.porous_layers + [self.ch]
        self.layers = self.components  # alias used by Cell.__post_init__
        self.o2_transport_resistance = 0
        self.h2ov_transport_resistance = 0
        self.h2_transport_resistance = 0   
        self.liquid_flux = 0
        self.gas_flux = 0
        self.s_relax = None
            
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
        return (sum(layer.gas_transport_resistance(layer, species) for layer in self.porous_layers) +
                self.ch.gas_transport_resistance(self.ch, species) +
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
        return sum(
            (layer.thermal_resistance() for layer in self.porous_layers if layer != self.cl) 
            if self.has_gdl else
            (layer.thermal_resistance() for layer in self.porous_layers)
            ) + self.thermal_contact_resistance
                
    def calculate_water_saturation(self): 
        """
        Compute and update the water saturation in each porous layer.

        Notes
        -----
        This method updates each layer's `water_saturation` attribute based on 
        the liquid flux and upstream capillary pressures. It assumes that 
        `self.liquid_flux` has been set externally.
        """
        for i, layer in enumerate(self.porous_layers): 
            layer.non_wetting_flux = self.liquid_flux if layer.contact_angle > 90 else self.gas_flux
            layer.downstream_saturation = np.zeros_like(layer.non_wetting_flux)
            layer.upstream_saturation = np.zeros_like(layer.non_wetting_flux)
            layer.non_wetting_saturation = np.zeros_like(layer.non_wetting_flux)
            layer.downstream_capillary_pressure = np.zeros_like(layer.non_wetting_flux)
        if self.has_gdl: 
            self.gdl.two_phase_transport_model.calculate_non_wetting_saturation(self.gdl, self.gdl.non_wetting_flux, upstream_capillary_pressure=np.zeros_like(self.cl.non_wetting_flux))
            if self.has_mpl: 
                self.mpl.two_phase_transport_model.calculate_non_wetting_saturation(self.mpl, self.mpl.non_wetting_flux, upstream_capillary_pressure=self.gdl.downstream_capillary_pressure)
                self.cl.two_phase_transport_model.calculate_non_wetting_saturation(self.cl, self.liquid_flux, upstream_capillary_pressure=self.mpl.downstream_capillary_pressure)
            else:
                self.cl.two_phase_transport_model.calculate_non_wetting_saturation(self.cl, self.liquid_flux, upstream_capillary_pressure=self.gdl.downstream_capillary_pressure)
        else: 
            self.cl.two_phase_transport_model.calculate_non_wetting_saturation(self.cl, self.cl.non_wetting_flux, upstream_capillary_pressure=np.zeros_like(self.cl.non_wetting_flux))
        for layer in self.porous_layers: 
            layer.liquid_saturation = layer.non_wetting_saturation if layer.contact_angle > 90. else (1-layer.non_wetting_saturation)
            layer.electrolyte_saturation = layer.liquid_saturation

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
        total_equivalent_flow_resistance = np.zeros_like(
            self.porous_layers[0].saturation_flow_resistance
        )
        for k, layer in enumerate(self.porous_layers[::-1]): 
            layer.equivalent_flow_resistance = layer.saturation_flow_resistance
            total_equivalent_flow_resistance += layer.equivalent_flow_resistance
        return total_equivalent_flow_resistance

    def max_water_vapor_removal(self):
        """
        Calculate the maximum removable water vapor rate.

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
class FuelCell(Cell):
    """
    Proton exchange membrane fuel cell: a :class:`~marapendi.cell.Cell` with
    the explicit-steady-state physics attached.

    Static geometry (``ca``, ``an``, ``membrane``, ``area``,
    ``electrical_resistance``) is inherited from :class:`~marapendi.cell.Cell`.
    Runtime state (temperatures, fluxes, voltages) is stored directly on this
    object and its component instances, to be migrated to
    :class:`~marapendi.state.CellState` in a later refactoring step.

    Model orchestration is delegated to
    :class:`~marapendi.model.ExplicitSteadyStateModel` and
    :class:`~marapendi.voltage.VoltageModel`.
    """

    # Override parent field types to accept FuelCellSide
    ca: FuelCellSide = field(default_factory=FuelCellSide)
    an: FuelCellSide = field(default_factory=FuelCellSide)
    membrane: Membrane = field(default_factory=Membrane)

    # FuelCell-specific fields (all optional so Cell fields keep their defaults)
    cell_number: int = 1
    h2_permeation_flux: float = 0.
    crossover_current: float = 0.
    thermal_resistance: float = 0.
    heat_release_rate: float = 0.
    mea_temperature_increase: float = 0.
    mea_temperature: float = 0.
    mea_surface_heat_capacity: float = 10000.
    use_eq_water_content_for_ionomer: bool = True

    def __post_init__(self):
        super().__post_init__()   # builds porous_layers, layers, sides from Cell
        self._voltage_model = VoltageModel()
        self._thermal_model = ThermalModel()
        self._model = ExplicitSteadyStateModel(self._voltage_model, self._thermal_model)

    # ------------------------------------------------------------------
    # Voltage methods — delegate to VoltageModel
    # ------------------------------------------------------------------

    def reversible_cell_voltage(self):
        return self._voltage_model.reversible_cell_voltage(self)

    def reversible_voltage_vs_RHE(self):
        return self._voltage_model.reversible_voltage_vs_RHE(self)

    def activation_overpotential(self, theta_PtO=0):
        return self._voltage_model.activation_overpotential(self, theta_PtO)

    def high_frequency_resistance(self):
        return self._voltage_model.high_frequency_resistance(self)

    def ohmic_overpotential(self):
        return self._voltage_model.ohmic_overpotential(self)

    def calculate_theta_PtO(self):
        return self._voltage_model.calculate_theta_PtO(self)

    def calculate_cell_voltage(self):
        return self._voltage_model.compute_cell_voltage(self)
    
    def set_mea_temperature(self, mea_temperature):
        """Delegate to :class:`~marapendi.thermal.ThermalModel`."""
        self._thermal_model.set_mea_temperature(mea_temperature, self)

    def calculate_water_transport(self, dynamic=False): 
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
       
        if not dynamic: 
            self.ca.calculate_equivalent_flow_resistance()
            self.ca.calculate_water_saturation()
            self.ca.cl.set_water_film_thickness(self.ca.cl.non_wetting_saturation)
            self.ca.h2ov_transport_resistance = self.ca.gas_transport_resistance('h2o')
            self.an.h2ov_transport_resistance = self.an.gas_transport_resistance('h2o')
    
        for cl in (self.ca.cl, self.an.cl): 
            if self.use_eq_water_content_for_ionomer: 
                cl.ionomer_water_content = cl.eq_water_content
            else: 
                cl.ionomer_water_content = cl.memb_interface_water_content
            cl.set_ionomer_wet_properties(cl.ionomer_water_content, cl.temperature)

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
            side.cl.gas.X[...,species_indexes['h2o']] = np.where(side.cl.liquid_saturation > 0, 
                side.cl.saturation_concentration()  / gas_concentration,                                    
                np.maximum(1e-12,
                side.ch.species_concentration('h2o') +
                side.vapor_flux * side.h2ov_transport_resistance) / gas_concentration)
            side.cl.gas.calculate_relative_humidity()

            # n2 concentration (from species balance)
            side.cl.gas.X[...,species_indexes['n2']] = (1 - side.cl.gas.X[...,species_indexes['o2']] -
                                                        side.cl.gas.X[...,species_indexes['h2o']] -
                                                        side.cl.gas.X[...,species_indexes['h2']])
    
    def calculate_heat_transfer_resistance(self):
        """Delegate to :class:`~marapendi.thermal.ThermalModel`."""
        self.thermal_resistance = self._thermal_model.heat_transfer_resistance(self)
    
    def calculate_heat_transport(self, dynamic=False): 
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
        self.heat_release_rate = (-h2_lhv(self.temperature) / (2 * ct.faraday) - self.cell_voltage) * self.current_density
        if not dynamic: 
            self.mea_temperature_increase = self.heat_release_rate * self.thermal_resistance

    def solve_transport(self):
        def f(dT): 
            mea_temperature = np.minimum(self.temperature + np.maximum(dT, 0), 373.15) 
            self.set_mea_temperature(mea_temperature)
            self.calculate_water_transport()
            self.calculate_gas_concentrations_at_cl()
            self.calculate_cell_voltage()
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
        elif model == 'explicit_steady_state_with_mea_temp_estimation': 
            return self.explicit_steady_state_model(mea_tempearture_estimation=True)
        elif model == 'steady_state_with_temp_solution': 
            self.solve_transport()
            return self.cell_voltage()
        
    def explicit_steady_state_model(self, mea_tempearture_estimation=False):
        """
        Simplified steady-state model where all calculations are explicit.

        Delegates to :class:`~marapendi.model.ExplicitSteadyStateModel`.
        """
        return self._model.solve(self, mea_temperature_estimation=mea_tempearture_estimation)

    def temperature_rate_of_change(self):
        self.calculate_heat_transport(dynamic=True) 
        return (self.heat_release_rate -  
                (self.mea_temperature_increase / self.thermal_resistance)) / self.mea_surface_heat_capacity
    
    def membrane_water_rate_of_change(self, water_profile, n_membrane_mesh):        
        return (self.membrane.water_balance_model.membrane_water_net_flux /
                   (self.membrane.surface_concentration / n_membrane_mesh))
    
    def saturation_rate_of_change(self): 
        dsdt = []
        for side in (self.an,self.ca): 
            for layer in side.porous_layers: 
                layer.flow_resistance_with_rel_permeability = layer.saturation_flow_resistance * layer.capillary_pressure_J_ratio / np.maximum(layer.non_wetting_saturation,1e-1) ** (layer.relative_permeability_exponent + 2)
                layer.capillary_pressure = layer.capillary_pressure_from_saturation(layer.non_wetting_saturation)
            
            if side.has_mpl: 
                side.cl_to_mpl_liquid_flux = (2/(side.cl.flow_resistance_with_rel_permeability + side.mpl.flow_resistance_with_rel_permeability) *
                                    (side.cl.capillary_pressure - side.mpl.capillary_pressure))
                side.mpl_to_gdl_liquid_flux = (2/(side.mpl.flow_resistance_with_rel_permeability + side.gdl.flow_resistance_with_rel_permeability) *
                                (side.mpl.capillary_pressure - side.gdl.capillary_pressure))
            else:
                side.cl_to_gdl_liquid_flux = (2/(side.cl.flow_resistance_with_rel_permeability + side.gdl.flow_resistance_with_rel_permeability) *
                                                    (side.cl.capillary_pressure - side.gdl.capillary_pressure))
            side.gdl_to_ch_liquid_flux = (2/side.gdl.flow_resistance_with_rel_permeability *
                                                (side.gdl.capillary_pressure - 0))
            if side.has_mpl: 
                side.cl.liquid_balance = (side.liquid_flux - side.cl_to_gdl_liquid_flux)
                side.mpl.liquid_balance = (side.cl_to_mpl_liquid_flux - side.mpl_to_gdl_liquid_flux)
                side.gdl.liquid_balance = (side.mpl_to_gdl_liquid_flux - side.gdl_to_ch_liquid_flux)
            else:
                side.cl.liquid_balance = (side.liquid_flux - side.cl_to_gdl_liquid_flux)
                side.gdl.liquid_balance = (side.cl_to_gdl_liquid_flux - side.gdl_to_ch_liquid_flux)
        
            for layer in side.porous_layers: 
                dsdt.append(layer.liquid_balance / (layer.porosity * layer.thickness) * self.mea_water_molar_volume)
        return np.array(dsdt)
    
    def relaxation_rate_of_change(self): 
        dxdt = []
        for side in (self.ca, self.an):
            side.t_relax = (self.membrane.relaxation_time_constant * 
                            np.exp((self.membrane.relaxation_time_activation_energy/ct.gas_constant)/self.mea_temperature) / 
                            np.where(side.membrane_water_flux < 0,1.,2.))
            dxdt += [-(side.s_relax - self.membrane.uptake_relaxed_fraction_constant * side.est_water_content)/side.t_relax]
        return np.array(dxdt)
    
    def set_water_saturation_in_porous_layers(self, saturation_profile): 
        k = 0
        for side in (self.an,self.ca): 
            for layer in side.porous_layers:
                layer.non_wetting_saturation = saturation_profile[k,...]
                k+=1
        for side in (self.an,self.ca): 
            side.h2ov_transport_resistance = side.gas_transport_resistance('h2o')
            side.cl.set_water_film_thickness(side.cl.non_wetting_saturation)
        
        
    def f_transient(self, t,x,u,p, n_memb_mesh=3): 
        self.set_conditions_from_input_dict(u,t * np.ones_like(x[0,...]))
        # Get variables 
        k = 1 + n_memb_mesh
        water_profile = x[1:k,...]
        saturation_profile = np.clip(x[k:k+len(self.porous_layers),...],0,0.9)
        self.ca.s_relax = x[-2,...]
        self.an.s_relax = x[-1,...]
        
        # Set conditions
        self.set_mea_temperature(x[0,...])
        self.set_water_saturation_in_porous_layers(saturation_profile)
        self.membrane.water_balance_model.solve_water_balance(self,water_profile,True)
        self.calculate_gas_concentrations_at_cl()
        self.calculate_cell_voltage()

        # Calculate rates of change
        dlmbddt = self.membrane_water_rate_of_change(water_profile, n_memb_mesh)
        dTdt = self.temperature_rate_of_change()
        dsdt = self.saturation_rate_of_change()
        dsrlxdt = self.relaxation_rate_of_change()
        return [dTdt] + list(dlmbddt) + list(dsdt) + list(dsrlxdt)

    def f_relax(self, t,x,u,p, n_memb_mesh=3): 
        self.set_conditions_from_input_dict(u,t * np.ones_like(x[0,...]))
        self.explicit_steady_state_model()

        self.ca.s_relax = x[-2,...]
        self.an.s_relax = x[-1,...]
        
        dsrlxdt = self.relaxation_rate_of_change()
        return list(dsrlxdt)

    def set_conditions_from_input_functions(self, u, t): 
        self.set_conditions_from_input_dict({key: u_function(t) for key, u_function in u.items()})

    def set_conditions_from_input_dict(self, u): 
        current_density = u['current-density']
        stack_temperature = u['cell-temperature']
        # for side in ('ca','an'):
        #     try: 
        #         u[f'{side}-inlet-liquid']
        #     except KeyError: 
        #         u[f'{side}-inlet-liquid'] = lambda t: ElectrolyteSolution()

        cathode_conditions, anode_conditions =  [OperatingConditions(
                inlet_temperature=u[f'{side}-inlet-temperature'],
                inlet_pressure=u[f'{side}-inlet-pressure'],
                outlet_pressure=u[f'{side}-outlet-pressure'],
                inlet_relative_humidity=u[f'{side}-inlet-rh'],      
                stoichiometry=u[f'{side}-stoichiometry'],
                dry_o2_mole_fraction=u[f'{side}-dry-o2-mole-fraction'],
                dry_h2_mole_fraction=u[f'{side}-dry-h2-mole-fraction'],
                inlet_liquid_saturation=u[f'{side}-inlet-liquid-saturation'],
                inlet_gas_flow_rate=u[f'{side}-inlet-gas-flow-rate'],
                inlet_liquid_flow_rate=u[f'{side}-inlet-liquid-flow-rate'],
                inlet_liquid=u[f'{side}-inlet-liquid'] if f'{side}-inlet-liquid' in u.keys() else  ElectrolyteSolution(),
            ) for side in ('ca','an')]
        self.set_conditions(stack_temperature, current_density, cathode_conditions, anode_conditions)


    def set_consumption_production(self, current_density): 
        self.o2_consumption = current_density / (4 * ct.faraday)
        self.h2_consumption = 2 * self.o2_consumption
        self.ca.reactant_consumption = self.o2_consumption
        self.an.reactant_consumption = self.h2_consumption
        self.h2o_production = self.h2_consumption
        self.ca.h2o_production = self.h2o_production
        self.an.h2o_production = np.zeros_like(self.h2o_production)

    def set_flow_rates(self,cathode_conditions, anode_conditions): 
        for cell_side, conditions in zip((self.ca, self.an), (cathode_conditions, anode_conditions)): 
            cell_side.ch.set_fixed_inlet_liquid_flow_rate(conditions.inlet_liquid_flow_rate)
            cell_side.ch.set_inlet_gas_flow_rate_from_stoichiometry(
                (self.o2_consumption if cell_side == self.ca else self.h2_consumption) * self.area, conditions.stoichiometry,
                fixed_inlet_gas_flow_rate=conditions.inlet_gas_flow_rate
            )

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
        self.set_consumption_production(current_density)

        self.temperature = stack_temperature
        self.membrane.temperature = stack_temperature

        for side in (self.ca, self.an): 
            for layer in side.components: 
                layer.non_wetting_saturation = np.zeros_like(self.current_density)
                layer.liquid_saturation = np.zeros_like(self.current_density)
                layer.temperature = stack_temperature
            side.cl.set_water_film_thickness(0)
            side.is_liquid_equilibrated = False

        for cell_side, conditions in zip((self.ca, self.an), (cathode_conditions, anode_conditions)): 
            cell_side.cl.ionomer_water_content = 10
            cell_side.cl.set_ionomer_wet_properties(cell_side.cl.ionomer_water_content, self.temperature)
            cell_side.electrolyte = conditions.inlet_liquid
            cell_side.electrolyte.set_temperature(self.membrane.temperature)

            for component in cell_side.components: 
   
                cd = np.atleast_1d(self.current_density)
                component.gas.X = np.zeros((cd.size, 4))
                component.gas.X[:, 0] = 1.0
                                
                component.set_gas_temperature_and_pressure(conditions.inlet_temperature, conditions.inlet_pressure)
                component.set_gas_composition(conditions.dry_o2_mole_fraction, 
                                              conditions.dry_h2_mole_fraction,
                                              conditions.inlet_relative_humidity)
                component.set_gas_temperature_and_pressure(conditions.inlet_temperature, conditions.inlet_pressure)
                # component.set_gas_composition(conditions.dry_o2_mole_fraction, 
                #                               conditions.dry_h2_mole_fraction,
                #                               conditions.inlet_relative_humidity)
                component.set_gas_temperature_and_pressure(stack_temperature, conditions.inlet_pressure)
            
        self.set_flow_rates(cathode_conditions, anode_conditions)
        for cell_side in (self.ca, self.an):
            for component in cell_side.components: 
                component.electrolyte = cell_side.electrolyte
                if component.contact_angle < 90: 
                    component.non_wetting_saturation = 1-cell_side.ch.inlet_liquid_saturation * np.ones_like(self.current_density)
                else: 
                    component.non_wetting_saturation = cell_side.ch.inlet_liquid_saturation * np.ones_like(self.current_density)
            

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



class DynamicOperatingConditions:
    """
    Represents the operating conditions of a fuel cell system side (anode/cathode), considering 
    dynamic data as a function of time.

    Attributes
    ----------
    inlet_temperature : function
        The temperature of the gas at the inlet (K).
    inlet_pressure : function
        The pressure at the inlet (Pa). Defaults to outlet_pressure if not provided.
    outlet_pressure : function
        The pressure at the outlet (Pa). Defaults to inlet_pressure if not provided.
    dry_o2_mole_fraction : function
        The mole fraction of oxygen in the dry gas mixture.
    dry_h2_mole_fraction : function
        The mole fraction of hydrogen in the dry gas mixture.
    inlet_relative_humidity : function
        The relative humidity of the gas at the inlet (0 to 1).
    stoichiometry : function
        The stoichiometric ratio of reactant.
    average_pressure : function
        The average pressure between inlet and outlet (Pa).
    inlet_liquid_saturation : function 
        The volume fraction of the liquid phase at the inlet. Defaults to zero. 
    inlet_liquid : ElectrolyteSolution
        The nature of the liquid phase.
    inlet_liquid_volume_flow_rate : function 
        The inlet liquid flow rate (m3/s).
    inlet_liquid_volume_flow_rate : function 
        The inlet liquid flow rate (m3/s).
    inlet_gas_volume_flow_rate : function 
        The inlet liquid flow rate (m3/s).
    """


    def __init__(
                self, 
                inlet_temperature = lambda t: 353.15,
                inlet_pressure = None,
                outlet_pressure =  None,
                dry_o2_mole_fraction = lambda t: 0.2,
                dry_h2_mole_fraction = lambda t: 0.0,
                inlet_relative_humidity = lambda t: 0.5,
                stoichiometry = lambda t: 2,
                inlet_liquid_saturation = lambda t: 0,
                inlet_liquid = ElectrolyteSolution(),
                inlet_liquid_flow_rate = lambda t: 0,
                inlet_gas_flow_rate = lambda t: 0
            ):
        self.inlet_temperature = inlet_temperature
        self.inlet_pressure = inlet_pressure
        self.outlet_pressure =  outlet_pressure
        self.dry_o2_mole_fraction = dry_o2_mole_fraction
        self.dry_h2_mole_fraction = dry_h2_mole_fraction
        self.inlet_relative_humidity = inlet_relative_humidity
        self.stoichiometry = stoichiometry
        self.inlet_liquid_saturation = inlet_liquid_saturation
        self.inlet_liquid = inlet_liquid
        self.inlet_liquid_flow_rate = inlet_liquid_flow_rate
        self.inlet_gas_flow_rate = inlet_gas_flow_rate
        
        if self.inlet_pressure is None:
            self.inlet_pressure = self.outlet_pressure if self.outlet_pressure is not None else lambda t: 101325.0
        if self.outlet_pressure is None:
            self.outlet_pressure = self.inlet_pressure

    def get_operating_conditions(self, t): 
        return OperatingConditions(
                self.inlet_temperature(t),
                self.inlet_pressure(t),
                self.outlet_pressure(t),
                self.dry_o2_mole_fraction(t),
                self.dry_h2_mole_fraction(t),
                self.inlet_relative_humidity(t),
                self.stoichiometry(t),
                self.inlet_liquid_saturation(t),
                self.inlet_liquid,
                self.inlet_liquid_flow_rate(t),
                self.inlet_gas_flow_rate(t)
            )
    
