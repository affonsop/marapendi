"""
Module providing a fuel cell class intended to be the base class for different fuel cell models.
"""
from dataclasses import dataclass, field
from scipy.optimize import root, least_squares
import numpy as np
from ..porous.porous_layers import PorousLayer
from ..porous.cl.catalyst_layers import PtCCatalystLayer
from ..channel.flow_channels import FlowChannel
from ..membrane.membrane import Membrane
from .cell import Cell, CellSide
from ...models.cell.voltage import VoltageModel
from ...models.cell.thermal import ThermalModel
from ...models.cell.explicit_steady_state import ExplicitSteadyStateModel
from ...simulation.state import (
    CellState, CellSideState, LayerState, CatalystLayerState,
    FlowChannelState, MembraneState,
)
from ...models.gas import GasState

@dataclass
class FuelCellSide(CellSide):
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
    thermal_contact_resistance : float
        Additional thermal contact resistance at interfaces.
    """

    cl: PorousLayer = field(default_factory=PtCCatalystLayer)
    gdl: PorousLayer = field(default_factory=PorousLayer)
    mpl: PorousLayer = field(default_factory=PorousLayer)
    ch: FlowChannel = field(default_factory=FlowChannel)
    has_mpl: bool = False
    has_gdl: bool = True
    is_wet: bool = False
    thermal_contact_resistance: float = 0

    def __post_init__(self):
        pass

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
    mea_surface_heat_capacity: float = 10000.
    use_eq_water_content_for_ionomer: bool = True

    def __post_init__(self):
        self.ca.reactant = 'o2'
        self.an.reactant = 'h2'
        super().__post_init__()   # builds porous_layers, layers, sides from Cell
        self._voltage_model = VoltageModel()
        self._thermal_model = ThermalModel()
        self._model = ExplicitSteadyStateModel(
            self._voltage_model,
            self._thermal_model,
            self.membrane.water_balance_model,
        )
        self._gas_transport_model = self._model.gas_transport_model
        self.state = CellState()

    # ------------------------------------------------------------------
    # Voltage methods — delegate to VoltageModel
    # ------------------------------------------------------------------

    def reversible_cell_voltage(self):
        return self._voltage_model.reversible_cell_voltage(self, self.state)

    def reversible_voltage_vs_RHE(self):
        return self._voltage_model.reversible_voltage_vs_RHE(self, self.state)

    def activation_overpotential(self, theta_PtO=0):
        return self._voltage_model.activation_overpotential(self, self.state, theta_PtO)

    def high_frequency_resistance(self):
        return self._voltage_model.high_frequency_resistance(self, self.state)

    def ohmic_overpotential(self):
        return self._voltage_model.ohmic_overpotential(self, self.state)

    def calculate_theta_PtO(self):
        return self._voltage_model.calculate_theta_PtO(self, self.state)

    def calculate_cell_voltage(self):
        return self._voltage_model.compute_cell_voltage(self, self.state)

    def set_mea_temperature(self, mea_temperature):
        """Delegate to :class:`~marapendi.thermal.ThermalModel`."""
        self._thermal_model.set_mea_temperature(mea_temperature, self, self.state)

    def calculate_water_transport(self, dynamic=False):
        """Delegate to :class:`~marapendi.water_balance_models.MembraneWaterBalanceModel`."""
        self._model.water_balance_model.calculate_water_transport(self, self.state, dynamic)

    def calculate_gas_concentrations_at_cl(self):
        """Delegate to :class:`~marapendi.transport.GasTransportModel`."""
        self._gas_transport_model.calculate_gas_concentrations(self, self.state)
    
    def calculate_heat_transfer_resistance(self):
        """Delegate to :class:`~marapendi.thermal.ThermalModel`."""
        self.thermal_resistance = self._thermal_model.heat_transfer_resistance(self)
    
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
        
    def explicit_steady_state_model(self, mea_tempearture_estimation=False):
        """
        Simplified steady-state model where all calculations are explicit.

        Delegates to :class:`~marapendi.model.ExplicitSteadyStateModel`.
        """
        voltage = self._model.solve(self, self.state, mea_temperature_estimation=mea_tempearture_estimation)
        self._sync_state_to_fc()
        return voltage

    def _sync_state_to_fc(self):
        """Copy key values from ``self.state`` back to legacy ``FuelCell`` attributes.

        Keeps backward compatibility for callers that read ``fc.cell_voltage``,
        ``fc.mea_temperature``, etc. directly.
        """
        s = self.state
        self.cell_voltage = s.cell_voltage
        self.mea_temperature = s.mea_temperature if s.mea_temperature is not None else self.temperature
        self.mea_temperature_increase = s.mea_temperature_increase if s.mea_temperature_increase is not None else 0.
        self.thermal_resistance = s.thermal_resistance if s.thermal_resistance is not None else self.thermal_resistance
        self.crossover_current = s.crossover_current if s.crossover_current is not None else 0.
        if s.membrane.h2_permeation_flux is not None:
            self.h2_permeation_flux = s.membrane.h2_permeation_flux

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
        self._model.water_balance_model.solve_water_balance(self, water_profile=water_profile, dynamic=True)
        self.calculate_gas_concentrations_at_cl()
        self.calculate_cell_voltage()

        # Calculate rates of change
        wbm = self._model.water_balance_model
        dlmbddt = wbm.membrane_water_rate_of_change(self, n_memb_mesh)
        dTdt = self._thermal_model.temperature_rate_of_change(self)
        dsdt = wbm.saturation_rate_of_change(self)
        dsrlxdt = wbm.relaxation_rate_of_change(self)
        return [dTdt] + list(dlmbddt) + list(dsdt) + list(dsrlxdt)

    def f_relax(self, t,x,u,p, n_memb_mesh=3): 
        self.set_conditions_from_input_dict(u,t * np.ones_like(x[0,...]))
        self.explicit_steady_state_model()

        self.ca.s_relax = x[-2,...]
        self.an.s_relax = x[-1,...]
        
        dsrlxdt = self._model.water_balance_model.relaxation_rate_of_change(self)
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


    def populate_state(self):
        """Populate ``self.state`` from the component tree after ``set_conditions``.

        Called at the end of :meth:`set_conditions` so that the pure-data
        :class:`~marapendi.state.CellState` always mirrors the current
        operating point, ready for model code that consumes state objects
        rather than component instances.
        """
        def _layer(component):
            return LayerState(
                gas=GasState(X=component.gas.X.copy()),
                temperature=component.temperature,
                pressure=component.pressure,
                liquid_saturation=np.asarray(component.liquid_saturation).copy(),
                non_wetting_saturation=np.asarray(component.non_wetting_saturation).copy(),
            )

        def _cl(cl):
            return CatalystLayerState(
                gas=GasState(X=cl.gas.X.copy()),
                temperature=cl.temperature,
                pressure=cl.pressure,
                liquid_saturation=np.asarray(cl.liquid_saturation).copy(),
                non_wetting_saturation=np.asarray(cl.non_wetting_saturation).copy(),
                ionomer_water_content=cl.ionomer_water_content,
            )

        def _ch(ch):
            return FlowChannelState(
                gas=GasState(X=ch.gas.X.copy()),
                temperature=ch.temperature,
                pressure=ch.pressure,
                inlet_gas_flow_rate=ch.inlet_gas_flow_rate,
                inlet_liquid_flow_rate=ch.inlet_liquid_flow_rate,
                inlet_liquid_saturation=ch.inlet_liquid_saturation,
                inlet_stoichiometry=ch.inlet_stoichiometry,
            )

        def _side(fc_side):
            return CellSideState(
                cl=_cl(fc_side.cl),
                gdl=_layer(fc_side.gdl) if fc_side.has_gdl else None,
                mpl=_layer(fc_side.mpl) if fc_side.has_mpl else None,
                ch=_ch(fc_side.ch),
                h2o_production=fc_side.h2o_production,
                reactant_consumption=fc_side.reactant_consumption,
                s_relax=fc_side.s_relax,
            )

        self.state = CellState(
            ca=_side(self.ca),
            an=_side(self.an),
            membrane=MembraneState(temperature=self.membrane.temperature),
            current_density=self.current_density,
            temperature=self.temperature,
        )

    def set_conditions(self, stack_temperature, current_density, cathode_conditions, anode_conditions):
        """Delegate to :meth:`~marapendi.model.ExplicitSteadyStateModel.set_initial_state`."""
        self._model.set_initial_state(
            self, self.state, stack_temperature, current_density,
            cathode_conditions, anode_conditions,
        )
            

from ..electrolyte.electrolyte import ElectrolyteSolution

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
    
