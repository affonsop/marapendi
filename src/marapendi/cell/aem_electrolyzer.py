"""
AEM water electrolyzer cell model.

:class:`ElectrolyzerCellSide` extends :class:`FuelCellSide` for wet-side
operation and adds the dry-gas partial pressure calculation.

:class:`ElectrolyzerCell` extends :class:`FuelCell` with AEM-specific
electrochemistry: Nernst voltage from gas activities corrected for water
activity, ohmic overpotential via hydroxide conductance, and a two-phase
water balance loop that couples cathode and anode water fluxes through
a root-finding iteration.
"""

from dataclasses import dataclass, field
from scipy.optimize import root_scalar
import numpy as np
from ..thermo.constants import FARADAY_CONSTANT

from .fuelcell import FuelCell, FuelCellSide
from ..thermo.electrochemistry import calculate_reversible_cell_voltage, STD_PRESSURE
from ..thermo.gas import GasModel
from ..thermo.water import water_molar_volume
from ..thermo.gas import species_indexes

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
    is_wet: bool = True
    has_gdl: bool = True

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
                        self.cl.pressure - GasModel.vapor_pressure(self.cl))

@dataclass
class ElectrolyzerCell(FuelCell):
    """
    AEM water electrolyzer cell.

    Extends :class:`FuelCell` with electrolyzer-specific voltage model and
    a coupled water balance. The cathode operates in gas mode and the anode
    in liquid mode. Gas activities at each electrode are computed from the
    dry partial pressure after subtracting the solution saturation pressure.

    Attributes
    ----------
    electrolyte_saturation_exponent : float
        Exponent for the electrolyte saturation correction on the activation
        overpotential (n.d.).
    """
    electrolyte_saturation_exponent: float = 2 
    
    def reversible_cell_voltage(self):
        """
        Reversible (Nernst) cell voltage for the water splitting reaction.

        Gas activities at each electrode are derived from the dry gas partial
        pressure at the catalyst layer. The water activity is the ratio of the
        electrolyte solution saturation pressure to the pure water saturation
        pressure, following Lawand et al. (2024).

        Returns
        -------
        float
            Reversible cell voltage (V).

        References
        ----------
        Lawand, K. et al. J. Power Sources 595, 234047 (2024).
        """
        h2_activity = self.ca.calculate_dry_gas_pressure() / STD_PRESSURE
        o2_activity = self.an.calculate_dry_gas_pressure() / STD_PRESSURE
        h2o_activity = self.ca.electrolyte.solution_sat_pressure / GasModel.saturation_pressure(self.ca.cl)
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

    def calculate_cell_voltage(self):
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
        self.cell_voltage = np.maximum(0, E_rev + eta_act + eta_ohm)
        return self.cell_voltage
    
    def activation_overpotential(self):
        """
        Compute the activation overpotential of the electrolyzer.

        Returns
        -------
        float
            Activation overpotential in volts.
        """
        self.h2_permeation_flux = self.membrane.hydrogen_permeation_flux(
            GasModel.species_partial_pressure(self.an.cl, 'h2'),
            self.membrane.temperature,
            self.an.cl.pressure - self.ca.cl.pressure,
            self.membrane.water_vol_fraction(
                self.membrane.water_content,
                water_molar_volume(self.membrane.temperature)))

        self.crossover_current = self.h2_permeation_flux * (2 * FARADAY_CONSTANT)

        unity_activity = 1.0
        tafel_overpotential_ca = self.ca.cl.activation_overpotential(self.current_density / self.ca.cl.electrolyte_saturation ** self.electrolyte_saturation_exponent, self.ca.cl.electrolyte.molarity)
        tafel_overpotential_an = self.an.cl.activation_overpotential(self.current_density / self.an.cl.electrolyte_saturation ** self.electrolyte_saturation_exponent, self.ca.cl.electrolyte.molarity)

        return tafel_overpotential_ca + tafel_overpotential_an
    
    def calculate_bubble_transport(self): 
        for side in (self.ca, self.an): 
            # side.calculate_phase_saturation()
            side.cl.electrolyte_saturation = 1#(1 - side.cl.non_wetting_saturation) if side.cl.contact_angle < 90 else side.cl.non_wetting_saturation

    def set_consumption_production(self, current_density): 
        self.o2_production = current_density / (4 * FARADAY_CONSTANT)
        self.h2_production = 2 * self.o2_production
        self.o2_consumption = 0
        self.h2_consumption = 0
        self.an.gas_production = self.o2_production
        self.ca.gas_production = self.h2_production
        self.an.h2o_production = self.h2_production
        self.ca.h2o_production = - 2 * self.h2_production

    def set_flow_rates(self,cathode_conditions, anode_conditions): 
        for side, gas in ((self.ca, 'h2'), (self.an, 'o2')): # Neglects crossover for now
            side.gas_flux = side.gas_production * side.cl.gas.X[...,species_indexes[gas]] / (1-side.cl.gas.X[...,species_indexes['h2o']])

        for cell_side, conditions in zip((self.ca, self.an), (cathode_conditions, anode_conditions)): 
            cell_side.ch.set_fixed_inlet_liquid_flow_rate(conditions.inlet_liquid_flow_rate)
            cell_side.ch.set_fixed_inlet_gas_flow_rate(conditions.inlet_gas_flow_rate)

    def calculate_gas_concentrations_at_cl(self): 
        pass

    def calculate_water_transport(self, dynamic=False):
        """
        Solve the coupled water balance across both electrodes and the membrane.

        Iterates over each current-density point to find the membrane water
        flux that satisfies the overall water balance. Updates gas compositions,
        liquid saturations, and ionomer water content on both sides.
        """

        self.ca.is_wet=False
        self.an.is_wet=True
        cathode_water_flux = []
        for side in (self.an, self.ca): 
            side.h2ov_transport_resistance = 1e-12
            side.cl.two_phase_transport_model.calculate_equivalent_flow_resistance(side)

        for i in range(len(self.current_density)): 
            def f(cathode_membrane_water_flux, i): 
                for side in (self.ca, self.an): 
                    water_flux = (
                        cathode_membrane_water_flux * (1 if side == self.ca else -1) 
                        + side.h2o_production[i]
                    )

                    max_vapor_flux = (side.gas_production) / (GasModel.concentration(side.cl) / GasModel.saturation_concentration(side.cl) - 1)
                    side.liquid_flux = np.maximum(water_flux - max_vapor_flux, 0)
                    side.vapor_flux = water_flux - side.liquid_flux

                    if not side.is_wet:
                        rh = side.vapor_flux / (side.gas_production + side.vapor_flux) * GasModel.concentration(side.cl) / GasModel.saturation_concentration(side.cl)
                        side.calculate_water_saturation()
                        GasModel.set_composition(side.cl, 0, 1, rh[i] * np.ones_like(self.current_density),
                                                 side.cl.pressure, side.cl.temperature)

                    else:
                        # Gas leaving the cell is at 100% RH
                        side.gas_flux = side.gas_production / (1 - GasModel.saturation_concentration(side.cl) / GasModel.concentration(side.cl))
                        side.calculate_water_saturation()
                        
                self._model.water_balance_model.solve_water_balance(cell=self)
                print('xxxx', self.an.water_flux[i])
                return self.ca.membrane_water_flux[i] - cathode_membrane_water_flux
          
            sol = root_scalar(f, x0=-self.ca.h2o_production[i], args=(i,), xtol=1e-12)
            cathode_water_flux.append(sol.root)
            # print(sol)
            
        #     print(sol)
        f(np.array(cathode_water_flux), np.arange(len(cathode_water_flux)))

        self.an.gas_flux = self.an.gas_production
        self.an.calculate_water_saturation()
        
        for side in (self.an,self.ca): 
            side.cl.set_water_film_thickness(self.ca.cl.non_wetting_saturation)
               
        for cl in (self.ca.cl, self.an.cl): 
            if self.use_eq_water_content_for_ionomer: 
                cl.ionomer_water_content = cl.eq_water_content
            else: 
                cl.ionomer_water_content = cl.memb_interface_water_content
            cl.set_ionomer_wet_properties(cl.ionomer_water_content, cl.temperature)