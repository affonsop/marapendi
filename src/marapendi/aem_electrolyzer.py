"""
Module providing a AEM water electrolyzer class. 
"""
from dataclasses import dataclass, field
from scipy.optimize import root
import numpy as np 
import cantera as ct

from .fuelcell import FuelCell, FuelCellSide
from .electrochemistry import calculate_reversible_cell_voltage, h2_lhv
from .porous_layers import PorousLayer, PtCCatalystLayer
from .flow_channels import GasFlowChannel
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
        return calculate_reversible_cell_voltage(
            self.mea_temperature,
            self.an.cl.species_partial_pressure('o2'),
            self.ca.cl.species_partial_pressure('h2')
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
        self.ca.cl.proton_resistance = self.ca.cl.effective_charge_resistance(
            self.current_density, 
            self.ca.cl.ionomer_water_content, 
            self.ca.cl.temperature
        )
        self.an.cl.proton_resistance = self.ca.cl.effective_charge_resistance(
            self.current_density, 
            self.ca.cl.ionomer_water_content, 
            self.ca.cl.temperature
        )
        return self.current_density * (self.ca.cl.proton_resistance + 
                                       self.high_frequency_resistance() + 
                                       self.an.cl.proton_resistance)
    
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
        eta_act = self.activation_overpotential(self.ca.cl.theta_catalyst)
        return np.maximum(0, E_rev + eta_act + eta_ohm)
    