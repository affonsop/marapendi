"""
Module providing a membrane class intended to be the base class for different membrane models. 
"""

import numpy as np
import cantera as ct 
from dataclasses import dataclass, field
from coulomb.tools import calculate_arrhenius_term
from coulomb.water import water_saturation_concentration, water_molar_volume

@dataclass
class HydrogenPermeationModel:
    """
    A dataclass representing the properties of membrane hydrogen permeability model.
    
    For details, see Trinke et al. (2016)
    
    Attributes:
    -----------
    permeability_reference_temperature : float = 333.15
        The reference temperature for reference permeability coefficients in K. 
    permeability_reference_water_vol_fraction: float = 0.37
        The reference water volume fraction for reference permeability coefficients (n.d.). 
    reference_diffusion_permeability_coefficient: float = 2.95e-17
        Reference hydrogen diffusion permeability coefficient in kmol/(m.s.Pa). 
    diffusion_permeability_activation_energy: float = 27e6
        Diffusion permeability activation energy in J/kmol. 
    reference_convection_permeability_coefficient: float = 9.01e-24
        Reference hydrogen convection permeability coefficient in kmol/(m.s.Pa). 
    convection_permeability_activation_energy: float = 2.7e6
        Convection permeability activation energy in J/kmol.   
    interface_resistance: float = 1.6e12
        Interface resistance in (m.s.Pa)/kmol.
    permeability_correction_factor: float = 1.
        A correction factor to the permeation flux that can be fitted. 

    Methods:
    --------
    calculate_permeability_coefficient(self, temperature, pressure_difference):
        Calculate the effective hydrogen permeability coefficient
    permeation_flux(membrane_thickness, partial_pressure_h2, temperature, 
                    pressure_difference, water_vol_fraction): 
        Calculate the hydrogen permeation flux through the membrane.

    References:
    -----------
    Kang, Z., Pak, M. & Bender, G. Int. J. Hydrogen Energy 46, 15161–15167 (2021).
    Trinke, P. et al. J. Electrochem. Soc. 163, F3164–F3170 (2016).
    """

    # Default values from table II in Trinke et al. (2016)
    permeability_reference_temperature: float = 333.15
    reference_diffusion_permeability_coefficient: float = 2.95e-17
    diffusion_permeability_activation_energy: float = 27e6
    reference_convection_permeability_coefficient: float = 9.01e-24
    convection_permeability_activation_energy: float = 2.7e6
    permeability_reference_water_vol_fraction: float = 0.37
    interface_resistance: float = 1.6e12
    permeability_correction_factor: float = 1.

    def calculate_permeability_coefficient(self, temperature, pressure_difference):
        """
        Calculate the effective hydrogen permeability coefficient. 

        We adopt the model proposed by Trinke et al. (2016) with temperature dependency.
        The effective hydroogne permeability coefficient is calculated as the sum of 
        diffusion and convection contributions. 

        Parameters:
        -----------
        temperature : float
            The temperature in K. 
        
        pressure_difference : float
            The pressure difference between anode and cathode. 
            Positive when the pressure on the hydrogen side is higher. 
         
        Returns:
        --------
        float
            The hydrogen permeability coefficient in kmol/(m·s·Pa).

        References:
        -----------
        Trinke, P. et al. J. Electrochem. Soc. 163, F3164–F3170 (2016).
        """

        diffusion_permeability_coeff = (
            self.reference_diffusion_permeability_coefficient *
            calculate_arrhenius_term(
                self.diffusion_permeability_activation_energy,
                temperature,
                self.permeability_reference_temperature
            )
        )

        convection_permeability_coeff = (
            self.reference_convection_permeability_coefficient *
            calculate_arrhenius_term(
                self.convection_permeability_activation_energy,
                temperature,
                self.permeability_reference_temperature
            )
        )
        return diffusion_permeability_coeff + convection_permeability_coeff * pressure_difference

    def permeation_flux(self,
                        membrane_thickness: float,
                        partial_pressure_h2: float,
                        temperature: float,
                        pressure_difference: float,
                        water_vol_fraction: float
        ) -> float:
        """
        Calculate the hydrogen permeation flux through the membrane. 

        We adopt the model proposed by Trinke et al. (2016) with temperature dependency, 
        as well as the parameters given in table II of that paper. However, we add an interface 
        resistance in series to avoid overestimating the hydrogen crossover in thin membranes and
        to match (within 10%) experimental data in Kang et al. (2021). A formal validation of this 
        approach would be welcome. 

        Parameters:
        -----------
        membrane_thickness : float 
            The thickness of the membrane in meters (m). 

        partial_pressure_h2 : float
            The partial pressure of hydrogen in Pascals (Pa).

        temperature : float
            The temperature in K. 
        
        pressure_difference : float
            The pressure difference between anode and cathode. 
            Positive when the pressure on the hydrogen side is higher. 
        
        water_vol_fraction : float
            The membrane water volume fraction. 

        Returns:
        --------
        float
            The hydrogen permeation flux in kmol/(m²·s).

        References:
        -----------
        Kang, Z., Pak, M. & Bender, G. Int. J. Hydrogen Energy 46, 15161–15167 (2021).
        Trinke, P. et al. J. Electrochem. Soc. 163, F3164–F3170 (2016).
        """

        permeability_coefficient = self.calculate_permeability_coefficient(
            temperature,
            pressure_difference
        )

        # The interface resistance is set to reproduce (within 10%) results from Kang et al. (2021).
        # A formal parameter estimation would be better.
        h2_permeability_resistance = (membrane_thickness / permeability_coefficient +
                                        self.interface_resistance)

        return (partial_pressure_h2 / h2_permeability_resistance  *
                    water_vol_fraction  / self.permeability_reference_water_vol_fraction *
                    self.permeability_correction_factor)

@dataclass
class MembraneWaterBalanceModel:
    reference_water_chemical_diffusion_coefficient: float = 4.3e-10
    reference_temperature: float = 303.15
    water_diffusivity_activation_energy: float = 27.8e6
    # Eq. 31b in Ferrara et al. (2018), equal to 0.2 for Nafion and water contents above 5
    # This assumption tends to overestimate water diffusivity and therefore backdiffusion for 
    # dry membranes. 
    gamma_function_ferrara: float = 0.2 
    reference_absorption_coefficient: float = 5e-5
                                

    def wet_chemical_diffusion_coefficient(self, temperature):
        # Corresponds to D_chem^T in Ferrara et al. (2018)
        return (self.reference_water_chemical_diffusion_coefficient *
                calculate_arrhenius_term(self.water_diffusivity_activation_energy,
                                         temperature,
                                         self.reference_temperature))
    
    
    def water_diffusivity(self, temperature):
        # According to Ferrara et al. (2018)
        return self.wet_chemical_diffusion_coefficient(temperature) * self.gamma_function_ferrara 
    
    def electroosmotic_drag_coefficient(self, temperature, water_content):
        return (0.02 * temperature - 3.86) / 22.5 * water_content

    def electroosmotic_drag_speed(self, temperature, current_density, membrane):
        return (0.02 * temperature - 3.86) / 22.5 * current_density / ct.faraday  / membrane.dry_concentration
    
    def peclet_number(self, temperature, current_density, membrane, water_diffusivity):
        return self.electroosmotic_drag_speed(temperature, current_density, membrane) * membrane.dry_thickness / water_diffusivity 
    
    def biot_number(self, absorption_coefficient, membrane, water_diffusivity):
        return absorption_coefficient * membrane.dry_thickness / water_diffusivity

    def water_balance(self, cell):
        self.absorption_coefficient = self.reference_absorption_coefficient * calculate_arrhenius_term(29e6, cell.membrane.temperature, 353.15)
        self.membrane_water_diffusivity = self.water_diffusivity(cell.membrane.temperature) 
        i_star = cell.current_density / (2*ct.faraday) * cell.membrane.dry_thickness / self.membrane_water_diffusivity / cell.membrane.dry_concentration 

        bi = self.biot_number(self.absorption_coefficient, cell.membrane, self.membrane_water_diffusivity)
        Pe =  self.peclet_number(cell.membrane.temperature, cell.current_density, cell.membrane, self.membrane_water_diffusivity) 
        ePe = np.exp(Pe)      
        
        cell.membrane.pe = Pe
        cell.membrane.bi = bi 

        for side in (cell.ca, cell.an):
  
            c_sat_cl = side.cl.saturation_concentration()

            side.rh_at_cl_without_crossover = np.minimum(
                (side.ch.vapor_concentration() + ((cell.current_density / (2*ct.faraday) * side.h2ov_transport_resistance)
                                                if side == cell.ca else np.zeros_like(cell.current_density)))
                
                / c_sat_cl,
                1
            )
            side.equiv_water_content = cell.membrane.equilibrium_water_content(
                side.rh_at_cl_without_crossover
            )
            side.equiv_water_content_derivative = cell.membrane.equilibrium_water_content_derivative(
                side.rh_at_cl_without_crossover
            )
            side.R_v_star = side.h2ov_transport_resistance * self.absorption_coefficient * cell.membrane.dry_concentration / c_sat_cl / side.equiv_water_content_derivative
            side.bi = bi

        Bi_ca = cell.ca.bi# * cell.ca.cl.porosity + cell.membrane.dry_thickness / cell.ca.cl.thickness * cell.ca.cl.ionomer_vol_fraction
        Bi_an = -cell.an.bi #* cell.an.cl.porosity - cell.membrane.dry_thickness / cell.an.cl.thickness * cell.an.cl.ionomer_vol_fraction

        # linear version: K = 1 / (1 / Bi_ca - 1 / Bi_an + 1) # 
        K = Pe * (Pe / Bi_an + 1) / (ePe * (Pe / Bi_ca + 1) - (Pe / Bi_an + 1))

        lmbd_eq_ch_ca = cell.ca.equiv_water_content #+ i_star * cell.ca.R_v_star / Bi_ca
        lmbd_eq_ch_an = cell.an.equiv_water_content
        
        lmbd_eq_cl_ca = ((lmbd_eq_ch_ca * Bi_ca * (-Bi_an  + (K + Pe) * cell.an.R_v_star) - lmbd_eq_ch_an * Bi_an * (K + Pe) * cell.ca.R_v_star) / 
                  (Bi_ca * (-Bi_an + (K + Pe) * cell.an.R_v_star) - Bi_an * K * cell.ca.R_v_star))
        lmbd_eq_cl_an = ((lmbd_eq_ch_ca * Bi_ca * K * cell.an.R_v_star - lmbd_eq_ch_an * Bi_an * (K * cell.ca.R_v_star + Bi_ca )) / 
                  (Bi_ca * (-Bi_an + (K + Pe) * cell.an.R_v_star) - Bi_an * K * cell.ca.R_v_star))
        # lmbd_m = (lmbd_eq_cl_ca - lmbd_eq_cl_an) * (1/2 - 1/Bi_an) * K + lmbd_eq_cl_an
        
        xi = np.linspace(0,np.ones_like(cell.current_density),10)
        
        #linear version : self.water_content_profile = (lmbd_eq_cl_ca - lmbd_eq_cl_an) * (xi - 1/Bi_an) * K + lmbd_eq_cl_an
        self.water_content_profile = (lmbd_eq_cl_ca - lmbd_eq_cl_an) * (np.expm1(xi*Pe) - Pe/Bi_an) / (Pe/Bi_ca * ePe - Pe/Bi_an + ePe-1) + lmbd_eq_cl_an

        cell.membrane.K = K
    
        cell.membrane.i_star = i_star
        cell.membrane.water_content = np.mean(self.water_content_profile, axis=0)
        cell.ca.cl.ionomer_water_content = lmbd_eq_cl_ca
        cell.an.cl.ionomer_water_content = lmbd_eq_cl_an
        self.calculate_cell_water_fluxes(cell)

        return self.water_content_profile

    def calculate_cell_water_fluxes(self, cell): 
        cell.ca.water_flux = cell.h2o_production + self.cathode_flux(cell)
        cell.an.water_flux = - self.cathode_flux(cell)
        self.calculate_cell_side_liquid_flux(cell.ca)
        self.calculate_cell_side_liquid_flux(cell.an)
        
    def cathode_flux(self, cell): 
        lmbd_ca = self.water_content_profile[-1,:]
        return (self.absorption_coefficient * (lmbd_ca-cell.ca.cl.ionomer_water_content) #* cell.ca.cl.porosity +
                #self.membrane_water_diffusivity * (lmbd_ca-cell.ca.membrane_surface_water_content) * cell.ca.cl.ionomer_vol_fraction / cell.ca.cl.thickness + #/ cell.ca.equiv_water_content_derivative
                + self.electroosmotic_drag_speed(cell.membrane.temperature, cell.current_density, cell.membrane) * lmbd_ca) * cell.membrane.dry_concentration


    def calculate_cell_side_liquid_flux(self,cell_side): 
        cl_sat_concentration = cell_side.cl.saturation_concentration()
        ch_vapor_concentration = cell_side.ch.vapor_concentration()
        max_vapor_removal_flux = (cl_sat_concentration - ch_vapor_concentration) / cell_side.h2ov_transport_resistance
        cell_side.liquid_flux = np.maximum(cell_side.water_flux-max_vapor_removal_flux,1e-12)
        cell_side.vapor_flux = cell_side.water_flux - cell_side.liquid_flux

@dataclass
class SimpleMembraneWaterBalanceModel(MembraneWaterBalanceModel): 
    
    def water_balance(self, cell): 
        for side in (cell.ca, cell.an):
            side.rh_at_cl_without_crossover = (side.ch.vapor_pressure() / side.cl.gas.saturation_pressure +
                                               (cell.current_density / (2*ct.faraday) * side.h2ov_transport_resistance / side.cl.saturation_concentration() 
                                                if side == cell.ca else np.zeros_like(cell.current_density)))
            side.cl.ionomer_water_content = cell.membrane.equilibrium_water_content(side.rh_at_cl_without_crossover)

        
        xi = np.linspace(0,np.ones_like(cell.current_density),10)
        self.water_content_profile = cell.ca.cl.ionomer_water_content * xi + cell.an.cl.ionomer_water_content * (1-xi)
        cell.membrane.water_content = np.mean(self.water_content_profile, axis=0)
        self.calculate_cell_water_fluxes(cell)

        return self.water_content_profile
    
    def cathode_flux(self, cell): 
        self.membrane_water_diffusivity = self.water_diffusivity(cell.membrane.temperature) 
        lmbd_ca = self.water_content_profile[-1,:]
        lmbd_an =  self.water_content_profile[0,:]
        return (-self.membrane_water_diffusivity * (lmbd_ca-lmbd_an) / cell.membrane.dry_thickness + #/ cell.ca.equiv_water_content_derivative
                + self.electroosmotic_drag_speed(cell.membrane.temperature, cell.current_density, cell.membrane) * lmbd_ca) * cell.membrane.dry_concentration


@dataclass
class NewMembraneWaterBalanceModel(MembraneWaterBalanceModel): 

    def electroosmotic_drag_coefficient(self, temperature, water_content=0):
        return 1
    
    def water_balance(self, cell): 
        n_d = self.electroosmotic_drag_coefficient(cell.membrane.temperature)
        for side in (cell.ca, cell.an):
            side.cl.ionomer_equiv_length = 0.5 * (cell.membrane.dry_thickness +
                                            cell.membrane.dry_concentration * side.cl.thickness /
                                            side.cl.ionomer_vol_fraction / side.cl.ionomer.dry_concentration)
        
        total_ionomer_equiv_length = (cell.ca.cl.ionomer_equiv_length + cell.an.cl.ionomer_equiv_length)
        R_mb = total_ionomer_equiv_length / cell.membrane.dry_concentration / self.water_diffusivity(cell.membrane.temperature)
        cell.membrane.water_content_resistance = (cell.membrane.dry_thickness / cell.membrane.dry_concentration) / (self.water_diffusivity(cell.membrane.temperature))
        for side in (cell.ca, cell.an):
            side.cl.ionomer_equiv_length = 0.5 * (cell.membrane.dry_thickness +
                                        cell.membrane.dry_concentration * side.cl.thickness /
                                        side.cl.ionomer_vol_fraction / side.cl.ionomer.dry_concentration)
                        
            side.ch_rh_at_cl_temp = side.ch.vapor_concentration() / side.cl.saturation_concentration()
            side.equiv_rh =  side.ch_rh_at_cl_temp + ((2 * n_d + 1) if side == cell.ca else (- 2 * n_d)) * cell.h2o_production * side.h2ov_transport_resistance / side.cl.saturation_concentration()
            
            side.equiv_water_content_derivative = cell.membrane.equilibrium_water_content_derivative(side.ch_rh_at_cl_temp)
            side.cl.nondim_ionomer_equiv_length = 0.5 * (1 +
                                        cell.membrane.dry_concentration / cell.membrane.dry_thickness *
                                        side.cl.thickness / side.cl.ionomer_vol_fraction / side.cl.ionomer.dry_concentration) / side.equiv_water_content_derivative
        for side in (cell.ca, cell.an):
            side.xi = side.h2ov_transport_resistance / side.cl.saturation_concentration() / cell.membrane.water_content_resistance / (cell.ca.cl.nondim_ionomer_equiv_length + cell.an.cl.nondim_ionomer_equiv_length)
        denom = (1 + cell.ca.xi + cell.an.xi)
        cell.ca.cl.water_activity = (cell.ca.equiv_rh * (1 + cell.an.xi) + cell.ca.xi * cell.an.equiv_rh) / denom
        cell.an.cl.water_activity = (cell.an.equiv_rh * (1 + cell.ca.xi) + cell.an.xi * cell.ca.equiv_rh) / denom

        cell.ca.liquid_flux = cell.ca.cl.saturation_concentration()/ cell.ca.h2ov_transport_resistance * np.maximum(cell.ca.cl.water_activity - 1, 0) * denom
        cell.ca.cl.water_activity = np.minimum(cell.ca.cl.water_activity, 1)    
        
        cell.an.liquid_flux = 0

        cell.membrane.water_activity = sum(side.cl.water_activity / side.cl.nondim_ionomer_equiv_length for side in (cell.ca, cell.an)) /  sum(1 / side.cl.nondim_ionomer_equiv_length for side in (cell.ca, cell.an))
        cell.membrane.crossover_flux = ((cell.an.cl.water_activity - cell.ca.cl.water_activity) / cell.membrane.water_content_resistance / (cell.ca.cl.nondim_ionomer_equiv_length + cell.an.cl.nondim_ionomer_equiv_length) +
                                         2 * n_d * cell.h2o_production)
        
        cell.ca.water_flux = cell.membrane.crossover_flux + cell.h2o_production
        cell.an.water_flux = -cell.membrane.crossover_flux
        for side in (cell.ca, cell.an):
            side.vapor_flux = side.water_flux - side.liquid_flux
        
        s = cell.ca.calculate_water_saturation()
        
        for side in (cell.ca, cell.an):
    
            side.cl.water_content = cell.membrane.equilibrium_water_content(side.cl.water_activity) * (1-s) + 24  * s
    
        cell.membrane.water_content = cell.membrane.equilibrium_water_content(cell.membrane.water_activity) * (1-s) + 24  * s

    def cathode_flux(self, cell): 
        self.membrane_water_diffusivity = self.water_diffusivity(cell.membrane.temperature) 
        lmbd_ca = self.water_content_profile[-1,:]
        lmbd_an =  self.water_content_profile[0,:]
        return (-self.membrane_water_diffusivity * (lmbd_ca-lmbd_an) / cell.membrane.dry_thickness + #/ cell.ca.equiv_water_content_derivative
                + self.electroosmotic_drag_speed(cell.membrane.temperature, cell.current_density, cell.membrane) * lmbd_ca) * cell.membrane.dry_concentration

@dataclass
class Membrane:
    """
    A dataclass representing the properties of a proton exchange membrane (PEM) 
    and methods for calculating water volume fraction, hydrogen permeability, 
    and hydrogen permeation flux.

    Attributes:
    -----------
    equivalent_weight : float
        Equivalent weight of the membrane in kg/kmol. Default is 1100 kg/kmol.
    density : float
        Density of the membrane in kg/m³. Default is 1980 kg/m³.
    thickness : float
        Thickness of the membrane in meters (m). Default is 25 µm.
    hydrogen_permeation_model: HydrogenPermeationModel
        A dataclass representing the properties of membrane hydrogen permeability model.
    water_content: float
        Water content of the membrane. 
    conductivity_correction: float
        Correction factor for the proton conductivity. 
    conductivity_correction: float
        Correction factor for the proton conductivity, to scale the expression from Vetter and Schumacher (2020).  
    conductivity_exp: float
        Exponent for the proton conductivity correlation. 

    Computed Attributes:
    --------------------
    dry_concentration : float
        Concentration of the membrane in mol/m³, computed during initialization.
    dry_molar_volume : float
        Molar volume of the membrane in m³/mol, computed during initialization.

    Methods:
    --------
    water_vol_fraction(water_content, water_molar_volume):
        Calculate the volume fraction of water in the membrane.

    hydrogen_permeation_flux(partial_pressure_h2, hydrogen_permeability):
        Calculate the hydrogen permeation flux through the membrane.
    """

    equivalent_weight: float = 1.1e3
    
    dry_density: float = 1980.
    dry_thickness: float = 25e-6
    h2_permeation_model: HydrogenPermeationModel = field(default_factory=HydrogenPermeationModel)
    water_balance_model: MembraneWaterBalanceModel = field(default_factory=MembraneWaterBalanceModel)
    water_content: float = 14
    conductivity_correction: float = 1
    conductivity_exp: float = 1.5

    def __post_init__(self):
        """
        Compute derived properties of the membrane after initialization.
        """
        self.dry_concentration = self.dry_density / self.equivalent_weight  # kmol/m³
        self.dry_molar_volume = 1. / self.dry_concentration  # m³/kmol
       
    def water_vol_fraction(self, water_content: float, water_molar_volume: float) -> float:
        """
        Calculate the volume fraction of water in the membrane.

        Parameters:
        -----------
        water_content : float
            The water content of the membrane, defined as the number of moles of water 
            per equivalent of the membrane.
        water_molar_volume : float
            The molar volume of water in m³/kmol.

        Returns:
        --------
        float
            The volume fraction of water in the membrane.
        """
        membrane_water_molar_volume = water_molar_volume * water_content
        return membrane_water_molar_volume / (self.dry_molar_volume +
                                               membrane_water_molar_volume)

    def hydrogen_permeation_flux(self,
                                 partial_pressure_h2: float,
                                 temperature: float,
                                 pressure_difference: float,
                                 water_vol_fraction: float,
                                 ) -> float:
        """
        Calculate the hydrogen permeation flux through the membrane. 

        Parameters:
        -----------
        partial_pressure_h2 : float
            The partial pressure of hydrogen in Pascals (Pa).

        temperature : float
            The temperature in K. 
        
        pressure_difference : float
            The pressure difference between anode and cathode. 
            Positive when the pressure on the hydrogen side is higher. 
        
        water_vol_fraction : float
            The membrane water volume fraction.  

        Returns:
        --------
        float
            The hydrogen permeation flux in kmol/(m²·s).
        """

        return self.h2_permeation_model.permeation_flux(self.dry_thickness,
                                                        partial_pressure_h2,
                                                        temperature,
                                                        pressure_difference,
                                                        water_vol_fraction)

    def equilibrium_water_content(self, rh):
        rh = np.minimum(np.maximum(rh, 0),1)
        return (0.043 + 17.18 * rh - 39.85 * rh**2 + 36 * rh**3)

    def equilibrium_water_content_derivative(self,rh): 
        rh = np.minimum(np.maximum(rh, 0),1)
        return (17.18 - 79.70 * rh + 108 * rh**2)

    def proton_conductivity(self, temperature, water_vol_fraction, water_content, use_water_profile=True): 
        if use_water_profile:
            fv = self.water_vol_fraction(self.water_balance_model.water_content_profile ,water_molar_volume(temperature))
            return 1/np.mean(1/(self.conductivity_correction * 50 * (np.maximum(fv, 0.061) - 0.06 ) ** self.conductivity_exp * calculate_arrhenius_term(15e6, temperature, 303.15)), axis=0)
        else: 
            fv = self.water_vol_fraction(water_content ,water_molar_volume(temperature))
            return self.conductivity_correction * 50 * (np.maximum(fv, 0.061) - 0.06 ) ** self.conductivity_exp * calculate_arrhenius_term(15e6, temperature, 303.15)
        
    def proton_resistance(self, temperature, water_vol_fraction, water_content): 
        return self.dry_thickness / self.proton_conductivity(temperature, water_vol_fraction, water_content)