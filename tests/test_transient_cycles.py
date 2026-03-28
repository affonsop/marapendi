import pytest
import numpy as np
import pandas as pd
import marapendi as mrpd
import matplotlib.pyplot as plt 
from dataclasses import dataclass 

@pytest.fixture
def fuel_cell(): 
    fc = mrpd.FuelCell(
        electrical_resistance=20e-7,
        cell_area = 25e-4, 
        cell_number = 1, 
        ca = mrpd.FuelCellSide(
            cl=mrpd.PtCCatalystLayer(
                ecsa=50e3, 
                platinum_loading=0.4e-2, 
                catalyst_platinum_weight_percent=0.4,
                carbon_agglomerate_radius=20e-9,
                thickness=10e-6,
                thermal_conductivity=0.25,
                ionomer = mrpd.PFSAIonomer(),
                ionomer_to_carbon_ratio=0.6,
                reaction=mrpd.ElectrochemicalReaction(
                    reference_exchange_current_density=2.45e-4,
                    reaction_order=0.54, 
                    activation_energy=67e6, 
                    reference_activity=1e5,
                    reference_temperature=353.15,
                    number_of_electrons=2,
                    charge_transfer_coeff=0.5
                ), 
                absolute_permeability=1e-13, 
                contact_angle=95.,
                transport_resistance_model=mrpd.PorousGasResistanceModel(water_saturation_exponent=1.5),
                two_phase_transport_model=mrpd.DarcyTransportModel(J_function_exponent=2),  
            ),
            gdl=mrpd.PorousLayer(
                thickness=150e-6,
                effective_gas_diffusion_ratio=0.20,
                thermal_conductivity=0.2,
                porosity=0.6,
                absolute_permeability=1e-12, 
                contact_angle=120.,
                transport_resistance_model=mrpd.PorousGasResistanceModel(water_saturation_exponent=1.5),
                two_phase_transport_model=mrpd.DarcyTransportModel(J_function_exponent=2),  
            ),
            has_mpl=False, 
            ch=mrpd.FlowChannel(
                height=1e-3,
                width=1e-3, 
                length=21 * 50e-3,
                n_parallel=1,
                reactant='o2', 
                transport_resistance_model = mrpd.ChannelGasResistanceModel(
                    sherwood=3.6, B_ch=1.0
                )
            ),
            thermal_contact_resistance=2e-4,
        ),
        an = mrpd.FuelCellSide(
            cl=mrpd.PtCCatalystLayer(
                thickness=6e-6,
                catalyst_platinum_weight_percent=0.4,
                carbon_agglomerate_radius=20e-9,
                ionomer_to_carbon_ratio=0.7,
                platinum_loading=0.1e-2, 
                thermal_conductivity=0.25,
                absolute_permeability=1e-13, 
                contact_angle=95.,
                transport_resistance_model=mrpd.PorousGasResistanceModel(water_saturation_exponent=1.5),
                two_phase_transport_model=mrpd.DarcyTransportModel(J_function_exponent=2),  
            ),
            gdl=mrpd.PorousLayer(
                thickness=200e-6,
                effective_gas_diffusion_ratio=0.20, 
                porosity=0.6,
                thermal_conductivity=0.2,
                absolute_permeability=1e-12, 
                contact_angle=120.,
                transport_resistance_model=mrpd.PorousGasResistanceModel(water_saturation_exponent=1.5),
                two_phase_transport_model=mrpd.DarcyTransportModel(J_function_exponent=2),  
            ),
            ch=mrpd.FlowChannel(
                height=1e-3,
                width=1e-3, 
                length=21 * 50e-3,
                n_parallel=1,
                reactant='h2', 
                transport_resistance_model = mrpd.ChannelGasResistanceModel(
                    sherwood=3.6, B_ch=1.0
                )
            ),
            thermal_contact_resistance=2e-4
        ),
        membrane = mrpd.PFSA(
            equivalent_weight=1100.,
            dry_density=1980., 
            dry_thickness=12e-6,
            h2_permeation_model=mrpd.HydrogenPermeationModel(
                permeability_correction_factor=1
            ), 
            water_balance_model=mrpd.MembraneWaterBalanceModel()
        )
    )
    return fc

time = np.linspace(0,1000,1001)

@pytest.fixture 
def cycle(fuel_cell): 
    cycle = mrpd.LoadCycle(
        duration=500.,
        time_step=1.,
    )
    current_density = lambda t: np.where( t > 400., 1.2e4, 0.2e4)
    cell_temperature = 353.15
    cell_pressure = 1.5e5
    
    cycle.set_input_dict({
            'cell-temperature':             lambda t: cell_temperature,
            'current-density':              current_density, 
            'ca-inlet-temperature':         lambda t: cell_temperature,
            'ca-inlet-rh':                  lambda t: 0.8,
            'ca-inlet-pressure':            lambda t: cell_pressure,
            'ca-outlet-pressure':           lambda t: None,
            'ca-dry-o2-mole-fraction':      lambda t: 0.21,
            'ca-dry-h2-mole-fraction':      lambda t: 0.,
            'ca-inlet-gas-flow-rate':       lambda t: 0.,
            'ca-inlet-liquid-flow-rate':    lambda t: 0.,
            'ca-inlet-liquid-saturation':   lambda t: 0.,
            'ca-stoichiometry':             lambda t: np.maximum(2, 4 * (fuel_cell.cell_area/25e-4) / 24.5 / 3600. * 0.21 / (current_density(t) * fuel_cell.cell_area / 4 / 96485)), 
            'an-inlet-temperature':         lambda t: cell_temperature,
            'an-inlet-rh':                  lambda t: 0.8,
            'an-inlet-pressure':            lambda t: cell_pressure,
            'an-outlet-pressure':           lambda t: None,
            'an-dry-o2-mole-fraction':      lambda t: 0.,
            'an-dry-h2-mole-fraction':      lambda t: 1.,
            'an-inlet-gas-flow-rate':       lambda t: 0.,
            'an-inlet-liquid-flow-rate':    lambda t: 0.,
            'an-inlet-liquid-saturation':   lambda t: 0.,
            'an-stoichiometry':             lambda t: np.maximum(1.5, 4 * (fuel_cell.cell_area/25e-4)/ 24.5 / 3600 * 1.00 / (current_density(t) * fuel_cell.cell_area / 2 / 96485)), 
            })
   
    return cycle

@pytest.fixture
def model(fuel_cell, cycle):
    
    def f(t,x,u,p=None): 

        fuel_cell.set_conditions_from_input_functions(u,t)
        fuel_cell.ca.s_relax = x[0]
        fuel_cell.an.s_relax = x[1]
        fuel_cell.explicit_steady_state_model()
        dxdt = []
        for side in (fuel_cell.ca, fuel_cell.an):

            side.t_relax = 0.067 * np.exp(28000/8.314/fuel_cell.membrane.temperature) / np.where(side.membrane_water_flux < 0,1.,2.)

            dxidt = -(side.s_relax - fuel_cell.membrane.phi * 
                      fuel_cell.membrane.equilibrium_water_content(fuel_cell.ca.cl.relative_humidity(), fuel_cell.mea_temperature,None))/side.t_relax
            
            dxdt += [np.mean(dxidt)]
        return dxdt
    
    def h(t,x,u,p=None):
        return x 
        
    return mrpd.DynamicModel(f=f, h=h, u = cycle.u)

def test_equilibrium(model, fuel_cell, cycle):   
    t, x, y = model.solve(np.linspace(0,3*3600,120), x0 = [0,0], u=cycle.u, rtol=1e-3)
    
    u_cycles = cycle.repeat_cycles(25)
    
    fuel_cell.set_conditions_from_input_dict(u_cycles)
    fuel_cell.ca.s_relax = np.interp(u_cycles['total_time'], t, x[0,...])
    fuel_cell.an.s_relax = np.interp(u_cycles['total_time'], t, x[1,...])
    fuel_cell.explicit_steady_state_model()
        

    plt.plot(t,x[0])
    plt.plot(t,x[1])
    plt.plot(u_cycles['total_time'], fuel_cell.cell_voltage)
    plt.show()
    
    # Check that the value 
    assert np.isclose(fuel_cell.membrane.equilibrium_water_content(fuel_cell.ca.cl.relative_humidity()[-1], fuel_cell.mea_temperature[-1],None), 
                      fuel_cell.membrane.equilibrium_water_content(fuel_cell.ca.cl.relative_humidity()[-1], fuel_cell.mea_temperature[-1],fuel_cell.ca.s_relax[-1]), 1e-3)