import pytest
import numpy as np
import pandas as pd
import marapendi as mrpd
import matplotlib.pyplot as plt 

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
def u(fuel_cell): 

    current_density = lambda t: 1e-6 * np.ones_like(t)
    cell_temperature = 353.15
    cell_pressure = 1.5e5 
   
    return {
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
        }

@pytest.fixture
def model(fuel_cell, u):
    
    def f(t,x,u,p=None): 
        fuel_cell.set_conditions_from_input_dict(u,t)
        fuel_cell.mea_surface_heat_capacity = 1000.
        T_mea = x[0,...]
        water_profile = x[1:4,...]
        k = 4
        for side in (fuel_cell.ca,fuel_cell.an): 
            for layer in side.porous_layers:
                layer.non_wetting_saturation = x[k,...]
                k+=1
        fuel_cell.ca.s_relax = x[-2]
        fuel_cell.an.s_relax = x[-1]
        dxdt = []

        fuel_cell.set_mea_temperature(T_mea)
        for side in (fuel_cell.ca,fuel_cell.an): 
            side.h2ov_transport_resistance = side.gas_transport_resistance('h2o')
            side.cl.set_water_film_thickness(side.cl.non_wetting_saturation)

        fuel_cell.membrane.water_balance_model.solve_water_balance(fuel_cell,water_profile,True)
        fuel_cell.calculate_gas_concentrations_at_cl()
        fuel_cell.calculate_heat_transport(dynamic=True)
        
        # Temperature balance
        
        dxdt = [(fuel_cell.heat_release_rate - 
                (fuel_cell.mea_temperature_increase / fuel_cell.thermal_resistance)) / 
                fuel_cell.mea_surface_heat_capacity]

        dlmbddt = (fuel_cell.membrane.water_balance_model.membrane_water_net_flux /
                   (fuel_cell.membrane.surface_concentration / 3))
        
        # Water content balance
        dxdt += list(dlmbddt)
            
        # Water saturation balance
        for side in (fuel_cell.ca,fuel_cell.an): 
            for layer in side.porous_layers: 
                layer.flow_resistance_with_rel_permeability = layer.saturation_flow_resistance * layer.capillary_pressure_J_ratio / np.maximum(layer.non_wetting_saturation,1e-1) ** (layer.relative_permeability_exponent + 2)
                layer.capillary_pressure = layer.capillary_pressure_from_saturation(layer.non_wetting_saturation)
            
            side.cl_to_gdl_liquid_flux = (2/(side.cl.flow_resistance_with_rel_permeability + side.gdl.flow_resistance_with_rel_permeability) *
                                                (side.cl.capillary_pressure - side.gdl.capillary_pressure))
            side.gdl_to_ch_liquid_flux = ( 2/side.gdl.flow_resistance_with_rel_permeability *
                                                (side.gdl.capillary_pressure - 0))
            side.cl.liquid_balance = (side.liquid_flux - side.cl_to_gdl_liquid_flux)
            side.gdl.liquid_balance = (side.cl_to_gdl_liquid_flux - side.gdl_to_ch_liquid_flux)
        
            for layer in side.porous_layers: 
                
                dxdt.append(layer.liquid_balance / (layer.porosity * layer.thickness) * mrpd.water_molar_volume(layer.temperature))
        # Membrane relaxation

        for side in (fuel_cell.ca, fuel_cell.an):
            side.t_relax = 0.067 * np.exp(28000/8.314/fuel_cell.membrane.temperature) / np.where(side.membrane_water_flux < 0,1.,2.)
            dxdt += [-(side.s_relax - fuel_cell.membrane.xi_phi * side.est_water_content)/side.t_relax]
        return dxdt
    h = lambda t,x,u,p: x
    

    return mrpd.DynamicModel(f=f, h=h, u = u, ode_solution_method='BDF')

def test_equilibrium(model, fuel_cell, u):   
    u['current_density'] = lambda t: 1e-6 * np.ones_like(t)
    t, x, y = model.solve(np.linspace(0,1200000,12000), x0 = [343.15,4,4,4,0,0,0,0,0,0], u=u, rtol=1e-3)
    assert np.isclose(x[1,-1], fuel_cell.membrane.equilibrium_water_content(u['ca-inlet-rh'](t[-1]), x[0,-1],x[-2,-1],x[1,-1]), 1e-3)
    assert np.isclose(x[-2,-1], fuel_cell.membrane.xi_phi * x[1,-1], 1e-3)
    assert np.isclose(x[0,-1], u['ca-inlet-temperature'](t[-1]), 1e-3)