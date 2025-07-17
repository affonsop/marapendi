import pytest
import numpy as np
import cantera as ct 
import matplotlib.pyplot as plt 

import marapendi as mrpd

@pytest.fixture
def toray_gdl_060(): 
    lmbd = 0.86 # Data for figure 9 in Baker et al. (2009)
    f = 1 + 0.803 * np.exp(-1.17 * lmbd) + 0.197 * np.exp(-0.164 * lmbd)
    gdl = mrpd.PorousLayer(thickness=160e-6, 
                         absolute_permeability=1e-13,
                         thermal_conductivity=1.24,
                         contact_angle=115.,
                         gas=mrpd.GasComposition(temperature=343.15, pressure=3.0e5), 
                         effective_gas_diffusion_ratio=0.15/f) # D_OM / D_OMy = 5 in Chuang et al. (2020)
    return gdl 

@pytest.fixture
def cl(): 
    return mrpd.PtCCatalystLayer(thickness=10e-6,
                            platinum_loading=0.3e-2, 
                            ionomer_to_carbon_ratio=0.7, 
                            catalyst_platinum_weight_percent=0.4,
                            thermal_conductivity=0.25,
                            ecsa=45e3,
                            ionomer=mrpd.PFSAIonomer(),     
                            carbon_agglomerate_radius=25e-9, 
                            absolute_permeability=1e-13,
                            contact_angle=95,
                            reaction = mrpd.ElectrochemicalReaction(reference_exchange_current_density=2.47e-8 * 3e11 * 10e-6,
                                                                activation_energy=67e6,
                                                                reaction_order=0.54,
                                                                reference_activity=1.,
                                                                reference_temperature=353.15,
                                                                number_of_electrons=2,
                                                                charge_transfer_coeff=1))

@pytest.fixture
def fc(cl, toray_gdl_060): 
    fc = mrpd.FuelCell(2e-4, 1.)
    fc.membrane.temperature = 343.15
    fc.current_density = 1.e4
    fc.ca.stoichiometry = 33.0
    fc.ca.thermal_contact_resistance = 0
    fc.ca.set_catalyst_layer(cl)
    fc.ca.set_gas_diffusion_layer(toray_gdl_060)
    fc.ca.set_channel(mrpd.FlowChannel(width=0.1e-2, height=0.1e-2, length=3.7e-2, n_parallel=6)) # Values from Baker et al. (2009)
    fc.ca.ch.transport_resistance_model = mrpd.ChannelGasResistanceModel(sherwood=3.6, B_ch=1.01)
    fc.ca.ch.gas.set_temperature_and_pressure(343.15, 3.0e5)
    fc.ca.ch.set_inlet_stoichiometry(33) 
    return fc

def test_gas_porous_transport_resistance(toray_gdl_060, fc, cl): 
    for layer in fc.ca.components:
        layer.gas.set_composition(0.2,0,.64)
    
    # Test dry resistance measurements (fig 2c in Chuang et al. (2020))
    for p in (1.0e5, 2.0e5, 3.0e5): 
        for layer in fc.ca.components:
            layer.gas.set_temperature_and_pressure(353.15, p)
        experimental_resistance = 0.457 * p / 1000 + 27.7
        dry_resistance = (fc.ca.gdl.gas_transport_resistance(species='o2') +
                    fc.ca.cl.gas_transport_resistance(species='o2') +
                    fc.ca.cl.o2_ionomer_film_resistance(14, temperature=353.))
        assert np.isclose(dry_resistance, experimental_resistance, 20e-2) 

    # Test wet conditions resistance, low current densities. A small saturation of 0.1 is needed, probably because of very wet conditions
    for layer in fc.ca.components:
            layer.gas.set_composition(0.2,0,1)
            layer.gas.set_temperature_and_pressure(343.15, 300e3)
    fc.ca.gdl.liquid_saturation = 0.05
    dry_resistance = (fc.ca.gdl.gas_transport_resistance(species='o2') +
                    fc.ca.cl.gas_transport_resistance(species='o2') +
                    fc.ca.cl.o2_ionomer_film_resistance(14, fc.ca.cl.gas_temperature()))
    assert np.isclose(dry_resistance, 254, 10e-2)

    # Test Damkholer is close to 1 at transition points in figure 3a
    # In theory Da should be equal to 1. I think that we cannot reproduce results from Chuang et al. (2020)
    # if we don't consider different sigmoids for land and channel regions. See for instance works of 
    # Owejan et al. (2014) and Xu et al. (2021)
    
    fc.ca.thermal_contact_resistance =  2e-4
    fc.an.thermal_contact_resistance =  2e-4
    fc.calculate_heat_transfer_resistance()
    fc.ca.h2ov_transport_resistance = fc.ca.gas_transport_resistance('h2o')
    for rh, i_cell in ((1., 0.7e4), (0.9, 0.91e4), (0.8, 1.18e4)):
        
        temperature = 343.15 + 0.7 * i_cell * fc.thermal_resistance
        for layer in fc.ca.components:
            layer.gas.set_composition(0.2,0,rh)
        fc.ca.cl.gas.set_temperature(temperature)
       
        fc.ca.liquid_transport_model = mrpd.DarcyLiquidTransportModel() 
        da = fc.ca.liquid_transport_model.calculate_damkholer_number(fc.ca, 0.5*i_cell/(2 * ct.faraday))


    # Test wet conditions resistance
    i_cell = 2e4
    for layer in fc.ca.components:
            layer.gas.set_composition(0.2,0,1)
            layer.gas.set_temperature_and_pressure(343.15, 300e3)
    temperature = 343.15 + 0.7 * i_cell * fc.thermal_resistance
    fc.ca.cl.gas.set_temperature(temperature)  
    fc.ca.calculate_equivalent_flow_resistance()
    
    fc.ca.gdl.liq_transport_model.calculate_water_saturation(fc.ca.gdl, i_cell/(2*ct.faraday), 
                                                                                               0)
    fc.ca.cl.liq_transport_model.calculate_water_saturation(fc.ca.cl, i_cell/(2*ct.faraday), 
                                                                                               0)
    r1 = fc.ca.cl.o2_ionomer_film_resistance(14, fc.ca.cl.gas_temperature())
    fc.ca.cl.set_water_film_thickness(fc.ca.cl.liquid_saturation)
    print(fc.ca.cl.liquid_saturation, fc.ca.cl.water_film_thickness,mrpd.o2_water_diffusivity(fc.ca.cl.temperature))
    r2 = fc.ca.cl.o2_ionomer_film_resistance(14, fc.ca.cl.gas_temperature())
    print(r2-r1, (fc.ca.cl.ionomer_k3 + 1) * fc.ca.cl.water_film_thickness/mrpd.o2_water_diffusivity(fc.ca.cl.temperature))
    wet_resistance = (fc.ca.gdl.gas_transport_resistance(species='o2') +
                    fc.ca.cl.gas_transport_resistance(species='o2') +
                    fc.ca.cl.o2_ionomer_film_resistance(14, fc.ca.cl.gas_temperature()))

    assert np.isclose(wet_resistance,277, 10e-2)