import pytest
import numpy as np
import coulomb as cb
import matplotlib.pyplot as plt 

@pytest.fixture
def cathode_conditions():
    return cb.OperatingConditions(
        inlet_temperature = 353.15,
        inlet_relative_humidity=.5,
        outlet_pressure=1.5e5
    )
@pytest.fixture
def anode_conditions():
    return cb.OperatingConditions(
        inlet_temperature = 353.15,
        inlet_relative_humidity=1,
        dry_h2_mole_fraction=1, 
        dry_o2_mole_fraction=0,
        outlet_pressure=1.5e5
    )
@pytest.fixture
def fuel_cell(cathode_conditions, anode_conditions): 

    fc = cb.FuelCell(
        electrical_resistance=20e-7,
        cell_area = 25e-4, 
        cell_number = 1, 
        ca = cb.FuelCellSide(
            cl=cb.CatalystLayer(
                ecsa=50e3, 
                platinum_loading=0.4e-2, 
                carbon_agglomerate_radius=60e-9,
                thickness=10e-6,
                thermal_conductivity=0.25,
                reaction=cb.ElectrochemicalReaction(
                    reference_exchange_current_density=2.45e-4,
                    reaction_order=0.54, 
                    activation_energy=67e6, 
                    reference_activity=1e5,
                    reference_temperature=353.15,
                    number_of_electrons=2,
                    charge_transfer_coeff=0.5
                ), 
            ),
            gdl=cb.PorousLayer(
                thickness=200e-6,
                effective_gas_diffusion_ratio=0.25,
                thermal_conductivity=5.75
            ),
            has_mpl=False, 
            ch=cb.GasFlowChannel(
                height=1e-3,
                width=1e-3, 
                length=0.1,
                n_parallel=20,
                reactant='o2', 
            ),
            liq_transport_model=cb.PorousLiquidTransportModel(
                critical_damkholer=1,
                dry_wet_transition_parameter=0.1
            ),
            thermal_contact_resistance=2e-4,
        ),
        an = cb.FuelCellSide(
            cl=cb.CatalystLayer(
                thickness=6e-6, 
                thermal_conductivity=0.25,
            ),
            gdl=cb.PorousLayer(
                thickness=200e-6,
                effective_gas_diffusion_ratio=0.25, 
                thermal_conductivity=5.75
            ),
            ch=cb.GasFlowChannel(
                height=1e-3,
                width=1e-3, 
                length=0.1,
                n_parallel=20,
                reactant='h2', 
            ),
            thermal_contact_resistance=2e-4
        ),
        membrane = cb.Membrane(
            equivalent_weight=1100,
            density=1980, 
            dry_thickness=25e-6,
            h2_permeation_model=cb.HydrogenPermeationModel(
                permeability_correction_factor=1
            ), 
            water_balance_model=cb.SimpleMembraneWaterBalanceModel()
        )
    )
    


    return fc

def test_polarization_curve(fuel_cell, cathode_conditions, anode_conditions):

    for k, rh_cathode in enumerate((0.5,0.7,0.9)): 
        cathode_conditions.inlet_relative_humidity = rh_cathode
        fuel_cell.set_conditions(353.15, np.linspace(0.01,2e4,200),cathode_conditions, anode_conditions)
        fuel_cell.solve_transport()
        plt.figure(1)
        plt.plot(fuel_cell.current_density, fuel_cell.cell_voltage())
        plt.figure(2)
        plt.plot(fuel_cell.current_density, fuel_cell.ca.gdl.water_saturation, 'C{}'.format(k))
        plt.plot(fuel_cell.current_density, fuel_cell.ca.rh_at_cl_without_crossover, '--C{}'.format(k))
        plt.plot(fuel_cell.current_density, fuel_cell.an.rh_at_cl_without_crossover, '-.C{}'.format(k))
        plt.figure(3)
        plt.plot(fuel_cell.current_density, fuel_cell.membrane.water_content)
        plt.figure(4)
        plt.plot(fuel_cell.current_density, fuel_cell.ca.cl.get_o2_mole_fraction())
        plt.figure(5)
        plt.plot(fuel_cell.current_density, fuel_cell.ca.cl.get_gas_temperature())
    
    plt.show()
    

