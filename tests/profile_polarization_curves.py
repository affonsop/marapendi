import pytest
import numpy as np
import marapendi as mrpd
import matplotlib.pyplot as plt 


def cathode_conditions():
    return mrpd.OperatingConditions(
        inlet_temperature = 353.15,
        inlet_relative_humidity=.5,
        outlet_pressure=1.5e5
    )

def anode_conditions():
    return mrpd.OperatingConditions(
        inlet_temperature = 353.15,
        inlet_relative_humidity=1,
        dry_h2_mole_fraction=1, 
        dry_o2_mole_fraction=0,
        outlet_pressure=1.5e5
    )

def fuel_cell(): 

    fc = mrpd.FuelCell(
        electrical_resistance=20e-7,
        cell_area = 25e-4, 
        cell_number = 1, 
        ca = mrpd.FuelCellSide(
            cl=mrpd.PtCCatalystLayer(
                ecsa=50e3, 
                platinum_loading=0.4e-2, 
                carbon_agglomerate_radius=60e-9,
                thickness=10e-6,
                thermal_conductivity=0.25,
                reaction=mrpd.ElectrochemicalReaction(
                    reference_exchange_current_density=2.45e-4,
                    reaction_order=0.54, 
                    activation_energy=67e6, 
                    reference_activity=1e5,
                    reference_temperature=353.15,
                    number_of_electrons=2,
                    charge_transfer_coeff=0.5
                ), 
            ),
            gdl=mrpd.PorousLayer(
                thickness=200e-6,
                effective_gas_diffusion_ratio=0.25,
                thermal_conductivity=5.75
            ),
            has_mpl=False, 
            ch=mrpd.FlowChannel(
                height=1e-3,
                width=1e-3, 
                length=0.1,
                n_parallel=20,
                reactant='o2', 
            ),
            liq_transport_model=mrpd.PorousLiquidTransportModel(
                critical_damkholer=1,
                dry_wet_transition_parameter=0.1
            ),
            thermal_contact_resistance=2e-4,
        ),
        an = mrpd.FuelCellSide(
            cl=mrpd.PtCCatalystLayer(
                thickness=6e-6, 
                thermal_conductivity=0.25,
            ),
            gdl=mrpd.PorousLayer(
                thickness=200e-6,
                effective_gas_diffusion_ratio=0.25, 
                thermal_conductivity=5.75
            ),
            ch=mrpd.FlowChannel(
                height=1e-3,
                width=1e-3, 
                length=0.1,
                n_parallel=20,
                reactant='h2', 
            ),
            thermal_contact_resistance=2e-4
        ),
        membrane = mrpd.Membrane(
            equivalent_weight=1100,
            density=1980, 
            dry_thickness=25e-6,
            h2_permeation_model=mrpd.HydrogenPermeationModel(
                permeability_correction_factor=1
            ), 
            water_balance_model=mrpd.SimpleMembraneWaterBalanceModel()
        )
    )
    


    return fc

def test_polarization_curve(fuel_cell, cathode_conditions, anode_conditions):
    for k, rh_cathode in enumerate((0.5,0.7,0.9)): 
        cathode_conditions.inlet_relative_humidity = rh_cathode
        fuel_cell.set_conditions(353.15, np.linspace(0.01,1.5e4,50),cathode_conditions, anode_conditions)
        fuel_cell.solve_transport()
        
    
if __name__ == "__main__":
    import cProfile
    cProfile.run('test_polarization_curve(fuel_cell(), cathode_conditions(), anode_conditions())')
