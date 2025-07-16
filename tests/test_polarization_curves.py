import pytest
import numpy as np
import marapendi as mrpd
import matplotlib.pyplot as plt 

@pytest.fixture
def cathode_conditions():
    return mrpd.OperatingConditions(
        inlet_temperature = 353.15,
        inlet_relative_humidity=.5,
        outlet_pressure=1.5e5
    )
@pytest.fixture
def anode_conditions():
    return mrpd.OperatingConditions(
        inlet_temperature = 353.15,
        inlet_relative_humidity=1,
        dry_h2_mole_fraction=1, 
        dry_o2_mole_fraction=0,
        outlet_pressure=1.5e5
    )
@pytest.fixture
def fuel_cell(cathode_conditions, anode_conditions): 

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
                ionomer = mrpd.PFSAIonomer(),
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
                effective_gas_diffusion_ratio=0.20,
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
            thermal_contact_resistance=2e-4,
        ),
        an = mrpd.FuelCellSide(
            cl=mrpd.PtCCatalystLayer(
                thickness=6e-6, 
                thermal_conductivity=0.25,
            ),
            gdl=mrpd.PorousLayer(
                thickness=200e-6,
                effective_gas_diffusion_ratio=0.20, 
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
        membrane = mrpd.PFSA(
            equivalent_weight=1100,
            dry_density=1980, 
            dry_thickness=25e-6,
            h2_permeation_model=mrpd.HydrogenPermeationModel(
                permeability_correction_factor=1
            ), 
            water_balance_model=mrpd.MembraneWaterBalanceModel()
        )
    )
    


    return fc

def test_polarization_curve(fuel_cell, cathode_conditions, anode_conditions):
    fig, ax = plt.subplots(2,3,figsize=(8,4))
    for k, rh_cathode in enumerate((0.5,0.7,0.9)): 
        cathode_conditions.inlet_relative_humidity = rh_cathode
        fuel_cell.set_conditions(353.15, np.linspace(0.01,1.6e4,50),cathode_conditions, anode_conditions)
        fuel_cell.solve_transport()
        
        ax[0,0].plot(fuel_cell.current_density * 1e-4, fuel_cell.cell_voltage(), label=f'{rh_cathode * 100:.0f} %')
        ax[0,0].set_ylabel('Cell voltage (V)')
        ax[0,0].set_xlabel('Curent density (A/cm$^2$)')

        ax[0,1].plot(fuel_cell.current_density * 1e-4, fuel_cell.ca.gdl.liquid_saturation, label=f'{rh_cathode * 100:.0f} %')
        ax[0,1].set_ylabel('GDL water\nsaturation (n.d.)')
        ax[0,1].set_xlabel('Curent density (A/cm$^2$)')

        ax[0,2].plot(fuel_cell.current_density * 1e-4, fuel_cell.membrane.water_content, label=f'{rh_cathode * 100:.0f} %')
        ax[0,2].set_ylabel('Membrane\nwater content (n.d.)')
        ax[0,2].set_xlabel('Curent density (A/cm$^2$)')
        ax[0,2].legend(loc='upper left', bbox_to_anchor=(1,1.0), title='RH$_{in,ca}$')
        
        ax[1,0].plot(fuel_cell.current_density * 1e-4, fuel_cell.ca.cl.o2_mole_fraction(), label=f'{rh_cathode * 100:.0f} %')
        ax[1,0].set_ylabel('Cathode CL\nO$_2$ mole fraction (n.d.)')
        ax[1,0].set_xlabel('Curent density (A/cm$^2$)')
     
        ax[1,1].plot(fuel_cell.current_density * 1e-4, fuel_cell.ca.cl.gas_temperature()-273.15)
        ax[1,1].set_ylabel(u'Cathode CL\ntemperature (\u00B0C)')
        ax[1,1].set_xlabel('Curent density (A/cm$^2$)')
    
        ax[1,2].plot(fuel_cell.current_density * 1e-4, 1e7 * fuel_cell.high_frequency_resistance())
        ax[1,2].set_ylabel(r'HFR (m$\Omega$.cm$^2$)')
        ax[1,2].set_xlabel('Curent density (A/cm$^2$)')
    plt.tight_layout()
    plt.savefig('tests/figures/test_polarization_curves.png',dpi=300)

    
if __name__ == "__main__":
    test_polarization_curve(fuel_cell(), cathode_conditions(), anode_conditions())
