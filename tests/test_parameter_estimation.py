import pytest
import numpy as np
import pandas as pd
import marapendi as mrpd
import matplotlib.pyplot as plt 


    
pressure_list = (1.5,2,2.25,2.5)

exp_data = {p: pd.read_csv(f'data/gass_et_al_2024/gass_et_al_data_{p*1e3:.0f}.csv',skiprows=1).iloc[::4] 
            for p in pressure_list}
    

def compute_ui_curve(current_density, fuel_cell, stack_pressure): 
    for p in pressure_list: 
        stack_temperature = 74.15 + 273.15
        cathode_conditions = mrpd.OperatingConditions(
            inlet_temperature = stack_temperature,
            inlet_relative_humidity=.6,
            outlet_pressure=stack_pressure,
            stoichiometry=np.maximum(2., 2 * 0.1e4 / (current_density+1e-4))
        )
        anode_conditions = mrpd.OperatingConditions(
            inlet_temperature = stack_temperature,
            inlet_relative_humidity=0.4,
            dry_h2_mole_fraction=1, 
            dry_o2_mole_fraction=0,
            outlet_pressure=stack_pressure,
            stoichiometry=np.maximum(1.2, 1.2 * 0.1e4 / (current_density+1e-4))
        )

        fuel_cell.compute_ui_curve(current_density, stack_temperature, cathode_conditions, anode_conditions)
        return fuel_cell.cell_voltage


def create_fuel_cell(params): 
    fc = mrpd.FuelCell(
        electrical_resistance=20e-7,
        area = 25e-4, 
        cell_number = 1, 
        ca = mrpd.FuelCellSide(
            cl=mrpd.PtCCatalystLayer(
                ecsa=params['ecsa'], 
                L_Pt=0.4e-2, 
                r_C=60e-9,
                thickness=10e-6,
                thermal_conductivity=0.25,
                ionomer=mrpd.PFSAIonomer(),
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
        membrane = mrpd.PFSA(
            equivalent_weight=1100,
            bulk_density=1980, 
            thickness=25e-6,
            h2_permeation_model=mrpd.HydrogenPermeationModel(
                permeability_correction_factor=params['crossover-correction']
            ), 
            water_balance_model=mrpd.MembraneWaterBalanceModel()
        )
    )
    return fc

    
def h(params): 
    cell_voltages = []
    for p in pressure_list: 
        fuel_cell = create_fuel_cell(params)
        cell_voltages.append(compute_ui_curve(exp_data[p]['i'].values*1e4, fuel_cell, p*1e5))
    return np.concatenate(
         cell_voltages
    )

exp_voltage_list = np.concatenate(
         [exp_data[p]['U'] for p in pressure_list]
    )

rng = np.random.default_rng()
simulated_data = h({'ecsa':70e3, 'crossover-correction':1})
simulated_data *= (1 + .00 * rng.standard_normal(len(simulated_data))) 

@pytest.fixture
def estimator(): 
    return mrpd.SteadyStateModel(h, {'ecsa':70e3, 'crossover-correction':1.})

def test_model_to_model_validation(estimator): 
    estimator.set_unknown_params(
        [('ecsa', (40e3, 80e3), True, '$ECSA$'),]
    )
    sol, p = estimator.estimate(simulated_data, t=0, print_iterations=False, popsize=20, ftol=1e-8)
    
    assert np.isclose(p[0], 70.e3, atol = 5.e3) 

def test_global_sensitivity(estimator): 
   
    estimator.set_unknown_params(
        [('ecsa', (40e3, 80e3), True, '$ECSA$')]
    )
    cosPhi_med_ij, norm_s_i, S_med, S_std, S_med_i, S_std_i, S_n, n_valid, n_opt = estimator.compute_global_sensitivity(t=0, m=2,  check_samples=False, y_exp=simulated_data, rmse_limit=0.02)
    fig1, ax1 = estimator.plot_global_sensitivity(xlabel_angle=0) 
    fig2, ax2 = estimator.plot_colinearity_map(xlabel_angle=0, cmap='Blues',figsize=(5,4))
    fig1.tight_layout()
    fig2.tight_layout()
    # plt.show()