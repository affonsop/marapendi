import pytest
import numpy as np
import pandas as pd
import coulomb as cb
import matplotlib.pyplot as plt 


    
pressure_list = (1.5,2,2.25,2.5)

exp_data = {p: pd.read_csv(f'data/gass_et_al_2024/gass_et_al_data_{p*1e3:.0f}.csv',skiprows=1).iloc[::4] 
            for p in pressure_list}
    

def compute_ui_curve(current_density, fuel_cell, stack_pressure): 
    for p in pressure_list: 
        stack_temperature = 74.15 + 273.15
        cathode_conditions = cb.OperatingConditions(
            inlet_temperature = stack_temperature,
            inlet_relative_humidity=.6,
            outlet_pressure=stack_pressure,
            stoichiometry=np.maximum(2., 2 * 0.1e4 / (current_density+1e-4))
        )
        anode_conditions = cb.OperatingConditions(
            inlet_temperature = stack_temperature,
            inlet_relative_humidity=0.4,
            dry_h2_mole_fraction=1, 
            dry_o2_mole_fraction=0,
            outlet_pressure=stack_pressure,
            stoichiometry=np.maximum(1.2, 1.2 * 0.1e4 / (current_density+1e-4))
        )

        fuel_cell.set_conditions(stack_temperature, current_density,cathode_conditions, anode_conditions)
        fuel_cell.solve_transport() 
        return fuel_cell.cell_voltage()


def create_fuel_cell(params): 
    fc = cb.FuelCell(
        electrical_resistance=20e-7,
        cell_area = 25e-4, 
        cell_number = 1, 
        ca = cb.FuelCellSide(
            cl=cb.CatalystLayer(
                ecsa=params['ecsa'], 
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
                permeability_correction_factor=params['crossover-correction']
            ), 
            water_balance_model=cb.SimpleMembraneWaterBalanceModel()
        )
    )
    


    return fc

    
def h(params): 
    fuel_cell = create_fuel_cell(params)
    return np.concatenate(
         [compute_ui_curve(exp_data[p]['i']*1e4, fuel_cell, p*1e5)
          for p in pressure_list]
    )

exp_voltage_list = np.concatenate(
         [exp_data[p]['U'] for p in pressure_list]
    )

estimator = cb.ParameterEstimationSteadyState(h, {'ecsa':70e3})
estimator.set_unknown_params(
    [('ecsa', (40e3, 80e3), True, '$ECSA$'),
     ('crossover-correction', (0,2), True, '$k_x$')]
)

sol, p = estimator.estimate(h({'ecsa': 50e3, 'crossover-correction':1}), t=0, print_iterations=True, popsize=5, ftol=1e-8)