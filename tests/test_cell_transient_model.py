import pytest
import numpy as np
import cantera as ct
import pandas as pd
from scipy.interpolate import interp1d

from scipy.integrate import solve_ivp
import marapendi as mrpd
import matplotlib.pyplot as plt



@pytest.fixture
def toray_gdl_060(): 
    lmbd = 0.86 # Data for figure 9 in Baker et al. (2009)
    f = 1 + 0.803 * np.exp(-1.17 * lmbd) + 0.197 * np.exp(-0.164 * lmbd)
    gdl = mrpd.PorousLayer(thickness=160e-6, 
                         porosity=0.72,
                         absolute_permeability=1e-12,
                         thermal_conductivity=1.24,
                         contact_angle=115.,
                         gas=mrpd.GasComposition(temperature=343.15, pressure=3.0e5), 
                         effective_gas_diffusion_ratio=0.25) # D_OM / D_OMy = 5 in Chuang et al. (2020)
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
                            reaction = mrpd.ElectrochemicalReaction(reference_exchange_current_density=2.47e-8,
                                                                activation_energy=67e6,
                                                                reaction_order=0.54,
                                                                reference_activity=1.,
                                                                reference_temperature=353.15,
                                                                number_of_electrons=2,
                                                                charge_transfer_coeff=1))

@pytest.fixture
def cell(cl, toray_gdl_060): 
    return mrpd.TransientCellModel(cell_area=25e-4,cell_number=1, electrical_resistance=30e-7, thermal_resistance=2e-4,
                                   ca=mrpd.FuelCellSide(
                                        cl=mrpd.PtCCatalystLayer(thickness=10e-6,
                                            density=2010., 
                                            specific_heat_capacity=710.,
                                            platinum_loading=0.3e-2, 
                                            ionomer_to_carbon_ratio=0.7, 
                                            catalyst_platinum_weight_percent=0.4,
                                            thermal_conductivity=0.25,
                                            ecsa=45e3,
                                            ionomer=mrpd.PFSAIonomer(),     
                                            carbon_agglomerate_radius=25e-9, 
                                            absolute_permeability=1e-13,
                                            contact_angle=95,
                                            reaction = mrpd.ElectrochemicalReaction(reference_exchange_current_density=2.47e-8 * 10e-6,
                                                                                activation_energy=67e6,
                                                                                reaction_order=0.54,
                                                                                reference_activity=1.,
                                                                                reference_temperature=353.15,
                                                                                number_of_electrons=2,
                                                                                charge_transfer_coeff=1)), 
                                        gdl = mrpd.PorousLayer(thickness=160e-6, 
                                            porosity=0.72,
                                            density=440., 
                                            specific_heat_capacity=710.,
                                            absolute_permeability=1e-12,
                                            thermal_conductivity=1.24,
                                            contact_angle=115.,
                                            gas=mrpd.GasComposition(temperature=343.15, pressure=3.0e5), 
                                            effective_gas_diffusion_ratio=0.25), 
                                        ch = mrpd.FlowChannel(height=1e-3, thermal_conductivity=100.),
                                        has_mpl=False), 
                                        
                                    an=mrpd.FuelCellSide(
                                        cl=mrpd.PtCCatalystLayer(thickness=10e-6,
                                            density=2010., 
                                            specific_heat_capacity=710.,
                                            platinum_loading=0.3e-2, 
                                            ionomer_to_carbon_ratio=0.7, 
                                            catalyst_platinum_weight_percent=0.4,
                                            thermal_conductivity=0.25,
                                            ecsa=45e3,
                                            ionomer=mrpd.PFSAIonomer(),     
                                            carbon_agglomerate_radius=25e-9, 
                                            absolute_permeability=1e-13,
                                            contact_angle=95,
                                            reaction = mrpd.ElectrochemicalReaction(reference_exchange_current_density=2.47e-8 * 10e-6,
                                                                                activation_energy=67e6,
                                                                                reaction_order=0.54,
                                                                                reference_activity=1.,
                                                                                reference_temperature=353.15,
                                                                                number_of_electrons=2,
                                                                                charge_transfer_coeff=1)), 
                                        gdl = mrpd.PorousLayer(thickness=160e-6, 
                                            absolute_permeability=1e-12,
                                            thermal_conductivity=1.24,
                                            contact_angle=115.,
                                            gas=mrpd.GasComposition(temperature=343.15, pressure=3.0e5), 
                                            effective_gas_diffusion_ratio=0.25), 
                                        ch = mrpd.FlowChannel(height=1e-3, thermal_conductivity=100.),
                                        has_mpl=False),
                                   membrane=mrpd.PFSA())
df = pd.read_csv('data/test_ui_curve.csv')

def test(cell):
    i = interp1d(df['t_step'].values, 1e4*np.maximum(0, df['I_step']).values,fill_value=0,bounds_error=False)
    x0 = np.array(
        [[[14] * cell.n_layers,
        [353.15] * cell.n_layers ,
        [1e5/ct.gas_constant/353.15 * .4] * cell.n_layers , 
         [1e5/ct.gas_constant/353.15 * .2] * cell.n_layers, 
         [1e5/ct.gas_constant/353.15 * .0] * cell.n_layers, 
         [1e5/ct.gas_constant/353.15 * 0.4] * cell.n_layers , 
        [0.1] * cell.n_layers],
         [[10] * cell.n_layers,
        [343.15] * cell.n_layers ,
        [1e5/ct.gas_constant/353.15 * .3] * cell.n_layers , 
         [1e5/ct.gas_constant/353.15 * .2] * cell.n_layers, 
         [1e5/ct.gas_constant/353.15 * .2] * cell.n_layers, 
         [1e5/ct.gas_constant/353.15 * 0.3] * cell.n_layers , 
        [0.1] * cell.n_layers]]
    ).transpose()
    
    def f(t,x): 
        
        dxdt = cell.rates_of_change(x, current_density=i(t))

        return dxdt 
    import time
    t1 = time.time()
    tf=df['t_step'].values[-1]

    sol = solve_ivp(f,t_span=(0,tf), t_eval = df['t_step'].values,y0=(x0[...,1] / cell.norm_factor).reshape(cell.n_layers* cell.n_variables), method='BDF', vectorized=True, max_step=10)
    lmbd, T, cg_k, s = cell.get_states_from_x(sol.y.reshape(cell.n_layers, cell.n_variables, sol.y.shape[-1]) * cell.norm_factor[...,np.newaxis])
    c_g = np.sum(cg_k, axis=1)
    p_g = c_g * ct.gas_constant * T
    x_g_k = cg_k / c_g[...,np.newaxis,:] 
    p_g_k = p_g[...,np.newaxis,:] * x_g_k 
    current_density = i(df['t_step'])
    iF = current_density / ct.faraday
    R_o2_local = cell.ca.cl.o2_ionomer_film_resistance(lmbd[cell.ca.cl.ix,...], T[cell.ca.cl.ix,...])
    c_o2_local = cg_k[cell.ca.cl.ix, 0, ...] - R_o2_local * iF/4
    p_o2_local = c_o2_local * ct.gas_constant * T[cell.ca.cl.ix, 0, ...]

    V_cell, eta_ohm, eta_act, S_T_losses = cell.calculate_cell_voltage(T, lmbd, p_g, p_g_k, p_o2_local, s, current_density)
    t2 = time.time()
    print(sol, tf/(t2-t1))
    plt.figure()
    plt.plot(sol.t, lmbd[2,...])
    plt.plot(sol.t, lmbd[3,...])
    plt.plot(sol.t, lmbd[4,...])
    ax2 = plt.gca().twinx()
    ax2.plot(df['t_step'], i(df['t_step']))
    plt.figure()
    plt.plot(sol.t, s[cell.ca.cl.ix,...])
    plt.plot(sol.t, s[cell.ca.gdl.ix,...])
    plt.figure()
    plt.plot(sol.t, T[cell.ca.ch.ix,...])
    plt.plot(sol.t, T[cell.ca.cl.ix,...])
    plt.plot(sol.t, T[cell.membrane.ix,...])
    plt.plot(sol.t, T[cell.an.cl.ix,...])
    plt.plot(sol.t, T[cell.an.ch.ix,...])
    plt.figure()
    for k in [3]:
        plt.plot(sol.t, cg_k[cell.ca.cl.ix,k,...] / mrpd.water_saturation_concentration(T[cell.ca.cl.ix,...]),label='ca')
        plt.plot(sol.t, cg_k[cell.ca.gdl.ix,k,...]  / mrpd.water_saturation_concentration(T[cell.ca.cl.ix,...]),label='ca')
        plt.plot(sol.t, cg_k[cell.an.cl.ix,k,...]  / mrpd.water_saturation_concentration(T[cell.ca.cl.ix,...]),label='an')
        plt.legend()
    plt.figure()
    plt.plot(current_density,V_cell, '.', alpha=0.1)
    
    plt.show()
    assert False