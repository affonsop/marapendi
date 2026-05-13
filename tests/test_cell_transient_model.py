import pytest
import numpy as np
import cantera as ct
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
                            reaction = mrpd.ElectrochemicalReaction(reference_exchange_current_density=2.47e-8 * 3e11 * 10e-6,
                                                                activation_energy=67e6,
                                                                reaction_order=0.54,
                                                                reference_activity=1.,
                                                                reference_temperature=353.15,
                                                                number_of_electrons=2,
                                                                charge_transfer_coeff=1))

@pytest.fixture
def cell(cl, toray_gdl_060): 
    return mrpd.TransientCellModel(cell_area=25e-4,cell_number=1, 
                                   ca=mrpd.FuelCellSide(
                                        cl=mrpd.PtCCatalystLayer(thickness=10e-6,
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
                                                                                charge_transfer_coeff=1)), 
                                        gdl = mrpd.PorousLayer(thickness=160e-6, 
                                            porosity=0.72,
                                            absolute_permeability=1e-12,
                                            thermal_conductivity=1.24,
                                            contact_angle=115.,
                                            gas=mrpd.GasComposition(temperature=343.15, pressure=3.0e5), 
                                            effective_gas_diffusion_ratio=0.25), has_mpl=False), 
                                    an=mrpd.FuelCellSide(
                                        cl=mrpd.PtCCatalystLayer(thickness=10e-6,
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
                                                                                charge_transfer_coeff=1)), 
                                        gdl = mrpd.PorousLayer(thickness=160e-6, 
                                            absolute_permeability=1e-12,
                                            thermal_conductivity=1.24,
                                            contact_angle=115.,
                                            gas=mrpd.GasComposition(temperature=343.15, pressure=3.0e5), 
                                            effective_gas_diffusion_ratio=0.25), has_mpl=False),
                                   membrane=mrpd.PFSA())
def test(cell):

    x0 = np.array(
        [[[10] * cell.n_layers,
        [353.15] * cell.n_layers ,
        [1e5/ct.gas_constant/353.15 * .4] * cell.n_layers , 
         [1e5/ct.gas_constant/353.15 * .2] * cell.n_layers, 
         [1e5/ct.gas_constant/353.15 * .0] * cell.n_layers, 
         [1e5/ct.gas_constant/353.15 * 0.4] * cell.n_layers , 
        [0.1] * cell.n_layers],
         [[10] * cell.n_layers,
        [353.15] * cell.n_layers ,
        [1e5/ct.gas_constant/353.15 * .4] * cell.n_layers , 
         [1e5/ct.gas_constant/353.15 * .2] * cell.n_layers, 
         [1e5/ct.gas_constant/353.15 * .0] * cell.n_layers, 
         [1e5/ct.gas_constant/353.15 * 0.4] * cell.n_layers , 
        [0.1] * cell.n_layers]]
    ).transpose()

    def f(t,x): 

        dxdt = cell.rates_of_change(x, current_density=np.where(t > 50, 1e4,0))

        return dxdt 
    import time
    t1 = time.time()
    tf=200

    sol = solve_ivp(f,t_span=(0,tf), y0=x0[...,1].reshape(cell.n_layers* cell.n_variables), method='BDF', vectorized=True)
    lmbd, T, cg, s = cell.get_states_from_x(sol.y.reshape(cell.n_layers, cell.n_variables, sol.y.shape[-1]))
    t2 = time.time()
    print(sol, tf/(t2-t1))
    plt.figure()
    plt.plot(sol.t, lmbd[2,...])
    plt.plot(sol.t, lmbd[3,...])
    plt.plot(sol.t, lmbd[4,...])
    plt.figure()
    plt.plot(sol.t, s[cell.ca.cl.ix,...])
    plt.plot(sol.t, s[cell.ca.gdl.ix,...])
    plt.figure()
    for k in [3]:
        plt.plot(sol.t, cg[cell.ca.cl.ix,k,...] / mrpd.water_saturation_concentration(T[cell.ca.cl.ix,...]),label='ca')
        plt.plot(sol.t, cg[cell.ca.gdl.ix,k,...]  / mrpd.water_saturation_concentration(T[cell.ca.cl.ix,...]),label='ca')
        plt.plot(sol.t, cg[cell.an.cl.ix,k,...]  / mrpd.water_saturation_concentration(T[cell.ca.cl.ix,...]),label='an')
        plt.legend()
    # plt.figure()
    # plt.plot(sol.t, np.sum(cg[cell.ca.cl.ix,...], axis=0) * ct.gas_constant * T[cell.ca.cl.ix,...],label='ca')
    # plt.plot(sol.t, np.sum(cg[cell.ca.gdl.ix,...], axis=0) * ct.gas_constant * T[cell.ca.gdl.ix,...],label='ca')
    plt.show()
    assert False