import pytest
import numpy as np
import pandas as pd
import marapendi as mrpd
import matplotlib.pyplot as plt 


@pytest.fixture
def cycle(): 
    cycle = mrpd.LoadCycle(
        duration=6,
        time_step=.01,
    )
    step_duration = 3. 
    lower_potential = 0.6 
    upper_potential = 0.95
    potential_change_duration = 0.6
    cell_potential = lambda t: np.interp(t - np.floor(t / cycle.duration) * cycle.duration, 
                                         [0, step_duration/2-potential_change_duration, step_duration/2, 1.5 * step_duration-potential_change_duration, 1.5*step_duration, 2 * step_duration], 
                                         [lower_potential, lower_potential, upper_potential, upper_potential, lower_potential, lower_potential])
    cell_temperature = 353.15
    cell_pressure = 1.5e5
    
    cycle.set_input_dict({
            'cell-potential':               cell_potential,
            'cell-temperature':             lambda t: cell_temperature,
            'current-density':              lambda t: 0, 
            'ca-inlet-temperature':         lambda t: cell_temperature,
            'ca-inlet-rh':                  lambda t: 1,
            'ca-inlet-pressure':            lambda t: cell_pressure,
            'ca-outlet-pressure':           lambda t: None,
            'ca-dry-o2-mole-fraction':      lambda t: 0.,
            'ca-dry-h2-mole-fraction':      lambda t: 0.,
            'ca-inlet-gas-flow-rate':       lambda t: 0.,
            'ca-inlet-liquid-flow-rate':    lambda t: 0.,
            'ca-inlet-liquid-saturation':   lambda t: 0.,
            'ca-stoichiometry':             lambda t: 2, 
            'an-inlet-temperature':         lambda t: cell_temperature,
            'an-inlet-rh':                  lambda t: 1,
            'an-inlet-pressure':            lambda t: cell_pressure,
            'an-outlet-pressure':           lambda t: None,
            'an-dry-o2-mole-fraction':      lambda t: 0.,
            'an-dry-h2-mole-fraction':      lambda t: 1.,
            'an-inlet-gas-flow-rate':       lambda t: 0.,
            'an-inlet-liquid-flow-rate':    lambda t: 0.,
            'an-inlet-liquid-saturation':   lambda t: 0.,
            'an-stoichiometry':             lambda t: 2, 
            })
   
    return cycle

@pytest.fixture 
def catalyst_layer():
    cl = mrpd.CatalystLayer(ionomer_vol_fraction=0.3, 
                            ionomer=mrpd.PFSAIonomer(equivalent_weight=1100.))
    cl.platinum_size_distribution = mrpd.PtSizeDistribution(
        number_density_array=np.array([]),
        r_array=np.array([]),
        n_points=32, r_mean=5.5e-9, r_std=1.5e-9, initial_ecsa=40e3,
    )
    return cl 

@pytest.fixture
def model(cycle, catalyst_layer):
    dissol = mrpd.PtDissolution(catalyst_layer=catalyst_layer,
                                platinum_dissolution=mrpd.PlatinumDissolution(rate_constant=1e-9,
                                                                              transfer_coeff_an=0.3,
                                                                              transfer_coeff_ca=0.5, 
                                                                              reference_potential=1.188,
                                                                              ), 
                                platinum_oxide_formation=mrpd.PlatinumOxideFormation(rate_constant=1.4e-10,
                                                                                     transfer_coeff_an=0.35, 
                                                                                     transfer_coeff_ca=0.5,
                                                                                     omega_platinum_oxide_formation=27e6,
                                                                                     reference_potential=0.98,
                                ),
                                platinum_oxide_dissolution=mrpd.PlatinumOxideDissolution(rate_constant=3e-23)) 
    u_cycles = cycle.u
    
    def f(t,x,u,p=None): 
        
        nr = catalyst_layer.platinum_size_distribution.n_points
        r = x[:nr,...] * 1e-9 
        cdiss = x[nr,...] * 1e-5
        pto = x[nr+1:,...] 
        catalyst_layer.platinum_size_distribution.r_array = r 
        catalyst_layer.dissolution_model = dissol 

        drdt, dcdissdt, dptodt = dissol.time_derivatives(
            cdiss,
            pto,
            1./(catalyst_layer.ionomer.dry_molar_volume
                + catalyst_layer.ionomer.equilibrium_water_content(1.)
                * mrpd.water_molar_volume(catalyst_layer.temperature)),
            u['cell-potential'](t),
            u['cell-temperature'](t),
            u['ca-inlet-rh'](t)
        )
        return np.concat((drdt * 1e9, [dcdissdt * 1e5] , dptodt), axis=0)
    
    def h(t,x,u,p=None):
        return x 
    return mrpd.DynamicModel(f=f, h=h, u = u_cycles, ode_solution_method='BDF')

@pytest.mark.skip(reason="PtDissolution.time_derivatives API changed; needs update to new signature")
def test_integration(model, catalyst_layer):
    r_0 = catalyst_layer.platinum_size_distribution.r_array
    n_r = catalyst_layer.platinum_size_distribution.n_points
    t, x, y = model.solve(np.linspace(0,18,1000), x0 = np.concat((r_0 * 1e9, [0], np.zeros_like(r_0)), axis=0), rtol=1e-10, 
                          vectorized=False)




    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(t, x[n_r+5,...], label=f'r = {x[4,0]:.1f} nm')
    ax.plot(t, x[n_r+12,...], label=f'r = {x[11,0]:.1f} nm')
    ax.plot(t, x[2*n_r-14,...], label=f'r = {x[n_r-15,0]:.1f} nm')
    ax2 = ax.twinx()
    ax2.plot(t, model.u['cell-potential'](t), 'k', label='$V_{cell}$')
    ax2.set_ylabel('Cell voltage (V)')
    ax2.set_ylim([0,1.2])
    ax.set_ylim([0,0.6])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'$\theta_\text{PtO}$ (n.d.)')
    ax.set_xlim([0,18])
    ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1.0))
    ax2.legend(loc='upper right')
    fig.tight_layout()

    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(t, x[n_r,...] * 1e12 * 1e-6 / 1e5, label=r'$C_{\text{Pt}^{2+}}$')
    ax2 = ax.twinx()
    ax2.plot(t, model.u['cell-potential'](t), 'k', label='$V_{cell}$')
    ax2.set_ylabel('Cell voltage (V)')
    ax2.set_ylim([0,1.0])
    ax.set_ylim([0,15])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'$C_{\text{Pt}^{2+}}$ (nmol/cm$^3$)')
    ax.set_xlim([0,18])
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    fig.tight_layout()


    R_diss = catalyst_layer.dissolution_model.platinum_dissolution.rate_of_reaction(
        x[n_r]*1e-5,
        x[n_r+1:,...],
        model.u['cell-potential'](t),
        model.u['cell-temperature'](t),
        model.u['ca-inlet-rh'](t), 
        x[0:n_r,...] * 1e-9,
    ) * 1e11

    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(t, R_diss[4,...], label=f'r = {x[4,0]:.1f} nm')
    ax.plot(t, R_diss[11,...], label=f'r = {x[11,0]:.1f} nm')
    ax.plot(t, R_diss[n_r-15,...], label=f'r = {x[n_r-15,0]:.1f} nm')
    ax2 = ax.twinx()
    ax2.plot(t, model.u['cell-potential'](t), 'k', label='$V_{cell}$')
    ax2.set_ylabel('Cell voltage (V)')
    ax2.set_ylim([0,1.2])
    ax.set_ylim([-2,6])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'$R_{diss}$ (pmol/cm$^2$.s)')
    ax.set_xlim([0,18])
    ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1.0))
    ax2.legend(loc='upper right')
    fig.tight_layout()
    plt.show()

    assert False

# def test_potential_cycle(cycle, catalyst_layer): 
#     u_cycles = cycle.repeat_cycles(3)
    
#     fig, ax = plt.subplots() 
#     ax.plot(u_cycles['total_time'], u_cycles['cell-potential'])
    
    
#     dissol = mrpd.PtDissolution(catalyst_layer=catalyst_layer) 
    
#     drdt, dcdissdt, dptodt = dissol.time_derivatives(
#             0,
#             0,
#             1.,
#             u_cycles['cell-potential'],
#             u_cycles['cell-temperature'],
#             u_cycles['ca-inlet-rh']
#         )
#     print(drdt.shape, dcdissdt.shape, dptodt.shape)
#     ax2 = ax.twinx()
#     ax2.plot(u_cycles['total_time'], dcdissdt, 'C1')
#     plt.show()
#     assert False