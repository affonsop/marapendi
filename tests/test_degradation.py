import pytest
import numpy as np
import pandas as pd
import marapendi as mrpd
import matplotlib.pyplot as plt 


# def test_pt_distribution():
#     dist = mrpd.PtRadiusDistribution()
#     dist.plot_distribution()
  
#     # print(dist.initial_dist.interval(0.999))
#     # print(dist.platinum_radius_distribution)
#     print(dist.utilization_ratio)
#     print(dist.geometrical_specific_surface_area())
#     print(dist.ecsa())
#     plt.show()
#     assert False 
# def test_pt_dissolution(): 
#     dissol = mrpd.PtDissolution()
#     phi = np.linspace(0,2,10)
#     plt.figure()
#    #plt.plot(phi, dissol.platinum_dissolution_rate_of_reaction(0,0.,phi, 353.15, 4e-9))
#     #plt.semilogy(phi, dissol.platinum_oxide_formation_rate_of_reaction(0,0,0,phi, 353.15, 4e-9))
#     plt.semilogy(phi, np.ones_like(phi) * dissol.platinum_oxide_dissolution.rate_of_reaction(0,.8,0,353.15, 4e-9, dissol.platinum_dissolution, dissol.platinum_oxide_formation))

#     plt.figure()
#     plt.plot(np.linspace(2e-9,6e-9,10), dissol.platinum_dissolution.equilibrium_potential(np.linspace(2e-9,6e-9,10)))
#     plt.show()
#     assert False

@pytest.fixture
def cycle(): 
    cycle = mrpd.LoadCycle(
        duration=6,
        time_step=.01,
    )
    step_duration = 3. 
    lower_potential = 0.6 
    upper_potential = 0.9
    potential_change_duration = 0.6
    cell_potential = lambda t: np.interp(t, [0, step_duration-potential_change_duration, step_duration, 2 * step_duration-potential_change_duration, 2 * step_duration], [lower_potential, lower_potential, upper_potential, upper_potential, lower_potential])
    cell_temperature = 353.15
    cell_pressure = 1.5e5
    
    cycle.set_input_dict({
            'cell-potential':               cell_potential,
            'cell-temperature':             lambda t: cell_temperature,
            'current-density':              lambda t: 0, 
            'ca-inlet-temperature':         lambda t: cell_temperature,
            'ca-inlet-rh':                  lambda t: 0.8,
            'ca-inlet-pressure':            lambda t: cell_pressure,
            'ca-outlet-pressure':           lambda t: None,
            'ca-dry-o2-mole-fraction':      lambda t: 0.,
            'ca-dry-h2-mole-fraction':      lambda t: 0.,
            'ca-inlet-gas-flow-rate':       lambda t: 0.,
            'ca-inlet-liquid-flow-rate':    lambda t: 0.,
            'ca-inlet-liquid-saturation':   lambda t: 0.,
            'ca-stoichiometry':             lambda t: 2, 
            'an-inlet-temperature':         lambda t: cell_temperature,
            'an-inlet-rh':                  lambda t: 0.8,
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


def test_potential_cycle(cycle): 
    u_cycles = cycle.repeat_cycles(3)
    print(cycle.cycle_time)
    fig, ax = plt.subplots() 
    ax.plot(u_cycles['total_time'], u_cycles['cell-potential'])
    
    cl = mrpd.CatalystLayer(ionomer_vol_fraction=0.3, 
                            ionomer=mrpd.CatalystLayerIonomer(equivalent_weight=1100.))

    cl.platinum_size_distribution = mrpd.PtSizeDistribution()
    dissol = mrpd.PtDissolution(catalyst_layer=cl) 
    
    drdt, dcdissdt, dptodt = dissol.time_derivatives(
            0,
            0,
            1.,
            u_cycles['cell-potential'],
            u_cycles['cell-temperature'],
            u_cycles['ca-inlet-rh']
        )
    
    ax2 = ax.twinx()
    ax2.plot(u_cycles['total_time'], dcdissdt, 'C1')
    plt.show()
    assert False