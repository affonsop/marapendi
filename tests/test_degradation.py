import pytest
import numpy as np
import pandas as pd
import marapendi as mrpd
import matplotlib.pyplot as plt 


def test_pt_dissolution(): 
    dissol = mrpd.PtDissolution()
    phi = np.linspace(0,2,10)
    plt.plot(phi, dissol.platinum_dissolution_rate_of_reaction(0,0.,phi, 353.15, 4e-9))
    plt.semilogy(phi, dissol.platinum_oxide_formation_rate_of_reaction(0,0,0,phi, 353.15, 4e-9))
    plt.figure()
    plt.plot(np.linspace(2e-9,6e-9,10), dissol.platinum_dissolution_equilibrium_potential(np.linspace(2e-9,6e-9,10)))
    plt.show()
    assert False
