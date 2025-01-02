import pytest
import numpy as np
import coulomb as cb
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression


@pytest.fixture
def cl(): 
    return cb.CatalystLayer(thickness=5.4e-6,
                            platinum_loading=0.12e-2, 
                            ionomer_to_carbon_ratio=0.75, 
                            catalyst_platinum_weight_percent=0.3,
                            carbon_agglomerate_radius=26e-9, 
                            reaction = cb.ElectrochemicalReaction(reference_exchange_current_density=2.47e-8 * 3e11 * 10e-6,
                                                                activation_energy=67e6,
                                                                reaction_order=0.54,
                                                                reference_activity=1.,
                                                                reference_temperature=353.15,
                                                                number_of_electrons=2,
                                                                charge_transfer_coeff=0.5)
                            ) #TEC10V30E, 0.2 mgPt/cm2, 0.75 I/C


relative_humidity = [0.3, 0.6, 0.9] 
ionomer_water_content = [3.8, 6.4, 11.6] # From figure 3a in supplementary material
proton_conductivity = [0.45, 2.5, 8.3]  # From figure 3b in supplementary material
ionomer_film_resistance = [6200., 5660., 3125.] # From figure 4b 
proton_resistance = [545e-7, 137e-7, 24e-7] # From figure 3c
local_resistance = [28.5, 18.7, 13.8] # From figure 3b 

# model = LinearRegression()
# model.fit(np.array(lambda_nafion).reshape(-1, 1), np.array(ionomer_film_resistance))
# lmbd = np.linspace(0,14,10).reshape(-1, 1)
# plt.plot(lambda_nafion, ionomer_film_resistance, 's', label='Jinnouchi et al. (2021)')
# plt.plot(lmbd, model.predict(lmbd), label='({:.0f}$\lambda$ + {:.0f}) s/m'.format(model.coef_[0], model.intercept_))
# plt.xlabel('Ionomer water content (n.d.)')
# plt.ylabel('Ionomer interfacial resistance (s/m)')
# plt.legend()
# plt.tight_layout()
# plt.show()

def test_catalyst_layer(cl): 
    
    for k in range(3):
        print(relative_humidity[k])
        assert np.isclose(cl.ionomer.o2_film_resistance(ionomer_water_content[k], temperature=353.), ionomer_film_resistance[k], 10e-2)
        assert np.isclose(cl.calculate_film_resistance(ionomer_water_content[k], temperature=353.15), local_resistance[k], 20e-2)

        assert np.isclose(cl.ionomer.proton_conductivity(relative_humidity[k], 0, temperature=353.15),  proton_conductivity[k], 12e-2)
        assert np.isclose(cl.calculate_ionomer_sheet_proton_resistance(relative_humidity[k], ionomer_water_content[k], temperature=353.15), 
                           proton_resistance[k], rtol=25e-2, atol=5e-7)
        
def test_neyerlin_correction(cl): 
    assert np.isclose(cl.reaction.tafel_slope(353.15), 70e-3, atol=2e-3)
    R_cl_sheet = cl.calculate_ionomer_sheet_proton_resistance(1, 0, 353.15)
    nu = 1e4 * cl.calculate_ionomer_sheet_proton_resistance(1, 0, 353.15) / cl.reaction.tafel_slope(temperature=353.15)
    assert np.isclose(nu, 0.36, atol=0.001)
    assert np.isclose(cl.calculate_effective_proton_resistance(1e4, 1., 0, temperature=353.15), R_cl_sheet / (3 + 0.246), rtol=1e-2)

