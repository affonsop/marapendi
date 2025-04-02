import pytest
import numpy as np
import coulomb as cb
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression


@pytest.fixture
def cl(): 
    return cb.CatalystLayer(thickness=2.7*1.2*4/3*1e-6,
                            platinum_loading=0.12e-2, 
                            ionomer_to_carbon_ratio=0.75, 
                            catalyst_platinum_weight_percent=0.3,
                            carbon_agglomerate_radius=25e-9, 
                            ionomer=cb.CatalystLayerIonomerModel(dry_density=2004, equivalent_weight=952, 
                                                                 conductivity_correction=1., conductivity_exp=1.5),
                            reaction = cb.ElectrochemicalReaction(reference_exchange_current_density=2.47e-8 * 3e11 * 10e-6,
                                                                activation_energy=67e6,
                                                                reaction_order=0.54,
                                                                reference_activity=1.,
                                                                reference_temperature=353.15,
                                                                number_of_electrons=2,
                                                                charge_transfer_coeff=0.5)
                            ) #TEC10V30E, 0.2 mgPt/cm2, 0.75 I/C

# Data from Jinnouchi et al. (2021)
relative_humidity = [0.3, 0.6, 0.9] 
ionomer_water_content = [3.8, 6.4, 11.6] # From figure 3a in supplementary material
proton_conductivity = [0.45, 2.5, 8.3]  # From figure 3b in supplementary material
ionomer_film_resistance = [6200., 5660., 3125.] # From figure 4b 
proton_resistance = [545e-7, 137e-7, 24e-7] # From figure 3c
local_resistance = [28.5, 18.7, 13.8] # From figure 3b 
o2_diff_coeff =[2e-10, 3.5e-10, 5.5e-10]
o2_perm=[10e-15,13e-15,18e-15] # From Kudo et al. (2016)
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

def test_ionomer_o2_transport(cl): 
    
    for k in range(3):
        
        assert np.isclose(cl.ionomer_film_thickness, 6e-9, atol=2e-10)
        assert np.isclose(cl.ionomer.o2_permeability(ionomer_water_content[k], temperature=353.),  1e-3*o2_perm[k], atol=5e-15)
        assert np.isclose(cl.ionomer.o2_film_diffusion_coefficient(ionomer_water_content[k], temperature=353.), o2_diff_coeff[k], atol=1e-10)
        #assert np.isclose(cl.o2_ionomer_film_resistance(ionomer_water_content[k], temperature=353.), ionomer_film_resistance[k], 10e-2)
        assert np.isclose(cl.o2_ionomer_film_resistance(ionomer_water_content[k], temperature=353.15), local_resistance[k], rtol=0.3)

        # assert np.isclose(cl.ionomer.proton_conductivity(relative_humidity[k], 0, temperature=353.15),  proton_conductivity[k], 12e-2)
        # assert np.isclose(cl.ionomer_sheet_proton_resistance(relative_humidity[k], ionomer_water_content[k], temperature=353.15), 
        #                    proton_resistance[k], rtol=25e-2, atol=5e-7)
        
def test_neyerlin_correction(cl): 
    assert np.isclose(cl.reaction.tafel_slope(353.15), 70e-3, atol=2e-3)
    R_cl_sheet = 1e-6
    nu = 1e4 * R_cl_sheet / cl.reaction.tafel_slope(temperature=353.15)
    assert np.isclose(nu, 0.14184, atol=0.001)

    # nu = 1e4 * cl.ionomer_sheet_proton_resistance(14, temperature=353.15) / cl.reaction.tafel_slope(temperature=353.15)
    # assert np.isclose(cl.effective_proton_resistance(1e4, 14, temperature=353.15), 
    #                   cl.ionomer_sheet_proton_resistance(14, temperature=353.15) / (3 + nu), rtol=1e-2)

def test_ionomer_proton_conductivity(cl): 
    for k in range(3):
        print(cl.ionomer_vol_fraction)
        assert np.isclose(cl.ionomer_sheet_proton_resistance( ionomer_water_content[k], temperature=298.15),  proton_resistance[k], rtol=0.6)
        plt.plot(relative_humidity[k], cl.ionomer_sheet_proton_resistance(ionomer_water_content[k], temperature=298.15),'C0o')
        plt.semilogy(relative_humidity[k], proton_resistance[k],'C0s')

    assert np.isclose(cl.ionomer.proton_conductivity(cl.ionomer.equilibrium_water_content(0.97), temperature=298.15),  5.6, 10e-2)
    assert np.isclose(cl.ionomer.proton_conductivity(cl.ionomer.equilibrium_water_content(0.81), temperature=298.15),  3.7, 10e-2)
    assert np.isclose(cl.ionomer.proton_conductivity(cl.ionomer.equilibrium_water_content(0.60), temperature=298.15),  1.91, 10e-2)
    
# ## RH vs sigma_p ionomer at 298.15 K, Jinnouchi et al. (2021), sup material
# 0,9662698412698414; 5,633142670601358
# 0,8115079365079365; 3,72023668141307
# 0,6031746031746033; 1,9155005555735298
# 0,41269841269841273; 0,8588828559546259
# 0,18650793650793657; 0,20956623994804352
# 0,09920634920634924; 0,05256791122018435
    

# RH	Lmbd
#  0,0164  	 0,6593  
#  0,0509  	 1,8132  
#  0,1147  	 2,5275  
#  0,1712  	 3,1044  
#  0,2096  	 3,3242  
#  0,2296  	 3,5165  
#  0,2770  	 3,8187  
#  0,3016  	 3,9835  
#  0,3536  	 4,2857  
#  0,4257  	 4,7802  
#  0,5013  	 5,3297  
#  0,5797  	 5,9890  
#  0,6627  	 6,6209  
#  0,7411  	 7,3901  
#  0,8122  	 8,3516  
#  0,8787  	 9,3132  
#  0,9343  	 10,4121  
#  0,9761  	 11,9231    