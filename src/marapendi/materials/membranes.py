"""
Data from Wei et al. J. Power Sources 557, nᵒ 232494 (2023): 232494. https://doi.org/10.1016/j.jpowsour.2022.232494.
"""
from marapendi.components.membrane import PFSAMembrane, Membrane

import numpy as np

Nafion_N115 = PFSAMembrane(
    thickness=127e-6,                                               # m
    rho_dry_ion=2.0 * 1e3,                                         # kg/m³
    EW_ion=1100,                                         # kg/kmol
    darken_num_ion=np.array([1, -0.9211, 0.5870, -0.1291, 0.01090]),
    darken_den_ion=np.array([1, -0.8424, 0.3858, -0.07920, 6.066e-3, 8.154e-4]),
    sorption_coeffs_ion=np.array([0.043, 17.81, -39.85, 36.0]),
    lmbd_liq_ref_ion=22,                  # kmol_H2O/kmol_ion
    D_lmbd_ref_ion=3.74 * 2.72e-5 * 1e-4,             # m²/s
    k_des_ref_ion=0.0184 * 4.59e-5,      # m/s
    E_act_ion=2.29 * 20e6,                                  # J/kmol
    sigma_ref_ion = 50., 
    f_v_perc_ion= 0.1,
    n_sigma_ion=1.5,
    T_ref_sigma_ion=298.15,
    T_ref_D_ion=303.15,
    T_ref_des_ion=303.15,   
)

Nafion_N117 = PFSAMembrane(
    thickness=183e-6,                                               # m
    rho_dry_ion=2.0 * 1e3,                                         # kg/m³
    EW_ion=1100,                                         # kg/kmol
    darken_num_ion=np.array([1, -0.9211, 0.5870, -0.1291, 0.01090]),
    darken_den_ion=np.array([1, -0.8424, 0.3858, -0.07920, 6.066e-3, 8.154e-4]),
    sorption_coeffs_ion=np.array([0.043, 17.81, -39.85, 36.0]),
    lmbd_liq_ref_ion=22,                  # kmol_H2O/kmol_ion
    D_lmbd_ref_ion=2.74 * 2.72e-5 * 1e-4,             # m²/s
    k_des_ref_ion=0.0224 * 4.59e-5,      # m/s
    E_act_ion=2.18 * 20e6,        
    sigma_ref_ion = 50., 
    f_v_perc_ion= 0.1,
    n_sigma_ion=1.5,
    T_ref_sigma_ion=298.15,
    T_ref_D_ion=303.15,
    T_ref_des_ion=303.15,                         
)

Nafion_N211 = PFSAMembrane(
    thickness=25e-6,                                                # m
    rho_dry_ion=2.0 * 1e3,                                         # kg/m³
    EW_ion=1100,                                         # kg/kmol
    darken_num_ion=np.array([1, -0.9211, 0.5870, -0.1291, 0.01090]),
    darken_den_ion=np.array([1, -0.8424, 0.3858, -0.07920, 6.066e-3, 8.154e-4]),
    sorption_coeffs_ion=np.array([0.043, 17.81, -39.85, 36.0]),
    lmbd_liq_ref_ion=22,                  # kmol_H2O/kmol_ion
    D_lmbd_ref_ion=0.260 * 2.72e-5 * 1e-4,            # m²/s
    k_des_ref_ion=0.0160 * 4.59e-5,      # m/s
    E_act_ion=2.79 * 20e6,                                  # J/kmol
    sigma_ref_ion = 50., 
    f_v_perc_ion= 0.1,
    n_sigma_ion=1.5,
    T_ref_sigma_ion=298.15,
    T_ref_D_ion=303.15,
    T_ref_des_ion=303.15,    
)

Nafion_N212 = PFSAMembrane(
    thickness=50e-6,                                                # m
    rho_dry_ion=2.0 * 1e3,                                         # kg/m³
    EW_ion=1100,                                         # kg/kmol
    darken_num_ion=np.array([1, -0.9211, 0.5870, -0.1291, 0.01090]),
    darken_den_ion=np.array([1, -0.8424, 0.3858, -0.07920, 6.066e-3, 8.154e-4]),
    sorption_coeffs_ion=np.array([0.043, 17.81, -39.85, 36.0]),
    lmbd_liq_ref_ion=22,                  # kmol_H2O/kmol_ion
    D_lmbd_ref_ion=0.314 * 2.72e-5 * 1e-4,            # m²/s
    k_des_ref_ion=0.0211 * 4.59e-5,      # m/s
    E_act_ion=2.54 * 20e6,                                  # J/kmol
    sigma_ref_ion = 50., 
    f_v_perc_ion= 0.1,
    n_sigma_ion=1.5,
    T_ref_sigma_ion=298.15,
    T_ref_D_ion=303.15,
    T_ref_des_ion=303.15,    
)

Aemion_AH1_HNN8_50_X = Membrane(
    thickness=50e-6,                                                # m
    rho_dry_ion=1.15 * 1e3,                                        # kg/m³
    EW_ion=441,                                          # kg/kmol
    darken_num_ion=np.array([1, 0.5167, 0.01733, -0.01172, 1.091e-3]),
    darken_den_ion=np.array([1, 0.07799, 0.01949, 2.814e-3, 1.091e-3, 0.0]),
    sorption_coeffs_ion=np.array([0.0, 18.31, -32.57, 28.06]),
    lmbd_liq_ref_ion=25,                  # kmol_H2O/kmol_ion
    D_lmbd_ref_ion=0.111 * 2.72e-5 * 1e-4,            # m²/s
    k_des_ref_ion=0.0109 * 4.59e-5,      # m/s
    E_act_ion=2.51 * 20e6,                                  # J/kmol
)

Fumapem_FAA3_30 = Membrane(
    thickness=30e-6,                                                # m
    rho_dry_ion=1.46 * 1e3,                                        # kg/m³
    EW_ion=552,                                          # kg/kmol
    darken_num_ion=np.array([1, -0.05586, 0.05000, -0.01140, 1.006e-3]),
    darken_den_ion=np.array([1, -0.2384, 0.07011, -0.006739, 1.006e-3, 0.0]),
    sorption_coeffs_ion=np.array([0.0, 12.11, -15.76, 13.56]),
    lmbd_liq_ref_ion=12.5,                # kmol_H2O/kmol_ion
    D_lmbd_ref_ion=0.366 * 2.72e-5 * 1e-4,            # m²/s
    k_des_ref_ion=0.0358 * 4.59e-5,      # m/s
    E_act_ion=2.29 * 20e6,                                  # J/kmol
)

Fumapem_FAA3_50 = Membrane(
    thickness=50e-6,                                                # m
    rho_dry_ion=1.46 * 1e3,                                        # kg/m³
    EW_ion=552,                                          # kg/kmol
    darken_num_ion=np.array([1, -0.05586, 0.05000, -0.01140, 1.006e-3]),
    darken_den_ion=np.array([1, -0.2384, 0.07011, -0.006739, 1.006e-3, 0.0]),
    sorption_coeffs_ion=np.array([0.0, 12.11, -15.76, 13.56]),
    lmbd_liq_ref_ion=12.5,                # kmol_H2O/kmol_ion
    D_lmbd_ref_ion=0.577 * 2.72e-5 * 1e-4,            # m²/s
    k_des_ref_ion=0.0448 * 4.59e-5,      # m/s
    E_act_ion=2.07 * 20e6,                                  # J/kmol
)

PiperION_A40 = Membrane(
    thickness=40e-6,                                                # m
    rho_dry_ion=1.30 * 1e3,                                        # kg/m³
    EW_ion=469,                                          # kg/kmol
    darken_num_ion=np.array([1, -0.3692, 0.1007, -0.01468, 9.297e-4]),
    darken_den_ion=np.array([1, -0.4411, 0.1085, -0.01334, 9.297e-4, 0.0]),
    sorption_coeffs_ion=np.array([0.0, 13.13, -14.81, 14.41]),
    lmbd_liq_ref_ion=13.8,                # kmol_H2O/kmol_ion
    D_lmbd_ref_ion=0.270 * 2.72e-5 * 1e-4,            # m²/s
    k_des_ref_ion=0.0307 * 4.59e-5,      # m/s
    E_act_ion=2.31 * 20e6,                                  # J/kmol
)