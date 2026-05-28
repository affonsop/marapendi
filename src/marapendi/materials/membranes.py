"""
Data from Wei et al. J. Power Sources 557, nᵒ 232494 (2023): 232494. https://doi.org/10.1016/j.jpowsour.2022.232494.
"""
from marapendi.components.membrane import Membrane
import numpy as np

Nafion_N115 = Membrane(
    thickness=127e-6,                                               # m
    dry_density=2.0 * 1e3,                                         # kg/m³
    equivalent_weight=1100,                                         # kg/kmol
    darken_num=np.array([1, -0.9211, 0.5870, -0.1291, 0.01090]),
    darken_den=np.array([1, -0.8424, 0.3858, -0.07920, 6.066e-3, 8.154e-4]),
    sorption_isotherm_coeffs=np.array([0.043, 17.81, -39.85, 36.0]),
    reference_liquid_equilibrium_water_content=22,                  # kmol_H2O/kmol_ion
    reference_water_diffusivity=3.74 * 2.72e-5 * 1e-4,             # m²/s
    reference_desorption_coefficient=0.0184 * 4.59e-2 * 1e-2,      # m/s
    ionomer_activation_energy=2.29 * 20e6,                                  # J/kmol
)

Nafion_N117 = Membrane(
    thickness=183e-6,                                               # m
    dry_density=2.0 * 1e3,                                         # kg/m³
    equivalent_weight=1100,                                         # kg/kmol
    darken_num=np.array([1, -0.9211, 0.5870, -0.1291, 0.01090]),
    darken_den=np.array([1, -0.8424, 0.3858, -0.07920, 6.066e-3, 8.154e-4]),
    sorption_isotherm_coeffs=np.array([0.043, 17.81, -39.85, 36.0]),
    reference_liquid_equilibrium_water_content=22,                  # kmol_H2O/kmol_ion
    reference_water_diffusivity=2.74 * 2.72e-5 * 1e-4,             # m²/s
    reference_desorption_coefficient=0.0224 * 4.59e-2 * 1e-2,      # m/s
    ionomer_activation_energy=2.18 * 20e6,                                  # J/kmol
)

Nafion_N211 = Membrane(
    thickness=25e-6,                                                # m
    dry_density=2.0 * 1e3,                                         # kg/m³
    equivalent_weight=1100,                                         # kg/kmol
    darken_num=np.array([1, -0.9211, 0.5870, -0.1291, 0.01090]),
    darken_den=np.array([1, -0.8424, 0.3858, -0.07920, 6.066e-3, 8.154e-4]),
    sorption_isotherm_coeffs=np.array([0.043, 17.81, -39.85, 36.0]),
    reference_liquid_equilibrium_water_content=22,                  # kmol_H2O/kmol_ion
    reference_water_diffusivity=0.260 * 2.72e-5 * 1e-4,            # m²/s
    reference_desorption_coefficient=0.0160 * 4.59e-2 * 1e-2,      # m/s
    ionomer_activation_energy=2.79 * 20e6,                                  # J/kmol
)

Nafion_N212 = Membrane(
    thickness=50e-6,                                                # m
    dry_density=2.0 * 1e3,                                         # kg/m³
    equivalent_weight=1100,                                         # kg/kmol
    darken_num=np.array([1, -0.9211, 0.5870, -0.1291, 0.01090]),
    darken_den=np.array([1, -0.8424, 0.3858, -0.07920, 6.066e-3, 8.154e-4]),
    sorption_isotherm_coeffs=np.array([0.043, 17.81, -39.85, 36.0]),
    reference_liquid_equilibrium_water_content=22,                  # kmol_H2O/kmol_ion
    reference_water_diffusivity=0.314 * 2.72e-5 * 1e-4,            # m²/s
    reference_desorption_coefficient=0.0211 * 4.59e-2 * 1e-2,      # m/s
    ionomer_activation_energy=2.54 * 20e6,                                  # J/kmol
)

Aemion_AH1_HNN8_50_X = Membrane(
    thickness=50e-6,                                                # m
    dry_density=1.15 * 1e3,                                        # kg/m³
    equivalent_weight=441,                                          # kg/kmol
    darken_num=np.array([1, 0.5167, 0.01733, -0.01172, 1.091e-3]),
    darken_den=np.array([1, 0.07799, 0.01949, 2.814e-3, 1.091e-3, 0.0]),
    sorption_isotherm_coeffs=np.array([0.0, 18.31, -32.57, 28.06]),
    reference_liquid_equilibrium_water_content=25,                  # kmol_H2O/kmol_ion
    reference_water_diffusivity=0.111 * 2.72e-5 * 1e-4,            # m²/s
    reference_desorption_coefficient=0.0109 * 4.59e-2 * 1e-2,      # m/s
    ionomer_activation_energy=2.51 * 20e6,                                  # J/kmol
)

Fumapem_FAA3_30 = Membrane(
    thickness=30e-6,                                                # m
    dry_density=1.46 * 1e3,                                        # kg/m³
    equivalent_weight=552,                                          # kg/kmol
    darken_num=np.array([1, -0.05586, 0.05000, -0.01140, 1.006e-3]),
    darken_den=np.array([1, -0.2384, 0.07011, -0.006739, 1.006e-3, 0.0]),
    sorption_isotherm_coeffs=np.array([0.0, 12.11, -15.76, 13.56]),
    reference_liquid_equilibrium_water_content=12.5,                # kmol_H2O/kmol_ion
    reference_water_diffusivity=0.366 * 2.72e-5 * 1e-4,            # m²/s
    reference_desorption_coefficient=0.0358 * 4.59e-2 * 1e-2,      # m/s
    ionomer_activation_energy=2.29 * 20e6,                                  # J/kmol
)

Fumapem_FAA3_50 = Membrane(
    thickness=50e-6,                                                # m
    dry_density=1.46 * 1e3,                                        # kg/m³
    equivalent_weight=552,                                          # kg/kmol
    darken_num=np.array([1, -0.05586, 0.05000, -0.01140, 1.006e-3]),
    darken_den=np.array([1, -0.2384, 0.07011, -0.006739, 1.006e-3, 0.0]),
    sorption_isotherm_coeffs=np.array([0.0, 12.11, -15.76, 13.56]),
    reference_liquid_equilibrium_water_content=12.5,                # kmol_H2O/kmol_ion
    reference_water_diffusivity=0.577 * 2.72e-5 * 1e-4,            # m²/s
    reference_desorption_coefficient=0.0448 * 4.59e-2 * 1e-2,      # m/s
    ionomer_activation_energy=2.07 * 20e6,                                  # J/kmol
)

PiperION_A40 = Membrane(
    thickness=40e-6,                                                # m
    dry_density=1.30 * 1e3,                                        # kg/m³
    equivalent_weight=469,                                          # kg/kmol
    darken_num=np.array([1, -0.3692, 0.1007, -0.01468, 9.297e-4]),
    darken_den=np.array([1, -0.4411, 0.1085, -0.01334, 9.297e-4, 0.0]),
    sorption_isotherm_coeffs=np.array([0.0, 13.13, -14.81, 14.41]),
    reference_liquid_equilibrium_water_content=13.8,                # kmol_H2O/kmol_ion
    reference_water_diffusivity=0.270 * 2.72e-5 * 1e-4,            # m²/s
    reference_desorption_coefficient=0.0307 * 4.59e-2 * 1e-2,      # m/s
    ionomer_activation_energy=2.31 * 20e6,                                  # J/kmol
)