"""
Physical constants used throughout :mod:`marapendi`.

Values match Cantera's SI (kmol-based) constants so that no Cantera import
is required.
"""

GAS_CONSTANT = 8314.46261815324
"""Universal gas constant (J / (kmol K))."""

FARADAY_CONSTANT = 96485332.12331001
"""Faraday constant (C / kmol)."""

WATER_MOLECULAR_WEIGHT = 18.016
"""Molecular weight of water (kg / kmol)."""

# ---------------------------------------------------------------------------
# Standard thermodynamic quantities for the reaction H2 + 0.5 O2 → H2O
# at T_ref = 298.15 K, P_ref = 1e5 Pa.
# Values computed from Cantera's GRI-3.0 / water.yaml databases and
# hardcoded here to remove the runtime dependency.
# Units: J / kmol  and  J / (kmol K)
# ---------------------------------------------------------------------------

STD_TEMPERATURE = 298.15
"""Standard reference temperature (K)."""

STD_FORMATION_ENTHALPY_H2OV = -241_824_621.6  # J/kmol  (H2O gas)
STD_FORMATION_ENTROPY_H2OV  =     -44_426.4   # J/(kmol·K)
STD_FORMATION_ENTHALPY_H2OL = -285_828_371.0  # J/kmol  (H2O liquid)
STD_FORMATION_ENTROPY_H2OL  =    -163_315.7   # J/(kmol·K)

STD_FORMATION_GIBBS_H2OV = STD_FORMATION_ENTHALPY_H2OV - STD_TEMPERATURE * STD_FORMATION_ENTROPY_H2OV
STD_FORMATION_GIBBS_H2OL = STD_FORMATION_ENTHALPY_H2OL - STD_TEMPERATURE * STD_FORMATION_ENTROPY_H2OL

# Quadratic polynomial fits for H2 enthalpy of combustion as a function of T (K).
# h2_lhv(T) = np.polyval(H2_LHV_COEFFS, T)   — product H2O(g), J/kmol
# h2_hhv(T) = np.polyval(H2_HHV_COEFFS, T)   — product H2O(l), J/kmol
# Fit from Cantera; max error < 0.003 % over 300–473 K.
H2_LHV_COEFFS = (5.77677039e-01, -1.04830008e+04, -2.38748300e+08)
H2_HHV_COEFFS = (1.15544507e+01,  2.37559933e+04, -2.93929942e+08)
