"""
***************************************
Membrane and ionomer correlations
***************************************

:doc:`/science/membrane_correlations` documents the stateless empirical
correlations attached to :class:`~marapendi.membrane.pem.PFSAIonomer`
(equilibrium sorption isotherm, proton conductivity, electroosmotic drag,
water diffusivity/absorption, H\\ :sub:`2`/O\\ :sub:`2` permeability). This
example evaluates each of them for a typical Nafion 1100 EW ionomer at
several temperatures, over the water-content (or water-activity, for the
isotherm) range relevant to operation.
"""

# %%
# Ionomer
# =======
#
# Same Nafion 1100 EW parameterization used in
# :doc:`plot_07_pwl_membrane`.

import numpy as np
import matplotlib.pyplot as plt
import marapendi as mrpd
from marapendi.models.thermo.water import water_molar_volume

nafion = mrpd.PFSAIonomer(
    equivalent_weight=1100.,
    dry_density=1980,
    reference_conductivity=50.,
    residual_conductivity=0.3,
    conductivity_fv_threshold=0.04,
    conductivity_exp=1.5,
    reference_conductivity_temperature=300.,
    conductivity_activation_energy=10.540e6,
    reference_water_absorption_coefficient=1e-5,
    reference_water_absorption_temperature=303.15,
    water_absorption_activation_energy=20e6,
    reference_water_diffusivity=2e-10,
    reference_water_diffusivity_temperature=300.,
    water_diffusivity_activation_energy=20e6,
    vapor_equilibrium_polynomial=[36, -39.85, 17.18, 0.043],
)

temperatures = np.array([303.15, 333.15, 343.15, 353.15])  # 30, 50, 80, 100 degC
colors = ["C0", "C1", "C2", "C3"]



# %%
# Equilibrium sorption isotherm
# ==============================
#
# :meth:`~marapendi.membrane.pem.PFSAIonomer.vapor_equilibrium_water_content`
# (Springer et al. 1991) does not depend on temperature by construction, so a
# single curve covers all vapor-equilibrated conditions. The
# liquid-equilibrated water content
# (:meth:`~marapendi.membrane.pem.PFSAIonomer.liquid_equilibrium_water_content`,
# Goshtasbi et al. 2020) does depend on temperature and is shown as dashed
# horizontal markers at RH = 1 (Schroeder's paradox: liquid-equilibrated
# uptake exceeds the RH -> 1 limit of the vapor isotherm).

fig, ax = plt.subplots(1,1, figsize=(5, 4))
rh = np.linspace(0.0, 1.0, 200)
ax.plot(rh, nafion.vapor_equilibrium_water_content(rh, temperatures[0]),
        "k", lw=1.5, label="Vapor isotherm (all T)")
for T, c in zip(temperatures, colors):
    lmbd_liq = nafion.liquid_equilibrium_water_content(T)
    ax.plot(1.0, lmbd_liq, "o", color=c, label=f"Liquid, T={T - 273.15:.0f} °C")
ax.set_xlabel("Relative humidity / water activity (-)")
ax.set_ylabel(r"$\lambda_{eq}$ (mol H$_2$O / mol SO$_3^-$)")
ax.set_title("Equilibrium sorption isotherm")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.4)

# %%
# Proton conductivity
# =====================
#
# :meth:`~marapendi.membrane.pem.PFSAIonomer.proton_conductivity` combines a
# water-volume-fraction power law with an Arrhenius temperature correction
# (Kusoglu and Weber, 2017).

fig, ax = plt.subplots(1,1, figsize=(5, 4))
lmbd = np.linspace(0.5, 20, 200)
for T, c in zip(temperatures, colors):
    ax.plot(lmbd, nafion.proton_conductivity(lmbd, T), color=c, label=f"T={T - 273.15:.0f} °C")
ax.set_xlabel(r"$\lambda$ (mol H$_2$O / mol SO$_3^-$)")
ax.set_ylabel(r"$\sigma_\mathrm{ion}$ (S/m)")
ax.set_title("Proton conductivity")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.4)

# %%
# Electroosmotic drag coefficient and H\ :sub:`2`/O\ :sub:`2` permeability
# ===========================================================================
#
# :meth:`~marapendi.membrane.pem.PFSAIonomer.calculate_electroosmotic_drag_coefficient`
# is linear in both :math:`\lambda` and :math:`T`.
# :meth:`~marapendi.membrane.pem.PFSAIonomer.h2_permeability` and
# :meth:`~marapendi.membrane.pem.PFSAIonomer.o2_permeability` (Goshtasbi et
# al., 2020) are shown together, in solid/dashed line style, against the
# water volume fraction they actually depend on
# (:meth:`~marapendi.membrane.ionomer_base.Ionomer.water_vol_fraction`).

fig, ax = plt.subplots(1,1, figsize=(5, 4))
for T, c in zip(temperatures, colors):
    fv = nafion.water_vol_fraction(lmbd, water_molar_volume(T))
    ax.plot(fv, nafion.h2_permeability(lmbd, T) * 1e15, color=c, ls="-",
             label=f"H$_2$, T={T - 273.15:.0f} °C")
    ax.plot(fv, nafion.o2_permeability(lmbd, T) * 1e15, color=c, ls="--",
             label=f"O$_2$, T={T - 273.15:.0f} °C")
ax.set_xlabel(r"Water volume fraction $f_v$ (-)")
ax.set_ylabel(r"Permeability ($\times 10^{-15}$ kmol/m/s/Pa)")
ax.set_title(r"H$_2$/O$_2$ permeability (solid/dashed)")
ax.legend(fontsize=6, ncol=2)
ax.grid(True, alpha=0.4)

# %%
# Water diffusivity and absorption coefficient
# ===============================================
#
# :meth:`~marapendi.membrane.ionomer_base.Ionomer.calculate_water_diffusivity`
# and
# :meth:`~marapendi.membrane.ionomer_base.Ionomer.calculate_water_absorption_coefficient`
# depend only on temperature (Arrhenius corrections on a reference value),
# so they are plotted against T directly rather than water content.

fig, ax = plt.subplots(1,1, figsize=(5, 4))
T_range = np.linspace(temperatures[0], temperatures[-1], 100)
ax.plot(T_range - 273.15, nafion.calculate_water_diffusivity(T_range) * 1e10,
        "C0", label=r"$D_w$ ($\times 10^{-10}$ m$^2$/s)")
ax2 = ax.twinx()
ax2.plot(T_range - 273.15, nafion.calculate_water_absorption_coefficient(T_range) * 1e5,
         "C1", label=r"$k_{abs}$ ($\times 10^{-5}$ m/s)")
ax.set_xlabel("Temperature (°C)")
ax.set_ylabel(r"$D_w$ ($\times 10^{-10}$ m$^2$/s)", color="C0")
ax2.set_ylabel(r"$k_{abs}$ ($\times 10^{-5}$ m/s)", color="C1")
ax.set_title("Water diffusivity / absorption coefficient")
ax.grid(True, alpha=0.4)

fig.tight_layout()
plt.show()