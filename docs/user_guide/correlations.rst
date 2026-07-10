Verifying correlations
=======================

Every physical correlation in **marapendi** is exposed as a method on the
relevant component or ionomer object, so it can be called directly without
running the full simulation.  This lets you plot and verify correlations against
literature data or your own measurements before setting up a simulation.

.. seealso::

   The derivations and literature references behind these correlations are
   documented in :doc:`/science/membrane_correlations`, :doc:`/science/two_phase_flow`,
   :doc:`/science/gas_transport`, and :doc:`/science/orr_kinetics`.
   :doc:`/auto_examples/plot_09_membrane_correlations` reproduces the
   membrane/ionomer plots below (and a few more) for several temperatures in
   one figure.

Membrane conductivity vs water content
---------------------------------------

:meth:`~marapendi.membrane.pem.PFSAIonomer.proton_conductivity` returns the
proton conductivity in S/m as a function of the local water content and
temperature:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    import marapendi as mrpd

    ionomer = mrpd.PFSAIonomer(equivalent_weight=1100, dry_density=1980)

    lmbd = np.linspace(0, 22, 200)   # mol H₂O / mol SO₃⁻

    fig, ax = plt.subplots(figsize=(6, 4))
    for T_C, c in [(25, "C0"), (60, "C1"), (80, "C2")]:
        T = T_C + 273.15
        sigma = ionomer.proton_conductivity(lmbd, T)
        ax.plot(lmbd, sigma, color=c, label=f"{T_C} °C")

    ax.set_xlabel("Water content λ (mol/mol)")
    ax.set_ylabel("Proton conductivity σ (S/m)")
    ax.set_title("PFSA membrane conductivity (Springer et al. 1991 / Arrhenius)")
    ax.legend()
    ax.grid(True, alpha=0.3)

To plot conductivity against **water volume fraction** f_v (the variable used
in the empirical fit):

.. code-block:: python

    from marapendi.models.thermo.water import water_molar_volume

    T_ref = 353.15
    fv = ionomer.water_vol_fraction(lmbd, water_molar_volume(T_ref))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fv, ionomer.proton_conductivity(lmbd, T_ref))
    ax.set_xlabel("Water volume fraction f_v")
    ax.set_ylabel("Conductivity σ (S/m)")
    ax.set_title(f"Conductivity vs water volume fraction at {T_ref - 273.15:.0f} °C")
    ax.grid(True, alpha=0.3)

Equilibrium isotherm λ(RH)
---------------------------

:meth:`~marapendi.membrane.pem.PFSAIonomer.vapor_equilibrium_water_content`
evaluates the cubic polynomial from Springer et al. (1991):

.. code-block:: python

    rh = np.linspace(0, 1, 200)

    fig, ax = plt.subplots(figsize=(6, 4))
    for T_C, c in [(25, "C0"), (60, "C1"), (80, "C2")]:
        lmbd_eq = ionomer.vapor_equilibrium_water_content(rh, T_C + 273.15)
        ax.plot(rh * 100, lmbd_eq, color=c, label=f"{T_C} °C")

    ax.set_xlabel("Relative humidity (%)")
    ax.set_ylabel("Equilibrium water content λ (mol/mol)")
    ax.set_title("PFSA equilibrium isotherm")
    ax.legend()
    ax.grid(True, alpha=0.3)

Compare the piecewise-linear approximation used by the standard water-balance
model against the cubic polynomial (see :doc:`/auto_examples/plot_07_pwl_membrane`
for a more complete version of this comparison, including the resulting
polarization curve):

.. code-block:: python

    lmbd_pwl = ionomer.linear_rh_from_water_content(lmbd)
    rh_cubic = ionomer.vapor_equilibrium_water_content(rh, ionomer.pwl_temperature) / lmbd

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    T_ref = ionomer.pwl_temperature

    lmbd_ref = ionomer.vapor_equilibrium_water_content(rh, T_ref)
    rh_from_pwl = ionomer.linear_rh_from_water_content(lmbd_ref)

    axes[0].plot(lmbd_ref, rh, "k-", label="Cubic (Springer)")
    axes[0].plot(lmbd_ref, rh_from_pwl, "C0--", label="PWL")
    for b in ionomer.lmbd_pwl_breaks:
        axes[0].axvline(b, color="C0", lw=0.6, ls=":")
    axes[0].set_xlabel("λ (mol/mol)")
    axes[0].set_ylabel("RH")
    axes[0].legend()

    err = (rh_from_pwl - rh) * 100
    axes[1].plot(lmbd_ref, err, "C3-")
    axes[1].axhline(0, color="k", lw=0.7, ls="--")
    axes[1].set_xlabel("λ (mol/mol)")
    axes[1].set_ylabel("PWL error (%)")
    fig.tight_layout()

Membrane through-plane proton resistance
-----------------------------------------

:meth:`~marapendi.membrane.pem.PFSA.proton_resistance` integrates the local
conductivity over a through-plane water-content profile read from a
:class:`~marapendi.simulation.state.MembraneState`-like object (as populated
by :doc:`/science/water_balance` during a solve). To verify the underlying
conductivity correlation in isolation for a *uniform* water content, it is
simpler to call
:meth:`~marapendi.membrane.pem.PFSAIonomer.proton_conductivity` directly and
divide the membrane thickness by it (see :doc:`/science/membrane_correlations`
for the equation):

.. code-block:: python

    membrane = mrpd.PFSA(ionomer=ionomer, dry_thickness=25e-6)

    lmbd_vals = np.linspace(3, 20, 50)
    T = 353.15

    sigma = ionomer.proton_conductivity(lmbd_vals, T)
    R_proton = membrane.dry_thickness / sigma

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(lmbd_vals, R_proton * 1e4, "C1-")
    ax.set_xlabel("Mean water content λ (mol/mol)")
    ax.set_ylabel("Proton resistance (mΩ cm²)")
    ax.set_title("Membrane proton resistance vs water content")
    ax.grid(True, alpha=0.3)

.. note::

   ``proton_resistance`` itself takes a full state object exposing
   ``water_content_profile``/``temperature`` (and optionally
   ``water_saturation`` to blend in the liquid-equilibrated conductivity —
   see the "Liquid-equilibrated water content" correlation in
   :doc:`/science/membrane_correlations`); it is normally called by
   :doc:`/science/water_balance`'s solve pipeline rather than directly.

Capillary pressure vs saturation (GDL)
----------------------------------------

:class:`~marapendi.models.darcy.DarcyTransportModel` relates capillary
pressure and non-wetting (liquid water) saturation through a power-law
J-function; the layer's breakthrough pressure
(:attr:`~marapendi.porous_layers.porous_layers.PorousLayer.breakthrough_pressure`,
derived from its geometry) sets the scale — see :doc:`/science/two_phase_flow`
for the full derivation:

.. code-block:: python

    from marapendi.models.darcy import DarcyTransportModel

    liq = DarcyTransportModel(J_function_exponent=0.4)
    gdl = mrpd.GasDiffusionLayer(
        thickness=200e-6, porosity=0.6, contact_angle=120.,
        absolute_permeability=1e-12, tortuosity=1.5,
        thermal_conductivity=0.5, two_phase_transport_model=liq,
    )

    s_arr = np.linspace(0.001, 0.5, 200)

    # Capillary pressure at each saturation (gdl itself provides breakthrough_pressure)
    pc = liq.capillary_pressure_from_saturation(gdl, s_arr)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(s_arr, pc / 1e3)
    ax.set_xlabel("Liquid saturation s")
    ax.set_ylabel("Capillary pressure (kPa)")
    ax.set_title(f"Capillary pressure J-function (θ = {gdl.contact_angle}°)")
    ax.grid(True, which="both", alpha=0.3)

Gas diffusion resistance vs saturation
-----------------------------------------

Effective gas diffusion resistance increases with liquid saturation through
the ``water_saturation_correction`` factor :math:`f(s)` — see
:doc:`/science/gas_transport` for the full porous-layer diffusion model
(molecular + Knudsen, in series):

.. code-block:: python

    from marapendi.models.diffusion import PorousGasDiffusionModel

    diff_model = PorousGasDiffusionModel(water_saturation_exponent=3.0)

    s_arr = np.linspace(0, 0.4, 100)
    f_s = diff_model.water_saturation_correction(s_arr)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(s_arr, f_s, "C0-")
    ax.set_xlabel("Liquid saturation s")
    ax.set_ylabel(r"$f(s) = [\max(1-s,\,10^{-6})]^{n_s}$")
    ax.set_title("Diffusivity correction factor vs liquid saturation")
    ax.grid(True, alpha=0.3)

Electrochemical kinetics
-------------------------

:class:`~marapendi.models.thermo.electrochemistry.ElectrochemicalReaction` computes the
ORR exchange current density as a function of temperature and O₂ partial
pressure (see :doc:`/science/orr_kinetics` for the Neyerlin et al. (2006)
correlation and the Tafel activation overpotential built on top of it):

.. code-block:: python

    reaction = mrpd.ElectrochemicalReaction(
        reference_exchange_current_density=2.5e-4,
        reaction_order=0.54,
        activation_energy=67e6,
        reference_activity=1e5,
        reference_temperature=353.15,
        number_of_electrons=2,
        charge_transfer_coeff=0.5,
    )

    T_arr = np.linspace(323.15, 373.15, 50)
    p_O2  = 0.21 * 1.5e5   # Pa (21 % O₂ at 1.5 bar total)

    i0_T = np.array([reaction.exchange_current_density(p_O2, T) for T in T_arr])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(T_arr - 273.15, i0_T * 1e3, "C3-")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Exchange current density i₀ (mA/m²)")
    ax.set_title("ORR exchange current density vs temperature")
    ax.grid(True, alpha=0.3)
