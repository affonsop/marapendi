Membrane and ionomer correlations
==================================

These are the stateless input correlations consumed by the membrane
:doc:`water_balance` solve: proton conductivity as a function of water
content and temperature, and the equilibrium sorption isotherm relating water
content to relative humidity.

.. seealso::

   :doc:`/auto_examples/plot_09_membrane_correlations` plots every
   correlation on this page against water content (or water activity, for
   the isotherm) at several temperatures, for a typical Nafion 1100 EW
   ionomer.

Proton conductivity
------------------------

Membrane proton conductivity uses an Arrhenius temperature correction on top
of a water-content-dependent term
(:meth:`~marapendi.components.membrane.pem.PFSAIonomer.proton_conductivity`), giving the
:math:`\sigma_\mathrm{ion}(\lambda, T)` used both for the through-plane
membrane resistance and for the catalyst-layer ionomer resistance in
:doc:`catalyst_layer`. The same functional form is used for the membrane and
the catalyst-layer ionomer film, following Kusoglu and Weber (2017):

.. math::

    \sigma_{mb/ion} = \xi_\sigma^{mb/ion}\, \sigma_0\, (f_v - f_0)^{n_\sigma^{mb/ion}}
        \exp\!\left[\frac{E_{act,\sigma}^{mb/ion}}{R}
        \left(\frac{1}{T_\mathrm{ref}} - \frac{1}{T}\right)\right],

where :math:`f_v` is the water volume fraction in the ionomer
(:meth:`~marapendi.components.membrane.ionomer_base.Ionomer.water_vol_fraction`) and
:math:`f_0` the percolation threshold below which the ionomer does not
conduct. Kusoglu and Weber report :math:`n = 1.0` for low-EW membranes and
:math:`n = 1.5` for high-EW membranes such as Nafion (1100 EW), with an
average :math:`f_0 = 0.10`; we adopt their high-EW prefactor
:math:`\sigma_0 = 50\ \mathrm{S/m}` and fit the correction factors
:math:`\xi_\sigma^{mb}` and :math:`\xi_\sigma^{ion}` separately for the
membrane and the catalyst-layer ionomer film.

Electroosmotic drag coefficient
-----------------------------------

The electroosmotic drag coefficient
(:meth:`~marapendi.components.membrane.pem.PFSAIonomer.calculate_electroosmotic_drag_coefficient`,
overriding the neutral, water-content-independent default of unity in
:meth:`~marapendi.components.membrane.ionomer_base.Ionomer.calculate_electroosmotic_drag_coefficient`)
is taken linear in both temperature and local water content:

.. math::

    \xi_{EOD}(\lambda, T) = \frac{0.02\,T - 3.86}{22.5}\,\lambda.

:meth:`~marapendi.components.membrane.ionomer_base.Ionomer.calculate_electroosmotic_drag_speed`
evaluates :math:`\xi_{EOD}` at :math:`\lambda = 1` to convert a current
density into an electroosmotic drag speed,

.. math::

    v_{EOD} = \xi_{EOD}(1, T)\, \frac{i}{F\, c_\mathrm{dry}},

with :math:`c_\mathrm{dry}` the dry-ionomer molar concentration of sulfonate
sites (equivalent weight :math:`/` dry density), used as a boundary condition
of the membrane water-transport problem in :doc:`water_balance`.

Water diffusivity
-----------------------------------

The adsorbed-water diffusivity in the ionomer follows an Arrhenius
correction on a reference value defined at
:attr:`~marapendi.components.membrane.pem.PFSAIonomer.reference_water_diffusivity_temperature`
(:meth:`~marapendi.components.membrane.ionomer_base.Ionomer.calculate_water_diffusivity`):

.. math::

    D_w(T) = D_{w,\mathrm{ref}}\,
        \exp\!\left[\frac{E_{act,D_w}}{R}
        \left(\frac{1}{T_\mathrm{ref}} - \frac{1}{T}\right)\right],

with :math:`D_{w,\mathrm{ref}}` =
:attr:`~marapendi.components.membrane.pem.PFSAIonomer.reference_water_diffusivity` and
activation energy
:attr:`~marapendi.components.membrane.pem.PFSAIonomer.water_diffusivity_activation_energy`.
This enters the non-dimensional Péclet and Biot numbers
(:meth:`~marapendi.models.water_balance.membrane.MembraneWaterBalanceModel.calculate_peclet_number`,
:meth:`~marapendi.models.water_balance.membrane.MembraneWaterBalanceModel.calculate_biot_number`)
of the membrane water-transport problem described in :doc:`water_balance`.

Water absorption/desorption rate constant
-------------------------------------------

The interfacial water absorption/desorption rate constant
(:meth:`~marapendi.components.membrane.ionomer_base.Ionomer.calculate_water_absorption_coefficient`)
follows the same Arrhenius form:

.. math::

    k_{abs}(T) = k_{abs,\mathrm{ref}}\,
        \exp\!\left[\frac{E_{act,k}}{R}
        \left(\frac{1}{T_\mathrm{ref}} - \frac{1}{T}\right)\right],

with :math:`k_{abs,\mathrm{ref}}` =
:attr:`~marapendi.components.membrane.pem.PFSAIonomer.reference_water_absorption_coefficient`
and
:attr:`~marapendi.components.membrane.pem.PFSAIonomer.water_absorption_activation_energy`,
setting the Biot number that couples the membrane's boundary water content
to its equilibrium value at each catalyst-layer interface.

Hydrogen permeability
-----------------------------------

Following Goshtasbi et al. (2020), H\ :sub:`2` permeability through the
ionomer (:meth:`~marapendi.components.membrane.pem.PFSAIonomer.h2_permeability`) sums a
dry and a water-volume-fraction-weighted wet term, each with its own
Arrhenius activation energy:

.. math::

    P_{H_2}(f_v, T) = 15.7\times10^{-15}\exp\!\left(-\frac{20280\times10^3\ \mathrm{J/kmol}}{RT}\right)
        + f_v\, 45\times10^{-15}\exp\!\left(-\frac{18930\times10^3\ \mathrm{J/kmol}}{RT}\right)
        \quad [\mathrm{kmol/m/s/Pa}],

where :math:`f_v` is the water volume fraction
(:meth:`~marapendi.components.membrane.ionomer_base.Ionomer.water_vol_fraction`). This
sets the crossover current density subtracted from the Faradaic current in
the cell-voltage model.

Oxygen permeability
-----------------------------------

The membrane-scale O\ :sub:`2` permeability
(:meth:`~marapendi.components.membrane.pem.PFSAIonomer.o2_permeability`) follows the
same functional form, with its own fitted coefficients:

.. math::

    P_{O_2}(f_v, T) = 6.74\times10^{-15}\exp\!\left(-\frac{21280\times10^3\ \mathrm{J/kmol}}{RT}\right)
        + f_v\, 50.5\times10^{-15}\exp\!\left(-\frac{20470\times10^3\ \mathrm{J/kmol}}{RT}\right)
        \quad [\mathrm{kmol/m/s/Pa}].

A separate, catalyst-layer-specific O\ :sub:`2` diffusivity through the thin
ionomer film coating the carbon/Pt agglomerates — used for the ionomer-film
transport resistance in :doc:`catalyst_layer` — is given by
(:meth:`~marapendi.components.membrane.pem.PFSAIonomer.o2_film_diffusion_coefficient`):

.. math::

    D_{O_2,film}(\lambda, T) = D_{O_2,ref}\left(\frac{\lambda}{14}\right)^{n_{D_{O_2}}}
        \exp\!\left[\frac{E_{act,D_{O_2}}}{R}
        \left(\frac{1}{353.15} - \frac{1}{T}\right)\right],

with :attr:`~marapendi.components.membrane.pem.PFSAIonomer.hydrated_o2_diffusion`,
:attr:`~marapendi.components.membrane.pem.PFSAIonomer.o2_diffusion_exponent` and
:attr:`~marapendi.components.membrane.pem.PFSAIonomer.o2_diffusion_activation_energy`.

Equilibrium sorption isotherm
-----------------------------------

The reference equilibrium sorption isotherm :math:`\lambda_\mathrm{eq}(\mathrm{RH})`
is the cubic polynomial of Springer et al. (1991)
(:meth:`~marapendi.components.membrane.pem.PFSAIonomer.vapor_equilibrium_water_content`),
with the temperature-dependent correction of Goshtasbi et al. (2019):

.. math::

    \lambda_\mathrm{eq}(\mathrm{RH}) = a_0\,\mathrm{RH}^3 + a_1\,\mathrm{RH}^2
        + a_2\,\mathrm{RH} + a_3,

with fitted coefficients :math:`[a_0, a_1, a_2, a_3]` stored in
:attr:`~marapendi.components.membrane.pem.PFSAIonomer.vapor_equilibrium_polynomial`
(default ``[36, -39.85, 17.18, 0.043]``, from Springer et al. 1991). This
cubic form is temperature-independent; it is also re-fit to a
piecewise-linear approximation at construction time — see `Piecewise-linear
fit`_ below.

For a membrane face in contact with liquid water rather than vapor, the
equilibrium (Schroeder's paradox) water content
(:meth:`~marapendi.components.membrane.pem.PFSAIonomer.liquid_equilibrium_water_content`)
uses the linear-in-temperature fit of Goshtasbi et al. (2020):

.. math::

    \lambda_\mathrm{liq}(T) = 9.22 + 0.181\,(T - 273.15),\quad T\ \mathrm{in\ K},

used e.g. by :meth:`~marapendi.components.membrane.pem.PFSA.proton_resistance` to blend
liquid- and vapor-equilibrated conductivities by local water saturation.

Piecewise-linear fit
~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~marapendi.models.water_balance.membrane_pwl.MembraneWaterBalanceModelPiecewise`
(the default water-balance model) replaces this cubic isotherm with a
piecewise-*linear* fit
(:meth:`~marapendi.components.membrane.pem.PFSAIonomer.fit_rh_piecewise_linear`), selecting
the active linear segment self-consistently — iterating down from the highest
segment until the equilibrium water content falls inside that segment's
validity range, guaranteed to converge in at most ``n_segments`` iterations.

Because :math:`\lambda_\mathrm{eq}(\mathrm{RH})` is then linear in
:math:`\mathrm{RH}` on the active segment, the whole non-dimensional membrane
system in :doc:`water_balance` stays linear in the unknown boundary water
content — this closed-form piecewise-linear treatment is one of the
contributions described in Affonso Nobrega et al. (2026), and is what
:doc:`../user_guide/polarization_curve` refers to as the "first-order linear
expansion."

References
--------------

Springer, T. E. et al. *J. Electrochem. Soc.* **138**, 2334 (1991).

Goshtasbi, A. et al. *J. Electrochem. Soc.* **166**, F3154 (2019).

Goshtasbi, A. et al. *J. Electrochem. Soc.* **167**, 024518 (2020).

Kusoglu, A. and Weber, A. Z. *Chem. Rev.* **117**, 987 (2017).

Affonso Nobrega, P. et al. *J. Electrochem. Soc.* **173**, 114503 (2026).
