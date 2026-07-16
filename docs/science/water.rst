Water properties
=========================

:mod:`marapendi.models.thermo.water` collects the pure-water thermophysical
correlations (saturation pressure, density, viscosity, surface tension, O₂
diffusivity) that underlie the membrane and liquid-water transport models in
:doc:`water_balance` and :doc:`two_phase_flow`. All correlations are pure
NumPy (no Cantera dependency) but validated against Cantera's ``Water`` phase
object, to the tolerances stated in each function's docstring
(``tests/test_water_properties.py``).

Saturation pressure and concentration
------------------------------------------

The saturation vapor pressure follows the Buck equation
(:func:`~marapendi.models.thermo.water.water_saturation_pressure`):

.. math::

    p_\mathrm{sat}(T) = 611.21\,\exp\!\left[
        \left(18.678 - \frac{T_C}{234.5}\right)
        \left(\frac{T_C}{257.14 + T_C}\right)
    \right] \quad [\mathrm{Pa}],

with :math:`T_C = T - \SI{273.15}{K}`, accurate to better than 0.1 % against
Cantera over 274–373 K. This is the basis for the ideal-gas saturation
concentration
(:func:`~marapendi.models.thermo.water.water_saturation_concentration`),

.. math::

    c_\mathrm{sat}(T) = \frac{p_\mathrm{sat}(T)}{RT},

used throughout :doc:`water_balance` and :doc:`gas_transport` (e.g.
:attr:`~marapendi.simulation.state.GasState.saturation_concentration`) to
detect liquid-water formation and to set the water-vapor driving
concentration at the catalyst layer, and by :doc:`heat_transfer` to evaluate
the MEA saturation pressure for the temperature increase calculation.

The inverse mapping — dew-point temperature from a water vapor partial
pressure — is a cubic polynomial fit in :math:`\ln p` calibrated against
Cantera over 700–101 325 Pa, accurate to within 0.05 K
(:func:`~marapendi.models.thermo.water.water_dew_point`):

.. math::

    T_\mathrm{dew}(p) = \left[\left(a_0 \ln p - a_1\right)\ln p + a_2\right]\ln p + a_3,

with :math:`[a_0, a_1, a_2, a_3] = [0.101006,\ 1.376370,\ 19.219599,\ 179.796275]`.

Liquid density and molar volume
-------------------------------------

Liquid water density follows a quadratic fit in Celsius temperature after
Kell (1975), as cited in IAPWS-IF97
(:func:`~marapendi.models.thermo.water.water_density`), accurate to better
than 0.05 % against Cantera over 274–373 K:

.. math::

    \rho_w(T) = \left(-2.658\times10^{-3}\,T_C - 0.155\right) T_C + 1001.3
    \quad [\mathrm{kg/m^3}].

The molar volume
(:func:`~marapendi.models.thermo.water.water_molar_volume`),
:math:`\bar V_w(T) = M_w / \rho_w(T)`, is used throughout the ionomer water
volume-fraction calculation
(:meth:`~marapendi.components.membrane.ionomer_base.Ionomer.water_vol_fraction`, feeding
the conductivity and permeability correlations in
:doc:`membrane_correlations`), the wet ionomer expansion factor
(:meth:`~marapendi.components.membrane.ionomer_base.Ionomer.wet_expansion_factor`, used
by :doc:`catalyst_layer`'s microstructure), and to convert liquid volumetric
flow rates to molar flow rates in
:class:`~marapendi.simulation.state.GasFlowState`.

Viscosity and surface tension
------------------------------------

Dynamic viscosity follows a three-parameter Vogel-equation fit
(:func:`~marapendi.models.thermo.water.water_dynamic_viscosity`), accurate to
within 1.1 % against Cantera over 274–373 K:

.. math::

    \mu_l(T) = 3.16222\times10^{-5}\,
        \exp\!\left(\frac{482.6125}{T - 153.5669}\right) \quad [\mathrm{Pa \cdot s}],

with the kinematic viscosity
(:func:`~marapendi.models.thermo.water.water_kinematic_viscosity`) simply
:math:`\nu_l = \mu_l/\rho_w`. Both feed the Darcy liquid-water flux and
breakthrough-pressure calculations of :doc:`two_phase_flow`
(:meth:`~marapendi.components.porous_layers.porous_layers.PorousLayer.calculate_saturation_flow_resistance`).

Surface tension follows a linear fit
(:func:`~marapendi.models.thermo.water.water_surface_tension`):

.. math::

    \gamma(T) = 0.076 - 1.677\times10^{-4}\,T_C \quad [\mathrm{N/m}],

used directly in the breakthrough-pressure correlation
(:meth:`~marapendi.components.porous_layers.porous_layers.PorousLayer._compute_breakthrough_pressure`,
see :doc:`two_phase_flow`) and in the electrolyte capillary/wetting
calculations for alkaline layers
(:mod:`marapendi.components.electrolyte.electrolyte`).

Oxygen diffusivity in liquid water
------------------------------------

The O₂ diffusivity in bulk liquid water follows an Arrhenius-type fit to the
298 K value tabulated by Tsimpanogiannis et al. (2021)
(:func:`~marapendi.models.thermo.water.o2_water_diffusivity`):

.. math::

    D_{\mathrm{O_2},w}(T) = 4.6\times10^{-7}\,\exp\!\left(-\frac{1550}{T}\right)
    \quad [\mathrm{m^2/s}],

used by the catalyst-layer water-film term of the O₂ ionomer-film resistance
(:meth:`~marapendi.components.porous_layers.catalyst_layers.PtCCatalystLayer.o2_ionomer_film_resistance`,
see :doc:`catalyst_layer`) to account for the additional O₂ transport
resistance through a liquid water film covering the agglomerates when the
cathode catalyst layer is flooded.

References
--------------

Buck, A. L. *J. Appl. Meteorol.* **20**, 1527 (1981).

Kell, G. S. *J. Chem. Eng. Data* **20**, 97 (1975).

Tsimpanogiannis, I. N. et al. (2021).
