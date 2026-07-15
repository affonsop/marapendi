Water balance
=================
**marapendi** solves membrane water balance assuming dry conditions (no liquid water). If the 
water flux leaving the MEA on either side exceeds the maximum removable vapor flux, then 
some of the water will be leaving in liquid form and liquid water saturation is calculated for each layer. 

Membrane water balance
----------------------
The first step of the water balance is to solve the membrane water balance. 
:class:`~marapendi.models.water_balance.membrane.MembraneWaterBalanceModel`
solves the steady 1D diffusion + electroosmotic-drag (EOD) problem across the
membrane thickness analytically, giving the water-content profile
:math:`\lambda(\xi)` where :math:`\xi \in [0, 1]` is the normalised
through-plane coordinate (:math:`\xi = 0` at the anode interface,
:math:`\xi = 1` at the cathode interface), in an extension of the work of Ferrara et al. (2018) to 
non-equilibrium boundary conditions.

The governing steady-state balance is a 1D diffusion–EOD equation for the
membrane water content, with an EOD velocity
:math:`u_d` (:meth:`~marapendi.components.membrane.membrane_base.Membrane.calculate_electroosmotic_drag_speed`)
that is proportional to current density:

.. math::

    0 = -\frac{d}{dx}\left[-D_\lambda \frac{d\lambda}{dx} + u_d\,\lambda\right]

with non-equilibrium sorption boundary conditions:

.. math::

    \pm\left[-D_\lambda \frac{d\lambda}{dx} + u_d\,\lambda\right]
        = k_{abs}\left(\lambda - \lambda_{eq}\right)

with :math:`k_{abs}` the water-absorption rate coefficient
(:meth:`~marapendi.components.membrane.membrane_base.Membrane.calculate_water_absorption_coefficient`).
Membrane swelling is neglected. :math:`D_\lambda`
(:meth:`~marapendi.components.membrane.membrane_base.Membrane.calculate_water_diffusivity`)
and :math:`k_{abs}` are independent of the water content :math:`\lambda` but
are corrected for temperature with an Arrhenius term (see
:doc:`membrane_correlations`).

Equilibrium water content
~~~~~~~~~~~~~~~~~~~~~~~~~

The equilibrium water content at each catalyst layer,
:math:`\lambda_{eq}`, is not known *a priori*: it depends on the
local relative humidity at the catalyst layer, which itself depends on the
water flux crossing the membrane — the same flux the boundary condition
above is trying to solve for.

**marapendi** proposes two methods to deal with this problem. The first one,
described in Affonso Nobrega et al. (2026), uses a first-order Taylor expansion of the
sorption isotherm around estimated water activity values:

.. math::
    :nowrap:

    \begin{aligned}
        a_{w,\mathrm{est}}^{an} &= \mathrm{RH}_\mathrm{CH}^{an}
            \frac{c_{\mathrm{sat}}(T_{st})}{c_{\mathrm{sat}}(T_{\mathrm{MEA}})} \\
        a_{w,\mathrm{est}}^{ca} &= \mathrm{RH}_\mathrm{CH}^{ca}
            \frac{c_{\mathrm{sat}}(T_{st})}{c_{\mathrm{sat}}(T_{\mathrm{MEA}})}
            + \frac{i}{2F}\frac{R_v^{ca}}{c_{\mathrm{sat}}(T_{\mathrm{MEA}})}
    \end{aligned}

with :math:`c_\mathrm{sat}` the water saturation concentration
(:func:`~marapendi.models.thermo.water.water_saturation_concentration`) and
:math:`R_v` the vapor transport resistance
(:meth:`~marapendi.models.gas_transport_resistance.GasTransportModel.gas_transport_resistance`).

This allows us to define a vapor transport resistance on a water-content basis, which can be treated as being in
series with the adsorption resistance, such that the sorption boundary conditions can be rewritten as:

.. math::

    \pm\left[-D_\lambda \frac{d\lambda}{dx} + u_d\,\lambda\right]
        = k_{eq}\left(\lambda - \lambda_{\mathrm{est}}\right)

with:

.. math::

    k_{eq}=\left[\frac{1}{k_{abs}} + \frac{R_v}{c_{\mathrm{sat}}\left(T_\mathrm{MEA}\right)}
        \frac{\partial \lambda_{eq}}{\partial a_w}\left(a_{w,\mathrm{est}}\right)\right]^{-1}

and:

.. math::

    \lambda_{\mathrm{est}} = \lambda_{eq}(a_{w,\mathrm{est}})

The analytical solution to the boundary value problem gives the membrane water-content profile:

.. math::

    \lambda(\xi) = \frac{
            \lambda_{eq,\mathrm{est}}^{an} \left(e^{Pe\xi}Pe/Bi^{ca} - e^{Pe\xi} +e^{Pe}\right) +
            \lambda_{eq,\mathrm{est}}^{ca}\left(e^{Pe\xi}Pe/Bi^{an} + e^{Pe\xi} - 1\right)
        }
        {e^{Pe}-1 + e^{Pe}Pe/Bi^{an} + Pe/Bi^{ca}}

with the non-dimensional quantities:

.. math::

    Pe = u_d\delta_{mb}/D_\lambda \quad Bi^{ca/an} = k_{eq}^{ca/an}\delta_{mb}/D_\lambda \quad \xi = x/\delta_{mb}

computed by
:meth:`~marapendi.models.water_balance.membrane.MembraneWaterBalanceModel.calculate_peclet_number` and
:meth:`~marapendi.models.water_balance.membrane.MembraneWaterBalanceModel.calculate_biot_number`.

This first method is implemented in :class:`~marapendi.models.water_balance.membrane.MembraneWaterBalanceModel`. Its
main drawback is that the first-order Taylor expansion is not accurate given the shape of the water
equilibrium isotherm for PFSA membranes.

A second method uses instead a piecewise-linear approximation of the equilibrium isotherm, with two segments: one
for high and one for low water activities. For each segment, the equilibrium isotherm is written:

.. math::

    \lambda_{eq}(a_w) = \alpha\, a_{w} + \beta

so that:

.. math::

    a_{w} = \frac{\lambda_{eq}(a_w) - \beta}{\alpha}

It can be shown after some manipulation that the boundary conditions can be written exactly as in the first method:

.. math::

    \pm\left[-D_\lambda \frac{d\lambda}{dx} + u_d\,\lambda\right]
        = k_{eq}\left(\lambda - \lambda_{\mathrm{est}}\right)

but with the equilibrium isotherm derivative replaced by :math:`\alpha`:

.. math::

    k_{eq}=\left[\frac{1}{k_{abs}} + \frac{R_v}{c_{\mathrm{sat}}\left(T_\mathrm{MEA}\right)}
        \alpha\right]^{-1}

In other words, instead of using the value of the derivative :math:`\partial \lambda_{eq}/\partial a_w` at
:math:`a_{w,\mathrm{est}}`, we use an average derivative for the segment of the isotherm curve — high or low water
activity — that :math:`a_{w,\mathrm{est}}` falls into, which gives a better approximation.

The implementation of this second method,
:meth:`~marapendi.models.water_balance.membrane_pwl.MembraneWaterBalanceModelPiecewise.solve_membrane_water_balance`,
starts by assuming the water activity is high and using the corresponding segment. The membrane water balance is
solved so that the membrane water flux, and then the water activity at each catalyst layer
(:meth:`~marapendi.models.water_balance.membrane_pwl.MembraneWaterBalanceModelPiecewise.estimate_equilibrium_water_contents`),
can be calculated. If the calculated water activity falls in the low-activity segment instead, :math:`\alpha` and
:math:`\beta` are updated and the water balance is recalculated.

A piecewise-linear fit of the equilibrium isotherm with 3 linear segments is automatically calculated at
initialization by :class:`~marapendi.components.membrane.pem.PFSAIonomer`.

.. seealso::

    :doc:`/auto_examples/plot_07_pwl_membrane` inspects the quality of the
    piecewise-linear fit and compares the two models on a full polarization
    curve.

Water fluxes
------------
Once the profile and equilibrium water contents are determined, the membrane water flux on each side is
calculated by
:meth:`~marapendi.models.water_balance.membrane.MembraneWaterBalanceModel.update_membrane_water_fluxes`
from the Biot number and the jump between the membrane-interface water content and the equilibrium water
content found in `Equilibrium water content`_:

.. math::

    J_{w}^{mb} = k_{abs}\frac{\lambda - \lambda_{eq}}{R_{\lambda}}

The total water flux leaving each catalyst layer adds the electrochemical water production
:math:`J_{w}^{rxn} = i/2F` (positive at the cathode, from the ORR; zero at the anode):

.. math::

    J_w = J_{w}^{mb} + J_{w}^{rxn}

:meth:`~marapendi.models.water_balance.water_balance.WaterBalanceModel.update_cell_side_water_fluxes`
then splits :math:`J_w` into its liquid and vapor parts. Liquid water is assumed to appear once
:math:`J_w` exceeds the maximum flux that can be removed as vapor,
:math:`J_{v,\max}` — the flux obtained when the vapor concentration at the catalyst
layer equals the saturation concentration:

.. math::

    J_{w,\max}^\mathrm{vap} = \frac{c_\mathrm{sat}(T_{\mathrm{MEA}}) - c_{v}(T_{st})}{R_v}

with :math:`c_\mathrm{sat}` and :math:`c_v` the saturation and vapor concentrations
(:meth:`~marapendi.simulation.state.GasState.saturation_concentration` and
:meth:`~marapendi.simulation.state.GasState.vapor_concentration`) and :math:`R_v` the (dry)
vapor transport resistance between the catalyst layer and the channel. The liquid and vapor
fluxes follow as:

.. math::

    J_l = \max\left(J_w - J_{v,\max},\ 0\right)
    \qquad
    J_v = J_w - J_w^\mathrm{liq}


Catalyst layer ionomer water content
------------------------------------ 
The ionomer water contents in the anode and cathode catalyst layers are assumed equal to the equilibrium values:  

.. math::   

    \lambda_{ion, \mathrm{CL}} = \lambda_{eq}

A discussion on this hypothesis can be found in Affonso Nobrega et al. (2026).

References
--------------

Affonso Nobrega, P. et al. *J. Electrochem. Soc.* **173**, 114503 (2026).

Ferrara, A. et al. *J. Power Sources* **390**, 197–207 (2018).
