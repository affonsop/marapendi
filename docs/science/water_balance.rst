Water balance
=================

:class:`~marapendi.models.water_balance.membrane.MembraneWaterBalanceModel`
solves the steady 1D diffusion + electroosmotic-drag (EOD) problem across the
membrane thickness analytically, giving the water-content profile
:math:`\lambda(\xi)` where :math:`\xi \in [0, 1]` is the normalised
through-plane coordinate (:math:`\xi = 0` at the anode interface,
:math:`\xi = 1` at the cathode interface), following Ferrara et al., *J.
Power Sources* **390**, 197–207 (2018).
:class:`~marapendi.models.water_balance.membrane_pwl.MembraneWaterBalanceModelPiecewise`
(the default used by both steady-state models) is the same solution with the
piecewise-linear isotherm from :doc:`membrane_correlations` substituted in as
the boundary condition, which is what keeps everything below in closed form.

Non-dimensional numbers
----------------------------

The analytical solution
(:meth:`~marapendi.models.water_balance.membrane.MembraneWaterBalanceModel.update_non_dimensional_parameters`)
is built from the membrane's EOD velocity :math:`v_\mathrm{EOD}`, dry thickness
:math:`\delta_\mathrm{mb}`, and water diffusivity :math:`D_\lambda`:

.. math::

    Pe = \frac{v_\mathrm{EOD}\, \delta_\mathrm{mb}}{D_\lambda}
    \qquad\text{(Péclet number, } \texttt{peclet\_number} \text{)},

and, for each side :math:`s \in \{\mathrm{ca}, \mathrm{an}\}`, a Biot number
from the water-absorption coefficient :math:`k_s` at that interface:

.. math::

    Bi_s = \frac{k_s\, \delta_\mathrm{mb}}{D_\lambda}
    \qquad\text{(} \texttt{biot\_number} \text{)}.

The non-dimensional vapor transport resistance :math:`R_{v,s}^*` (Ferrara et
al. 2018, eq. 13;
:meth:`~marapendi.models.water_balance.membrane.MembraneWaterBalanceModel.update_non_dim_vapor_resistance`)
folds the gas-phase vapor resistance :math:`R_{v,s}` on that side (from
:doc:`gas_transport`) into the same non-dimensional framework:

.. math::

    R_{v,s}^* = \frac{R_{v,s}}{c_\mathrm{sat}(T_{\mathrm{cl},s})\, R_D}
        \left.\frac{\partial \lambda_\mathrm{eq}}{\partial (\mathrm{RH})}\right|_s,
    \qquad
    R_D = \frac{\delta_\mathrm{mb}}{D_\lambda\, c_\mathrm{mb}^\mathrm{dry}},

where :math:`R_D` is the membrane's water diffusion resistance
(``water_diffusion_resistance``) and :math:`c_\mathrm{mb}^\mathrm{dry}` its dry
concentration. The equivalent (combined vapor + absorption) resistance on each
side is :math:`1/Bi^\mathrm{eq}_s = R^*_{v,s} + 1/Bi_s`
(``non_dim_equiv_resistance``).

Water-content profile
--------------------------

With :math:`\widetilde{Pe}_s \equiv Pe / Bi^\mathrm{eq}_s`
(``peclet_over_modified_biot``) and :math:`\alpha_s = 1 -
\mathbb{1}[\text{EOD parallel to sorption}] / (Bi_s\, Bi^\mathrm{eq}_s)`, the
closed-form profile
(:meth:`~marapendi.models.water_balance.membrane.MembraneWaterBalanceModel.update_water_profile`)
is

.. math::

    \lambda(\xi) = \frac{
        \lambda_\mathrm{eq,an}\left[(e^{Pe} - e^{Pe\,\xi})(1 - \alpha_\mathrm{ca}\widetilde{Pe}_\mathrm{ca}) + e^{Pe}\widetilde{Pe}_\mathrm{ca}\right]
        + \lambda_\mathrm{eq,ca}\left[(e^{Pe\,\xi} - 1)(1 + \alpha_\mathrm{an}\widetilde{Pe}_\mathrm{an}) + \widetilde{Pe}_\mathrm{an}\right]
    }{D},

with :math:`\lambda_{\mathrm{eq},s}` the estimated catalyst-layer equilibrium
water content on side :math:`s` (from :doc:`membrane_correlations`) and
:math:`D` the shared denominator built from the same non-dimensional groups
(see
:meth:`~marapendi.models.water_balance.membrane.MembraneWaterBalanceModel.update_water_profile`
for the full expression). Once :math:`\lambda(\xi)` is known, the membrane
water flux on each side follows Fickian + EOD transport
(:meth:`~marapendi.models.water_balance.membrane.MembraneWaterBalanceModel.update_membrane_water_fluxes`):

.. math::

    J_{w,s} = Bi_s \, \frac{\lambda_{\mathrm{mb},s} - \lambda_{\mathrm{eq},s}^\ast}{R_D},

where :math:`\lambda_{\mathrm{mb},s}` is the membrane water content at
interface :math:`s` and :math:`\lambda_{\mathrm{eq},s}^\ast` is the
self-consistent equilibrium water content once coupled back to the membrane
interface value. Through-plane proton resistance, obtained by integrating
:math:`\sigma_\mathrm{ion}(\lambda(\xi), T)` (:doc:`membrane_correlations`)
over this profile, is the membrane's contribution to :math:`\eta_\mathrm{ohm}`
in :doc:`cell_voltage`.

See :doc:`transient_model` for how this same physics is re-used — with a
*prescribed* rather than solved-for profile — during transient integration.
