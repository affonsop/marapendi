Catalyst layer
==================

Microstructure
---------------------

:class:`~marapendi.porous_layers.catalyst_layers.PtCCatalystLayer` derives an
explicit Pt/C agglomerate microstructure from four inputs — platinum loading
:math:`L_\mathrm{Pt}`, platinum weight percent in the catalyst powder,
carbon-agglomerate radius :math:`r_\mathrm{C}`, and ionomer-to-carbon mass
ratio :math:`\mathrm{IC}` — following Hao et al. (2015). Carbon and platinum
volume fractions follow directly from the loadings and layer thickness
:math:`\delta_\mathrm{CL}`; the dry ionomer volume fraction is then scaled up
by the volumetric expansion due to water sorption
(:meth:`~marapendi.membrane.ionomer_base.Ionomer.wet_expansion_factor`),
computed once at construction and refreshed at every operating point by
:meth:`~marapendi.porous_layers.catalyst_layers.PtCCatalystLayer.set_ionomer_wet_properties`:

.. math::

    \varepsilon_\mathrm{C} = \frac{L_\mathrm{Pt}}{\delta_\mathrm{CL}\,\rho_\mathrm{C}}
        \left(\frac{1}{\mathrm{Pt\ wt\%}} - 1\right), \qquad
    \varepsilon_\mathrm{Pt} = \frac{L_\mathrm{Pt}}{\delta_\mathrm{CL}\,\rho_\mathrm{Pt}},

.. math::

    \varepsilon_\mathrm{ion} = \frac{\varepsilon_\mathrm{C}\,\rho_\mathrm{C}\,\mathrm{IC}}{\rho_\mathrm{ion}^\mathrm{dry}}
        \left(1 + \lambda_\mathrm{ion}\,\frac{\bar V_w}{\bar V_\mathrm{ion}}\right).

The resulting ionomer film thickness coating each carbon agglomerate,

.. math::

    \delta_\mathrm{ion} = r_\mathrm{C}\left[\left(\frac{\varepsilon_\mathrm{ion}}{\varepsilon_\mathrm{C}} + 1\right)^{1/3} - 1\right],

and its specific surface area,

.. math::

    a_\mathrm{ion} = 4\pi(r_\mathrm{C} + \delta_\mathrm{ion})^2 N_\mathrm{C},

with :math:`N_\mathrm{C}` the carbon-agglomerate number density, are the
geometric inputs to the ionomer sheet resistance below and to the O₂
ionomer-film resistance in `Local oxygen transport`_. The remaining pore
volume fraction — used for the liquid- and gas-phase transport properties in
:doc:`two_phase_flow` and :doc:`gas_transport` — closes the volume balance:

.. math::

    \varepsilon_\mathrm{CL} = 1 - \varepsilon_\mathrm{Pt} - \varepsilon_\mathrm{C} - \varepsilon_\mathrm{ion}.

The ionomer film has its own, generally much lower, tortuosity than the bulk
pore network, following Hao et al. (2016)
(:meth:`~marapendi.porous_layers.catalyst_layers.PtCCatalystLayer.ionomer_tortuosity`):

.. math::

    \tau_\mathrm{ion} = \begin{cases}
        0.0845\,(\varepsilon_\mathrm{ion} - 0.04)^{-1.17} & \varepsilon_\mathrm{ion} < 0.16, \\
        1 & \varepsilon_\mathrm{ion} \geq 0.16.
    \end{cases}

The bulk pore network instead uses the standard Bruggeman relation between
porosity and tortuosity, :math:`\varepsilon_\mathrm{CL}/\tau_\mathrm{CL} =
\varepsilon_\mathrm{CL}^{1.5}` (Andersson et al., 2016) — the
:attr:`~marapendi.porous_layers.porous_layers.PorousLayer.tortuosity`
default in
:meth:`~marapendi.porous_layers.catalyst_layers.PtCCatalystLayer.set_ionomer_wet_properties`
when no tortuosity is supplied explicitly.

Charge transport
---------------------

Ionomer (proton) charge transport through the catalyst-layer film follows a
simple effective-medium resistance
(:meth:`~marapendi.porous_layers.catalyst_layers.CatalystLayer.ionomer_sheet_charge_resistance`):

.. math::

    R_\mathrm{ion} = \frac{\delta_\mathrm{CL}}{
        \dfrac{\varepsilon_\mathrm{ion}}{\tau_\mathrm{ion}} \, \sigma_\mathrm{ion}(\lambda, T)
    },

where :math:`\delta_\mathrm{CL}` is the catalyst-layer thickness,
:math:`\varepsilon_\mathrm{ion}` the ionomer volume fraction,
:math:`\tau_\mathrm{ion}` the ionomer-phase tortuosity (Bruggeman-type
correction), and :math:`\sigma_\mathrm{ion}(\lambda, T)` the ionomer proton
conductivity (:doc:`membrane_correlations`). In series with any parallel
electrolyte (liquid-filled) resistance :math:`R_\mathrm{elyte}` — from the
two-phase saturation in :doc:`two_phase_flow` — the combined sheet resistance
is :math:`R_\mathrm{sheet} = \left(1/R_\mathrm{ion} + 1/R_\mathrm{elyte}\right)^{-1}`.

The *effective* charge resistance actually seen by the reaction accounts for
the reaction-current distribution through the catalyst-layer depth, following
Neyerlin et al. (2007) as parameterised by Goshtasbi et al. (2020)
(:meth:`~marapendi.porous_layers.catalyst_layers.CatalystLayer.effective_charge_resistance`):

.. math::

    \nu = \min\!\left(\frac{R_\mathrm{sheet}\, i}{b},\, 10\right), \qquad
    \xi = \nu\,(-8.287\times10^{-3}\,\nu + 0.7184) - 2.072\times10^{-3}, \qquad
    R_\mathrm{eff} = \frac{R_\mathrm{sheet}}{3 + \xi},

with :math:`b` the Tafel slope from :doc:`orr_kinetics`. This effective
resistance is the catalyst layer's contribution to :math:`\eta_\mathrm{ohm}`
in :doc:`cell_voltage`.

Local oxygen transport
----------------------------

Oxygen transport through the ionomer film covering the Pt/C agglomerates
follows Hao et al. (2015, 2016)
(:meth:`~marapendi.porous_layers.catalyst_layers.PtCCatalystLayer.o2_ionomer_film_resistance`):
Pt, carbon, ionomer and pore volume fractions are computed directly from the
platinum loading, ionomer-to-carbon ratio and catalyst density, and the local
O₂ transport resistance combines a bulk-film diffusion term with an
interfacial (gas/ionomer, ionomer/Pt) term via their equation 32. This
resistance is added in series with the porous-layer and channel resistances
from :doc:`gas_transport` to give the total O₂ transport resistance feeding
the ORR kinetics in :doc:`orr_kinetics`.

References
--------------

Neyerlin, K. C. et al. *J. Electrochem. Soc.* **154**, B279 (2007).

Goshtasbi, A. et al. *J. Electrochem. Soc.* **167**, 024518 (2020).

Hao, L. et al. *J. Electrochem. Soc.* **162**, F854 (2015).

Hao, L. et al. *J. Electrochem. Soc.* **163**, F744 (2016).

Andersson, M. et al. *Appl. Energy* **180**, 757 (2016).
