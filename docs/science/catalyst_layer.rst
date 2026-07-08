Catalyst layer
==================

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
Neyerlin et al., *J. Electrochem. Soc.* **154**, B279 (2007) as parameterised
by Goshtasbi et al., *J. Electrochem. Soc.* **167**, 024518 (2020)
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
follows Hao et al., *J. Electrochem. Soc.* **162**, F854 (2015) and **163**,
F744 (2016)
(:meth:`~marapendi.porous_layers.catalyst_layers.PtCCatalystLayer.o2_ionomer_film_resistance`):
Pt, carbon, ionomer and pore volume fractions are computed directly from the
platinum loading, ionomer-to-carbon ratio and catalyst density, and the local
O₂ transport resistance combines a bulk-film diffusion term with an
interfacial (gas/ionomer, ionomer/Pt) term via their equation 32. This
resistance is added in series with the porous-layer and channel resistances
from :doc:`gas_transport` to give the total O₂ transport resistance feeding
the ORR kinetics in :doc:`orr_kinetics`.
