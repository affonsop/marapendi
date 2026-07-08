Membrane correlations
=========================

These are the stateless input correlations consumed by the membrane
:doc:`water_balance` solve: proton conductivity as a function of water
content and temperature, and the equilibrium sorption isotherm relating water
content to relative humidity.

Proton conductivity
------------------------

Membrane proton conductivity uses an Arrhenius temperature correction on top
of a water-content-dependent term
(:meth:`~marapendi.membrane.pem.PFSAIonomer.proton_conductivity`), giving the
:math:`\sigma_\mathrm{ion}(\lambda, T)` used both for the through-plane
membrane resistance and for the catalyst-layer ionomer resistance in
:doc:`catalyst_layer`.

Equilibrium sorption isotherm
-----------------------------------

The reference equilibrium sorption isotherm :math:`\lambda_\mathrm{eq}(\mathrm{RH},
T)` is the cubic polynomial of Springer et al., *J. Electrochem. Soc.*
**138**, 2334 (1991)
(:meth:`~marapendi.membrane.pem.PFSAIonomer.vapor_equilibrium_water_content`),
with the temperature-dependent correction of Goshtasbi et al., *J.
Electrochem. Soc.* **166**, F3154 (2019).

Piecewise-linear fit
--------------------------

:class:`~marapendi.models.water_balance.membrane_pwl.MembraneWaterBalanceModelPiecewise`
(the default water-balance model) replaces this cubic isotherm with a
piecewise-*linear* fit
(:meth:`~marapendi.membrane.pem.PFSAIonomer.fit_rh_piecewise_linear`), selecting
the active linear segment self-consistently — iterating down from the highest
segment until the equilibrium water content falls inside that segment's
validity range, guaranteed to converge in at most ``n_segments`` iterations.

Because :math:`\lambda_\mathrm{eq}(\mathrm{RH})` is then linear in
:math:`\mathrm{RH}` on the active segment, the whole non-dimensional membrane
system in :doc:`water_balance` stays linear in the unknown boundary water
content — this closed-form piecewise-linear treatment is one of the
contributions described in Affonso Nobrega et al., *J. Electrochem. Soc.*
**173**, 114503 (2026), and is what :doc:`../user_guide/polarization_curve`
refers to as the "first-order linear expansion."
