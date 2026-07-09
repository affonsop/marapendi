Additional models: AEM and KOH (in preparation)
======================================================

These correlations are not part of the 13-topic model description above —
they are in preparation for AEM electrolyzer support and are documented here
for completeness.

Anion-exchange membrane (AEM) correlations
------------------------------------------------

:class:`~marapendi.membrane.aem.PAPIonomer` implements a poly(aryl
piperidinium) (PAP-85) ionomer with an Arrhenius hydroxide conductivity
(:meth:`~marapendi.membrane.aem.PAPIonomer.hydroxide_conductivity`):

.. math::

    \sigma_\mathrm{OH^-}(T) = \sigma_0 \exp\!\left[-\frac{E_a}{R}
        \left(\frac{1}{T} - \frac{1}{T^\mathrm{ref}}\right)\right],

with :math:`\sigma_0 = 5.8\ \mathrm{S/m}` and :math:`E_a = 22.5\
\mathrm{MJ/kmol}`, following Luo et al. (2020) and Khalid et al. (2022);
water uptake/vapor-equilibrium and water-transport activation energies
follow Eon Chae et al. (2024). Proton conductivity is set to zero for this
ionomer (hydroxide is the majority charge carrier).

KOH electrolyte
--------------------

Aqueous KOH electrolyte density, ionic conductivity and surface tension
(:mod:`marapendi.electrolyte.koh`) follow the correlations of Hodges et al.
(2023), parameterised by KOH molality and temperature.

References
--------------

Luo, X. et al. *J. Memb. Sci.* **598**, 117680 (2020).

Khalid, H. et al. *Membranes* **12**, 989 (2022).

Eon Chae, J. et al. *J. Ind. Eng. Chem.* **133**, 255–262 (2024).

Hodges, A. et al. *J. Chem. Eng. Data* **68**, 1485–1506 (2023).
