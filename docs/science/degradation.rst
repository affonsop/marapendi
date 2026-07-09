Degradation
===============

:mod:`marapendi.degradation.degradation` models electrochemical Pt
electrochemical-surface-area (ECSA) loss following Darling & Meyers (2003)
and Schneider et al. (2019). Platinum dissolution
(:class:`~marapendi.degradation.degradation.PlatinumDissolution`) follows a
Butler–Volmer-type rate law referenced to a particle-size-dependent
equilibrium potential with a Gibbs–Thomson correction
(:meth:`~marapendi.degradation.degradation.PlatinumDissolution.equilibrium_potential`):

.. math::

    E_\mathrm{eq}(r) = E^\circ - \frac{\Delta E_\gamma(r)}{2F},

where :math:`\Delta E_\gamma(r)` is the surface-tension potential shift for a
particle of radius :math:`r` and :math:`E^\circ` the reference (bulk)
dissolution potential. The net dissolution rate
(:meth:`~marapendi.degradation.degradation.PlatinumDissolution.rate_of_reaction`)
scales with the potential departure from :math:`E_\mathrm{eq}(r)`, is
suppressed by platinum-oxide surface coverage :math:`\theta_\mathrm{ox}`, and
carries an empirical relative-humidity power-law dependence
(exponent 1.7, from Schneider et al. 2019):

.. math::

    r_\mathrm{diss}(r) = k\;\mathrm{RH}^{1.7}\,(1-\theta_\mathrm{ox})\,
        g\!\left(E - E_\mathrm{eq}(r),\; c_\mathrm{Pt^{2+}}/c_\mathrm{Pt^{2+}}^\mathrm{ref}\right),

with :math:`g` the Butler–Volmer-type forward/backward rate difference
(anodic/cathodic transfer coefficients ``transfer_coeff_ca`` /
``transfer_coeff_an``). Particle-size evolution then follows from a
population balance over dissolution, chemical (oxide-mediated) and
electrochemical placing/re-deposition rates, with dissolved Pt²⁺ transported
into the membrane and removed by a characteristic-time sink representing
Pt-band formation (since explicit diffusivity/band-distance data are not
available in Schneider et al. 2019, a characteristic time of ≈3 was
back-fitted from their Figure 9).

Active-site density for the oxide-coverage balance assumes
:math:`210\ \mathrm{\mu C/cm^2}` of Pt in the hydrogen-adsorption region,
also from Schneider et al. (2019).

References
--------------

Darling, R. M. & Meyers, J. P. *J. Electrochem. Soc.* **150**, A1523 (2003).

Schneider, P. et al. *J. Electrochem. Soc.* **166**, F322–F333 (2019).
