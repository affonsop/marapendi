ORR kinetics
================

:class:`~marapendi.models.thermo.electrochemistry.ElectrochemicalReaction`
parameterises the electrode reaction — the oxygen reduction reaction (ORR) at
the cathode, hydrogen oxidation (HOR) at the anode — with an Arrhenius/power-law
exchange current density
(:func:`~marapendi.models.thermo.electrochemistry.calculate_exchange_current_density`):

.. math::

    i_0(T, a) = i_0^\mathrm{ref} \,
        \left(\frac{a}{a^\mathrm{ref}}\right)^{\gamma}
        \exp\!\left[-\frac{E_a}{R}\left(\frac{1}{T} - \frac{1}{T^\mathrm{ref}}\right)\right],

where :math:`\gamma` is ``reaction_order``, :math:`E_a` is
``activation_energy``, and :math:`a` is the reactant activity — the local O₂
(or H₂) partial pressure at the catalyst layer, normalised by
``reference_activity``, after subtracting the transport losses from
:doc:`catalyst_layer` and :doc:`gas_transport`.

Tafel slope and activation overpotential
----------------------------------------------

The Tafel slope
(:meth:`~marapendi.models.thermo.electrochemistry.ElectrochemicalReaction.tafel_slope`)
is

.. math::

    b = \frac{2.303\, R T}{n \alpha F},

with :math:`n` the number of electrons transferred (``number_of_electrons``)
and :math:`\alpha` the charge-transfer coefficient (``charge_transfer_coeff``).
The activation overpotential
(:func:`~marapendi.models.thermo.electrochemistry.calculate_tafel_overpotential`)
switches between the symmetric Butler–Volmer form and the high-overpotential
Tafel approximation:

.. math::

    \eta_\mathrm{act} =
    \begin{cases}
        \dfrac{RT}{n\alpha F} \sinh^{-1}\!\left(\dfrac{i}{2\,i_0}\right)
            & \alpha = 0.5 \\[6pt]
        \dfrac{RT}{n\alpha F} \ln\!\left(\dfrac{i}{i_0}\right)
            & \alpha \neq 0.5
    \end{cases}

where :math:`i` is the local current density (``current_density +
crossover_current``, i.e. including the H₂-crossover equivalent current). A
linearised form is also available for low overpotentials
(:math:`|\eta_\mathrm{act}| \ll RT/(n\alpha F)`,
:func:`~marapendi.models.thermo.electrochemistry.calculate_linear_overpotential`):

.. math::

    \eta_\mathrm{act} = \frac{RT}{n\alpha F}\frac{i}{i_0}.

:math:`\eta_\mathrm{act}` is the activation term subtracted in the cell
voltage assembly — see :doc:`cell_voltage`.
