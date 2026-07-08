Cell voltage and reaction kinetics
======================================

Reversible (Nernst) voltage
--------------------------------

:func:`~marapendi.models.thermo.electrochemistry.calculate_reversible_cell_voltage`
computes :math:`E_\mathrm{rev}` from the standard Gibbs energy and entropy of
formation of liquid water:

.. math::

    E_\mathrm{rev}(T, Q) = \frac{
        -\Delta G^\circ_{f,\mathrm{H_2O(l)}}
        + \Delta S^\circ_{f,\mathrm{H_2O(l)}} \,(T - T^\circ)
        + R T \ln Q
    }{2F},

where :math:`Q = a_{\mathrm{H_2}} \, a_{\mathrm{O_2}}^{1/2} / a_{\mathrm{H_2O}}`
is the reaction quotient (``activities_ratio`` in the code, built from partial
pressures normalised by a reference pressure), :math:`F` is the Faraday
constant, :math:`R` the gas constant, and :math:`T^\circ` the standard-state
temperature.

Butler–Volmer / Tafel activation overpotential
----------------------------------------------------

:class:`~marapendi.models.thermo.electrochemistry.ElectrochemicalReaction`
gives each electrode reaction (ORR at the cathode, HOR at the anode) an
Arrhenius/power-law exchange current density
(:func:`~marapendi.models.thermo.electrochemistry.calculate_exchange_current_density`):

.. math::

    i_0(T, a) = i_0^\mathrm{ref} \,
        \left(\frac{a}{a^\mathrm{ref}}\right)^{\gamma}
        \exp\!\left[-\frac{E_a}{R}\left(\frac{1}{T} - \frac{1}{T^\mathrm{ref}}\right)\right],

where :math:`\gamma` is ``reaction_order``, :math:`E_a` is
``activation_energy``, and :math:`a` is the reactant activity (partial
pressure normalised by ``reference_activity``). The Tafel slope
(:meth:`~marapendi.models.thermo.electrochemistry.ElectrochemicalReaction.tafel_slope`)
is

.. math::

    b = \frac{2.303\, R T}{n \alpha F},

with :math:`n` the number of electrons transferred (``number_of_electrons``)
and :math:`\alpha` the charge-transfer coefficient (``charge_transfer_coeff``).
The activation overpotential itself
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
:math:`\eta_\mathrm{act} = \frac{RT}{n\alpha F}\frac{i}{i_0}`.

The full cell voltage assembled by
:class:`~marapendi.models.voltage.VoltageModel` is exactly

.. math::

    V_\mathrm{cell} = E_\mathrm{rev} - \eta_\mathrm{act} - \eta_\mathrm{ohm},

with no additional fitted terms — every other loss mechanism (membrane
resistance, catalyst-layer charge transport, mass transport) is folded into
:math:`\eta_\mathrm{ohm}` via the models described in :doc:`membrane` and
:doc:`transport`.

Catalyst-layer charge transport
------------------------------------

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
conductivity (see :doc:`membrane`). In series with any parallel electrolyte
(liquid-filled) resistance :math:`R_\mathrm{elyte}`, the combined sheet
resistance is :math:`R_\mathrm{sheet} = \left(1/R_\mathrm{ion} +
1/R_\mathrm{elyte}\right)^{-1}`.

The *effective* charge resistance actually seen by the reaction accounts for
the reaction-current distribution through the catalyst-layer depth, following
Neyerlin et al., *J. Electrochem. Soc.* **154**, B279 (2007) as parameterised
by Goshtasbi et al., *J. Electrochem. Soc.* **167**, 024518 (2020)
(:meth:`~marapendi.porous_layers.catalyst_layers.CatalystLayer.effective_charge_resistance`):

.. math::

    \nu = \min\!\left(\frac{R_\mathrm{sheet}\, i}{b},\, 10\right), \qquad
    \xi = \nu\,(-8.287\times10^{-3}\,\nu + 0.7184) - 2.072\times10^{-3}, \qquad
    R_\mathrm{eff} = \frac{R_\mathrm{sheet}}{3 + \xi}.

Oxygen transport through the ionomer film covering the Pt/C agglomerates
follows Hao et al., *J. Electrochem. Soc.* **162**, F854 (2015) and **163**,
F744 (2016)
(:meth:`~marapendi.porous_layers.catalyst_layers.PtCCatalystLayer.o2_ionomer_film_resistance`):
Pt, carbon, ionomer and pore volume fractions are computed directly from the
platinum loading, ionomer-to-carbon ratio and catalyst density, and the local
O₂ transport resistance combines a bulk-film diffusion term with an
interfacial (gas/ionomer, ionomer/Pt) term via their equation 32.
