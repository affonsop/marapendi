Cell voltage
================

In :class:`~marapendi.models.voltage.VoltageModel` the fuel cell voltage calculated
as: 

.. math::

    V_\mathrm{cell} = E_\mathrm{rev} - \eta_\mathrm{act} - \eta_\mathrm{ohm},


Reversible cell voltage
------------------------

The reversible cell voltage is calculated with Nernst equation in 
:func:`~marapendi.models.thermo.electrochemistry.calculate_reversible_cell_voltage`: 

.. math::

    E_{rev} = -\frac{\Delta G^\circ}{2F}  + 
     \frac{\Delta S^\circ}{2F} (T - T_0) + 
     \frac{RT}{2F} 
     \ln Q,

where: 

.. math::

    Q = \left(\frac{p^\text{CL}_{\mathrm{H_2}}}{p_0}\right)
        \left(\frac{p^\text{Pt}_{\mathrm{O_2}}}{p_0}\right)^{0.5}

is the reaction quotient (``activities_ratio`` in the code, built from partial
pressures normalised by a reference pressure), :math:`F` is the Faraday
constant, :math:`R` the gas constant, and :math:`T_0` and :math:`p_0` the
standard-state reference temperature and pressure. The reaction Gibbs free
energy change is :math:`\Delta G^\circ = -237.14\ \mathrm{kJ/mol}` and the
entropy change is :math:`\Delta S^\circ = -163.31\ \mathrm{J/(mol \cdot K)}`,
both at the reference temperature :math:`T_0 = 298.15\ \mathrm{K}` and
pressure :math:`p_0 = 10^5\ \mathrm{Pa}`.


Activation overpotential
------------------------

The anode (HOR) activation overpotential is neglected; only the cathode (ORR)
overpotential is retained, in the Tafel form
(:func:`~marapendi.models.thermo.electrochemistry.calculate_tafel_overpotential`):

.. math::

    \eta_\mathrm{act} = \frac{RT}{\alpha F}
        \ln\left(\frac{i + i_x}{i_0}\right),

where :math:`i` is the current density, :math:`i_x` the H₂-crossover
equivalent current density (below), and :math:`\alpha` the cathode
charge-transfer coefficient. The exchange current density :math:`i_0` follows
the ORR kinetics fitted by Neyerlin et al. (2007) on Pt/C catalysts — an
Arrhenius temperature dependence combined with a power-law dependence on the
O₂ partial pressure at the Pt surface:

.. math::

    i_0 = i_{0,\mathrm{ref}}
        \left(\frac{p^\mathrm{Pt}_{\mathrm{O_2}}}{p_\mathrm{ref}}\right)^{\gamma_c}
        \exp\left[\frac{E_{\mathrm{act,ca}}}{R}
        \left(\frac{1}{T_\mathrm{ref}} - \frac{1}{T}\right)\right],

with :math:`\gamma_c` the cathode reaction order and
:math:`E_\mathrm{act,ca}` the cathode activation energy — these are exactly
``reaction_order``, ``activation_energy`` and
``reference_exchange_current_density`` on
:class:`~marapendi.models.thermo.electrochemistry.ElectrochemicalReaction`.
See :doc:`orr_kinetics` for the symmetric Butler–Volmer alternative used away
from the high-overpotential Tafel limit, and :doc:`catalyst_layer` for how
the O₂ activity reaching the Pt surface already accounts for transport losses
through the ionomer film.

.. note:: 
    Mass transport losses are accounted for by the use of the local oxygen 
    partial pressure :math:`p^\mathrm{Pt}_{\mathrm{O_2}}` in the calculation 
    of the reversible voltage and of the exchange current density. 


Hydrogen crossover current
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Following Goshtasbi et al. (2020), the crossover current density is the H₂
permeation flux through the membrane expressed as an equivalent current:

.. math::

    i_x = 2F\, J_{\mathrm{H_2},\mathrm{mb}},

with the flux itself driven by the anode-to-cathode H₂ partial-pressure
difference across the membrane
(:meth:`~marapendi.components.membrane.membrane_base.Membrane.hydrogen_permeation_flux`):

.. math::

    J_{\mathrm{H_2},\mathrm{mb}} = \Psi_{\mathrm{H_2},\mathrm{mb}}(\lambda, T)\,
        \frac{p_{\mathrm{H_2},\mathrm{an}}}{\delta_\mathrm{mb}},

where :math:`\delta_\mathrm{mb}` is the dry membrane thickness and the
pressure driving force is the full anode H₂ partial pressure
:math:`p_{\mathrm{H_2},\mathrm{an}}` since the cathode-side H₂ partial
pressure is negligible in comparison. The membrane's H₂ permeability
:math:`\Psi_{\mathrm{H_2},\mathrm{mb}}(\lambda, T)`
(:meth:`~marapendi.components.membrane.pem.PFSAIonomer.h2_permeability`) is itself an
Arrhenius fit with a water-content-dependent term, also from Goshtasbi et al.
(2020):

.. math::

    \Psi_{\mathrm{H_2},\mathrm{mb}}(\lambda, T) =
        15.7\times10^{-15}\, e^{-20280\times10^{3}/(RT)}
        + f_v(\lambda, T)\; 45\times10^{-15}\, e^{-18930\times10^{3}/(RT)}
        \quad \left[\mathrm{\frac{kmol}{m\,s\,Pa}}\right],

where: 

.. math::
    
    f_v(\lambda, T) = \frac{\lambda V_w(T)}
    {V_\mathrm{dry} + \lambda V_w(T)}

is the water volume fraction in the ionomer, calculated with 
:meth:`~marapendi.components.membrane.ionomer_base.Ionomer.water_vol_fraction`.

Ohmic overpotential
------------------------

The ohmic overpotential is the sum of three specific resistances in series,
scaled by the current density
(:meth:`~marapendi.models.voltage.VoltageModel.ohmic_overpotential`):

.. math::

    \eta_\mathrm{ohm} = \left(r_\mathrm{el} + r_\mathrm{mb} + r^\mathrm{ca}_\mathrm{CL}\right) i,

where :math:`r_\mathrm{el}` is the electric (electronic) resistance of the
cell — a fitting parameter
(:attr:`~marapendi.components.cell.cell.Cell.electric_resistance`); :math:`r_\mathrm{mb}`
is the membrane proton resistance
(:meth:`~marapendi.components.membrane.pem.PFSA.proton_resistance`, :doc:`membrane_correlations`);
and :math:`r^\mathrm{ca}_\mathrm{CL}` is the cathode catalyst-layer proton
resistance, following Neyerlin et al. (2007) as parameterised by Goshtasbi et
al. (2020) (:doc:`catalyst_layer`). Together, :math:`r_\mathrm{el} +
r_\mathrm{mb}` make up the high-frequency resistance
(:meth:`~marapendi.models.voltage.VoltageModel.high_frequency_resistance`).

The membrane resistance :math:`r_\mathrm{mb} = \delta_\mathrm{mb} /
\sigma_\mathrm{mb}^\mathrm{avg}` uses the through-plane average conductivity
:math:`\sigma_\mathrm{mb}^\mathrm{avg}`, which is calculated
(:meth:`~marapendi.components.membrane.pem.PFSA.proton_conductivity`) as the harmonic
mean of the local conductivity
(:meth:`~marapendi.components.membrane.pem.PFSAIonomer.proton_conductivity`,
:doc:`membrane_correlations`) over the membrane water-content profile from the
:doc:`water_balance` solve.

The cathode catalyst-layer proton resistance is given by
(:meth:`~marapendi.components.porous_layers.catalyst_layers.CatalystLayer.effective_charge_resistance`):

.. math::

    r^\mathrm{ca}_\mathrm{CL} = r_\mathrm{CL}^\mathrm{ca,sheet} / (3 + \zeta),

as proposed by Neyerlin et al. (2007), with :math:`\zeta` a parameter
accounting for the cathode catalyst-layer utilization. The polynomial fit to
the solution obtained by Neyerlin et al., provided by Goshtasbi et al. (2020),
is adopted:

.. math::

    \zeta = -8.287\times10^{-3}\, \nu^2 + 7.184\times10^{-1}\, \nu - 2.072\times10^{-3},

with :math:`\nu = i\, r_\mathrm{CL}^\mathrm{ca,sheet} / b` and :math:`b =
2.303\, RT / \alpha F` the Tafel slope — even though, as noted by Goshtasbi et
al., Neyerlin et al. obtained their results for simplified kinetics that does
not account for Pt oxidation. The sheet proton resistance
:math:`r_\mathrm{CL}^\mathrm{ca,sheet}`
(:meth:`~marapendi.components.porous_layers.catalyst_layers.CatalystLayer.ionomer_sheet_charge_resistance`)
is determined as:

.. math::

    r_\mathrm{CL}^\mathrm{ca,sheet} = \frac{\delta_\mathrm{CL}}
        {\left(\varepsilon_\mathrm{ion}/\tau_\mathrm{ion}\right)\,
        \sigma_\mathrm{ion}(\lambda_\mathrm{ion,CL})}.

See :doc:`catalyst_layer` for how this sheet resistance combines with the
parallel electrolyte (liquid-filled) resistance when the catalyst layer is
flooded. The ionomer water content :math:`\lambda_\mathrm{ion,CL})` is assumed equal to 
the equilibrium water content at the catalyst layer (see :doc:`water_balance`).

.. note:: 
    The calculations of :math:`\sigma_\mathrm{mb}^\mathrm{avg}` and of :math:`r_\mathrm{CL}^\mathrm{ca,sheet}` 
    differ from the ones
    in Affonso Nóbrega et al. (2026), as they do not account for the effect of 
    the liquid saturation in the catalyst layer. Indeed, we consider that there is too 
    much uncertainty on how local water saturation in the
    CL impact the hydration state of the membrane and ionomer thin layers, 
    specially considering Schröeder's paradox. Therefore, we neglect the effect of 
    liquid water in the catalyst layer for the sake of simplicity.



References
--------------

Affonso Nobrega, P. et al. *J. Electrochem. Soc.* **173**, 114503 (2026).

Goshtasbi, A. et al. *J. Electrochem. Soc.* **167**, 024518 (2020).

Neyerlin, K. C. et al. *J. Electrochem. Soc.* **154**, B279 (2007).