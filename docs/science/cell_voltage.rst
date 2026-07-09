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

Hydrogen crossover current
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Following Goshtasbi et al. (2020), the crossover current density is the H₂
permeation flux through the membrane expressed as an equivalent current:

.. math::

    i_x = 2F\, J_{\mathrm{H_2},\mathrm{mb}},

with the flux itself driven by the anode-to-cathode H₂ partial-pressure
difference across the membrane
(:meth:`~marapendi.membrane.membrane_base.Membrane.hydrogen_permeation_flux`):

.. math::

    J_{\mathrm{H_2},\mathrm{mb}} = \Psi_{\mathrm{H_2},\mathrm{mb}}(\lambda, T)\,
        \frac{p_{\mathrm{H_2},\mathrm{an}}}{\delta_\mathrm{mb}},

where :math:`\delta_\mathrm{mb}` is the dry membrane thickness and the
pressure driving force is the full anode H₂ partial pressure
:math:`p_{\mathrm{H_2},\mathrm{an}}` since the cathode-side H₂ partial
pressure is negligible in comparison. The membrane's H₂ permeability
:math:`\Psi_{\mathrm{H_2},\mathrm{mb}}(\lambda, T)`
(:meth:`~marapendi.membrane.pem.PFSAIonomer.h2_permeability`) is itself an
Arrhenius fit with a water-content-dependent term, also from Goshtasbi et al.
(2020):

.. math::

    \Psi_{\mathrm{H_2},\mathrm{mb}}(\lambda, T) =
        15.7\times10^{-15}\, e^{-20280\times10^{3}/(RT)}
        + f_v(\lambda, T)\; 45\times10^{-15}\, e^{-18930\times10^{3}/(RT)}
        \quad \left[\mathrm{\frac{kmol}{m\,s\,Pa}}\right],

where :math:`f_v(\lambda, T) = \lambda V_w(T) / (V_\mathrm{dry} + \lambda
V_w(T))` is the water volume fraction in the ionomer
(:meth:`~marapendi.membrane.ionomer_base.Ionomer.water_vol_fraction`):.

Ohmic overpotential
------------------------

Voltage vs. heat release
-----------------------------

The gap between the LHV-equivalent (thermoneutral) voltage and the actual
cell voltage is exactly the heat released at the MEA — see :doc:`heat_transfer`
for how that heat release drives the MEA temperature, and :doc:`steady_state_model`
for how :math:`V_\mathrm{cell}` and :math:`T_\mathrm{MEA}` are solved together
(or one after the other, depending on the model).

References
--------------

Neyerlin, K. C. et al. *J. Electrochem. Soc.* **154**, B279 (2007).

Goshtasbi, A. et al. *J. Electrochem. Soc.* **167**, 024518 (2020).
