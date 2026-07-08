Transport in porous layers, channels, and heat
====================================================

Two-phase (liquid water) transport
----------------------------------------

:class:`~marapendi.models.darcy.DarcyTransportModel` treats liquid water as
the non-wetting phase driven by capillary pressure. The entry (breakthrough)
capillary pressure of a layer is set by its geometry
(:meth:`~marapendi.porous_layers.porous_layers.PorousLayer._compute_breakthrough_pressure`):

.. math::

    P_\mathrm{bt} = \frac{\gamma(T)\,|\cos\theta|}{\sqrt{K/\varepsilon}},

with :math:`\gamma(T)` the water surface tension, :math:`\theta` the layer's
contact angle, :math:`K` the absolute permeability and :math:`\varepsilon` the
porosity. Saturation and capillary pressure are related by a power law with
exponent :math:`n` = ``J_function_exponent``
(:meth:`~marapendi.models.darcy.DarcyTransportModel.capillary_pressure_from_saturation`):

.. math::

    P_c(s) = P_\mathrm{bt}\, s^{\,n}.

Given the non-wetting (liquid) flux :math:`q` through the layer, the
saturation rise across the layer is obtained from a Darcy-flow resistance
:math:`R_s` (``saturation_flow_resistance``, from permeability and relative
permeability exponent :math:`q_\mathrm{rel}` =
``relative_permeability_exponent``) via
(:meth:`~marapendi.models.darcy.DarcyTransportModel.calculate_non_wetting_saturation`):

.. math::

    \Delta s = \left(R_s\, q\, \frac{q_\mathrm{rel}+n}{n}\right)^{1/(q_\mathrm{rel}+n)},
    \qquad
    s_\mathrm{avg} = s_\mathrm{up} + \Delta s\,\frac{q_\mathrm{rel}+n}{q_\mathrm{rel}+n+1}.

Gas-phase diffusion
------------------------

:class:`~marapendi.models.diffusion.PorousGasDiffusionModel` combines
molecular (Fickian) and Knudsen diffusion resistances in series, both
penalised by liquid saturation
(:meth:`~marapendi.models.diffusion.PorousGasDiffusionModel.total_diffusion_resistance`):

.. math::

    f(s) = \left[\max(1-s,\,10^{-6})\right]^{m}, \qquad
    L_\mathrm{eff} = \frac{\delta\,\tau}{\varepsilon\, f(s)},

.. math::

    R_\mathrm{diff} = L_\mathrm{eff}
        \left(\frac{1}{D_{ij}} + \frac{1}{D_\mathrm{Kn}}\right),
    \qquad
    D_\mathrm{Kn} = \frac{d_\mathrm{pore}}{3}
        \sqrt{\frac{8RT}{\pi\, M}},

where :math:`m` = ``water_saturation_exponent``, :math:`\delta` the layer
thickness, :math:`\tau` the tortuosity, :math:`d_\mathrm{pore}` the pore
diameter, :math:`M` the species molecular weight, and :math:`D_{ij}` the
binary molecular diffusivity.

Channel transport
----------------------

Species transport from the flow channel to the catalyst layer combines a
diffusive and a convective sub-resistance. The default Sherwood-number
approach
(:class:`~marapendi.channel.gas_transport_resistance.ChannelGasResistanceModel`)
gives

.. math::

    R_\mathrm{diff}^\mathrm{ch} = \frac{d_h}{Sh\, D},
    \qquad
    R_\mathrm{conv}^\mathrm{ch} = \frac{B_\mathrm{ch}\, L\, W}{2}
        \left(1 + \frac{1}{\mathrm{ch/land}}\right)\frac{n_\mathrm{ch}}{\dot V},

with :math:`d_h` the channel hydraulic diameter, :math:`Sh` the Sherwood
number, :math:`L, W` the channel length/width, ``channel_land_ratio`` the
channel-to-land width ratio, :math:`n_\mathrm{ch}` the number of parallel
channels, and :math:`\dot V` the volumetric flow rate. An alternative
empirical form follows Baker et al., *J. Electrochem. Soc.* **156**, B991
(2009)
(:class:`~marapendi.channel.gas_transport_resistance.BakerChannelGasResistanceModel`),
replacing :math:`d_h/Sh` with :math:`A_\mathrm{ch}\times(\text{channel
half-width})`. The end-to-end resistance seen by each species at the catalyst
layer — porous layers in series, plus the channel term, plus (for O₂) the
ionomer-film term from :doc:`kinetics` — is assembled by
:class:`~marapendi.models.gas_transport_resistance.GasTransportModel`.

Thermal model
-----------------

The MEA sits between two parallel thermal paths (cathode and anode stacks of
porous layers plus their contact resistances)
(:meth:`~marapendi.models.thermal.ThermalModel.heat_transfer_resistance`):

.. math::

    R_\mathrm{th}^{-1} = \sum_{s\,\in\,\{\mathrm{ca,an}\}}
        \left(\sum_{\text{layer} \neq \mathrm{cl}} R_\mathrm{th}^\mathrm{layer}
        + R_\mathrm{contact}\right)^{-1}_s.

The steady-state MEA temperature rise above the stack/coolant temperature
:math:`T` follows the lumped heat balance
(:meth:`~marapendi.models.thermal.ThermalModel.mea_temperature`):

.. math::

    \dot q = i\left(-\frac{\Delta H_\mathrm{LHV}(T)}{2F} - V_\mathrm{cell}\right),
    \qquad
    T_\mathrm{MEA} = T + \dot q\, R_\mathrm{th},

i.e. every irreversible loss (the gap between the LHV-equivalent voltage and
the actual cell voltage) is assumed to appear as heat at the MEA. This lumped
treatment follows Ferrara et al., *J. Power Sources* **390**, 197–207 (2018).
See :doc:`../user_guide/extending_models` (Pattern 4) for how to add an
explicit convective coolant term to :math:`R_\mathrm{th}` by subclassing
:class:`~marapendi.models.thermal.ThermalModel`.
