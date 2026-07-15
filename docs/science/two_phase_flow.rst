Two-phase flow
==================

:class:`~marapendi.models.darcy.DarcyTransportModel` treats liquid water as
the non-wetting phase driven by capillary pressure in the porous layers (GDL,
MPL, CL). The entry (breakthrough) capillary pressure of a layer is set by its
geometry
(:meth:`~marapendi.components.porous_layers.porous_layers.PorousLayer._compute_breakthrough_pressure`):

.. math::

    p_\mathrm{b} = \frac{\gamma(T)\,|\cos\theta_c|}{\sqrt{K_{abs}/\varepsilon}},

with :math:`\gamma(T)` the water surface tension, :math:`\theta_c` the layer's
contact angle, :math:`K_{abs}` the absolute permeability and :math:`\varepsilon` the
porosity. Saturation and capillary pressure are related by a power law with
exponent :math:`n` = ``J_function_exponent``
(:meth:`~marapendi.models.darcy.DarcyTransportModel.capillary_pressure_from_saturation`):

.. math::

    p_c(s) = p_\mathrm{b} J(s) = p_\mathrm{b}\, s^{\,n}.

The non-wetting (liquid) flux :math:`q_l` through each layer
is obtained with Darcy's law:

.. math::

    J_{l} =
    - \frac{\rho_l K_{abs}k_{rel}}{M_{w} \mu_l} \frac{dp_c}{dx}

with :math:`k_{rel} = s^m`, :math:`m` = ``relative_permeability_exponent``
(:attr:`~marapendi.components.porous_layers.porous_layers.PorousLayer.relative_permeability_exponent`),
so that we can write:

.. math::

    J_{l} = -\frac{\rho_l K_{abs} p_{b}}{M_{w} \mu_l} n s^{n-1 + m} \frac{ds}{dx}

The liquid water flux is assumed constant through all the layers such that water
saturation profiles can be obtained in each layer :math:`k`:

.. math::

    s_{k}(x) = s_{k}(0) +
    \left[-x J_{l} \frac{M_w \mu_l }{\rho_l K_{abs,\text{GDL}}p_{b,\text{GDL}}}\frac{n + m}{n}\right]^{1/\left(n + m\right)}

This saturation increment is what
:meth:`~marapendi.models.darcy.DarcyTransportModel.calculate_non_wetting_saturation`
evaluates at the layer's downstream face, using the lumped
:math:`M_w \mu_l / (\rho_l K_{abs} p_b)` prefactor pre-computed as
:attr:`~marapendi.components.porous_layers.porous_layers.PorousLayer.saturation_flow_resistance`
(:meth:`~marapendi.components.porous_layers.porous_layers.PorousLayer.calculate_saturation_flow_resistance`).

Capillary pressure continuity is enforced so that at the interface between adjacent
layers :math:`k` and :math:`k-1`:

.. math::

    s_{k}(0) = s_{k-1}(\delta_{k-1})\left(\frac{p_{b,k-1}}{p_{b,k}}\right)^{1/n}

This conversion is what
:meth:`~marapendi.models.darcy.DarcyTransportModel.saturation_from_capillary_pressure`
performs, translating the previous layer's
:attr:`~marapendi.simulation.state.LayerState.downstream_capillary_pressure`
into the ``upstream_capillary_pressure`` argument of
:meth:`~marapendi.models.darcy.DarcyTransportModel.calculate_non_wetting_saturation`
for the next layer. The channel/GDL interface uses
:math:`s_{\mathrm{GDL}}(0)=0` as boundary condition (i.e. zero upstream
capillary pressure, the default in
:meth:`~marapendi.models.darcy.DarcyTransportModel.calculate_non_wetting_saturation`),
as discussed in Rajora and Haverkort (2021).

The saturation profile :math:`s_k(x)` derived above closely reproduces the
profiles obtained numerically by Hu et al. (2016) for a fully saturated regime
using a one-dimensional model, which motivated the adoption of this simplified
approach here. The validity of the power-law capillary pressure/saturation
relationship, :math:`p_c(s) = p_\mathrm{b}\,s^{\,n}`, may be questioned, but
typical liquid-water transport models raise similar concerns while requiring
much higher computational cost. This power-law relationship is therefore
retained, since it allows the water-saturation profile in each layer to be
computed explicitly. Some flexibility can still be recovered by fitting the
exponent :math:`n` and, if needed, the layer material properties
(:math:`K_{abs,k}`, :math:`\theta_k`).

References
--------------

Rajora, A. and Haverkort, J. W. *J. Electrochem. Soc.* **168**, 034506 (2021).

Hu, J. et al. *Energy* **111**, 869–883 (2016).
