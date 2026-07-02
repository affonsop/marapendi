:orphan:

Examples
========

End-to-end examples demonstrating the `marapendi` API for PEM fuel-cell
modelling — from cell assembly to parameter estimation.

Each example is a self-contained Python script that produces publication-quality
plots.  Click on a thumbnail to see the full annotated example with downloadable
script and Jupyter notebook.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Assemble a PEM fuel cell from first principles and compute a steady-state polarization curve with a single vectorised call.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_01_polarization_curve_thumb.png
    :alt:

  :doc:`/auto_examples/plot_01_polarization_curve`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Polarization curve</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Fuel Cell Dynamic Load Cycle (FC-DLC) is the JRC/FCH-JU standard load profile for PEMFC endurance testing (Tsotridis et al., EUR 27632 EN, 2015, Appendix F).  It is derived from the New European Driving Cycle (NEDC) and consists of 35 piecewise-constant steps covering 1181 s, including four urban sub-cycles and one extra-urban sub-cycle.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_02_quasi_steady_thumb.png
    :alt:

  :doc:`/auto_examples/plot_02_quasi_steady`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Quasi-steady simulation — FC-DLC profile</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Both ExplicitSteadyStateModel and ImplicitSteadyStateModel compute a steady-state polarization curve from the same cell and conditions, but differ in how MEA temperature is determined:">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_03_implicit_vs_explicit_thumb.png
    :alt:

  :doc:`/auto_examples/plot_03_implicit_vs_explicit`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Implicit vs explicit steady-state model</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="TransientModel integrates coupled ODEs for MEA temperature T_\mathrm{MEA}(t) and the membrane water-content profile \lambda(\xi, t).">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_04_transient_thumb.png
    :alt:

  :doc:`/auto_examples/plot_04_transient`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Transient simulation — FC-DLC profile</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Simulate polarization curves across a range of operating conditions (temperature, pressure, inlet relative humidity) to understand the sensitivity of cell performance to each parameter.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_05_multi_condition_thumb.png
    :alt:

  :doc:`/auto_examples/plot_05_multi_condition`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Multi-condition polarization curves</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="marapendi provides two membrane water-balance models:">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_06_pwl_membrane_thumb.png
    :alt:

  :doc:`/auto_examples/plot_06_pwl_membrane`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Piecewise-linear membrane water balance</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="SteadyStatePolarizationCurveCalibration fits kinetic and transport parameters to multi-condition polarization and HFR data.  It uses scipy.optimize.differential_evolution as the global optimiser, with k-fold cross-validation and automatic complexity selection via the 1-SE rule.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_07_parameter_estimation_thumb.png
    :alt:

  :doc:`/auto_examples/plot_07_parameter_estimation`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Parameter estimation</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/plot_01_polarization_curve
   /auto_examples/plot_02_quasi_steady
   /auto_examples/plot_03_implicit_vs_explicit
   /auto_examples/plot_04_transient
   /auto_examples/plot_05_multi_condition
   /auto_examples/plot_06_pwl_membrane
   /auto_examples/plot_07_parameter_estimation


.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-gallery

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download all examples in Python source code: auto_examples_python.zip </auto_examples/auto_examples_python.zip>`

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download all examples in Jupyter notebooks: auto_examples_jupyter.zip </auto_examples/auto_examples_jupyter.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
