"""
Membrane component dataclasses.

These dataclasses store geometric and material parameters for proton- or
anion-exchange membranes.  Physics computations (water volume fraction,
diffusion coefficients, sorption isotherms, permeation flux, ionic
conductivity, etc.) are implemented as stateless strategy methods in
:mod:`marapendi.models.membrane` (``IonomerModel``, ``MembraneModel``,
``PFSAModel``).
"""

import numpy as np
import cantera as ct 
from dataclasses import dataclass, field, fields as dataclass_fields
from marapendi.components.ionomer import Ionomer, PFSAIonomer
from marapendi.components.porous_layers import PorousLayer

@dataclass
class Membrane(PorousLayer, Ionomer):
    """
    Base dataclass for proton- or anion-exchange membranes.

    Combines porous-layer geometry (:class:`~marapendi.components.porous_layers.PorousLayer`)
    with ionomer material parameters (:class:`~marapendi.components.ionomer.Ionomer`).

    All physics (water volume fraction, diffusion coefficients, sorption
    isotherms, H₂ permeation flux, ionic conductivity, electroosmotic drag)
    are computed by the corresponding model classes in
    :mod:`marapendi.models.membrane`.

    Attributes
    ----------
    thickness : float
        Membrane thickness [m].
    eps_ion : float
        Volume fraction of ionomer [-] (default 1 — pure ionomer).
    tort_ion : float
        Ionomer tortuosity factor [-] (default 1).
    ionomer : Ionomer, optional
        Ionomer parameter object whose fields are copied onto ``self``
        during ``__post_init__`` for direct attribute access.
    bulk_thermal_conductivity : float
        Thermal conductivity [W/(m·K)] (default 0.9).
    bulk_specific_heat_capacity : float
        Specific heat capacity [J/(kg·K)] (default 2000).
    """
    
    thickness: float
    eps_ion: float = 1.
    tort_ion: float = 1.
    ionomer: Ionomer = field(default=None)
    bulk_thermal_conductivity: float = 0.9
    bulk_specific_heat_capacity: float = 2000.
    
    def __eq__(self, other):
        """Use identity comparison — layers are unique instances in the cell model.
        This avoids ambiguous truth-value errors from numpy array fields during
        membership tests like `layer in self.ionomer_layers`."""
        return self is other

    def __hash__(self):
        return id(self)


    def __post_init__(self):
        """Resolve ionomer fields and wire bulk-property aliases.

        If an *ionomer* instance was supplied, all of its dataclass fields are
        copied onto ``self`` so that ionomer material parameters are directly
        accessible as attributes (e.g. ``self.EW_ion``, ``self.sigma_ref_ion``).
        This copy runs before the parent ``__post_init__`` calls so derived
        quantities (``c_ion``, ``V_ion``) are computed with the correct values.

        ``bulk_density`` is aliased to ``rho_dry_ion`` so that
        ``Cell.build_property_arrays`` picks it up correctly.
        """
        if self.ionomer is not None:
            for f in dataclass_fields(self.ionomer):
                setattr(self, f.name, getattr(self.ionomer, f.name))
        self.bulk_density = self.rho_dry_ion
        PorousLayer.__post_init__(self)
        Ionomer.__post_init__(self)


@dataclass
class PFSAMembrane(Membrane, PFSAIonomer):
    """Perfluorosulfonic acid membrane (e.g. Nafion NR-211/212).

    Inherits all ionomer parameters from :class:`PFSAIonomer` and all
    geometry/thermal parameters from :class:`Membrane`.  The default
    parameterisation targets Nafion NR-211 (25 µm) at 80 °C; override
    ``bulk_thermal_conductivity`` and ``thickness`` for other grades.
    """