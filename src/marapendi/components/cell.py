import numpy as np
from dataclasses import dataclass, field, fields
from .membrane import Membrane
from marapendi.tools.tools import Updatable
from .porous_layers import PorousLayer
from .flow_channels import FlowChannel


@dataclass
class CellSide(Updatable):
    """One side (anode or cathode) of an AEM/PEM cell."""
    name: str = "side"
    cl: PorousLayer = field(default_factory=PorousLayer)
    gdl: PorousLayer = field(default_factory=PorousLayer)
    mpl: PorousLayer = field(default_factory=PorousLayer)
    ch: FlowChannel = field(default_factory=FlowChannel)
    has_mpl: bool = False
    has_gdl: bool = True

    def __post_init__(self):
        self.porous_layers = (
            ([self.cl, self.mpl] if self.has_mpl else [self.cl])
            + ([self.gdl] if self.has_gdl else [])
        )
        self.layers = self.porous_layers + [self.ch]


@dataclass
class Cell(Updatable):
    """
    Geometry, material parameters, and layer stack for an AEM/PEM cell.

    Physics models (membrane transport, gas diffusion, liquid transport,
    catalyst-layer kinetics, voltage) are intentionally excluded from this
    class.  They belong on the :class:`~marapendi.models.model.BaseModel`
    that wraps the simulation, keeping component data separate from the
    equations that act on it.
    """
    name: str = "cell"
    area: float = 1.
    electrical_resistance: float = 0.
    thermal_resistance: float = 0.
    ca: CellSide = field(default_factory=CellSide)
    an: CellSide = field(default_factory=CellSide)
    memb: Membrane = field(default_factory=Membrane)
    charge: str = 'proton'

    def __post_init__(self):
        self.porous_layers = self.an.porous_layers[::-1] + self.ca.porous_layers
        self.layers = self.an.layers[::-1] + [self.memb] + self.ca.layers
        self.build_property_arrays()

    def get_property_array(self, property_name: str):
        """Get an array of a scalar or vector property across all layers.

        - Scalar property  → shape (n_layers, 1)
        - Length-N array   → shape (n_layers, N)

        Layers missing the attribute contribute ``np.nan``.
        """
        values = [getattr(layer, property_name, np.nan) for layer in self.layers]

        if any(isinstance(v, np.ndarray) for v in values):
            n = next(v.shape[0] for v in values if isinstance(v, np.ndarray))
            cols = [v if isinstance(v, np.ndarray) else np.full(n, np.nan)
                    for v in values]
            return np.stack(cols, axis=0)       # (n_layers, N)
        else:
            return np.array(values, dtype=float)[:, np.newaxis]  # (n_layers, 1)

    def build_property_arrays(self):
        _numeric = (int, float, bool, np.floating, np.integer, np.ndarray)
        for f in fields(PorousLayer):
            sample = next(
                (getattr(l, f.name) for l in self.layers
                 if getattr(l, f.name, None) is not None), None)
            if isinstance(sample, _numeric):
                self.__setattr__(f.name, self.get_property_array(f.name))
        for f in fields(Membrane):
            sample = next(
                (getattr(l, f.name) for l in self.layers
                 if getattr(l, f.name, None) is not None), None)
            if isinstance(sample, _numeric):
                self.__setattr__(f.name, self.get_property_array(f.name))
