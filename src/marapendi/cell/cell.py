"""
Cell components: assembly of layers into a fuel cell.

A :class:`CellSide` groups the porous layers and flow channel of one
electrode (anode or cathode). A :class:`Cell` combines a cathode
:class:`CellSide`, an anode :class:`CellSide`, and a
:class:`~marapendi.membrane.Membrane`.

This module defines the dataclasses and the lists used to loop over their
components. Orchestration of the calculations is left to
:class:`marapendi.model.CellModel`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..porous_layers.catalyst_layers import CatalystLayer, PtCCatalystLayer
from ..channel.flow_channels import FlowChannel
from ..membrane.membrane_base import Membrane
from ..membrane.pem import PFSA
from ..porous_layers.porous_layers import GasDiffusionLayer, MicroPorousLayer


@dataclass
class CellSide:
    """One side (anode or cathode) of a cell.

    Attributes
    ----------
    cl : CatalystLayer
        Catalyst layer.
    gdl : GasDiffusionLayer
        Gas diffusion layer.
    mpl : MicroPorousLayer
        Microporous layer.
    ch : FlowChannel
        Flow channel.
    has_mpl, has_gdl : bool
        Whether this side includes a microporous / gas diffusion layer.
    thermal_contact_resistance : float
        Thermal contact resistance between the GDL and the bipolar plate (m²·K/W).
    """

    cl: CatalystLayer = field(default_factory=PtCCatalystLayer)
    gdl: GasDiffusionLayer = field(default_factory=GasDiffusionLayer)
    mpl: MicroPorousLayer = field(default=None)
    ch: FlowChannel = field(default_factory=FlowChannel)

    thermal_contact_resistance: float = 0.

    @property
    def has_mpl(self) -> bool:
        return self.mpl is not None

    @property
    def has_gdl(self) -> bool:
        return self.gdl is not None

    @property
    def porous_layers(self) -> list:
        """Active porous layers, channel-to-CL order (gdl → [mpl →] cl)."""
        return [l for l in [self.gdl, self.mpl, self.cl] if l is not None]

    @property
    def layers(self) -> list:
        """Flow channel + active porous layers, channel-to-CL order."""
        return [self.ch] + self.porous_layers

@dataclass
class Cell:
    """A cell, combining a cathode, an anode and a membrane.

    Attributes
    ----------
    ca : CellSide
        Cathode.
    an : CellSide
        Anode.
    membrane : Membrane
        Membrane.
    area : float
        Active area (m²).
    electric_resistance : float
        Area-specific electrical resistance (Ω·m²).
    """

    ca: CellSide = field(default_factory=CellSide)
    an: CellSide = field(default_factory=CellSide)
    membrane: Membrane = field(default_factory=PFSA)
    area: float = 1.
    electric_resistance: float = 0.
    
    def __post_init__(self): 
        self.mea_surface_heat_capacity = (
            sum(
                layer.volume_heat_capacity * layer.thickness 
                for layer in self.porous_layers
            ) + self.membrane.volume_heat_capacity * self.membrane.dry_thickness
            )
                                        

    @property
    def porous_layers(self) -> list:
        """All porous layers, anode-to-cathode order."""
        return self.an.porous_layers[::-1] + self.ca.porous_layers

    @property
    def layers(self) -> list:
        """All layers (channels + porous + membrane), anode-to-cathode order."""
        return self.an.layers[::-1] + [self.membrane] + self.ca.layers

    @property
    def sides(self) -> list:
        return [self.ca, self.an]
