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

from .catalyst_layers import CatalystLayer, PtCCatalystLayer
from .flow_channels import FlowChannel
from .membrane import Membrane, PFSA
from .porous_layers import GasDiffusionLayer, MicroPorousLayer


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
    """

    cl: CatalystLayer = field(default_factory=PtCCatalystLayer)
    gdl: GasDiffusionLayer = field(default_factory=GasDiffusionLayer)
    mpl: MicroPorousLayer = field(default_factory=MicroPorousLayer)
    ch: FlowChannel = field(default_factory=FlowChannel)
    has_mpl: bool = False
    has_gdl: bool = True
    thermal_contact_resistance: float = 0.

    def __post_init__(self):
        self.porous_layers = (
            ([self.cl, self.mpl] if self.has_mpl else [self.cl])
            + ([self.gdl] if self.has_gdl else [])
        )
        self.layers = self.porous_layers + [self.ch]


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
        Active area (m^2).
    """

    ca: CellSide = field(default_factory=CellSide)
    an: CellSide = field(default_factory=CellSide)
    membrane: Membrane = field(default_factory=PFSA)
    area: float = 1.
    electrical_resistance: float = 0.

    def __post_init__(self):
        self.porous_layers = self.an.porous_layers[::-1] + self.ca.porous_layers
        self.layers = self.an.layers[::-1] + [self.membrane] + self.ca.layers
        self.sides = [self.ca, self.an]
