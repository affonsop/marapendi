"""
Gas transport resistance model.

:class:`GasTransportModel` computes the total gas-phase transport resistance
from the flow channel to the catalyst layer for a given cell side, delegating
per-layer resistance to :meth:`~marapendi.porous_layers.PorousLayer.gas_transport_resistance`
(which uses :class:`~marapendi.gas.GasModel` correlations) and ionomer film
resistance to the catalyst layer.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GasTransportModel:
    """Gas transport resistance from flow channel to catalyst layer."""

    def gas_transport_resistance(self, side, species: str = 'o2') -> float:
        """Total gas transport resistance for ``species`` on ``side`` (s/m).

        Parameters
        ----------
        side : FuelCellSide
            Cell side holding porous layers, flow channel, and catalyst layer.
        species : str
            Gas species identifier ('o2', 'h2', 'h2o').

        Returns
        -------
        float
            Sum of porous-layer, channel, and ionomer-film resistances.
        """
        return (
            sum(layer.gas_transport_resistance(layer, species) for layer in side.porous_layers)
            + side.ch.gas_transport_resistance(side.ch, species)
            + (side.cl.o2_ionomer_film_resistance(side.cl.ionomer_water_content, side.cl.temperature) if species == 'o2' else 0)
        )
