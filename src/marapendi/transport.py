"""
Gas transport resistance and concentration model.

:class:`GasTransportModel` computes the total gas-phase transport resistance
from the flow channel to the catalyst layer for a given cell side, and updates
the catalyst-layer gas composition from the computed transport resistances.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from .gas_composition import species_indexes


@dataclass
class GasTransportModel:
    """Gas transport resistance and CL gas concentration calculations."""

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

    def calculate_gas_concentrations(self, fc) -> None:
        """Update catalyst-layer gas composition for both cell sides.

        Derives reactant (O2/H2) and water vapor mole fractions at the CL
        from channel concentrations, reactant consumption, and the vapor flux
        already set by the water balance. Requires ``side.h2ov_transport_resistance``
        and ``side.vapor_flux`` to be set before calling.

        Parameters
        ----------
        fc : FuelCell
        """
        for side, reactant in [(fc.ca, 'o2'), (fc.an, 'h2')]:
            side.reactant_transport_resistance = self.gas_transport_resistance(side, reactant)
            gas_concentration = side.cl.gas.concentration()

            side.cl.gas.X[..., species_indexes[reactant]] = np.maximum(
                1e-12,
                side.ch.species_concentration(reactant)
                - side.reactant_consumption * side.reactant_transport_resistance,
            ) / gas_concentration

            side.cl.gas.X[..., species_indexes['h2o']] = np.where(
                side.cl.liquid_saturation > 0,
                side.cl.saturation_concentration() / gas_concentration,
                np.maximum(
                    1e-12,
                    side.ch.species_concentration('h2o')
                    + side.vapor_flux * side.h2ov_transport_resistance,
                ) / gas_concentration,
            )
            side.cl.gas.calculate_relative_humidity()

            side.cl.gas.X[..., species_indexes['n2']] = (
                1
                - side.cl.gas.X[..., species_indexes['o2']]
                - side.cl.gas.X[..., species_indexes['h2o']]
                - side.cl.gas.X[..., species_indexes['h2']]
            )
