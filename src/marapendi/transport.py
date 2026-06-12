"""
Gas transport model: gas-phase species transport resistance of a cell side.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cell import Cell, CellSide
from .gas import GasModel, species_indexes
from .constants import FARADAY_CONSTANT
from .state import CellSideState, CellState


@dataclass
class GasTransportModel:
    """Computes the gas transport resistance of a :class:`CellSide`."""

    def gas_transport_resistance(self, side: CellSide, side_state: CellSideState, species: str = 'o2') -> float:
        """Total gas transport resistance for ``species`` (s/m), from the channel
        to the catalyst layer.

        Includes the porous layers, the flow channel and, for O2, the
        catalyst-layer ionomer film resistance.
        """
        porous_layer_resistance = sum(
            layer.gas_transport_resistance(layer_state, species)
            for layer, layer_state in zip(side.porous_layers, side_state.porous_layers)
        )
        channel_resistance = side.ch.gas_transport_resistance(side_state.ch, species)
        ionomer_film_resistance = (
            side.cl.o2_ionomer_film_resistance(side_state.cl) if species == 'o2' else 0
        )
        return porous_layer_resistance + channel_resistance + ionomer_film_resistance

    def calculate_gas_concentrations(self, cell: Cell, state: CellState) -> None:
        """Update the catalyst-layer gas composition for both cell sides.

        Reactant (O2/H2) and water vapor concentrations at the catalyst
        layer are derived from the channel concentrations, the reactant
        consumption (from ``state.current_density``) and the water vapor
        flux from the membrane water balance. Requires
        ``side_state.h2ov_transport_resistance`` and ``side_state.vapor_flux``
        to already be set (i.e. after
        :meth:`~marapendi.water_balance.MembraneWaterBalanceModel.calculate_water_transport`).
        """
        o2_consumption = state.current_density / (4 * FARADAY_CONSTANT)
        for side, side_state, reactant, reactant_consumption in zip(
            cell.sides, state.sides, ('o2', 'h2'), (o2_consumption, 2 * o2_consumption),
        ):
            side_state.reactant_transport_resistance = self.gas_transport_resistance(side, side_state, reactant)
            cl_state = side_state.cl
            gas_concentration = GasModel.concentration(cl_state)

            x_reactant = np.maximum(
                1e-12,
                GasModel.species_concentration(side_state.ch, reactant)
                - reactant_consumption * side_state.reactant_transport_resistance,
            ) / gas_concentration

            x_h2o = np.where(
                cl_state.liquid_saturation > 0,
                GasModel.saturation_concentration(cl_state) / gas_concentration,
                np.maximum(
                    1e-12,
                    GasModel.species_concentration(side_state.ch, 'h2o')
                    + side_state.vapor_flux * side_state.h2ov_transport_resistance,
                ) / gas_concentration,
            )

            # The non-reactant species (H2 on the cathode, O2 on the anode) keeps its
            # channel-inlet value, broadcast to the shape of the other components.
            other_species = next(s for s in ('o2', 'h2') if s != reactant)
            x_other = np.asarray(cl_state.gas.X[..., species_indexes[other_species]])
            x_n2 = 1 - x_reactant - x_h2o - x_other

            shape = np.broadcast_shapes(
                np.shape(x_reactant), np.shape(x_h2o), np.shape(x_n2), x_other.shape,
            )
            gas_X = np.zeros(shape + (4,))
            gas_X[..., species_indexes[reactant]] = x_reactant
            gas_X[..., species_indexes['h2o']] = x_h2o
            gas_X[..., species_indexes['n2']] = x_n2
            gas_X[..., species_indexes[other_species]] = x_other
            cl_state.gas.X = gas_X

            cl_state.relative_humidity = GasModel.relative_humidity(cl_state)
