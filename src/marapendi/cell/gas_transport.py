"""
Gas transport resistance and concentration model.

:class:`GasTransportModel` computes the total gas-phase transport resistance
from the flow channel to the catalyst layer for a given cell side, and updates
the catalyst-layer gas composition from the computed transport resistances.

Static layer parameters (tortuosity, thickness, ionomer properties) are read
from the component tree (``cell``); runtime values (temperature, pressure,
gas composition, saturation) come from the :class:`~marapendi.state.CellState`.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from ..thermo.gas import GasModel, species_indexes


@dataclass
class GasTransportModel:
    """Gas transport resistance and CL gas concentration calculations."""

    def gas_transport_resistance(self, cell_side, side_state, species: str = 'o2') -> float:
        """Total gas transport resistance for ``species`` on one cell side (s/m).

        Parameters
        ----------
        cell_side : FuelCellSide
            Component side — provides static layer physics (tortuosity, thickness,
            ionomer properties).
        side_state : CellSideState
            Runtime state for this side — provides gas composition, temperature,
            pressure and saturation for each layer.
        species : str
            Gas species identifier ('o2', 'h2', 'h2o').

        Returns
        -------
        float
            Sum of porous-layer, channel, and (for O₂) ionomer-film resistances.
        """
        resistance = sum(
            layer.transport_resistance_model.gas_transport_resistance(layer, layer_state, species)
            for layer, layer_state in zip(cell_side.porous_layers, side_state.porous_layers)
        )
        resistance += cell_side.ch.transport_resistance_model.gas_transport_resistance(
            cell_side.ch, side_state.ch, species,
            volume_flow_rate=side_state.ch.inlet_gas_flow_rate,
        )
        if species == 'o2':
            resistance += cell_side.cl.o2_ionomer_film_resistance(
                side_state.cl.ionomer_water_content, side_state.cl.temperature,
            )
        return resistance

    def calculate_gas_concentrations(self, cell, state) -> None:
        """Update catalyst-layer gas composition for both cell sides.

        Derives reactant (O2/H2) and water-vapor mole fractions at the CL from
        channel concentrations, reactant consumption, and the vapor flux already
        set by the water balance.  Reads from ``state``; writes back to
        ``state.ca.cl.gas.X`` / ``state.an.cl.gas.X`` and also syncs the
        results to the component gas objects so that legacy code keeps working.

        Parameters
        ----------
        cell : FuelCell
            Component tree (static physics parameters).
        state : CellState
            Runtime state to read from and write to.
        """
        for cell_side, side_state in [
            (cell.ca, state.ca),
            (cell.an, state.an),
        ]:
            reactant = cell_side.reactant
            side_state.reactant_transport_resistance = self.gas_transport_resistance(
                cell_side, side_state, reactant,
            )
            side_state.h2ov_transport_resistance = self.gas_transport_resistance(
                cell_side, side_state, 'h2o',
            )
            gas_concentration = GasModel.concentration(side_state.cl)

            side_state.cl.gas.X[..., species_indexes[reactant]] = np.maximum(
                1e-12,
                GasModel.species_concentration(side_state.ch, reactant)
                - side_state.reactant_consumption * side_state.reactant_transport_resistance,
            ) / gas_concentration

            side_state.cl.gas.X[..., species_indexes['h2o']] = np.where(
                side_state.cl.liquid_saturation > 0,
                GasModel.saturation_concentration(side_state.cl) / gas_concentration,
                np.maximum(
                    1e-12,
                    GasModel.species_concentration(side_state.ch, 'h2o')
                    + side_state.vapor_flux * side_state.h2ov_transport_resistance,
                ) / gas_concentration,
            )

            side_state.cl.gas.X[..., species_indexes['n2']] = (
                1
                - side_state.cl.gas.X[..., species_indexes['o2']]
                - side_state.cl.gas.X[..., species_indexes['h2o']]
                - side_state.cl.gas.X[..., species_indexes['h2']]
            )

    def max_water_vapor_removal(self, cell_side, side_state):
        """Maximum water vapor removal rate for *cell_side* (kmol/m²/s)."""
        return (
            (GasModel.saturation_concentration(side_state.cl) - GasModel.vapor_concentration(side_state.ch))
            / self.gas_transport_resistance(cell_side, side_state, 'h2o')
        )
