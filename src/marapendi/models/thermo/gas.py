"""
Gas mixture composition and correlations.

The gas mixture is composed of O2, N2, H2 and H2O. :class:`GasState` holds
only the mixture's mole fractions (``X``) -- the variable that actually
needs to be tracked -- while temperature and pressure live on the
surrounding :class:`~marapendi.state.LayerState` /
:class:`~marapendi.state.FlowChannelState`.

:class:`GasModel` provides stateless correlations (relative humidity, vapor
pressure, diffusion coefficients, kinematic viscosity, ...) as pure
functions of a ``state`` object exposing ``gas: GasState``, ``temperature``
and ``pressure``. This avoids duplicating gas-composition bookkeeping
across layer getters, and ensures the saturation pressure is computed from
a single place.
"""
from __future__ import annotations

import numpy as np

from .water import water_saturation_pressure

species_list = ('o2', 'n2', 'h2', 'h2o')
species_indexes = dict(zip(species_list, range(4)))
index_o2, index_n2, index_h2, index_h2ov = (species_indexes[s] for s in species_list)

molecular_weights = np.array([32., 28., 2., 18.])
"""Molecular weights of (O2, N2, H2, H2O), in kg/kmol."""

_FICK_PT_REFERENCE_FACTOR = 1e5 / 353.15 ** 1.5
"""P_ref / T_ref**1.5 for the Fick's law T/P adjustment (T_ref=353.15 K, P_ref=1e5 Pa)."""

# Polynomial coefficients (highest degree first) for ``sqrt(kinematic_viscosity) = poly(log(T))``,
# fitted from the Cantera "gri30" transport data for O2, N2, H2 and H2O.
# Stored as plain Python tuples so Horner evaluation avoids numpy indexing overhead.
_viscosity_polynomials = {
    'o2':  (-1.951788060142541e-06,  6.0422679225074004e-05, -0.000698915553749822,
             0.003675525810527708,  -0.006280860305804308),
    'n2':  (-1.7418174930566134e-06, 5.344876287161394e-05,  -0.0006125055865647582,
             0.0031950685178810598, -0.005349912337402537),
    'h2':  (-3.323040125663745e-07,  9.673877158687006e-06,  -0.00010356810956997187,
             0.0005414323063318191, -0.00044135740261495426),
    'h2o': ( 4.616673944844547e-07, -3.274425664644034e-05,   0.0005317488930314697,
            -0.003007552407888745,   0.00621446598971834),
}


def _horner5(c, x):
    """Evaluate a degree-4 polynomial via Horner's method."""
    return ((((c[0] * x + c[1]) * x + c[2]) * x + c[3]) * x + c[4])


class GasModel:
    """Stateless correlations for a gas mixture's :class:`GasState`.

    All methods take a ``state`` object exposing ``gas: GasState``,
    ``temperature`` (K) and ``pressure`` (Pa) -- typically a
    :class:`~marapendi.state.LayerState` or
    :class:`~marapendi.state.FlowChannelState`.
    """

    @staticmethod
    def set_composition(state, dry_o2_mole_fraction: float, dry_h2_mole_fraction: float,
                         relative_humidity: float, inlet_pressure: float, inlet_temperature: float) -> None:
        """Set ``state.gas.X`` from a dry composition and relative humidity.

        The water vapor mole fraction is fixed by the relative humidity at
        the inlet conditions (``inlet_pressure``, ``inlet_temperature``),
        which may differ from ``state.pressure``/``state.temperature``
        (typically the average of the inlet and outlet conditions).
        """
        dry_mole_fractions = np.zeros_like(state.gas.X)
        dry_mole_fractions[..., index_o2] = dry_o2_mole_fraction
        dry_mole_fractions[..., index_h2] = dry_h2_mole_fraction
        dry_mole_fractions[..., index_n2] = 1 - dry_o2_mole_fraction - dry_h2_mole_fraction

        inlet_saturation_pressure = water_saturation_pressure(inlet_temperature)
        h2o_mole_fraction = relative_humidity * inlet_saturation_pressure / inlet_pressure
        vapor_mole_fractions = np.zeros_like(state.gas.X)
        vapor_mole_fractions[..., index_h2ov] = h2o_mole_fraction

        state.gas.X = (
            dry_mole_fractions * (1 - vapor_mole_fractions[..., index_h2ov, np.newaxis])
            + vapor_mole_fractions
        )

    @staticmethod
    def species_mole_fraction(state, species: str) -> float:
        """Mole fraction of ``species`` in the gas mixture."""
        return state.gas.X[..., species_indexes[species]]

    @staticmethod
    def species_partial_pressure(state, species: str) -> float:
        """Partial pressure of ``species`` (Pa)."""
        return GasModel.species_mole_fraction(state, species) * state.pressure

    @staticmethod
    def species_concentration(state, species: str) -> float:
        """Concentration of ``species`` (kmol/m^3)."""
        return GasModel.species_partial_pressure(state, species) / state.RT

    @staticmethod
    def vapor_pressure(state) -> float:
        """Partial pressure of water vapor (Pa)."""
        return GasModel.species_partial_pressure(state, 'h2o')

    @staticmethod
    def vapor_concentration(state) -> float:
        """Concentration of water vapor (kmol/m^3)."""
        return GasModel.species_concentration(state, 'h2o')

    @staticmethod
    def saturation_pressure(state) -> float:
        """Saturation pressure of water at ``state.temperature`` (Pa).

        Cached on ``state.saturation_pressure`` to avoid recomputing it.
        """
        if state.saturation_pressure is None:
            state.saturation_pressure = water_saturation_pressure(state.temperature)
        return state.saturation_pressure

    @staticmethod
    def saturation_concentration(state) -> float:
        """Saturation concentration of water vapor (kmol/m^3)."""
        return GasModel.saturation_pressure(state) / state.RT

    @staticmethod
    def relative_humidity(state) -> float:
        """Relative humidity (0 to 1)."""
        return GasModel.vapor_pressure(state) / GasModel.saturation_pressure(state)

    @staticmethod
    def mixture_molecular_weight(state) -> float:
        """Mean molecular weight of the gas mixture (kg/kmol)."""
        return np.sum(molecular_weights * state.gas.X, axis=-1)

    @staticmethod
    def concentration(state) -> float:
        """Total molar concentration of the gas mixture (kmol/m^3)."""
        return state.pressure / state.RT

    @staticmethod
    def density(state) -> float:
        """Mass density of the gas mixture (kg/m^3)."""
        return GasModel.concentration(state) * GasModel.mixture_molecular_weight(state)

    @staticmethod
    def species_kinematic_viscosity(state, species: str) -> float:
        """Kinematic viscosity of pure ``species`` at ``state.temperature`` (m^2/s)."""
        log_temperature = np.log(state.temperature)
        v = _horner5(_viscosity_polynomials[species], log_temperature)
        return v * v * np.sqrt(state.temperature)

    @staticmethod
    def mixture_kinematic_viscosity(state) -> float:
        """Kinematic viscosity of the gas mixture (m^2/s), as a mole-weighted average."""
        species_kinematic_viscosities = np.array(
            [GasModel.species_kinematic_viscosity(state, species) for species in species_list],
        ).transpose()
        return (
            np.sum(state.gas.X * species_kinematic_viscosities * molecular_weights, axis=-1)
            / GasModel.mixture_molecular_weight(state)
        )

    @staticmethod
    def diffusion_temp_and_pressure_correction(temperature, pressure) -> float:
        """Species-independent Fick's law adjustment ``T^1.5 / P`` for :meth:`species_diffusion_coefficient`.

        Cached on ``state.diffusion_temp_and_pressure_correction`` by
        :meth:`~marapendi.components.porous_layers.porous_layers.PorousLayer.update_state_at_temperature`,
        since up to 3 species share the same ``temperature``/``pressure`` per state.
        """
        # T**1.5 written as T * sqrt(T): np.sqrt dispatches faster than np.power
        # with a non-integer exponent, and this is on the transient ODE hot path.
        return _FICK_PT_REFERENCE_FACTOR * temperature * np.sqrt(temperature) / pressure

    @staticmethod
    def species_diffusion_coefficient(state, species: str) -> float:
        """Binary diffusion coefficient of ``species`` in the gas mixture (m^2/s).

        Uses empirical correlations based on reference values adjusted for
        temperature and pressure. Data from Vetter and Schumacher (2019).

        References
        ----------
        Vetter, R. & Schumacher, J. O. Comput. Phys. Commun. 234, 223-234 (2019).
        """
        if species == 'o2':
            reference_diffusion_coefficient = 0.28e-4
        elif species == 'h2':
            reference_diffusion_coefficient = 1.24e-4
        elif species == 'h2o':
            # If H2 is present, assume H2-H2O; else O2-H2O
            if np.max(GasModel.species_mole_fraction(state, 'h2')) > 0:
                reference_diffusion_coefficient = 1.24e-4
            else:
                reference_diffusion_coefficient = 0.36e-4

        return reference_diffusion_coefficient * state.diffusion_temp_and_pressure_correction
