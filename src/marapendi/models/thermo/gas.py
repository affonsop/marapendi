"""
Gas mixture composition constants and low-level correlations.

The gas mixture is composed of O2, N2, H2 and H2O. Species indices/weights
and the pure numeric correlations (viscosity polynomials, the Fick's-law
temperature/pressure factor) live here; the stateful, composition-aware
correlations (relative humidity, vapor pressure, diffusion coefficients,
kinematic viscosity, ...) are instance methods on
:class:`~marapendi.simulation.state.GasState` itself, which holds the
mixture's mole fractions (``X``) together with its own ``temperature`` and
``pressure``.
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

