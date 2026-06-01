"""
Base model for composing multiple submodels into a single ODE system.

Classes
-------
BaseModel
    Composes any number of submodels—each exposing ``rates_of_change`` and
    ``n_states``—into one combined ODE whose state vector is the
    concatenation of every submodel's state vector.

Protocol
--------
A submodel is compatible with ``BaseModel`` when it implements:

* ``rates_of_change(x, **inputs) -> np.ndarray``
  where ``x`` has shape ``(n_states, m)`` and the return has the same shape.
* ``n_states: int``
  total length of the normalised flat state vector.

``TransientCellModel`` satisfies both requirements.  ``BaseModel`` itself
also satisfies them, so models can be nested arbitrarily.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class BaseModel:
    """
    Compose multiple submodels into a single ODE system.

    The combined state vector is the concatenation of each submodel's
    normalised flat state vector.  ``rates_of_change`` dispatches each
    submodel's slice to the corresponding ``rates_of_change`` method, using
    time-dependent input functions to supply keyword arguments.

    Typical usage with ``scipy.integrate.solve_ivp``::

        base = BaseModel(
            submodels={
                'cell_a': model_a,
                'cell_b': model_b,
            },
            input_fns={
                'cell_a': lambda t: {'i': i_a(t)},
                'cell_b': lambda t: {'i': i_b(t)},
            },
        )

        y0 = base.initial_state(
            cell_a={'T': 353.15, 'p': 1.5e5},
            cell_b={'T': 343.15, 'p': 1.0e5},
        )

        sol = solve_ivp(
            base.rates_of_change,   # signature: (t, x) — matches solve_ivp
            t_span=(0, t_end),
            y0=y0,
            method='BDF',
            vectorized=False,
        )

        states = base.split_state(sol.y)   # {'cell_a': ..., 'cell_b': ...}

    Parameters
    ----------
    submodels : dict[str, object]
        Ordered mapping of ``name → submodel``.  Each submodel must expose
        ``rates_of_change(x, **inputs)`` and an integer attribute
        ``n_states``.
    input_fns : dict[str, callable], optional
        Mapping of ``name → f(t)`` where ``f`` returns a ``dict`` of keyword
        arguments forwarded to that submodel's ``rates_of_change``.  Any
        submodel not listed receives ``{}`` (no extra inputs).

    Attributes
    ----------
    n_states : int
        Total length of the combined normalised state vector.
    """

    submodels: dict[str, Any]
    input_fns: dict[str, Callable] = field(default_factory=dict)

    def __post_init__(self):
        offset = 0
        self._slices: dict[str, slice] = {}
        for name, model in self.submodels.items():
            if not hasattr(model, 'n_states'):
                raise AttributeError(
                    f"Submodel '{name}' must expose an 'n_states' integer attribute."
                )
            size = model.n_states
            self._slices[name] = slice(offset, offset + size)
            offset += size
        self.n_states: int = offset

    # ------------------------------------------------------------------
    # ODE interface
    # ------------------------------------------------------------------

    def rates_of_change(self, t, x):
        """
        Assemble dxdt by dispatching each slice to its submodel.

        The argument order ``(t, x)`` matches the convention expected by
        ``scipy.integrate.solve_ivp``, so ``base.rates_of_change`` can be
        passed directly as the ``fun`` argument.

        For non-vectorised integration ``x`` has shape ``(n_states,)``;
        for vectorised integration it has shape ``(n_states, m)``.  Both
        forms are handled transparently.

        Parameters
        ----------
        t : float
            Current time, passed to each ``input_fn`` to evaluate inputs.
        x : np.ndarray, shape (n_states,) or (n_states, m)
            Combined normalised state vector.

        Returns
        -------
        np.ndarray
            Combined derivative array, same shape as ``x``.
        """
        scalar = x.ndim == 1
        if scalar:
            x = x[:, np.newaxis]

        dxdt = np.empty_like(x)
        for name, model in self.submodels.items():
            sl = self._slices[name]
            inputs = self.input_fns.get(name, lambda _t: {})
            dxdt[sl] = model.rates_of_change(x[sl], **inputs(t))

        return dxdt[:, 0] if scalar else dxdt

    # ------------------------------------------------------------------
    # Initial state
    # ------------------------------------------------------------------

    def initial_state(self, **per_model_kwargs) -> np.ndarray:
        """
        Concatenate initial states from all submodels.

        Parameters
        ----------
        **per_model_kwargs
            Keyword arguments keyed by submodel name, forwarded to each
            submodel's ``initial_state`` method::

                base.initial_state(
                    cell_a={'T': 353.15, 'p': 1.5e5, 'rh': 0.7},
                    cell_b={'T': 343.15, 'p': 1.0e5, 'rh': 0.6},
                )

        Returns
        -------
        np.ndarray, shape (n_states,)
            Flat normalised initial-state vector ready for ``solve_ivp``.
        """
        parts = []
        for name, model in self.submodels.items():
            if not hasattr(model, 'initial_state'):
                raise AttributeError(
                    f"Submodel '{name}' has no 'initial_state' method."
                )
            parts.append(model.initial_state(**per_model_kwargs.get(name, {})))
        return np.concatenate(parts)

    # ------------------------------------------------------------------
    # Post-processing helpers
    # ------------------------------------------------------------------

    def split_state(self, x: np.ndarray) -> dict[str, np.ndarray]:
        """
        Split a combined state array into per-submodel slices.

        Parameters
        ----------
        x : np.ndarray, shape (n_states,) or (n_states, m)
            Combined state (e.g. ``sol.y`` from ``solve_ivp``).

        Returns
        -------
        dict[str, np.ndarray]
            ``{name: x[slice]}`` for each submodel.
        """
        return {name: x[sl] for name, sl in self._slices.items()}
