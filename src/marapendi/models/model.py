"""
Base model for composing multiple submodels into a single ODE system.

Classes
-------
BaseModel
    Generic ODE compositor.  Composes any number of submodels—each
    exposing ``rates_of_change`` and ``n_states``—into one combined ODE.
    If a submodel defines a ``get_inputs(t)`` method, ``BaseModel``
    registers it automatically in ``input_fns`` during ``__post_init__``;
    no manual ``input_fns`` dict is needed for the common case.

Protocol
--------
A submodel is compatible with ``BaseModel`` when it implements:

* ``rates_of_change(x, **inputs) -> np.ndarray``
  where ``x`` has shape ``(n_states, m)`` and the return has the same shape.
* ``n_states: int``
  total length of the normalised flat state vector.

Optionally, a submodel may implement:

* ``get_inputs(t: float) -> dict``
  returns keyword arguments forwarded to ``rates_of_change`` at time *t*.
  When present, this is registered in ``input_fns`` automatically (explicit
  entries in ``input_fns`` take precedence).
* ``base_model``
  any attribute by this name will be overwritten with a reference to the
  owning ``BaseModel`` so submodels can resolve shared model objects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class BaseModel:
    """
    Generic compositor: combine any number of ODE submodels.

    Parameters
    ----------
    submodels : dict[str, object]
        Ordered mapping of ``name → submodel``.  Each submodel must expose
        ``rates_of_change(x, **inputs)`` and ``n_states``.
    input_fns : dict[str, callable], optional
        Explicit ``name → f(t)`` overrides.  Submodels not listed here
        fall back to their own ``get_inputs`` method, if any.

    Attributes
    ----------
    n_states : int
        Total length of the combined normalised state vector.
    """

    submodels: dict[str, Any] = field(default_factory=dict)
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

        # Auto-register get_inputs from submodels (explicit input_fns take priority)
        for name, model in self.submodels.items():
            if hasattr(model, 'get_inputs') and name not in self.input_fns:
                self.input_fns[name] = model.get_inputs

        # Inject self into any submodel that declares a base_model slot
        for model in self.submodels.values():
            if hasattr(model, 'base_model'):
                model.base_model = self

    # ------------------------------------------------------------------
    # ODE interface
    # ------------------------------------------------------------------

    def rates_of_change(self, t, x):
        """
        Assemble dxdt by dispatching each slice to its submodel.

        ``(t, x)`` matches the ``solve_ivp`` convention, so
        ``base.rates_of_change`` can be passed directly as ``fun``.

        Parameters
        ----------
        t : float
            Current time, passed to each ``input_fn``/``get_inputs``.
        x : np.ndarray, shape (n_states,) or (n_states, m)

        Returns
        -------
        np.ndarray
            Same shape as ``x``.
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
            submodel's ``initial_state`` method.

        Returns
        -------
        np.ndarray, shape (n_states,)
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
    # Post-processing
    # ------------------------------------------------------------------

    def split_state(self, x: np.ndarray) -> dict[str, np.ndarray]:
        """Split a combined state array into per-submodel slices."""
        return {name: x[sl] for name, sl in self._slices.items()}
