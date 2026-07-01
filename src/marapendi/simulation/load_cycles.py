"""
Time-varying load cycle definition for transient PEMFC simulations.
"""
import numpy as np


class LoadCycle:
    """
    Class representing a load cycle defined over a given duration
    and time discretization.

    A LoadCycle stores time-dependent input functions and allows
    generation of input vectors over one or multiple repeated cycles.

    Notes
    -----
    Each input must be provided as a callable function of time.
    Returning ``None`` disables the corresponding input.

    Examples
    --------
    >>> cycle = LoadCycle(duration=30., time_step=1.)

    >>> current_density = lambda t: np.where(t > 10., 1e4, 0.5e4)
    >>> cell_temperature = 353.15
    >>> cell_pressure = 1.5e5

    >>> cycle.set_input_dict({
    ...     'cell-temperature': lambda t: cell_temperature,
    ...     'current-density': current_density,
    ...     'ca-inlet-temperature': lambda t: cell_temperature,
    ...     'ca-inlet-rh': lambda t: 0.8,
    ...     'ca-inlet-pressure': lambda t: cell_pressure,
    ...     'ca-outlet-pressure': lambda t: None,
    ... })

    >>> inputs = cycle.get_input_vectors(cycle.cycle_time)
    >>> multi_cycle = cycle.get_n_cycles(n_cycles=5)
    """

    def __init__(self, duration, time_step, u=None):
        """
        Initialize the load cycle.

        Parameters
        ----------
        duration : float
            Total duration of one cycle.
        time_step : float
            Time discretization step.
        u : dict, optional
            Dictionary of input functions. Each key corresponds
            to an input name and each value must be a callable
            function of time.
        """
        self.duration = duration
        self.time_step = time_step
        self.u = u

        # Time vector for one cycle
        self.cycle_time = np.linspace(
            0,
            self.duration,
            int(self.duration // self.time_step + 1)
        )

    def set_input_dict(self, u):
        """
        Set or update the input function dictionary.

        Parameters
        ----------
        u : dict
            Dictionary where:
                - keys are input names
                - values are callable functions of time
        """
        self.u = u

    def get_input_vectors(self, t):
        """
        Evaluate input functions at given time values.

        Parameters
        ----------
        t : array-like or float
            Time instant(s) at which inputs are evaluated.

        Returns
        -------
        dict
            Dictionary containing evaluated input vectors.
            If an input function returns None, the value
            is set to None.
        """
        u_vectors = {}

        for u_key, u_value in self.u.items():

            # Check if input is inactive (returns None)
            if np.any(u_value(0) == None):
                u_vectors[u_key] = None
            else:
                # Ensure output is at least 1D
                u_vectors[u_key] = np.array(
                    u_value(t),
                    ndmin=1
                )

        return u_vectors

    def repeat_cycles(self, n_cycles):
        """
        Generate inputs over multiple repeated load cycles.

        Parameters
        ----------
        n_cycles : int
            Number of cycles to concatenate.

        Returns
        -------
        dict
            Dictionary containing:
                - concatenated input vectors
                - total_time : cumulative simulation time
        """

        # Concatenate cycle time repeatedly and compute cumulative time

        time_blocks = [
            self.cycle_time + i * self.duration
            for i in range(n_cycles)
        ]

        total_time = np.concatenate(time_blocks)

        u_full = {}

        # Get inputs for one cycle
        for u_key, u_value in self.get_input_vectors(
            self.cycle_time
        ).items():

            if np.any(u_value == None):
                u_full[u_key] = None
            else:
                # Repeat input profile over n_cycles
                u_full[u_key] = np.concatenate(
                    [u_value * np.ones_like(self.cycle_time)] * n_cycles
                )

        # Store total simulation time
        u_full['total_time'] = total_time

        return u_full