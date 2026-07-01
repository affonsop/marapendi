from dataclasses import dataclass


@dataclass
class Parameter:
    """A fixed model parameter with display metadata."""
    value: float
    key: str = None      # dict key used in the params mapping passed to cell_creator
    symbol: str = None   # LaTeX symbol for plots
    units: str = 'n.d.'
    factor: float = 1    # display divisor (value / factor gives the human-readable number)


@dataclass
class UnknownParameter(Parameter):
    """A parameter to be estimated, with bounds and normalisation type."""
    initial_guess: float = None
    lower_bound: float = None
    upper_bound: float = None
    is_linear: bool = True   # False → log-scale normalisation via p_to_theta / theta_to_p

    def __post_init__(self):
        self.value = self.initial_guess
