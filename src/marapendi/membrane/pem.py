
"""PFSA ionomer (e.g. Nafion) material properties."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from ..tools import arrhenius_term
from ..thermo.constants import GAS_CONSTANT
from ..thermo.water import water_molar_volume

from .ionomer_base import Ionomer
from .membrane_base import Membrane

@dataclass
class PFSAIonomer(Ionomer):
    """PFSA ionomer (e.g. Nafion) with empirical fits for proton conductivity and O2 transport."""
    equivalent_weight: float = 952.
    dry_density: float = 2004.
    vapor_equilibrium_polynomial: list = field(default_factory=lambda: [36, -39.85, 17.18, 0.043])
    reference_conductivity: float = 50
    conductivity_correction: float = 1.0
    reference_water_diffusivity: float = 4.3e-10
    reference_water_absorption_coefficient: float = 1e-5   
    conductivity_exp: float = 1.5
    conductivity_fv_threshold: float = 0.11
    hydrated_proton_conductivity: float = 11
    conductivity_activation_energy: float = 11e6
    water_diffusivity_activation_energy: float = 20e6
    water_absorption_activation_energy: float = 20e6
    reference_conductivity_temperature: float = 353.15
    reference_water_absorption_temperature: float = 353.15
    reference_water_diffusivity_temperature: float = 353.15
    hydrated_o2_diffusion: float = 1.14698e-10 * 14 ** 0.708
    o2_diffusion_exponent: float = 0.708
    o2_diffusion_activation_energy: float = 24e6

    def __post_init__(self):
        super().__post_init__()
        self.fit_rh_piecewise_linear()

    # ------------------------------------------------------------------
    # Piecewise linear RH(λ) approximation
    # ------------------------------------------------------------------

    def fit_rh_piecewise_linear(
        self,
        n_segments: int = 3,
        temperature: float = 353.15,
        rh_breaks=None,
    ) -> None:
        """Fit and cache a continuous piecewise linear approximation of RH(λ_eq).

        The procedure has two distinct steps:

        1. **Fitting intervals** (in RH space, given by *rh_breaks*): each
           segment gets a least-squares line ``RH = a_k · λ + b_k`` fitted
           on the data in its interval.
        2. **Validity intervals** (in λ space): adjacent lines are extended
           until they intersect; each line is valid between the two nearest
           intersections.  Continuity is guaranteed at every transition.

        When *rh_breaks* is ``None``, interior fitting-interval boundaries
        are optimised to minimise the RMS approximation error under the
        intersection-based validity scheme.

        Results stored on self
        ----------------------
        fit_rh_breaks : ndarray, shape (n_segments + 1,)
            Fitting-interval boundaries in RH space (0 … 1).
        lmbd_pwl_breaks : ndarray, shape (n_segments + 1,)
            Validity-interval boundaries in λ space (domain endpoints +
            pairwise line intersections).
        rh_pwl_breaks : ndarray, shape (n_segments + 1,)
            RH values at the validity boundaries (equal from both sides by
            construction — continuity check).
        pwl_slopes : ndarray, shape (n_segments,)
            Regression slope *a_k* for each segment.
        pwl_intercepts : ndarray, shape (n_segments,)
            Regression intercept *b_k* for each segment.
        pwl_temperature : float
            Temperature (K) used for fitting.
        """
        rh_ref   = np.linspace(0.0, 1.0, 500)
        lmbd_ref = self.vapor_equilibrium_water_content(rh_ref, temperature)

        def _fit(rh_b):
            """Regression + intersection-based validity. Returns (a, b, lmbd_valid, rh_approx)."""
            lmbd_b = self.vapor_equilibrium_water_content(np.asarray(rh_b), temperature)
            n      = len(rh_b) - 1

            # Step 1: fit one regression line per fitting interval
            a = np.empty(n)
            b = np.empty(n)
            for k in range(n):
                mask = (lmbd_ref >= lmbd_b[k]) & (lmbd_ref <= lmbd_b[k + 1])
                if mask.sum() >= 2:
                    a[k], b[k] = np.polyfit(lmbd_ref[mask], rh_ref[mask], 1)
                else:
                    dl   = lmbd_b[k + 1] - lmbd_b[k]
                    a[k] = (rh_b[k + 1] - rh_b[k]) / dl if dl else 0.0
                    b[k] = rh_b[k] - a[k] * lmbd_b[k]

            # Step 2: compute intersections of adjacent lines
            # a[k]*λ + b[k] = a[k+1]*λ + b[k+1]  →  λ = (b[k+1]-b[k]) / (a[k]-a[k+1])
            da = a[:-1] - a[1:]
            crossings = np.where(
                np.abs(da) > 1e-12,
                (b[1:] - b[:-1]) / da,
                0.5 * (lmbd_b[1:-1] + lmbd_b[2:]),   # parallel lines: midpoint fallback
            )

            # Step 3: validity breakpoints = domain endpoints + intersections
            lmbd_valid = np.concatenate([[lmbd_ref[0]], crossings, [lmbd_ref[-1]]])

            # Evaluate on reference grid using validity intervals
            rh_approx = np.empty_like(lmbd_ref)
            for k in range(n):
                mask = (lmbd_ref >= lmbd_valid[k]) & (lmbd_ref <= lmbd_valid[k + 1])
                rh_approx[mask] = a[k] * lmbd_ref[mask] + b[k]

            return a, b, lmbd_valid, rh_approx

        def _rms(rh_b):
            _, _, _, rh_approx = _fit(rh_b)
            return float(np.sqrt(np.mean((rh_approx - rh_ref) ** 2)))

        if rh_breaks is not None:
            fit_breaks = np.asarray(rh_breaks, dtype=float)
        else:
            from scipy.optimize import minimize

            def _objective(x):
                interior = np.sort(np.clip(x, 1e-4, 1.0 - 1e-4))
                return _rms(np.concatenate([[0.0], interior, [1.0]]))

            x0  = np.linspace(0.0, 1.0, n_segments + 1)[1:-1]
            res = minimize(_objective, x0, method='Nelder-Mead',
                           options={'xatol': 1e-7, 'fatol': 1e-9, 'maxiter': 2000})
            interior   = np.sort(np.clip(res.x, 1e-4, 1.0 - 1e-4))
            fit_breaks = np.concatenate([[0.0], interior, [1.0]])

        slopes, intercepts, lmbd_valid, _ = _fit(fit_breaks)

        # RH at validity breakpoints (continuous by construction at intersections)
        rh_at_valid        = np.empty_like(lmbd_valid)
        rh_at_valid[0]     = slopes[0]  * lmbd_valid[0]  + intercepts[0]
        rh_at_valid[-1]    = slopes[-1] * lmbd_valid[-1] + intercepts[-1]
        for k in range(1, len(lmbd_valid) - 1):
            rh_at_valid[k] = slopes[k - 1] * lmbd_valid[k] + intercepts[k - 1]

        self.fit_rh_breaks   = fit_breaks
        self.lmbd_pwl_breaks = lmbd_valid
        self.rh_pwl_breaks   = rh_at_valid
        self.pwl_slopes      = slopes
        self.pwl_intercepts  = intercepts
        self.pwl_temperature = temperature

    def linear_rh_from_water_content(self, lmbd, interval=None) -> np.ndarray:
        """Evaluate the piecewise linear approximation RH(λ_eq).

        Parameters
        ----------
        lmbd : array_like
            Water content λ (mol H₂O / mol SO₃⁻).
        interval : int or None
            * ``None`` (default): select the segment automatically from the
              validity intervals (line intersections).  Result is clipped to
              [0, 1].
            * ``k`` (integer, supports negative indexing): always use segment
              *k*'s regression line regardless of the value of *lmbd*.
              No clipping applied — useful for deliberate extrapolation.
        """
        lmbd = np.asarray(lmbd, dtype=float)
        n    = len(self.pwl_slopes)
        if interval is None:
            idx = np.searchsorted(self.lmbd_pwl_breaks[1:-1], lmbd)
            return np.clip(self.pwl_slopes[idx] * lmbd + self.pwl_intercepts[idx], 0.0, 1.0)
        k = interval % n
        return self.pwl_slopes[k] * lmbd + self.pwl_intercepts[k]

    def linear_water_content_from_rh(self, rh, interval=None) -> np.ndarray:
        """Evaluate the inverse piecewise linear approximation λ_eq(RH).

        Inverts ``RH = a_k · λ + b_k``  →  ``λ = (RH − b_k) / a_k``.

        Parameters
        ----------
        rh : array_like
            Relative humidity (0–1).
        interval : int or None
            * ``None`` (default): select the segment automatically from the
              validity intervals in RH space.  No clipping applied.
            * ``k`` (integer, supports negative indexing): always use segment
              *k*'s regression line regardless of the value of *rh*.
        """
        rh = np.asarray(rh, dtype=float)
        n  = len(self.pwl_slopes)
        if interval is None:
            idx = np.searchsorted(self.rh_pwl_breaks[1:-1], rh)
        else:
            idx = interval % n
        return (rh - self.pwl_intercepts[idx]) / self.pwl_slopes[idx]

    def o2_film_diffusion_coefficient(self, water_content: float, temperature: float = 353.15) -> float:
        """Effective O2 diffusion coefficient in the hydrated ionomer film (m^2/s)."""
        return (
            self.hydrated_o2_diffusion * (water_content / 14) ** self.o2_diffusion_exponent
            * arrhenius_term(self.o2_diffusion_activation_energy, temperature, 353.15)
        )

    def h2_permeability(self, water_content: float, temperature: float = 353.15) -> float:
        """H2 permeability (kmol/m/s/Pa) from a volume-fraction approach.

        References
        ----------
        Goshtasbi, A. et al. J. Electrochem. Soc. 167, 024518 (2020).
        """
        fv = self.water_vol_fraction(water_content, water_molar_volume(temperature))
        RT = GAS_CONSTANT * temperature
        return (15.7e-15 * np.exp(-20280e3 / RT) + fv * 45e-15 * np.exp(-18930e3 / RT))
    
    def o2_permeability(self, water_content: float, temperature: float = 353.15) -> float:
        """O2 permeability (kmol/m/s/Pa) from a volume-fraction approach.

        References
        ----------
        Goshtasbi, A. et al. J. Electrochem. Soc. 167, 024518 (2020).
        """
        fv = self.water_vol_fraction(water_content, water_molar_volume(temperature))
        RT = GAS_CONSTANT * temperature
        return (6.74e-15 * np.exp(-21280e3 / RT) + fv * 50.5e-15 * np.exp(-20470e3 / RT))

    def calculate_electroosmotic_drag_coefficient(self, temperature: float, water_content: float) -> float:
        """Electroosmotic drag coefficient (n.d.) for a given ``water_content``."""
        return (0.02 * temperature - 3.86) / 22.5 * water_content

    def proton_conductivity(self, water_content: float, temperature: float) -> float:
        """Proton conductivity from empirical fits (S/m)."""
        fv = self.water_vol_fraction(water_content, water_molar_volume(temperature))
        return (
            self.conductivity_correction * 50 * (np.maximum(fv, 0.11) - 0.1) ** self.conductivity_exp
            * arrhenius_term(self.conductivity_activation_energy, temperature, 298.15)
        )

    def hydroxide_conductivity(self, water_content: float, temperature: float) -> float:
        """Hydroxide conductivity (S/m). PFSA ionomers do not conduct hydroxide."""
        return 1e-6


    def liquid_equilibrium_water_content(self, temperature):
        """Equilibrium water content in contact with liquid water.

        References
        ----------
        Goshtasbi, A. et al. J. Electrochem. Soc. 167, 024518 (2020).
        """
        return 9.22 + 0.181 * (temperature - 273.15)

    def vapor_equilibrium_water_content(self, rh: float, temperature) -> float:
        """Equilibrium water content as a function of relative humidity.

        References
        ----------
        
        """
        a = self.vapor_equilibrium_polynomial
        return  ((a[0] * rh + a[1]) * rh + a[2]) * rh + a[3]
    
    def vapor_equilibrium_water_content_derivative(self, rh, temperature):
        a = self.vapor_equilibrium_polynomial
        return (3 * a[0] * rh + 2 * a[1]) * rh + a[2]



@dataclass
class PFSA(Membrane):
    """Perfluorosulfonic-acid (PFSA, e.g. Nafion) membrane.

    Attributes
    ----------
    conductivity_correction : float
        Correction factor for the proton conductivity correlation
        (Vetter and Schumacher, 2020).
    conductivity_exp : float
        Exponent of the proton conductivity correlation.
    conductivity_activation_energy : float
        Activation energy for proton conductivity (J/kmol).
    phi : float
        Contribution of relaxation phenomena to the ionomer water uptake,
        according to Goshtasbi et al. (2019).

    References
    ----------
    Springer, T. E. et al. J. Electrochem. Soc. 138, 2334 (1991).
    Goshtasbi, A. et al. J. Electrochem. Soc. 167, 024518 (2020).
    """
    ionomer: PFSAIonomer = field(default_factory=PFSAIonomer)
    relaxation_time_constant: float = 0.067
    relaxation_time_activation_energy: float = 28e6
    uptake_relaxed_fraction_constant: float = 0.014
    phi: float = 0.15

    def equilibrium_water_content(self, rh, temperature, s_relax=None):
        """Equilibrium water content from the Springer et al. (1991) polynomial isotherm.

        References
        ----------
        Springer, T. E. et al. J. Electrochem. Soc. 138, 2334 (1991).
        Goshtasbi et al. J. Electrochem. Soc. 2019, 166 (7), F3154.
        """
        rh = np.clip(rh, 0, 1)
        lmbd_eq_relaxed = self.ionomer.vapor_equilibrium_water_content(rh, temperature)
        return ((1 - self.phi) * lmbd_eq_relaxed + s_relax) if s_relax is not None else lmbd_eq_relaxed

    def equilibrium_water_content_derivative(self, rh, temperature, s_relax=None):
        rh = np.clip(rh, 0, 1)
        d_lmbd_eq_relaxed = self.ionomer.vapor_equilibrium_water_content_derivative(rh, temperature)
        return ((1 - self.phi) * d_lmbd_eq_relaxed + s_relax) if s_relax is not None else d_lmbd_eq_relaxed


    def liquid_equilibrium_water_content(self, temperature):
        """Equilibrium water content in contact with liquid water — delegates to ionomer."""
        return self.ionomer.liquid_equilibrium_water_content(temperature)

    def proton_conductivity(self, water_content_profile, temperature):
        return 1 / np.mean(
            1 / (
                self.ionomer.charge_conductivity(water_content_profile, 
                                                 temperature, 'proton')
            ),
            axis=0,
        )

    def proton_resistance(self, state, water_saturation=0):
        """Through-plane proton resistance (Ω·m²).

        Weights liquid- and vapor-equilibrated conductivities by water saturation.
        """
        vapor_conductivity = self.proton_conductivity(state.water_content_profile, state.temperature)
        state.liquid_equilibrium_water_content = self.liquid_equilibrium_water_content(state.temperature)
        liquid_conductivity = self.proton_conductivity(state.liquid_equilibrium_water_content * np.ones_like(state.water_content_profile), state.temperature)
        average_conductivity = water_saturation * liquid_conductivity + (1-water_saturation) * vapor_conductivity 
        return self.dry_thickness / average_conductivity

NafionD2020 = PFSAIonomer(
    dry_density=2004., 
    equivalent_weight=952., 
    vapor_equilibrium_polynomial=[21.669, -27.692, 17.624, 0.688] # Jinnouchi, R. et al. Nat. Commun. 12, 4956 (2021).
)
