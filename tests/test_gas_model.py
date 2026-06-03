"""Tests for marapendi.models.gas_model — GasMixtureModel."""
import numpy as np
import pytest
import cantera as ct
from marapendi.models.gas import GasMixtureModel

# Species used throughout: cathode-side gas (O2, N2, H2O)
SPECIES = ['o2', 'n2', 'h2', 'h2o']
M_K = np.array([32., 28., 2., 18.])   # g/mol — consistent relative units


@pytest.fixture(scope="module")
def model() -> GasMixtureModel:
    return GasMixtureModel(species=SPECIES)


# ─── reference values via Cantera ─────────────────────────────────────────────

def _cantera_pure_viscosity(species: str, T: float) -> float:
    """Dynamic viscosity of a pure species from Cantera [Pa·s]."""
    gas = ct.Solution("gri30.yaml")
    gas.TPX = T, ct.one_atm, {species: 1.0}
    return gas.viscosity


def _cantera_mixture_viscosity(mole_fractions: dict, T: float) -> float:
    """Dynamic viscosity of a gas mixture from Cantera (Wilke rule) [Pa·s]."""
    gas = ct.Solution("gri30.yaml")
    gas.TPX = T, ct.one_atm, mole_fractions
    return gas.viscosity


# ─── output shape ─────────────────────────────────────────────────────────────

class TestOutputShape:
    @pytest.mark.parametrize("n_layers,n_meas", [(1, 1), (3, 5), (7, 10)])
    def test_output_shape(self, model, n_layers, n_meas):
        T = np.full((n_layers, n_meas), 500.)
        x_k = np.ones((n_layers, len(SPECIES), n_meas)) / len(SPECIES)
        result = model.mixture_dynamic_viscosity(T, x_k, M_K)
        assert result.shape == (n_layers, n_meas)

    def test_output_positive(self, model):
        T = np.linspace(300, 1200, 6).reshape(2, 3)
        x_k = np.ones((2, len(SPECIES), 3)) / len(SPECIES)
        result = model.mixture_dynamic_viscosity(T, x_k, M_K)
        assert np.all(result > 0)


# ─── pure-species accuracy vs Cantera ─────────────────────────────────────────
#
# For a pure species (x_k = 1, all others = 0) the weighting rule reduces to
#   μ_mix = x_k √M_k μ_k / (x_k √M_k) = μ_k
# so the result must match Cantera's pure-species viscosity exactly (within
# floating-point tolerance from the same polynomial coefficients).

class TestPureSpeciesVsCantera:
    @pytest.mark.parametrize("sp,sp_idx", [
        ("O2",  0),
        ("N2",  1),
        ("H2",  2),
        ("H2O", 3),
    ])
    @pytest.mark.parametrize("T_val", [400., 600., 900., 1200.])
    def test_pure_species_matches_cantera(self, model, sp, sp_idx, T_val):
        # Build x_k with x[sp_idx] = 1, rest = 0
        x_k = np.zeros((1, len(SPECIES), 1))
        x_k[0, sp_idx, 0] = 1.0

        T = np.array([[T_val]])
        result = model.mixture_dynamic_viscosity(T, x_k, M_K)
        expected = _cantera_pure_viscosity(sp, T_val)

        assert result.shape == (1, 1)
        assert result[0, 0] == pytest.approx(expected, rel=1e-6), (
            f"Pure {sp} at {T_val} K: got {result[0,0]:.4e}, "
            f"Cantera {expected:.4e}"
        )


# ─── mixture accuracy vs Cantera (Wilke rule) ─────────────────────────────────
#
# The √M weighting is an approximation to Wilke's rule.  For non-polar
# species at moderate pressures the error is typically < 5 %.

class TestMixtureVsCantera:
    @pytest.mark.parametrize("T_val,xO2,xN2", [
        (500.,  0.21, 0.79),
        (800.,  0.10, 0.90),
        (1000., 0.50, 0.50),
    ])
    def test_O2_N2_mixture_within_5pct_of_cantera(self, model, T_val, xO2, xN2):
        x_k = np.zeros((1, len(SPECIES), 1))
        x_k[0, 0, 0] = xO2  # O2
        x_k[0, 1, 0] = xN2  # N2

        T = np.array([[T_val]])
        result = model.mixture_dynamic_viscosity(T, x_k, M_K)
        expected = _cantera_mixture_viscosity({"O2": xO2, "N2": xN2}, T_val)

        assert result[0, 0] == pytest.approx(expected, rel=0.05), (
            f"O2/N2 mixture at {T_val} K (xO2={xO2}): "
            f"got {result[0,0]:.4e}, Cantera {expected:.4e}"
        )

    def test_mixture_bounded_by_pure_components(self, model):
        """Mixture viscosity should lie between the extremes of its components."""
        T_val = 700.
        x_k = np.zeros((1, len(SPECIES), 1))
        x_k[0, 0, 0] = 0.3   # O2
        x_k[0, 3, 0] = 0.7   # H2O

        T = np.array([[T_val]])
        result = model.mixture_dynamic_viscosity(T, x_k, M_K)

        mu_O2  = _cantera_pure_viscosity("O2",  T_val)
        mu_H2O = _cantera_pure_viscosity("H2O", T_val)

        assert min(mu_O2, mu_H2O) <= result[0, 0] <= max(mu_O2, mu_H2O)


# ─── vectorised consistency ────────────────────────────────────────────────────

class TestVectorisedConsistency:
    def test_stacked_layers_equal_independent_calls(self, model):
        """Processing multiple layers at once must give the same result as
        calling the model for each layer individually."""
        T_vals = [400., 700., 1000.]
        n_layers = len(T_vals)
        x_k_batch = np.zeros((n_layers, len(SPECIES), 1))
        x_k_batch[:, 0, 0] = [0.21, 0.10, 0.50]   # O2
        x_k_batch[:, 1, 0] = [0.79, 0.90, 0.50]   # N2

        T_batch = np.array(T_vals)[:, np.newaxis]   # (3, 1)
        result_batch = model.mixture_dynamic_viscosity(
            T_batch, x_k_batch, M_K)

        for i, T_i in enumerate(T_vals):
            x_k_i = x_k_batch[i:i+1]
            T_i_arr = np.array([[T_i]])
            result_i = model.mixture_dynamic_viscosity(
                T_i_arr, x_k_i, M_K)
            assert result_batch[i, 0] == pytest.approx(result_i[0, 0], rel=1e-12)

    def test_temperature_dependence_monotone(self, model):
        """Viscosity of a gas increases with temperature (kinetic theory)."""
        T = np.array([[300., 500., 700., 1000.]])     # (1, 4)
        x_k = np.ones((1, len(SPECIES), 4)) / len(SPECIES)
        result = model.mixture_dynamic_viscosity(T, x_k, M_K)
        assert np.all(np.diff(result[0]) > 0), "Viscosity should increase with T"
