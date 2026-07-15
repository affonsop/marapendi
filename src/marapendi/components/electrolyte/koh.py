"""
Aqueous KOH electrolyte solution with empirical property correlations.
"""
from dataclasses import dataclass, field
import numpy as np
from .electrolyte import ElectrolyteSolution


@dataclass
class KOH_solution(ElectrolyteSolution):
    """
    Aqueous KOH solution with empirical correlations for density,
    ionic conductivity, vapour pressure lowering, and surface tension.

    References
    ----------
    Hodges, A. et al. J. Chem. Eng. Data 68, 1485–1506 (2023).
    Balej, J. Int. J. Hydrogen Energy 10, 233–243 (1985).
    """

    def __post_init__(self):
        super().__post_init__()
        self.electrolyte_molecular_weight = 56.105

    def calculate_density(self, temperature=300):
        """
        Solution density using the correlation from Hodges et al. (2023).

        Returns
        -------
        float
            Density (kg/m³).

        References
        ----------
        Hodges, A. et al. J. Chem. Eng. Data 68, 1485–1506 (2023).
        """
        T = temperature - 273.15
        return (5.1998e-6 * T ** 3 - 39.771334e-4 * T ** 2 -
                848.089182e-4 * T + 1001.5409980109) * np.exp(0.0086 * self.weight_percent)

    def calculate_ionic_conductivity(self, temperature=300):
        """
        Ionic conductivity using the correlations from Hodges et al. (2023).

        Two separate polynomial fits are used below and above 353.15 K.

        Returns
        -------
        float
            Ionic conductivity (S/m).

        References
        ----------
        Hodges, A. et al. J. Chem. Eng. Data 68, 1485–1506 (2023).
        """
        low_temp_cond = 100 * self.molarity * (-2.041 - 0.0028 * self.molarity +
            0.005332 * temperature + 207.2 / temperature +
            0.001043 * self.molarity ** 2 - 3e-7 * self.molarity * temperature ** 2)

        high_temp_cond = 100 * self.weight_percent * (2.2204e-3 - 1.3077e-3 * self.weight_percent +
            3.3647e-4 * temperature - 10.7021 / temperature +
            7.0101e-6 * self.weight_percent ** 2 - 3.2033e-9 * self.weight_percent * temperature ** 2)

        return np.where(temperature <= 353.15, low_temp_cond, high_temp_cond)

    def calculate_saturation_pressure(self):
        """
        Vapour pressure lowering relative to pure water, from Balej (1985).

        Used to compute the water activity of the electrolyte solution.

        Returns
        -------
        float
            Solution saturation pressure (Pa).

        References
        ----------
        Balej, J. Int. J. Hydrogen Energy 10, 233–243 (1985).
        """
        m_mol_per_kg = self.molality * 1000.
        log_p_sat = np.log10(self.water_sat_pressure / 1e5)
        # Balej (1985) polynomial, valid for 0–18 mol/kg; clip exponent so the
        # result stays within [0, water_sat_pressure] outside the validity range.
        exponent = 5 + log_p_sat - m_mol_per_kg * (
            (0.01508 + 0.0012062 * log_p_sat) +
            (0.0016788 - 5.6024e-4 * log_p_sat) * m_mol_per_kg -
            (2.25887e-5 - 7.8228e-6 * log_p_sat) * m_mol_per_kg ** 2)
        return float(np.power(10.0, np.clip(exponent, -300, 5 + log_p_sat)))

    def calculate_surface_tension(self) -> float:
        """
        Surface tension of aqueous KOH using equation 10 in Hodges et al. (2023).

        Valid between 30 wt% and 50 wt%.

        Returns
        -------
        float
            Surface tension (N/m).

        References
        ----------
        Hodges, A. et al. J. Chem. Eng. Data 68, 1485–1506 (2023).
        """
        a = np.array([
            [75.4787, -0.138489, -3.38e-04, 4.75e-7, -2.64e-10],
            [-32.889, 1.34382, -9.10e-03, 3.96e-05, -5.74e-08],
            [614.527, -12.8736, 0.104855, -4.49e-04, 6.51e-07],
            [-1455.06, 39.8511, -0.344234, 1.44e-03, -2.08e-06],
            [1333.62, -38.3316, 0.335129, -1.37e-03, 1.95e-06]
        ])
        T = self.temperature - 273.15
        xi = self.weight_percent / 100.
        sigma = 0
        for i in range(5):
            for j in range(5):
                sigma += a[i, j] * (T ** j) * (xi ** i)
        return sigma * 1e-3


# Predefined solutions
KOH_1M = KOH_solution(temperature=298.15, weight_percent=5.3732)
KOH_2M = KOH_solution(temperature=298.15, weight_percent=10.3)
KOH_5M = KOH_solution(temperature=298.15, weight_percent=23.072)
KOH_20_wt_percent = KOH_solution(temperature=298.15, weight_percent=20.)
KOH_45_wt_percent = KOH_solution(temperature=298.15, weight_percent=45.)
