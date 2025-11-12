from dataclasses import dataclass, field
import numpy as np
from .water import water_saturation_pressure, water_surface_tension

@dataclass
class ElectrolyteSolution():
    """
    Class to represent an electrolyte solution, calculating various thermophysical
    and transport properties based on temperature and concentration.

    Attributes
    ----------
    temperature : float
        Temperature of the solution [K].
    molality : float
        Molality of the solution [kmol/kg].
    weight_percent : float
        Weight percent concentration [0-1].
    electrolyte_molecular_weight : float
        Molecular weight of the electrolyte [kg/kmol].
    """

    temperature: float = 298.15
    molality: float = 0
    weight_percent: float = 0
    electrolyte_molecular_weight: float = 56.105

    def __post_init__(self):
        """Initialize and compute dependent properties."""
        if self.molality == 0:
            self.molality = self.calculate_molality()
        elif self.weight_percent == 0:
            self.weight_percent = self.calculate_weight_percent()
        self.set_temperature(self.temperature)

    def set_temperature(self, temperature):
        """
        Set temperature and update dependent properties.

        Parameters
        ----------
        temperature : float
            New temperature [K].
        """
        self.temperature = temperature
        self.density = self.calculate_density()
        self.molarity = self.calculate_molarity()
        self.ionic_conductivity = self.calculate_ionic_conductivity()
        self.water_sat_pressure = water_saturation_pressure(self.temperature)
        self.solution_sat_pressure = self.calculate_saturation_pressure()
        self.surface_tension = self.calculate_surface_tension() 

    def calculate_ionic_conductivity(self, temperature=None):
        """Return a default very low ionic conductivity, override in subclasses."""
        return 1e-12

    def calculate_density(self):
        """Return a default water-like density, override in subclasses."""
        return 1000.

    def calculate_saturation_pressure(self):
        """Return water saturation pressure by default, override in subclasses."""
        return self.water_sat_pressure

    def calculate_surface_tension(self):
        """Return water surface tension by default, override in subclasses."""
        return water_surface_tension(self.temperature)

    def calculate_weight_percent(self):
        """
        Calculate weight percent from molality [kmol/kg].

        Returns
        -------
        float
            Weight percent [0-1].
        """
        return 100. * self.electrolyte_molecular_weight * self.molality / (1 + self.electrolyte_molecular_weight * self.molality)

    def calculate_molality(self):
        """
        Calculate molality from weight percent.

        Returns
        -------
        float
            Molality [kmol/kg].
        """
        return self.weight_percent / self.electrolyte_molecular_weight / (100 - self.weight_percent)

    def calculate_molarity(self):
        """
        Calculate molarity of the solution.

        Returns
        -------
        float
            Molarity [kmol/m³].
        """
        return self.weight_percent / 100. * self.calculate_density() / self.electrolyte_molecular_weight

    def calculate_weight_percent(self):
        """
        Calculate weight percent from molality [kmol/kg].

        Returns
        -------
        float
            Weight percent [0-1].
        """

@dataclass
class KOH_solution(ElectrolyteSolution):
    """
    Specialized class for aqueous KOH solutions, with empirical correlations
    for density, ionic conductivity and vapor pressure lowering.
    """
    
    def __post_init__(self):
        super().__post_init__()
        self.electrolyte_molecular_weight = 56.105

    def calculate_density(self):
        """
        Calculate solution density using correlation from Hodges et al. (2023).

        Parameters
        ----------
        weight_percent : float
            Weight percent [0-1].
        temperature : float
            Temperature [K].

        Returns
        -------
        float
            Density [kg/m³].
        
        Reference
        ---------
        Hodges, A. et al. J. Chem. Eng. Data 68, 1485–1506 (2023).
        """
        T = self.temperature - 273.15
        return (5.1998e-6 * T ** 3 - 39.771334e-4 * T ** 2 -
                848.089182e-4 * T + 1001.5409980109) * np.exp(0.0086 * self.weight_percent)

    def calculate_ionic_conductivity(self, temperature=None):
        """
        Calculate ionic conductivity using correlations from Hodges et al. (2023).

        Returns
        -------
        float
            Ionic conductivity [S/m].

        Reference
        ---------
        Hodges, A. et al. J. Chem. Eng. Data 68, 1485–1506 (2023).
        """
        if not temperature: 
            temperature = self.temperature
        low_temp_cond = 100 * self.molarity * (-2.041 - 0.0028 * self.molarity +
            0.005332 * temperature + 207.2 / temperature +
            0.001043 * self.molarity ** 2 - 3e-7 * self.molarity * temperature ** 2)

        high_temp_cond = 100 * self.weight_percent * (2.2204e-3 - 1.3077e-3 * self.weight_percent +
            3.3647e-4 * temperature - 10.7021 / temperature +
            7.0101e-6 * self.weight_percent ** 2 - 3.2033e-9 * self.weight_percent * temperature ** 2)

        return np.where(temperature <= 353.15, low_temp_cond, high_temp_cond)

    def calculate_saturation_pressure(self):
        """
        Calculate vapor pressure lowering using correlation from Balej (1985).
        Used to compute the water activity of the electrolyte solution. 

        Parameters
        ----------
        molality : float
            Molality [kmol/kg].
        water_sat_pressure : float
            Pure water saturation pressure [Pa].

        Returns
        -------
        float
            Solution saturation pressure [Pa].
        
        Reference
        --------- 
        Balej, J. International Journal of Hydrogen Energy 10, 233–243 (1985).
        """
        m_mol_per_kg = self.molality * 1000.
        log_p_sat = np.log10(self.water_sat_pressure / 1e5)
        return 10 ** (5 + log_p_sat - m_mol_per_kg * (
            (0.01508 + 0.0012062 * log_p_sat) +
            (0.0016788 - 5.6024e-4 * log_p_sat) * m_mol_per_kg -
            (2.25887e-5 - 7.8228e-6 * log_p_sat) * m_mol_per_kg ** 2))

    def calculate_surface_tension(self) -> float:
        """
        Calculate the surface tension of aqueous KOH using equation 10 in Hodges (2023).

        Equation valid between 30 % wt. and 50 % wt.

        Returns
        -------
        float
            Solution surface tension in N/m

        Reference
        ---------
        Hodges, A. et al. J. Chem. Eng. Data 68, 1485–1506 (2023).
        """
        # Coefficient matrix from the image
        a = np.array([
            [75.4787, -0.138489, -3.38e-04, 4.75e-7, -2.64e-10],
            [-32.889, 1.34382, -9.10e-03, 3.96e-05, -5.74e-08],
            [614.527, -12.8736, 0.104855, -4.49e-04, 6.51e-07],
            [-1455.06, 39.8511, -0.344234, 1.44e-03, -2.08e-06],
            [1333.62, -38.3316, 0.335129, -1.37e-03, 1.95e-06]
        ])

       
        T = self.temperature -273.15
        xi = self.weight_percent / 100.
        # Calculate surface tension using the double polynomial
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
