"""
Electrolyte solution base class with thermophysical property correlations.
"""
from dataclasses import dataclass, field
import numpy as np
from ..thermo.water import water_saturation_pressure, water_surface_tension


@dataclass
class ElectrolyteSolution():
    """
    Base class for an electrolyte solution with thermophysical property correlations.

    Stores concentration and computes density, ionic conductivity, saturation
    pressure, and surface tension.  Subclasses override the ``calculate_*``
    methods to provide electrolyte-specific correlations.

    Attributes
    ----------
    temperature : float
        Temperature of the solution (K).
    molality : float
        Molality of the electrolyte (kmol/kg).
    weight_percent : float
        Weight fraction of the electrolyte (0–100).
    electrolyte_molecular_weight : float
        Molecular weight of the dissolved electrolyte (kg/kmol).
    """

    temperature: float = 298.15
    molality: float = 0
    weight_percent: float = 0
    electrolyte_molecular_weight: float = 56.105

    def __post_init__(self):
        if self.molality == 0:
            self.molality = self.calculate_molality()
        elif self.weight_percent == 0:
            self.weight_percent = self.calculate_weight_percent()
        self.set_temperature(self.temperature)

    def set_temperature(self, temperature):
        """
        Set temperature and recompute all dependent properties.

        Parameters
        ----------
        temperature : float
            New temperature (K).
        """
        self.temperature = temperature
        self.density = self.calculate_density()
        self.molarity = self.calculate_molarity()
        self.ionic_conductivity = self.calculate_ionic_conductivity()
        self.water_sat_pressure = water_saturation_pressure(self.temperature)
        self.solution_sat_pressure = self.calculate_saturation_pressure()
        self.surface_tension = self.calculate_surface_tension()

    def calculate_ionic_conductivity(self, temperature=None):
        """Return a default very low ionic conductivity; override in subclasses."""
        return 1e-12

    def calculate_density(self):
        """Return a default water-like density; override in subclasses."""
        return 1000.

    def calculate_saturation_pressure(self):
        """Return water saturation pressure by default; override in subclasses."""
        return self.water_sat_pressure

    def calculate_surface_tension(self):
        """Return water surface tension by default; override in subclasses."""
        return water_surface_tension(self.temperature)

    def calculate_weight_percent(self):
        """
        Calculate weight percent from molality.

        Returns
        -------
        float
            Weight percent (0–100).
        """
        return 100. * self.electrolyte_molecular_weight * self.molality / (1 + self.electrolyte_molecular_weight * self.molality)

    def calculate_molality(self):
        """
        Calculate molality from weight percent.

        Returns
        -------
        float
            Molality (kmol/kg).
        """
        return self.weight_percent / self.electrolyte_molecular_weight / (100 - self.weight_percent)

    def calculate_molarity(self):
        """
        Calculate molar concentration of the solution.

        Returns
        -------
        float
            Molarity (kmol/m³).
        """
        return self.weight_percent / 100. * self.calculate_density() / self.electrolyte_molecular_weight
