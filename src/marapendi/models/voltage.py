"""
Electrochemical voltage model for PEM/AEM fuel cells and electrolysers.

Classes
-------
VoltageModel : Computes reversible voltage, overpotentials, and cell voltage.
    Stateless strategy object — all inputs are passed explicitly so methods
    can be tested and reused independently of the ODE machinery.
"""

from dataclasses import dataclass
import numpy as np
import cantera as ct

from marapendi.models.electrochemistry import calculate_reversible_cell_voltage, STD_PRESSURE
from marapendi.components.electrolyte import ElectrolyteSolution


@dataclass
class VoltageModel:
    """Electrochemical voltage model for a single PEM/AEM cell."""

    # ------------------------------------------------------------------
    # Reversible voltage
    # ------------------------------------------------------------------

    def calculate_reversible_cell_voltage(self, T_an_cl, T_ca_cl, p_h2, p_o2_local):
        """
        Thermodynamic reversible cell voltage and half-cell potentials.

        Returns
        -------
        E_rev, E_rev_ca, E_rev_an : ndarray
        """
        activity_o2 = p_o2_local / STD_PRESSURE
        activity_h2 = p_h2 / STD_PRESSURE
        E_rev_an = -(
            ct.gas_constant * T_an_cl * np.log(activity_h2) / (2 * ct.faraday)
        )
        E_rev_ca = calculate_reversible_cell_voltage(T_ca_cl, activity_o2 ** 0.5)
        return E_rev_ca - E_rev_an, E_rev_ca, E_rev_an

    # ------------------------------------------------------------------
    # Overpotentials
    # ------------------------------------------------------------------

    def calculate_orr_overpotential(self, T_ca_cl, p_o2_ca_cl, i,
                                    i_x, roughness_factor, reaction):
        """ORR Tafel overpotential at the cathode catalyst layer."""
        return reaction.tafel_overpotential(
            (i + i_x) / roughness_factor,
            T_ca_cl,
            p_o2_ca_cl,
        )

    def calculate_activation_overpotential(self, T_ca_cl, p_o2_ca_cl,
                                           i, i_x,
                                           theta_PtO, ca_cl):
        """Net activation overpotential including Pt-oxide voltage drop."""
        rf = ca_cl.ecsa * ca_cl.L_Pt * (1 - theta_PtO)
        orr_overpotential = self.calculate_orr_overpotential(
            T_ca_cl, p_o2_ca_cl, i, i_x,
            rf, ca_cl.reaction,
        )
        omega_PtO_voltage_drop = (
            ca_cl.omega_PtO * theta_PtO
            / (ca_cl.reaction.number_of_electrons
               * ca_cl.reaction.charge_transfer_coeff
               * ct.faraday)
        )
        return orr_overpotential + omega_PtO_voltage_drop

    def calculate_ohmic_overpotential(self, T_memb, f_v_memb, T_ca_cl, f_v_ca_cl, s_ca_cl,
                                      i, memb,
                                      electrical_resistance, membrane_model, ionomer_model, ca_cl_model, ca_cl, charge, use_neyerlin_correction=False):
        """
        Per-location ohmic overpotential.

        Returns
        -------
        eta_memb, eta_ca_cl, eta_gdl : ndarray
        """
        # TODO: remove workaround once electrolyte is set properly on CatalystLayer
        ca_cl.electrolyte = ElectrolyteSolution()

        eta_memb  = i * memb.thickness / membrane_model.charge_conductivity(
            f_v_memb, T_memb, charge, memb
        )
        eta_ca_cl = i * ca_cl_model.effective_charge_resistance(
            i, f_v_ca_cl, T_ca_cl, s_ca_cl, charge, ionomer_model, ca_cl, ca_cl.reaction, use_neyerlin_correction
        )
        eta_gdl = i * electrical_resistance / 2
        return eta_memb, eta_ca_cl, eta_gdl

    # ------------------------------------------------------------------
    # Cell voltage
    # ------------------------------------------------------------------

    def calculate_cell_voltage(self, T_an_cl, T_ca_cl, T_memb,
                               f_v_memb, f_v_ca_cl, s_ca_cl,
                               p_h2, p_o2_local, p_o2_ca_cl,
                               i, memb,
                               electrical_resistance, memb_model, ionomer_model, ca_cl_model,
                               ca_cl, charge,
                               theta_PtO=0, use_neyerlin_correction=False):
        """
        Cell voltage and per-location overpotential components.

        The caller assembles ``S_T_losses`` from the returned components.

        Returns
        -------
        V_cell, eta_ohm, eta_act, E_rev_ca, E_rev_an, eta_memb, eta_ca_cl, eta_gdl
        """
        E_rev, E_rev_ca, E_rev_an = self.calculate_reversible_cell_voltage(
            T_an_cl, T_ca_cl, p_h2, p_o2_local,
        )
        i_x = (
            memb_model.calculate_h2_permeation_flux(T_memb, f_v_memb, p_h2,
                                              memb.thickness)
            * (2 * ct.faraday)
        )
        eta_act = self.calculate_activation_overpotential(
            T_ca_cl, p_o2_ca_cl, i, i_x, theta_PtO, ca_cl,
        )
        eta_memb, eta_ca_cl, eta_gdl = self.calculate_ohmic_overpotential(
            T_memb, f_v_memb, T_ca_cl, f_v_ca_cl, 
            s_ca_cl, i, memb, electrical_resistance, 
            memb_model, ionomer_model, ca_cl_model, ca_cl, charge, 
            use_neyerlin_correction
        )
        eta_ohm = eta_memb + eta_ca_cl + 2 * eta_gdl
        V_cell  = E_rev - eta_ohm - eta_act

        return V_cell, eta_ohm, eta_act, E_rev_ca, E_rev_an, eta_memb, eta_ca_cl, eta_gdl
