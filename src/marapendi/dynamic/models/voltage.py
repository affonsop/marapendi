"""
Electrochemical voltage model for PEM/AEM fuel cells and electrolysers.

Classes
-------
VoltageModel : Computes reversible voltage, overpotentials, and cell voltage.
    Stateless strategy object — all inputs are passed explicitly so methods
    can be tested and reused independently of the ODE machinery.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import dataclass
import numpy as np
import cantera as ct

from marapendi.dynamic.models.electrochemistry import calculate_reversible_cell_voltage, STD_PRESSURE
from marapendi.dynamic.components.electrolyte import ElectrolyteSolution

if TYPE_CHECKING:
    from marapendi.dynamic.components.cell_state import CellState


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
        activity_o2 = np.maximum(p_o2_local / STD_PRESSURE, 1e-30)
        activity_h2 = np.maximum(p_h2 / STD_PRESSURE, 1e-30)
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

    def calculate_activation_overpotential(self, T_ca_cl, p_o2_local,
                                           i, i_x,
                                           theta_PtO, ca_cl):
        """Net activation overpotential including Pt-oxide voltage drop."""
        rf = ca_cl.ecsa * ca_cl.L_Pt * (1 - theta_PtO)
        orr_overpotential = self.calculate_orr_overpotential(
            T_ca_cl, p_o2_local, i, i_x,
            rf, ca_cl.reaction,
        )
        omega_PtO_voltage_drop = (
            ca_cl.omega_PtO * theta_PtO
            / (ca_cl.reaction.number_of_electrons
               * ca_cl.reaction.charge_transfer_coeff
               * ct.faraday)
        )
        return orr_overpotential + omega_PtO_voltage_drop

    def calculate_ohmic_overpotential(self, T_an_cl, f_v_an_cl, T_memb, f_v_memb,
                                      T_ca_cl, f_v_ca_cl, s_ca_cl,
                                      i, memb,
                                      electrical_resistance, membrane_model, ionomer_model,
                                      ca_cl_model, ca_cl, charge,
                                      use_neyerlin_correction=False):
        """Per-location ohmic overpotential.

        The membrane resistance is estimated by Simpson's rule over the three
        available σ_p sample points (anode-CL node, membrane centre node,
        cathode-CL node), which are used as proxies for the left boundary,
        midpoint, and right boundary of the membrane:

            R_memb ≈ (L/6) · (1/σ_left + 4/σ_centre + 1/σ_right)

        This captures the strongly non-linear variation of σ_p through the
        membrane that arises from the λ gradient driven by electro-osmotic
        drag, without requiring extra membrane nodes in the ODE.

        Returns
        -------
        eta_memb, eta_ca_cl, eta_gdl : ndarray
        """
        # TODO: remove workaround once electrolyte is set properly on CatalystLayer
        ca_cl.electrolyte = ElectrolyteSolution()

        def _sigma(f_v, T):
            return membrane_model.charge_conductivity(f_v, T, charge, memb)

        sigma_left   = _sigma(f_v_an_cl, T_an_cl)   # ACL/PEM boundary proxy
        sigma_center = _sigma(f_v_memb,  T_memb)     # membrane centre node
        sigma_right  = _sigma(f_v_ca_cl, T_ca_cl)    # PEM/CCL boundary proxy
        eta_memb = i * memb.thickness / 6 * (
            1 / sigma_left + 4 / sigma_center + 1 / sigma_right
        )
        eta_ca_cl = i * ca_cl_model.effective_charge_resistance(
            i, f_v_ca_cl, T_ca_cl, s_ca_cl, charge, ionomer_model, ca_cl, ca_cl.reaction,
            f_v_boundary=f_v_memb, T_boundary=T_memb,
            use_neyerlin_correction=use_neyerlin_correction,
        )
        eta_gdl = i * electrical_resistance / 2
        return eta_memb, eta_ca_cl, eta_gdl

    # ------------------------------------------------------------------
    # Cell voltage
    # ------------------------------------------------------------------

    def compute_cell_voltage(
        self,
        state: CellState,
        cell,
        memb_model,
        cl_model,
    ) -> None:
        """Compute cell voltage and all overpotentials; populate *state* in-place.

        Reads ``state.p_o2_local`` which must already be set before this call
        (e.g. by ``cl_model.compute_local_o2_partial_pressure``).

        Parameters
        ----------
        state : CellState
        cell : Cell
        memb_model :
            Membrane / ionomer model.
        cl_model :
            Catalyst-layer model used for the CL proton-resistance estimate.
        """
        i = state.iF * ct.faraday

        E_rev, state.E_rev_ca, state.E_rev_an = self.calculate_reversible_cell_voltage(
            state.T_an_cl, state.T_ca_cl, state.p_h2, state.p_o2_local,
        )
        i_x = (
            memb_model.calculate_h2_permeation_flux(
                state.T_memb, state.f_v_memb, state.p_h2, cell.memb.thickness,
            ) * (2 * ct.faraday)
        )
        state.eta_act = self.calculate_activation_overpotential(
            state.T_ca_cl, state.p_o2_local, i, i_x, theta_PtO=0, ca_cl=cell.ca.cl,
        )
        state.eta_memb, eta_ca_cl, state.eta_gdl = self.calculate_ohmic_overpotential(
            state.T_an_cl, state.f_v_an_cl, state.T_memb, state.f_v_memb,
            state.T_ca_cl, state.f_v_ca_cl,
            s_ca_cl=0, i=i, memb=cell.memb,
            electrical_resistance=cell.electrical_resistance,
            membrane_model=memb_model, ionomer_model=memb_model,
            ca_cl_model=cl_model, ca_cl=cell.ca.cl, charge=cell.charge,
        )
        state.eta_ohm = state.eta_memb + eta_ca_cl + 2 * state.eta_gdl
        state.V_cell  = E_rev - state.eta_ohm - state.eta_act
