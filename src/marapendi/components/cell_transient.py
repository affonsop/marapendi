import numpy as np 
import cantera as ct

from .cell import Cell 
from marapendi.models.gas_composition import species_names, calculate_species_diffusion_coefficient
from .water import water_saturation_concentration, water_density, water_molar_volume, water_kinematic_viscosity
from .electrolyte import ElectrolyteSolution
from marapendi.models.electrochemistry import calculate_reversible_cell_voltage, STD_PRESSURE, enthalpy_condensation

class TransientCellModel(Cell): 
    def __post_init__(self):
        super().__post_init__()
        self.n_layers = len(self.layers)
        
        self.ionomer_layers = [self.an.cl, self.memb, self.ca.cl]
        self.ionomer_domain = [l in self.ionomer_layers for l in self.layers] 
        self.porous_domain = [(l in self.ca.porous_layers) or 
                              (l in self.an.porous_layers) for l in self.layers] 
        self.channel_domain = [l in [self.ca.ch, self.an.ch] for l in self.layers]
        self.layer_thickness = np.array([self.an.ch.height] + 
                                        [l.thickness for l in self.an.porous_layers[::-1]] +
                                        [self.memb.thickness] + 
                                        [l.thickness for l in self.ca.porous_layers] +
                                        [self.ca.ch.height])[...,np.newaxis]
                                         
        self.x_h2_mask = np.array([
            1 if layer in self.an.porous_layers else 0
            for layer in self.layers
        ])
        for i, layer in enumerate(self.layers): 
            layer.ix = i 

        self.n_variables = (
                1 # ionomer
                + 1 # temperature 
                + 4 # gas concentrations 
                + 1 # liquid water
        )
        self.i_lmbd = 0 
        self.i_T = 1
        self.i_cg = [2,3,4,5]
        self.i_s = 6
        self.norm_factor = np.array([14,353.15,40e-3,40e-3,40e-3,40e-3,1.])[np.newaxis,:]
        
    def get_states_from_x(self,x): 
        lmbd = x[:, self.i_lmbd,...]
        T = x[:, self.i_T,...]
        cg_k = x[:, self.i_cg,...]
        s = x[:, self.i_s,...]
        return lmbd, T, cg_k, s
    
    def calculate_permeation_flux(self, T, lmbd, p_g, p_g_k): 
        return self.memb.hydrogen_permeation_flux(
            p_g_k[self.an.cl.ix, 2,...], 
            T[self.memb.ix, ...], 
            p_g[self.an.cl.ix]- p_g[self.ca.cl.ix],
            self.memb.water_vol_fraction(
                lmbd[self.memb.ix, ...], 
                water_molar_volume(T[self.memb.ix, ...])
                )
            )
    
    def calculate_reversible_cell_voltage(self, T, p_g_k, p_o2_local): 
        activity_o2 = p_o2_local / STD_PRESSURE
        activity_h2 = p_g_k[self.an.cl.ix,2,...] / STD_PRESSURE
        E_rev_an = - (
            ct.gas_constant * T[self.an.cl.ix,...] 
            * np.log(activity_h2) / (2 * ct.faraday)
        )
        E_rev_ca = calculate_reversible_cell_voltage(
            T[self.ca.cl.ix,...],
            activity_o2 ** 0.5
        )
        return E_rev_ca - E_rev_an, E_rev_ca, E_rev_an 

    def calculate_orr_overpotential(self, T, p_g_k, current_density, crossover_current, roughness_factor):
        return self.ca.cl.reaction.tafel_overpotential(
            (current_density + crossover_current) / roughness_factor,
            T[self.ca.cl.ix, ...],
            p_g_k[self.ca.cl.ix, 0,...]
        )

    def calculate_activation_overpotential(self,T, p_g_k, current_density, crossover_current, theta_PtO): 
        
        orr_overpotential = self.calculate_orr_overpotential(T, p_g_k, current_density, crossover_current, 
                                                             self.ca.cl.ecsa * self.ca.cl.platinum_loading * (1-theta_PtO))
        omega_PtO_voltage_drop = self.ca.cl.omega_PtO * theta_PtO / (self.ca.cl.reaction.number_of_electrons *
                                                                      self.ca.cl.reaction.charge_transfer_coeff * ct.faraday)
        
        hor_overpotential = 0
        eta_act = orr_overpotential + omega_PtO_voltage_drop + hor_overpotential
        return eta_act 
    
    def calculate_ohmic_overpotential(self, T, lmbd, current_density):
        r_ohm = np.zeros_like(T)
        self.ca.cl.electrolyte = ElectrolyteSolution() # Temporary workaround
        r_ohm[self.memb.ix, ...] = self.memb.calculate_proton_resistance(T[self.memb.ix, ...], lmbd[self.memb.ix, ...]) 
        r_ohm[self.ca.cl.ix, ...] = self.ca.cl.effective_charge_resistance(current_density, lmbd[self.ca.cl.ix, ...], T[self.ca.cl.ix, ...]) 
        r_ohm[self.ca.gdl.ix, ...] = self.electrical_resistance / 2
        r_ohm[self.an.gdl.ix, ...] = self.electrical_resistance / 2
        eta_ohm = r_ohm * current_density
        
    
        return eta_ohm
    
    def calculate_cell_voltage(self, T, lmbd, p_g, p_g_k, p_o2_local, s, current_density, theta_PtO=0): 
        reversible_cell_voltage, orr_reversible_potential, hor_reversible_potential = self.calculate_reversible_cell_voltage(T, p_g_k, p_o2_local)    
        eta_ohm= self.calculate_ohmic_overpotential(T, lmbd, current_density)
        crossover_current = self.calculate_permeation_flux(T, lmbd, p_g, p_g_k) * (2 * ct.faraday)
        eta_act = self.calculate_activation_overpotential(T, p_g_k, current_density, crossover_current, theta_PtO)

        S_T = eta_ohm * current_density
        S_T[self.ca.cl.ix,...] = (eta_act + orr_reversible_potential) * current_density 
        S_T[self.an.cl.ix,...] = (hor_reversible_potential) * current_density 
        eta_ohm = np.sum(eta_ohm, axis=0)
        return reversible_cell_voltage - eta_ohm - eta_act, eta_ohm, eta_act, S_T

    # ------------------------------------------------------------------
    # rates_of_change — sub-steps
    # ------------------------------------------------------------------

    def _compute_derived_quantities(self, x, current_density):
        """Unpack state vector and compute thermodynamic derived fields."""
        lmbd, T, cg_k, s = self.get_states_from_x(x)

        iF    = current_density / ct.faraday
        c_g   = np.sum(cg_k, axis=1)
        p_g   = c_g * ct.gas_constant * T
        x_g_k = cg_k / c_g[:,np.newaxis,...]
        p_g_k = p_g[:,np.newaxis,...] * x_g_k
        D_g   = calculate_species_diffusion_coefficient(T, p_g, x_h2=self.x_h2_mask)
        c_sat = water_saturation_concentration(T)
        c_v   = cg_k[:,-1, ...]
        rh    = c_v / c_sat
        rho_l = water_density(T)
        nu_l = water_kinematic_viscosity(T)
        M_w = x_g_k * np.array([32., 28., 2., 18.])[np.newaxis,:,np.newaxis]
        return lmbd, T, cg_k, s, iF, p_g, p_g_k, D_g, c_sat, c_v, rh, rho_l, nu_l, M_w

    def _compute_resistances(self, x, T, lmbd, s, D_g, nu_l, M_w):
        """Build layer resistance array R and harmonic-mean inter-layer eff_R."""
        R = np.zeros_like(x)

        # λ is only defined in the ionomer domain; block transport everywhere else so
        # the BDF Jacobian never sees a trivially-zero diagonal for those variables.
        R[:, self.i_lmbd, ...] = np.inf
       
        R[:, self.i_s,  ...] = self.darcy_transport_model.calculate_liquid_darcy_flow_resistance(
                    T, s, self.thickness, nu_l, self.absolute_permeability, self.relative_permeability_exponent)
        
        R[:, self.i_cg, ...] = self.gas_diffusion_model.total_diffusion_resistance(
                    T[:,np.newaxis, ...], 
                    s[:,np.newaxis, ...], 
                    D_g, 
                    M_w, 
                    self.thickness[:,np.newaxis, ...], 
                    self.porosity[:,np.newaxis, ...], 
                    self.tortuosity[:,np.newaxis, ...], 
                    self.pore_diameter[:,np.newaxis, ...], 
                    self.water_saturation_exponent[:,np.newaxis, ...])
        R[:, self.i_T, ...] = self.thickness / self.bulk_thermal_conductivity

        V_w = water_molar_volume(T)
        f_v = self.membrane_water_transport_model.water_vol_fraction(lmbd, V_w, self.ionomer_concentration)
        D_lmbd = self.membrane_water_transport_model.diffusion_coefficient(lmbd, f_v, T, self.darken_num, self.darken_den, 
                                                                           self.reference_water_diffusivity, self.ionomer_activation_energy)
        R[layer.ix, self.i_lmbd, ...] = self.membrane_water_transport_model.calculate_membrane_water_resistance(D_lmbd, self.thickness, self.ionomer_vol_fraction, self.ionomer_concentration, self.ionomer_tortuosity) 

        
        for layer in (self.ca.ch, self.an.ch):
                R[layer.ix, self.i_s,  ...] = 0
                R[layer.ix, self.i_cg, ...] = layer.hydraulic_diameter / D_g[layer.ix,...] / layer.sherwood
                R[layer.ix, self.i_T,  ...] = 2 * layer.height / layer.bulk_thermal_conductivity
        
        R[self.memb.ix,[self.i_s] + self.i_cg, ...] = np.inf
    
    
        eff_R = (R[:-1, ...] + R[1:, ...]) / 2
        eff_R[-1, self.i_T, ...] += self.thermal_resistance / 2
        eff_R[ 0, self.i_T, ...] += self.thermal_resistance / 2
        return R, eff_R

    def _compute_fluxes(self, x, eff_R, T, s, lmbd, p_g, p_g_k, cg_k, iF):
        """Compute inter-layer flux array J (diffusion + advection + EOD)
        and return also V_cell, eta_ohm, eta_act, S_T_losses, p_o2_local."""
        phi = x.copy()
        # Liquid-water driving potential: gas pressure + capillary pressure
        phi[:, self.i_s, ...] = p_g
        phi[self.porous_domain, self.i_s, ...] += self.darcy_transport_model.capillary_pressure_from_saturation(
            s, self.breakthrough_pressure, self.van_genuchten_m, self.van_genuchten_n)[self.porous_domain, ...]

        J = np.zeros((self.n_layers + 1, self.n_variables, x.shape[-1]))
        J[1:-1, ...] = -(phi[1:, ...] - phi[:-1, ...]) / eff_R


        # Local O2 partial pressure at catalyst surface
        R_o2_local = self.ca.cl.o2_ionomer_film_resistance(
            lmbd[self.ca.cl.ix, ...], T[self.ca.cl.ix, ...])
        c_o2_local = cg_k[self.ca.cl.ix, 0, ...] - R_o2_local * iF / 4
        p_o2_local = c_o2_local * ct.gas_constant * T[self.ca.cl.ix, 0, ...]

        V_cell, eta_ohm, eta_act, S_T_losses = self.calculate_cell_voltage(
            T, lmbd, p_g, p_g_k, p_o2_local, s, iF * ct.faraday)

        # Electro-osmotic drag
        J_eod = (self.memb.calculate_electroosmotic_drag_coefficient(T, lmbd)
                 [self.ionomer_domain, ...] * iF)
        J[[self.an.cl.ix + 1, self.memb.ix + 1], self.i_lmbd, ...] += J_eod[:1, ...]

        return J, V_cell, eta_ohm, eta_act, S_T_losses, p_o2_local

    def _compute_water_exchange(self, T, s, lmbd, rh):
        """Return ionomer absorption flux J_des (absorption/desorption at CL/membrane)."""
        k_abs   = self.memb.calculate_water_absorption_coefficient(T)
        lmbd_eq = ((1 - s) * self.memb.equilibrium_water_content(rh, T)
                   + s      * self.memb.liquid_equilibrium_water_content(T))
        J_des   = k_abs * (lmbd - lmbd_eq) * self.ionomer_concentration
        return J_des

    def _compute_phase_change(self, T, c_v, c_sat, s):
        """Return vapour–liquid phase-change source term S_vl.

        Positive when condensation occurs (c_v > c_sat), negative when evaporation.
        The `factor` selects liquid saturation s when condensing and (1-s) when evaporating,
        matching the original formulation.
        """
        c_sat  = water_saturation_concentration(T)
        factor = np.where(c_sat > c_v, s, 1 - s)
        return 1000.0 * (c_v - c_sat) * factor

    def _compute_sources(self, x, T, s, lmbd, iF, J_des, S_vl, S_T_losses):
        """Populate source term array S."""
        S = np.zeros_like(x)

        # Ionomer water: absorption sink ± electrolysis production
        S[self.an.cl.ix, self.i_lmbd, ...] += (       - J_des[self.an.cl.ix, ...]) / self.an.cl.thickness
        S[self.ca.cl.ix, self.i_lmbd, ...] += (iF / 2 - J_des[self.ca.cl.ix, ...]) / self.ca.cl.thickness

        # Liquid water: phase change (blocked in membrane)
        S[:, self.i_s, ...]              = S_vl
        S[self.memb.ix, self.i_s, ...] = 0

        # Water vapour: condensation sink
        S[:, self.i_cg[-1], ...] = -S_vl

        # Gas species: electrolysis consumption
        S[self.ca.cl.ix, self.i_cg[0], ...] = (-iF / 4) / self.ca.cl.thickness
        S[self.an.cl.ix, self.i_cg[2], ...] = (-iF / 2) / self.an.cl.thickness

        # Water vapour: re-evaporation from absorbed water
        for side in (self.an, self.ca):
            S[side.cl.ix, self.i_cg[-1], ...] += J_des[side.cl.ix, ...] / side.cl.thickness
        S[self.memb.ix, self.i_cg, ...] = 0

        # Temperature: ohmic/activation losses + phase-change enthalpy
        h_vl = enthalpy_condensation(T)
        S[:, self.i_T, ...] = S_T_losses / self.thickness + S_vl * h_vl
        for side in (self.an, self.ca):
            S[side.cl.ix, self.i_T, ...] -= (
                J_des[side.cl.ix, ...] / side.cl.thickness
                * self.memb.heat_of_adsorption(T[side.cl.ix, ...]))

        return S

    def _compute_capacities(self, x, rho_l):
        """Populate capacity array C."""
        C = np.ones_like(x)
        for side in (self.ca, self.an):
            C[side.cl.ix, self.i_lmbd, ...] = (
                side.cl.ionomer_vol_fraction
                * side.cl.ionomer.dry_concentration
                * side.cl.thickness)
        C[self.memb.ix, self.i_lmbd, ...] = (
            self.memb.dry_concentration * self.memb.thickness)
        for layer in self.layers:
            if layer != self.memb:
                C[layer.ix, self.i_s,  ...] = rho_l[layer.ix, ...] * layer.porosity
                C[layer.ix, self.i_cg, ...] = layer.porosity
            C[layer.ix, self.i_T] = layer.bulk_density * layer.bulk_specific_heat_capacity
        return C

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def rates_of_change(self, x, current_density=0.):
        
        """Return dxdt for all state variables across all layers.

        Parameters
        ----------
        x : ndarray, shape (n_layers * n_variables, m)
        current_density : float
        """
        x = (x.reshape(self.n_layers, self.n_variables, x.shape[-1])
             * self.norm_factor[..., np.newaxis])
        
        # 1. Derived thermodynamic quantities
        lmbd, T, cg_k, s, iF, p_g, p_g_k, D_g, c_sat, c_v, rh, rho_l, nu_l, M_w = (
            self._compute_derived_quantities(x, current_density))

        # 2. Transport resistances
        R, eff_R = self._compute_resistances(x, T, lmbd, s, D_g, nu_l, M_w)
        
        # 3. Inter-layer fluxes (including EOD)
        J, V_cell, eta_ohm, eta_act, S_T_losses, p_o2_local = (
            self._compute_fluxes(x, eff_R, T, s, lmbd, p_g, p_g_k, cg_k, iF))

        # 4. Water exchange between phases
        J_des = self._compute_water_exchange(T, s, lmbd, rh)
        S_vl  = self._compute_phase_change(T, c_v, c_sat, s)

        # 5. Source terms
        S = self._compute_sources(x, T, s, lmbd, iF, J_des, S_vl, S_T_losses)

        # 6. Capacities
        C = self._compute_capacities(x, rho_l)
        
        # 7. Assemble dxdt and zero out channel BCs
        dxdt = ((J[:-1, ...] - J[1:, ...]) / self.thickness[:,np.newaxis] + S) / C
        for ch in (self.ca.ch, self.an.ch):
            dxdt[ch.ix, self.i_T,  :] = 0
            dxdt[ch.ix, self.i_cg, :] = 0
            dxdt[ch.ix, self.i_s,  :] = 0

        return dxdt.reshape(self.n_layers * self.n_variables, x.shape[-1])
