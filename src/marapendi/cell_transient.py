import numpy as np 
import cantera as ct

from .fuelcell import FuelCell 
from .gas_composition import species_names, calculate_species_diffusion_coefficient
from .water import water_saturation_concentration, water_density, water_molar_volume
from .electrolyte import ElectrolyteSolution

class TransientCellModel(FuelCell): 
    def __post_init__(self):
        super().__post_init__()
        self.n_layers = len(self.layers)
        
        self.ionomer_layers = [self.an.cl, self.membrane, self.ca.cl]
        self.ionomer_domain = [l in self.ionomer_layers for l in self.layers] 
        self.porous_domain = [(l in self.ca.porous_layers) or 
                              (l in self.an.porous_layers) for l in self.layers] 
        self.channel_domain = [l in [self.ca.ch, self.an.ch] for l in self.layers]
        self.layer_thickness = np.array([self.an.ch.height] + 
                                        [l.thickness for l in self.an.porous_layers[::-1]] +
                                        [self.membrane.dry_thickness] + 
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
        self.thickness = np.array([layer.get_thickness() for layer in self.layers])
        self.norm_factor = np.array([14,353.15,40e-3,40e-3,40e-3,40e-3,1.])[np.newaxis,:]

    def get_states_from_x(self,x): 
        lmbd = x[:, self.i_lmbd,...]
        T = x[:, self.i_T,...]
        cg_k = x[:, self.i_cg,...]
        s = x[:, self.i_s,...]
        return lmbd, T, cg_k, s
    
    def cell_voltage_from_x(self, T, lmbd, p_g, s, current_density): 
        r_ohm = np.zeros_like(T)
        eta_act = np.zeros_like(T)
        self.ca.cl.electrolyte = ElectrolyteSolution() # Temporary workaround
        r_ohm[self.membrane.ix, ...] = self.membrane.calculate_proton_resistance(T[self.membrane.ix, ...], lmbd[self.membrane.ix, ...]) 
        r_ohm[self.ca.cl.ix, ...] = self.ca.cl.effective_charge_resistance(current_density, lmbd[self.ca.cl.ix, ...], T[self.ca.cl.ix, ...]) 
        r_ohm[self.ca.gdl.ix, ...] = self.electrical_resistance / 2
        r_ohm[self.an.gdl.ix, ...] = self.electrical_resistance / 2
        eta_ohm = r_ohm * current_density
        
        h2_permeation_flux = self.membrane.hydrogen_permeation_flux(self.an.cl.species_partial_pressure('h2'), 
                                                                        T[self.membrane.ix, ...], 
                                                                        p_g[self.an.cl.ix]- p_g[self.ca.cl.ix],
                                                                        self.membrane.water_vol_fraction(
                                                                            lmbd[self.membrane.ix, ...], 
                                                                            water_molar_volume(T[self.membrane.ix, ...])
                                                                            )
                                                                        )
        #eta_act = 
    def rates_of_change(self, x, current_density=0.):
        # x is a n_layers x n_states x m matrix     
        x = x.reshape(self.n_layers,self.n_variables, x.shape[-1]) * self.norm_factor[...,np.newaxis]
        
        dxdt = np.zeros_like(x) 
        R = np.zeros_like(x)
        J = np.zeros((self.n_layers+1,self.n_variables, x.shape[-1]))
        S = np.zeros_like(x) 
        C = np.ones_like(x)

        lmbd, T, cg_k, s = self.get_states_from_x(x)
        phi = x.copy()

        c_sat = water_saturation_concentration(T)
        c_v = cg_k[:,-1, ...]
        iF = current_density/ (ct.faraday)
        c_g = np.sum(cg_k, axis=1)
        
        p_g = c_g * ct.gas_constant * T
        
        x_g_k = cg_k / c_g[...,np.newaxis,:] 
        p_g_k = p_g[...,np.newaxis,:] * x_g_k 

        D_g = calculate_species_diffusion_coefficient(T, p_g, x_h2 = self.x_h2_mask)
        rh = c_v / c_sat
        rho_l = water_density(T)
        
        self.cell_voltage_from_x(T, lmbd, p_g, s, current_density)

        # Resistances
        for layer in self.layers:
            if layer in self.ionomer_layers:
                R[layer.ix, self.i_lmbd, ...] = layer.calculate_membrane_water_resistance(T[layer.ix,...], lmbd[layer.ix,...]) 

            if layer in self.porous_layers:
                R[layer.ix, self.i_s, ...] = layer.calculate_liquid_darcy_flow_resistance(T[layer.ix,...], s[layer.ix,...])
                R[layer.ix, self.i_cg, ...] = layer.calculate_gas_transport_resistance( 
                        D_g[:,layer.ix,...],
                        T[np.newaxis,layer.ix,...],
                        liquid_saturation=s[np.newaxis,layer.ix,...], 
                    )
                R[layer.ix, self.i_T,...] = layer.get_thickness() / layer.thermal_conductivity 
            
            elif layer in (self.ca.ch, self.an.ch): 
                R[layer.ix,self.i_s,...] = 0   
                R[layer.ix, self.i_cg, ...] = layer.transport_resistance_model.molecular_diffusion_resistance(layer, D_g[:,layer.ix,...])

            R[self.membrane.ix, self.i_T,...] = self.membrane.get_thickness() / self.membrane.thermal_conductivity 

        eff_R = (R[:-1,...] + R[1:,...]) / 2 + 1e-16
   
        # For water saturation, it is p_l which is conserved 
        phi[:,self.i_s,...] = p_g
        for l in self.porous_layers: 
            phi[l.ix,self.i_s,...] += l.capillary_pressure_from_saturation(s[l.ix,...])
        
        # Fluxes
        J[1:-1,...] = - (phi[1:,...] - phi[:-1,...]) / eff_R 
        
        # Flux boundary conditions 
        for i_var in (self.i_cg, self.i_s):
            J[self.an.cl.ix+1,i_var,...] = 0
            J[self.ca.cl.ix,i_var,...] = 0
        J[self.an.cl.ix,self.i_lmbd, ...] = 0
        J[self.ca.cl.ix+1,self.i_lmbd, ...] = 0

        # EOD flux 
        J_eod = self.membrane.calculate_electroosmotic_drag_coefficient(T, lmbd)[self.ionomer_domain,...] * iF
        J[[self.an.cl.ix+1, self.membrane.ix+1],self.i_lmbd,...] += J_eod[:1,...]

        # Membrane water absorption 
        k_abs = self.membrane.calculate_water_absorption_coefficient(T)
        lmbd_eq = ((1-s) * self.membrane.equilibrium_water_content(rh, T) 
                   + s * self.membrane.liquid_equilibrium_water_content(T))
        J_abs = (k_abs * (lmbd - lmbd_eq) * self.membrane.dry_concentration)
       
        # Water evaporation 
        factor = np.where(c_sat > c_v, s, 1 - s)
        S_vl = 1000.0 * (c_v - c_sat) * factor

        # Source terms
        S[self.an.cl.ix, self.i_lmbd, ...] +=  (      - J_abs[self.an.cl.ix,...]) / self.an.cl.thickness
        S[self.ca.cl.ix, self.i_lmbd, ...] +=  (iF/2  - J_abs[self.ca.cl.ix,...]) / self.ca.cl.thickness
        
        S[:, self.i_s, ...] = S_vl
        S[self.membrane.ix, self.i_s, ...] = 0

        S[:, self.i_cg[-1], ...] = -S_vl 
        
        S[self.ca.cl.ix,self.i_cg[0],...] = (-iF/4) / self.ca.cl.thickness
        S[self.an.cl.ix,self.i_cg[2],...] = (-iF/2) / self.an.cl.thickness
        
        for side in (self.an, self.ca):
            S[side.cl.ix, self.i_cg[-1], ...] += J_abs[side.cl.ix,...] / side.cl.thickness
        S[self.membrane.ix, self.i_cg, ...] = 0

        # Capacities
        for side in (self.ca, self.an): 
            C[side.cl.ix,self.i_lmbd,...] = (side.cl.ionomer_vol_fraction * side.cl.ionomer.dry_concentration * side.cl.thickness)
        C[self.membrane.ix,self.i_lmbd,...] = (self.membrane.dry_concentration * self.membrane.dry_thickness)
        for layer in self.layers: 
            if layer != self.membrane: 
                C[layer.ix, self.i_s, ...] = rho_l[layer.ix, ...] * layer.porosity
                C[layer.ix, self.i_cg, ...] = layer.porosity

        dxdt = ((J[:-1,...]-J[1:,...]) / self.thickness[:, np.newaxis, np.newaxis] + S) / C
        layer = self.ca.cl
        
        dxdt[:,self.i_T,:] = 0
        dxdt[self.ca.ch.ix, self.i_cg,:]=0
        dxdt[self.an.ch.ix, self.i_cg,:]=0
        dxdt[self.ca.ch.ix, self.i_s,:]=0
        dxdt[self.an.ch.ix, self.i_s,:]=0
        dxdt / self.norm_factor[...,np.newaxis]
        return dxdt.reshape(self.n_layers*self.n_variables, x.shape[-1]) 
