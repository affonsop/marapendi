import numpy as np 
import cantera as ct

from .fuelcell import FuelCell 
from .gas_composition import species_names, calculate_species_diffusion_coefficient
from .water import water_saturation_concentration, water_density

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

        self.x = np.zeros(
            self.n_layers * (
                1 # ionomer
                + 1 # temperature 
                + 4 # gas concentrations 
                + 1 # liquid water 
            )
        )

        self.dxdt = np.zeros_like(self.x)
        
    def get_states_from_x(self,x): 
        lmbd = x[:self.n_layers,...]
        T = x[self.n_layers:2*self.n_layers,...]
        cg = x[2*self.n_layers:6*self.n_layers,...]
        cg = cg.reshape(4, cg.shape[0] // 4, *cg.shape[1:])
        s = x[6*self.n_layers:7*self.n_layers,...]
        return lmbd, T, cg, s
    
    def rates_of_change(self, x, current_density=0.):
        lmbd, T, cg, s = self.get_states_from_x(x)
        c_sat = water_saturation_concentration(T)
        c_v = cg[-1, ...]
        iF = current_density/ (ct.faraday)

        rh = c_v / c_sat

        # membrane 
        dlmbddt = np.zeros_like(lmbd)
        R_lmbd = np.array([layer.calculate_membrane_water_resistance(T[layer.ix,...], lmbd[layer.ix,...]) 
                  for layer in self.ionomer_layers])
        eff_R_lmbd = (R_lmbd[:-1,...] + R_lmbd[1:,...])/2
        
        J_eod = self.membrane.calculate_electroosmotic_drag_coefficient(T, lmbd)[self.ionomer_domain,...] * iF
        J_lmbd = - np.diff(lmbd[self.ionomer_domain,...], axis=0) / eff_R_lmbd + J_eod[:1,...]
        k_abs = self.membrane.calculate_water_absorption_coefficient(T)
        lmbd_eq = ((1-s) * self.membrane.equilibrium_water_content(rh, T) 
                   + s * self.membrane.liquid_equilibrium_water_content(T))
        J_abs = (k_abs * (lmbd - lmbd_eq) * self.membrane.dry_concentration)
    
        dlmbddt[self.an.cl.ix,...] = (-J_lmbd[0,...] - J_abs[self.an.cl.ix,...]) / (self.an.cl.ionomer_vol_fraction * self.an.cl.ionomer.dry_concentration * self.an.cl.thickness)
        dlmbddt[self.membrane.ix,...] = (J_lmbd[0,...] - J_lmbd[1,...]) / (self.membrane.dry_concentration * self.membrane.dry_thickness)
        dlmbddt[self.ca.cl.ix,...] = (J_lmbd[1,...] - J_abs[self.ca.cl.ix,...] + iF/2) / (self.ca.cl.ionomer_vol_fraction * self.ca.cl.ionomer.dry_concentration * self.ca.cl.thickness)
        
        # liquid water 
        dsdt = np.zeros_like(s)
        p_c = np.zeros_like(s)
        for l in self.porous_layers: 
            p_c[l.ix,...] = l.capillary_pressure_from_saturation(s[l.ix,...])
        p_g = np.sum(cg, axis=0) * ct.gas_constant * T
        print(p_c, s ** 2 * self.ca.gdl.capillary_pressure_J_ratio)
        p_l = p_c + p_g

        R_l = np.zeros_like(s)
        for layer in self.layers:
            R_l[layer.ix, :] = layer.calculate_liquid_darcy_flow_resistance(T[layer.ix,...], s[layer.ix,...])
        for side in (self.an,self.ca): 
            R_l[side.ch.ix,...] = 0  

        eff_R_l = (R_l[:-1,...] + R_l[1:,...])/2
        dp_l = np.diff(p_l, axis=0) 
        J_l = -dp_l / eff_R_l
        J_l[self.ca.cl.ix-1,...] = 0
        J_l[self.an.cl.ix,...] = 0
        factor = np.where(c_sat > c_v, s, 1 - s)
        S_vl = 1000.0 * (c_v - c_sat) * factor
        rho_l = water_density(T)
        for layer in self.porous_layers: 
            dsdt[layer.ix, ...] = ((J_l[layer.ix-1,...] - J_l[layer.ix,...]) / layer.thickness + S_vl[layer.ix, ...]) / (rho_l[layer.ix, ...] * layer.porosity)

        # gas concentrations 
        dcgdt = np.zeros_like(cg)
        R_g = np.zeros_like(cg)
        D_g = calculate_species_diffusion_coefficient(T, p_g, x_h2 = self.x_h2_mask)
        for layer in self.porous_layers:
            R_g[:,layer.ix, ...] = layer.calculate_gas_transport_resistance( 
                T[layer.ix,...],
                D_g[:,layer.ix,...],
                liquid_saturation=s[layer.ix,...], 
                )
        R_g[:,self.membrane.ix, ...] = 1e18
        for side in (self.an, self.ca):
            R_g[:,side.ch.ix, ...] = side.ch.transport_resistance_model.molecular_diffusion_resistance(side.ch, D_g[:,side.ch.ix,...])
    
        eff_R_g = (R_g[:,:-1,...] + R_g[:,1:,...])/2
        dc_g = np.diff(cg, axis=1)
        J_g = -dc_g / eff_R_g
        S_g = np.zeros_like(cg)
        S_g[-1,...] = - S_vl
        for side in (self.an, self.ca):
            S_g[-1,side.cl.ix] += J_abs[side.cl.ix,...] / side.cl.thickness
        S_g[0,self.ca.cl.ix,...] = (-iF/4) / self.an.cl.thickness
        S_g[2,self.an.cl.ix,...] = (-iF/2) / self.ca.cl.thickness
        J_g[:,self.ca.cl.ix-1,...] = 0
        J_g[:,self.an.cl.ix,...] = 0
        for layer in self.porous_layers: 
            dcgdt[:,layer.ix, ...] = ((J_g[:,layer.ix-1,...] - J_g[:,layer.ix,...]) / layer.thickness + S_g[:,layer.ix, ...]) / layer.porosity
        return dlmbddt, dsdt, dcgdt
    
    def gas_concentration_rate_of_change(self,x, current_density=0.): 
        pass