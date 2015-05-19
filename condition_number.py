import Simulation as sm
import Apparatus as ap
import numpy as np
import seaborn as sb
import itertools as it
import cProfile
from utils import *

if __name__ == '__main__':
    #-----Parameters-----#
    eta = 1.
    chi = 1.
    
    #-----Basic Objects-----#
    t_b_diV_dict = {'delta': [np.sqrt(3.), -np.sqrt(3.)], 
                'chi': chi * np.ones((2,3)), 'kappa': [2., 2.],
                'gamma_1': [0.0, 0.0, 0.0], 'gamma_phi': [0.0, 0.0, 0.0],
                'purcell': np.zeros((2,3)), 'omega': [0., 0., 0.],
                'eta': eta, 'phi': 0.0}

    t_b_diV_app = ap.Apparatus(**t_b_diV_dict)

    #-----Set Pulse-----#
    tau = 30. / (eta * chi)
    t_on = 6. 
    t_off = tau - 6. # check butterfly plot for convergence.
    sigma = 30.
    e_ss = 2. * np.sqrt(1. / (eta * tau))
    # pulse = lambda t: arctan_updown(t, e_ss, sigma, t_on, t_off)    
    # pulse = lambda t: arctan_up(t, e_ss, sigma, t_on, t_off)    
    pulse = lambda t: cnst_pulse(t, e_ss)
    #-----Simulation Prep-----#
    t_b_diV_sim = sm.Simulation(t_b_diV_app, np.linspace(0., tau, 1000), pulse)
    t_b_diV_sim.set_lindblad_spr()

    #-----Condition Number-----#
    dt = t_b_diV_sim.times[1] - t_b_diV_sim.times[0]
    id_mat = np.eye(64, dtype=cpx)
    cond_nums = [np.linalg.cond(id_mat - 0.5 * dt * lind) 
                        for lind in t_b_diV_sim.lindblad_spr]
