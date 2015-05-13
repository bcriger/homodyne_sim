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
                'gamma_1': [0., 0., 0.], 'gamma_phi': [0., 0., 0.],
                'purcell': np.zeros((2,3)), 'omega': [0., 0., 0.],
                'eta': eta, 'phi': 0.0}

    t_b_diV_app = ap.Apparatus(**t_b_diV_dict)

    #-----Set Pulse-----#
    tau = 20./(eta * chi)
    t_on = 4. 
    t_off = tau - 4. # check butterfly plot for convergence.
    sigma = 20.
    e_ss = 2. * np.sqrt(1. / (eta * tau))
    pulse = lambda t: arctan_input(t, e_ss, sigma, t_on, t_off)    
    
    #-----Simulation Prep-----#
    t_b_diV_sim = sm.Simulation(t_b_diV_app, np.linspace(0., tau, 10000), pulse)
    t_b_diV_sim.set_operators()

    #-----Callback Functions-----#
    step_cb = lambda t, rho, dW: photocurrent(t, rho, dW, t_b_diV_sim)
    zzz_mat = all_zs(3)

    plus_state = np.array([1., 0., 0., 1., 0., 1., 1., 0.])/2.
    minus_state = np.array([0., 1., 1., 0., 1., 0., 0., 1.])/2.
    plus_mat = np.outer(plus_state, plus_state)
    minus_mat = np.outer(minus_state, minus_state)
    rho_init = 0.125 * np.ones((8,8), np.complex128)
    
    def end_cb(rho):
        zzz = overlap(zzz_mat, rho)
        f_plus = fidelity(rho, plus_mat)
        f_minus = fidelity(rho, minus_mat)
        f_input = fidelity(rho, rho_init)
        return zzz, f_plus, f_minus, f_input
    

    #cProfile.run("t_b_diV_sim.run(rho_init, step_cb, end_cb, 10, 'test.pkl')")
    t_b_diV_sim.run(rho_init, check_cb, end_cb, 100, 'test_100.pkl')
