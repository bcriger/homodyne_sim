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
    t_b_diV_sim.set_operators()

    #-----Callback Functions-----#
    step_cb = lambda t, rho, dW: photocurrent(t, rho, dW, t_b_diV_sim)
    zzz_mat = all_zs(3)

    plus_state = np.array([1., 0., 0., 1., 0., 1., 1., 0.])/2.
    minus_state = np.array([0., 1., 1., 0., 1., 0., 0., 1.])/2.
    plus_mat = np.outer(plus_state, plus_state)
    minus_mat = np.outer(minus_state, minus_state)
    rho_init = 0.125 * np.ones((64, ), np.complex128)
    
    def end_cb(rho):
        zzz = overlap(zzz_mat, rho, t_b_diV_app.nq)
        f_plus = np.sqrt(overlap(rho, plus_mat, t_b_diV_app.nq))
        f_minus = np.sqrt(overlap(rho, minus_mat, t_b_diV_app.nq))
        f_input = np.sqrt(overlap(rho, rho_init, t_b_diV_app.nq))
        return zzz, f_plus, f_minus, f_input
    

    #cProfile.run("t_b_diV_sim.run(rho_init, step_cb, end_cb, 10, 'test.pkl')")
    t_b_diV_sim.run(rho_init, check_cb, end_cb, 20, 'test.pkl', check_herm=False)
