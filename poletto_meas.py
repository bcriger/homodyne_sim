import Simulation as sm
import Apparatus as ap
import numpy as np
import seaborn as sb
from utils import *

cnst_pulse = lambda t: np.pi

plto_dict = {'delta': [0.], 'chi': [[1.]], 'kappa': [2.],
        'gamma_1': [0.], 'gamma_phi': [0.], 'purcell': [[0.]],
        'omega':[0.], 'eta': 1., 'phi': 0.0}

plto_app = ap.Apparatus(**plto_dict)

tau = 3. 
t_on = 0.5 
t_off = tau - 0.5 # check butterfly plot for convergence.
sigma = 100.
e_ss = 4. 
pulse = lambda t: arctan_up(t, e_ss, sigma, t_on, t_off) 

plto_sim = sm.Simulation(plto_app, np.linspace(0., tau, 2000), pulse)
plto_sim.set_operators()

z_mat = all_zs(1)

rho_init = 0.5 * np.ones((4, ), cpx)

def end_cb(rho):
    z = overlap(z_mat, rho, plto_app.nq)
    f_input = np.sqrt(overlap(rho, rho_init, plto_app.nq))
    return z, f_input

plto_sim.run(rho_init, check_cb, end_cb, 100, 'poletto_test.pkl')
