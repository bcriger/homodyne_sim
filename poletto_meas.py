import Simulation as sm
import Apparatus as ap
import numpy as np
import seaborn as sb
from utils import *

cnst_pulse = lambda t: np.pi

plto_dict = {'delta': [0.], 'chi': [[10. * np.pi]], 'kappa': [20. * np.pi],
            'gamma_1': [0.], 'gamma_phi': [0.], 'purcell': [[0.]],
            'omega':[0.], 'eta': 1., 'phi': 0.0}

plto_app = ap.Apparatus(**plto_dict)

tau = 0.3 
t_on = 0.05 
t_off = tau - 0.05 # check butterfly plot for convergence.
sigma = 100.
e_ss = 40. * np.pi
pulse = lambda t: arctan_up(t, e_ss, sigma, t_on, t_off) 

plto_sim = sm.Simulation(plto_app, np.linspace(0., 0.3, 2000), pulse)
plto_sim.set_operators()

_, rho_mats = plto_sim.run(0.5 * np.ones((4,), np.complex128),
                            lambda t, rho, dW: rho, lambda rho: None, 100)
rho_vecs = rho_mats
rho_mats = [[np.array(arr).reshape((2,2)).T for arr in traj] for traj in rho_vecs]
purities = [[np.trace(np.dot(rho, rho)) for rho in traj] for traj in rho_mats]
