import Simulation as sm
import Apparatus as ap
import numpy as np
import seaborn as sb

cnst_pulse = lambda t: np.pi

plto_dict = {'delta': [0.], 'chi': [[ np.pi]], 'kappa': [2. * np.pi],
            'gamma_1': [0.], 'gamma_phi': [0.], 'purcell': [0.],
            'omega':[0.], 'eta': 1., 'phi': 0.0}

plto_app = ap.Apparatus(**plto_dict)

plto_sim = sm.Simulation(plto_app, np.linspace(0., np.pi, 5000), cnst_pulse)
plto_sim.set_operators()
#print 0.5 * np.ones((4,1), np.complex128)
for _ in range(50):
    _, rho_mats = plto_sim.run(0.5 * np.ones((4,), np.complex128),
                                lambda t, rho, dW: rho, lambda rho: None, 1)
    rho_vecs = rho_mats[0]
    rho_mats = [np.array(arr).reshape((2,2)).T for arr in rho_vecs]
    purities = [np.trace(np.dot(rho, rho)) for rho in rho_mats]
    sb.plt.plot(purities)
sb.plt.show()