import homodyne_sim as hs
import numpy as np

try:
    import seaborn as sb
except ImportError:
    import matplotlib as sb

import cPickle as pkl

"""
This sheet replicates and saves figures from 
Gambetta/Blais/Boissoneault/Houck/Schuster/Girvin '07.
"""
units = 'MHz'
chi = 10. * np.pi
kappa = 20. * np.pi
#equation 3.5 and our definition differ by a factor of sqrt(kappa)
amp = 10. * np.pi / np.sqrt(kappa)
t_1 = 7. #microseconds, includes purcell
t_2 = 0.5 #microseconds

gamma_1 = 1. / t_1
gamma_phi = 1. / t_2 - gamma_1 / 2.

tau = 0.3 
t_on = 0.02 
sigma = 0.005
omega = 75. #MHz, result of fit-by-eye

pulse = lambda t: hs.tanh_up(t, amp, sigma, t_on) 
times = np.linspace(0., tau, 20000)

#Objects
#GBBHSG don't use an independent Purcell term, since it's identical to 
#T_1 in the Lindblad equation
gambetta_dict = {'delta': [0.], 'chi': [[chi]], 'kappa': [kappa],
        'gamma_1': [gamma_1], 'gamma_phi': [gamma_phi], 'purcell': [[0.]],
        'omega':[omega], 'eta': 1., 'phi': np.pi}

gambetta_app = hs.Apparatus(**gambetta_dict)

gambetta_sim = hs.Simulation(gambetta_app, times, pulse)
gambetta_sim.set_operators()

rho_init = 0.5 * np.ones((4, ), hs.cpx)
#Checking photocurrent to see if it integrates to Gaussian peaks
def pc_cb(t, rho, dW):
    return np.array([hs.alt_photocurrent(t, rho, dW, gambetta_sim)])

def bloch_cb(t, rho, dW=0.):
    x = hs.overlap(hs.sigma_x, rho, 1)
    y = hs.overlap(hs.sigma_y, rho, 1)
    z = hs.overlap(hs.sigma_z, rho, 1)
    p = hs.op_purity(rho)
    return np.array([x, y, z, p])

gambetta_sim.classical_sim(rho_init, bloch_cb, 'gambetta_fig_3_classical.pkl')

with open('gambetta_fig_3_classical.pkl', 'r') as phil:
    fig_3_dict = pkl.load(phil)

xs, ys, zs, ps = [[tpl[k] for tpl in fig_3_dict['step_result']]
                    for k in range(4)]

#fig. 3
#part a
sb.plt.plot(times * 1000, xs, label='x(t)')
sb.plt.plot(times * 1000, ys, label='y(t)')
sb.plt.plot(times * 1000, zs, label='z(t)')
sb.plt.xlabel('Time [ns]', fontsize=14)
sb.plt.legend()
sb.plt.savefig('gambetta_fig_3a_test.pdf')
sb.plt.show()
sb.plt.close()

#part b
sb.plt.plot(times * 1000, fig_3_dict['pulse_shape'] * np.sqrt(kappa)/(10. * np.pi), 'g--',
        label=r'Drive $\times \frac{\sqrt{\kappa}}{10\pi}$')
sb.plt.plot(times * 1000, [2 * p - 1 for p in ps], 'b-', label = r'2 $\times$ purity - 1')
sb.plt.xlabel('Time, t [ns]', fontsize=14)
sb.plt.legend()
sb.plt.savefig('gambetta_fig_3b_test.pdf')
sb.plt.show()
sb.plt.close()

#fig. 4
t_on = 0.05
amp = 2. * np.pi * np.sqrt(5. / kappa)
pulse = lambda t: hs.tanh_up(t, amp, sigma, t_on)
gambetta_sim = hs.Simulation(gambetta_app, times, pulse)
gambetta_sim.set_amplitudes()
a_e = gambetta_sim.amplitudes[:, 0, 0]
a_g = gambetta_sim.amplitudes[:, 0, 1]
gamma_d = 2 *chi* (np.multiply(a_e.conj(), a_g)).imag
gamma_m = kappa * np.abs(a_e - a_g)**2
sb.plt.plot(times * 1000, gamma_d/(2 * np.pi), label=r'$\Gamma_d(t)$ (MHz)')
sb.plt.plot(times * 1000, 0.5 * gamma_m/(2 * np.pi), label=r'$\Gamma_m(t)/2$ (MHz)')
sb.plt.legend()
sb.plt.savefig('gambetta_fig_4_test.pdf')
sb.plt.show()
sb.plt.close()

#fig. 5
t_on = 0.02
amp = 10. * np.pi / np.sqrt(kappa)
pulse = lambda t: hs.tanh_up(t, amp, sigma, t_on)
gambetta_sim = hs.Simulation(gambetta_app, times, pulse)
_, step_results = gambetta_sim.run(1, rho_init, bloch_cb)

xs, ys, zs, ps = [[tpl[k] for tpl in step_results[0]]
                    for k in range(4)]

#part a
sb.plt.plot(times * 1000, xs, label='x(t)')
sb.plt.plot(times * 1000, ys, label='y(t)')
sb.plt.plot(times * 1000, zs, label='z(t)')
sb.plt.xlabel('Time [ns]', fontsize=14)
sb.plt.legend()
sb.plt.savefig('gambetta_fig_5a_test.pdf')
sb.plt.show()
sb.plt.close()

#part b
sb.plt.plot(times * 1000, [2 * p - 1 for p in ps], 'b-', label = r'2 $\times$ purity - 1')
sb.plt.xlabel('Time, t [ns]', fontsize=14)
sb.plt.legend()
sb.plt.savefig('gambetta_fig_5b_test.pdf')
sb.plt.show()
sb.plt.close()

#But what if there was no T_1?
gambetta_dict['gamma_1'] = [0.]
gambetta_app = hs.Apparatus(**gambetta_dict)
gambetta_sim = hs.Simulation(gambetta_app, times, pulse)
gambetta_sim.set_operators()
_, step_results = gambetta_sim.run(1, rho_init, bloch_cb)

xs, ys, zs, ps = [[tpl[k] for tpl in step_results[0]]
                    for k in range(4)]

#part a
sb.plt.plot(times * 1000, xs, label='x(t)')
sb.plt.plot(times * 1000, ys, label='y(t)')
sb.plt.plot(times * 1000, zs, label='z(t)')
sb.plt.xlabel('Time [ns]', fontsize=14)
sb.plt.legend()
sb.plt.savefig('gambetta_fig_5a_no_T1_test.pdf')
sb.plt.show()
sb.plt.close()

#part b
sb.plt.plot(times * 1000, [2 * p - 1 for p in ps], 'b-', label = r'2 $\times$ purity - 1')
sb.plt.xlabel('Time, t [ns]', fontsize=14)
sb.plt.legend()
sb.plt.savefig('gambetta_fig_5b_no_T1_test.pdf')
sb.plt.show()
sb.plt.close()
