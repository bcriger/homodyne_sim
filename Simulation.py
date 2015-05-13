import numpy as np 
from numpy.random import randn
from sde_solve import platen_15_step
from Apparatus import Apparatus
from types import FunctionType
from odeintw import odeintw
from scipy.integrate import odeint
from itertools import product
import seaborn as sb
import cPickle as pkl
from os import getcwd
import ipy_progressbar as pb
from utils import *

class Simulation(object):
    """
    Handles homodyne measurement simulations.
    """
    def __init__(self, apparatus, times, pulse_fn):
        
        if not isinstance(apparatus, Apparatus):
            raise TypeError("input 'apparatus' must be an Apparatus"
                " object (see Apparatus.py). ")
        
        self.apparatus = apparatus
        self.times = np.array(times, dtype=np.float64)
        
        if not(type(pulse_fn) is FunctionType):
            raise TypeError("pulse_fn must be a function.")
        
        self.pulse_fn = pulse_fn

        self.amplitudes   = None
        self.lindbladian  = None
        self.measurement  = None
        
    def sizes(self):
        nq, nm = self.apparatus.nq, self.apparatus.nm
        ns = 2 ** nq
        nt = len(self.times)
        return nq, nm, ns, nt

    def set_amplitudes(self):
        """
        Formulates and solves deterministic DEs governing coherent 
        states in cavity modes.
        """
        nq, nm, ns, nt = self.sizes()

        #Use attributes of self.apparatus to determine d_alpha_dt
        def _d_alpha_dt(alpha, t):
            
            temp_mat = alpha.reshape((nm, ns))
            kappa = self.apparatus.kappa
            delta = self.apparatus.delta
            chi = self.apparatus.chi
            alpha_dot = np.zeros(temp_mat.shape, temp_mat.dtype)
            
            for k, i in product(range(nm), range(ns)):
                #drift term
                alpha_dot[k, i] = -1j * delta[k] * temp_mat[k, i]
                #qubit coupling term
                alpha_dot[k, i] += -1j * (sum([bt_sn(i, l) * chi[k, l]
                                               for l in range(nq)])
                                                * temp_mat[k, i])
                #driving term
                alpha_dot[k, i] += -1j * (np.sqrt(kappa[k]) *
                                             self.pulse_fn(t)) 
                # alpha_dot[k, i] += -1j * self.pulse_fn(t) 
                #damping term
                for kp in range(nm):
                    alpha_dot[k, i] -= 0.5 * np.sqrt(kappa[k] * kappa[kp]) * temp_mat[kp, i]    
            
            return alpha_dot.reshape(alpha.shape)

        alpha_0 = np.zeros((nm * ns,), np.complex128)
        self.amplitudes = odeintw(_d_alpha_dt, alpha_0, self.times).reshape((nt, nm, ns))
    
    def butterfly_plot(self, *args, **kwargs):
        """
        plots the `alpha_out`s corresponding to the conditional 
        amplitudes of the cavity coherent states depending on the 
        computational basis state of the register. Passes additional 
        arguments/keyword arguments straight to the plotting function.
        """
        if self.amplitudes is None:
            self.set_amplitudes()
        alpha = self.amplitudes
        
        nq, nm, ns, nt = self.sizes()
        kappa = self.apparatus.kappa

        alpha_outs = np.zeros((nt, ns), np.complex128)
        for tdx, k, i in product(range(nt), range(nm), range(ns)):
            alpha_outs[tdx, i] += np.sqrt(kappa[k]) * alpha[tdx, k, i]

        for i in range(ns):
            sb.plt.plot(alpha_outs[:,i].real, alpha_outs[:,i].imag,
                        *args, **kwargs)
        
        sb.plt.show()
        pass
    
    def set_lindbladian(self, addl_ti=None, addl_td=None):
        """
        Computes a 2+1 dimensional array of supermatrices, 
        incorporating:
         + The lindbladian resulting from the parameters of the 
           apparatus (drift hamiltonian, T_1, T_2, Purcell)
         + The coupling Lindbladian resulting from interaction with the
           cavity
         + An optional additional time-independent lindbladian
         + An optional additional time-dependent lindbladian
        """
        nq, nm, ns, nt = self.sizes()
        chi, omega = self.apparatus.chi, self.apparatus.omega
        g_1, g_p = self.apparatus.gamma_1, self.apparatus.gamma_phi
        purcell = self.apparatus.purcell

        if self.amplitudes is None:
            self.set_amplitudes()
        alpha = self.amplitudes
             
        self.lindbladian = np.zeros((ns**2, ns**2, nt), np.complex128)
        
        drift_lind = np.zeros((ns**2, ns**2), np.complex128)
        #Drift Hamiltonian
        drift_lind += sum([z_ham(omega[q], q, nq)
                                    for q in range(nq)])
        #T_1
        drift_lind += sum([gamma_1_lind(g_1[q], q, nq)
                                    for q in range(nq)])
        #Purcell
        drift_lind += sum([gamma_1_lind(purcell[q], q, nq)
                                    for q in range(nq)])
        #T_2
        drift_lind += sum([gamma_2_lind(g_p[q], q, nq)
                                    for q in range(nq)])
        
        if addl_ti:
            drift_lind += addl_ti

        for tdx in range(nt):
            self.lindbladian[:, :, tdx] = drift_lind
            if addl_td:
                self.lindbladian[:, :, tdx] += addl_td[:, :, tdx]
            cpl_l = np.zeros((ns, ns), dtype=np.complex128)
            for i, j in product(range(ns), repeat=2):
                '''
                ij = int_cat(i, j, nq)
                for k, l in product(range(nm), range(nq)):
                    self.lindbladian[ij, ij, tdx] += -1j * chi[k, l] * \
                        np.conj(alpha[tdx, k, i]) * alpha[tdx, k, j] * \
                        (bt_sn(j, l) - bt_sn(i, l))
                '''
                for k, l in product(range(nm), range(nq)):
                    cpl_l[i, j] += -1j * chi[k, l] * \
                        np.conj(alpha[tdx, k, j]) * alpha[tdx, k, i] * \
                        (bt_sn(i, l) - bt_sn(j, l))
            
            for i, j in product(range(ns), repeat=2):
                ij = int_cat(i, j, nq)
                self.lindbladian[ij, ij, tdx] += cpl_l[j, i]
        pass

    def set_measurement(self):
        """
        Prepares diag( A \otimes \id + \id \otimes A.conj() ), the 
        relevant array in determining the measurement contribution to 
        d rho. Storing this array means not having to perform Kronecker
        products in the main loop.
        """
        nq, nm, ns, nt = self.sizes()

        if self.amplitudes is None:
            self.set_amplitudes()
        alpha = self.amplitudes

        eta, phi = self.apparatus.eta, self.apparatus.phi
        kappa = self.apparatus.kappa

        self.measurement = np.zeros((ns, ns, nt), np.complex128)
        '''
        for i, j in product(range(ns), range(ns)):
            ij = int_cat(i, j, nq)
            for tdx in range(nt):
                elem = sum([np.sqrt(kappa[k]) * 
                            (np.exp(-1j * phi) * alpha[tdx, k, i] + 
                            np.exp(1j * phi) * alpha[tdx, k, j].conj())
                             for k in range(nm)])
                self.measurement[ij, tdx] += elem
        '''
        #form matrix and take diagonal like an animal
        for tdx in range(nt):
            c = np.zeros((ns, ns), dtype=np.complex128)
            for j in range(ns):
                c[j, j] = sum([np.sqrt(kappa[k]) * alpha[tdx, k, j]
                            for k in range(nm)])
            self.measurement[:, :, tdx] = c
        self.measurement *= np.sqrt(eta) * np.exp(-1j * phi)
        pass


    def set_operators(self):
        """
        Uses the saved coherent state amplitudes to formulate a 
        Lindbladian and measurement operator determining the SME.
        """
        if self.lindbladian is None:
            self.set_lindbladian()
        if self.measurement is None:
            self.set_measurement()
        pass

    def run(self, rho_vec_init, step_fn, final_fn, n_runs, flnm=None, pbar=True):
        self.set_operators()
        nq, nm, ns, nt = self.sizes()
        final_results = []
        step_results = []
        dt = self.times[1] - self.times[0]
        
        if pbar:
            #run_iter = pb.ProgressBar(range(run_iter), widgets = [pb.AdaptiveETA()])
            run_iter = range(n_runs)
        else:
            run_iter = range(n_runs)

        for run in run_iter:
            rho = np.copy(rho_vec_init)
            dWs = np.sqrt(dt) * randn(nt)
            step_result = [step_fn(self.times[0], rho, dWs[0])]
            
            for tdx in range(1, nt):
                #euler-maruyama step
                d_rho = np.zeros(rho.shape, rho.dtype)
                d_rho = dt * np.dot(self.lindbladian[:, :, tdx], rho)
                # d_rho += dWs[tdx] * m_c_rho(self.measurement[:, tdx], rho)
                d_rho += dWs[tdx] * m_c_rho_op(self.measurement[:, :, tdx], rho)
                rho += d_rho
                
                #callback
                if step_fn is not None:
                    step_result.append(step_fn(self.times[tdx], rho.copy(), dWs[tdx]))
                
            step_results.append(step_result)
            if final_fn is not None:
                final_results.append(final_fn(rho.copy()))
        
        if flnm is None:
            return final_results, step_results
        else:
            sim_dict = {'apparatus': self.apparatus,
                        'times': self.times,
                        'pulse_shape': self.pulse_fn(self.times),
                        'final_results': final_results,
                        'step_results': step_results}
            with open('/'.join([getcwd(),flnm]), 'w') as phil:
                pkl.dump(sim_dict, phil)
