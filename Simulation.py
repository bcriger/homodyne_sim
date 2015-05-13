import numpy as np 
from Apparatus import Apparatus
from types import FunctionType
from odeintw import odeintw
from scipy.integrate import odeint
from itertools import product
import seaborn as sb
import cPickle as pkl
from os import getcwd
import ipy_progressbar as pb
import utils as ut

class Simulation(object):
    """
    Handles homodyne measurement simulations. Stores the time-dependent
    parameters (times, pulse function, cavity coherent state 
    amplitudes, coupling Lindbladian, measurement operator)
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
        self.coupling_lindbladian = None
        self.measurement  = None
        
    def sizes(self):
        nq, nm = self.apparatus.nq, self.apparatus.nm
        ns = self.apparatus.ns
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
                alpha_dot[k, i] += -1j * (sum([ut.bt_sn(i, l, nq) * chi[k, l]
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

        alpha_0 = np.zeros((nm * ns,), ut.cpx)
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

        alpha_outs = np.zeros((nt, ns), ut.cpx)
        for tdx, k, i in product(range(nt), range(nm), range(ns)):
            alpha_outs[tdx, i] += np.sqrt(kappa[k]) * alpha[tdx, k, i]

        for i in range(ns):
            sb.plt.plot(alpha_outs[:,i].real, alpha_outs[:,i].imag,
                        *args, **kwargs)
        
        sb.plt.show()
        pass
    
    #TODO Refactor to use properties
    def set_coupling_lindbladian(self):
        """
        Computes a (1+2)-dimensional array, each slice across the last
        two dimensions containing the values of the lindbladian to be
        multiplied elementwise onto rho.
        """
        nq, nm, ns, nt = self.sizes()
        chi = self.apparatus.chi
        
        if self.amplitudes is None:
            self.set_amplitudes()
        alpha = self.amplitudes
             
        self.coupling_lindbladian = np.zeros((nt, ns, ns), ut.cpx)
        
        for tdx in range(nt):
            for i, j in product(range(ns), repeat=2):
                for k, l in product(range(nm), range(nq)):
                    self.coupling_lindbladian[tdx, i, j] += -1j * chi[k, l] * \
                        np.conj(alpha[tdx, k, j]) * alpha[tdx, k, i] * \
                        (ut.bt_sn(i, l, nq) - ut.bt_sn(j, l, nq))
        pass

    def set_measurement(self):
        """
        Prepares a (1+2)-dimensional array 
        """
        _, nm, ns, nt = self.sizes()

        if self.amplitudes is None:
            self.set_amplitudes()
        alpha = self.amplitudes

        eta, phi = self.apparatus.eta, self.apparatus.phi
        kappa = self.apparatus.kappa

        self.measurement = np.zeros((nt, ns, ns), ut.cpx)
        for tdx in range(nt):
            c = np.zeros((ns, ns), dtype=ut.cpx)
            for j in range(ns):
                self.measurement[tdx, j, j] = sum([
                    np.sqrt(kappa[k]) * alpha[tdx, k, j]
                            for k in range(nm)])

        self.measurement *= np.sqrt(eta) * np.exp(-1j * phi)
        pass


    def set_operators(self):
        """
        Uses the saved coherent state amplitudes to formulate a 
        Lindbladian and measurement operator determining the SME.
        """
        if self.coupling_lindbladian is None:
            self.set_coupling_lindbladian()
        if self.measurement is None:
            self.set_measurement()
        pass

    def run(self, rho_vec_init, step_fn, final_fn, n_runs, flnm=None, pbar=True):
        self.set_operators()
        nq, nm, ns, nt = self.sizes()
        final_results = []
        step_results = []
        dt = self.times[1] - self.times[0]
        drift_h = self.apparatus.drift_hamiltonian
        jump_ops = self.apparatus.jump_ops
        for run in range(n_runs):
            rho = np.copy(rho_vec_init)
            dWs = np.sqrt(dt) * np.random.randn(nt)
            step_result = [step_fn(self.times[0], rho, dWs[0])]
            
            for tdx in range(1, nt):
                #euler-maruyama step
                #TODO Break up / refactor
                d_rho = np.zeros(rho.shape, rho.dtype)
                d_rho += -1j * (np.dot(drift_h, rho) - np.dot(rho, drift_h)) * dt
                cpl_l = self.coupling_lindbladian[tdx, :, :]
                meas = self.measurement[tdx, :, :]
                meas_d = meas.conj().transpose()
                for op in jump_ops:
                    op_d = op.conj().transpose()
                    d_rho += dt * (np.dot(np.dot(op, rho), op_d)
                        - 0.5 * (np.dot(np.dot(op_d, op), rho) + 
                                    np.dot(rho, np.dot(op_d, op)) ))
                d_rho = dt * np.multiply(cpl_l, rho)
                d_rho += dWs[tdx] * ( np.dot(meas, rho) + np.dot(rho, meas_d) - 
                np.trace(np.dot(meas + meas_d, rho)) * rho )
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
