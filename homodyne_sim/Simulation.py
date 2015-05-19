import numpy as np 
from scipy.linalg import inv
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
from sde_solve import platen_15_step

__all__ = ['Simulation', '_platen_15_rho_step', '_non_lin_meas']

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
        self.lindblad_spr = None
        self.lin_meas_spr = None
        
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

        alpha_0 = np.zeros((nm * ns,), dtype=ut.cpx)
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
        Prepares a (1+2)-dimensional array which stores explicit 
        measurement operators.
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

    def set_lindblad_spr(self):
        self.set_coupling_lindbladian()

        _, _, ns, nt = self.sizes()
        self.lindblad_spr = np.zeros((nt, ns**2, ns**2), dtype=ut.cpx)

        drift_h = self.apparatus.drift_hamiltonian
        id_mat = np.eye(ns, dtype=ut.cpx)
        big_drift = np.zeros((ns**2, ns**2), dtype=ut.cpx)
        
        big_drift += -1j * (np.kron(id_mat, drift_h) - 
                            np.kron(drift_h.transpose(), id_mat))
        
        for op in self.apparatus.jump_ops:
            big_drift += np.kron(op.conj(), op)
            big_drift -= 0.5 * np.kron(id_mat, np.dot(op.conj().transpose(), op))
            big_drift -= 0.5 * np.kron(np.dot(op.transpose(), op.conj()), id_mat)
        
        for tdx, elem in enumerate(self.coupling_lindbladian):
            self.lindblad_spr[tdx, :, :] = big_drift.copy()
            for ddx, value in enumerate(ut.mat2vec(elem).flatten()):
                self.lindblad_spr[tdx, ddx, ddx] += value
        pass

    def set_lin_meas_spr(self):
        """
        Calculates the linear measurement superoperator
        (I kron c + c^* kron I) corresponding to the linear diffusion 
        term (c rho + rho c^+). In order to apply this, you'll have to 
        calculate the operator trace of the multiplication result, then
        multiply by rho and subtract.
        """
        self.set_measurement()
        _, _, ns, nt = self.sizes()
        self.lin_meas_spr = np.zeros((nt, ns**2, ns**2), dtype=ut.cpx)
        id_mat = np.eye(ns, dtype=ut.cpx)
        
        for tdx in range(nt):
            c = self.measurement[tdx, :, :]
            self.lin_meas_spr[tdx, :, :] = np.kron(id_mat, c) + np.kron(c.conj(), id_mat)
        
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

    def run(self, rho_init, step_fn, final_fn, n_runs, flnm=None, check_herm=False):
        nq, nm, ns, nt = self.sizes()
        final_results = []
        step_results = []
        rho_is_vec = (len(rho_init.shape) == 1)
        
        if rho_is_vec:
            self.set_lindblad_spr()
            self.set_lin_meas_spr()

        self.set_operators()
        
        for run in range(n_runs):
            rho = np.copy(rho_init)
            dWs = np.sqrt(self.times[1] - self.times[0]) * np.random.randn(nt)
            step_result = [step_fn(self.times[0], rho, dWs[0])]
            
            for tdx in range(1, nt):
                rho = _platen_15_rho_step(self, tdx, rho, dWs[tdx], 
                                            rho_is_vec=rho_is_vec,
                                            check_herm=check_herm)
                
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

def _platen_15_rho_step(sim, tdx, rho, dW, copy=True, rho_is_vec=True,
                    check_herm=False):
    
    if not(rho_is_vec):
        raise NotImplementedError("rho_is_vec must be True "
                                    "for _platen_15_step")
    
    dt = sim.times[1] - sim.times[0]
    rho_c = rho.copy() if copy else rho

    #Ito Integrals
    u_1, u_2 = dW/np.sqrt(dt), np.random.randn()
    I_10  = 0.5 * dt**1.5 * (u_1 + u_2/np.sqrt(3.)) 
    I_00  = 0.5 * dt**2 
    I_01  = dW * dt - I_10 
    I_11  = 0.5 * (dW**2 - dt) 
    I_111 = 0.5 * (dW**2/3. - dt) * dW 
    #Evaluations of DE functions
    det_v  = np.dot(sim.lindblad_spr[tdx,:,:], rho_c) #det_f(t, rho)
    stoc_v = _non_lin_meas(sim.lin_meas_spr[tdx,:,:], rho_c) #stoc_f(t, rho) 
    #Nasty hack to avoid last timestep error
    try:
        det_vp = np.dot(sim.lindblad_spr[tdx + 1,:,:], rho_c) #det_f(t + dt, rho)
        stoc_vp = _non_lin_meas(sim.lin_meas_spr[tdx + 1,:,:], rho_c) #stoc_f(t + dt, rho)
    except IndexError:
        det_vp = np.dot(sim.lindblad_spr[tdx,:,:], rho_c) #det_f(t + dt, rho)
        stoc_vp = _non_lin_meas(sim.lin_meas_spr[tdx,:,:], rho_c) #stoc_f(t + dt, rho)
    
    #Supporting Values
    u_p = rho_c + det_v * dt + stoc_v * np.sqrt(dt)
    u_m = rho_c + det_v * dt - stoc_v * np.sqrt(dt)
    det_u_p = np.dot(sim.lindblad_spr[tdx,:,:], u_p) #det_f(t, u_p)
    det_u_m = np.dot(sim.lindblad_spr[tdx,:,:], u_m) #det_f(t, u_m)
    stoc_u_p = _non_lin_meas(sim.lin_meas_spr[tdx,:,:], u_p) #stoc_f(t, u_p)
    stoc_u_m = _non_lin_meas(sim.lin_meas_spr[tdx,:,:], u_m) #stoc_f(t, u_m)
    phi_p = u_p + stoc_u_p * np.sqrt(dt)
    phi_m = u_p - stoc_u_p * np.sqrt(dt)
    stoc_phi_p = _non_lin_meas(sim.lin_meas_spr[tdx,:,:], phi_p)
    stoc_phi_m = _non_lin_meas(sim.lin_meas_spr[tdx,:,:], phi_m)
    #Euler term
    rho_c += det_v * dt + stoc_v * dW 
    #1/(2 * np.sqrt(dt)) term
    rho_c += ((det_u_p - det_u_m) * I_10 +
             (stoc_u_p - stoc_u_m) * I_11) / (2. * np.sqrt(dt)) 
    #1/dt term
    rho_c += ((det_vp - det_v) * I_00 +
             (stoc_vp - stoc_v) * I_01) / dt 
    #first 1/(2 dt) term
    rho_c += ((det_u_p - 2. * det_v + det_u_m) * I_00
             + (stoc_u_p - 2. * stoc_v + stoc_u_m) * I_01) / (2. * dt) 
    #second 1/(2 dt) term
    rho_c += (stoc_phi_p - stoc_phi_m 
                - stoc_u_p + stoc_u_m) * I_111 / (2. * dt) 

    return rho_c

def _non_lin_meas(lin_meas, rho):
    temp_vec = np.dot(lin_meas, rho)
    return temp_vec - ut.op_trace(temp_vec) * rho