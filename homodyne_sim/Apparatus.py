from __future__ import absolute_import

from sys import version_info
PY2 = version_info.major == 2
PY3 = version_info.major == 3

if PY2:
    import utils as ut
elif PY3:
    from . import utils as ut
else:
    raise RuntimeError("Version of Python not recognized: {}".format(version_info))

import numpy as np
from numpy.linalg import inv
import itertools as it
from scipy.signal import lti

__all__ = ['Apparatus', '_drift_h', '_jump_op_lst', 
            '_qubit_damping_ops', '_qubit_dephasing_ops', 
            '_purcell_ops']

class Apparatus(object):

    """
    Pseudo-container class for simulation parameters related to the 
    simulated device (i.e., independent of time, pulse). Contains 
    values of:
     + omega: uncorrected qubit frequencies  
     + delta: uncorrected cavity offset frequencies  
     + chi: cavity-qubit coupling frequencies  
     + kappa: cavity damping rates  
     + gamma_1: intrinsic qubit damping rates  
     + gamma_phi: intrinsic qubit dephasing rates  
     + purcell: cavity-qubit purcell factors  
     + eta: homodyne efficiency
     + phi: homodyne phase

    Important Note: It is assumed that all qubits are far-detuned, 
    leading to the elimination of the first-order qubit-qubit coupling
    Hamiltonian term and the reduction of the Purcell dissipator to 
    independent T_1 on each qubit, as derived in Criger/DiVincenzo.
    """

    def __init__(self, omega, delta, chi, kappa, gamma_1, 
                gamma_phi, purcell, eta, phi):
        arg_dict = {'delta': delta, 'chi': chi, 'kappa': kappa,
                    'omega': omega, 'gamma_1': gamma_1, 
                    'gamma_phi': gamma_phi, 'purcell': purcell}

        for key in arg_dict.keys():
            try:
                arg_dict[key] = np.array(arg_dict[key], dtype=ut.cpx)
            except Exception as e:
                pre_str = "Error while casting "+\
                            "{} to array:\n".format(key)
                e.message = pre_str + e.message
                raise e
        
        
        #Take delta to define the number of modes
        self.nm = arg_dict['delta'].shape[0]
        
        #Take omega to define the number of qubits
        self.nq = arg_dict['omega'].shape[0]
        self.ns = 2**self.nq #number of states FIXME USE PROPERTY
        
        #----Test shapes of other arguments to ensure consistency----#
        qubit_keys = ['gamma_1', 'gamma_phi']
        for key in qubit_keys:
            test_shape = arg_dict[key].shape
            if test_shape != (self.nq, ):
                raise ValueError("Number of qubits inconsistent, "
                    "{} has shape {} (should be ({}, )).".format(
                        key, test_shape, self.nq))

        kappa_shape = arg_dict['kappa'].shape
        if kappa_shape != (self.nm, ): 
            raise ValueError("Number of cavity modes inconsistent, "
                "kappa has shape {} (should be ({}, ))".format(
                    kappa_shape, self.nm))

        coupling_keys = ['chi', 'purcell']
        coupling_shape = (self.nm, self.nq)
        for key in coupling_keys:
            test_shape = arg_dict[key].shape
            if test_shape != coupling_shape:
                raise ValueError("Shapes inconsistent, "
                    "{}.shape = {} (should be {})".format(
                        key, test_shape, coupling_shape))
        
        #If these tests pass, we can set arrays:
        self._delta = arg_dict['delta']
        self._chi = arg_dict['chi']
        self._kappa = arg_dict['kappa']
        self._omega = arg_dict['omega']
        self._gamma_1 = arg_dict['gamma_1'] 
        self._gamma_phi = arg_dict['gamma_phi']
        self._purcell = arg_dict['purcell']
        
        #-----test and set scalars-----#
        scalar_dict = {'phi': phi, 'eta': eta}
        for key in scalar_dict.keys():
            test_type = type(scalar_dict[key])
            if not np.issubdtype(test_type, np.float):
                raise TypeError("Argument {} must be float".format(key)+
                    ", got {}".format(test_type))

        self.eta = eta
        self.phi = phi

        #-----derived properties-----#
        self.drift_hamiltonian = _drift_h(self)
        self.jump_ops = _jump_op_lst(self)

    #convenience methods
    def sizes(self):
        return self.nq, self.nm, self.ns

    def cav_params(self):
        return self.delta, self.kappa, self.chi

    @property
    def omega(self):
        return self._omega
    @omega.setter #FIXME: Setter can result in unchecked bad dimensions
    def omega(self, new_val):
        self._omega = new_val
        self.drift_hamiltonian = _drift_h(self)
    
    @property
    def chi(self):
        return self._chi
    @chi.setter #FIXME: Setter can result in unchecked bad dimensions
    def chi(self, new_val):
        self._chi = new_val
        self.drift_hamiltonian = _drift_h(self)

    @property
    def delta(self):
        return self._delta

    @property
    def kappa(self):
        return self._kappa
    @kappa.setter #FIXME: Setter can result in unchecked bad dimensions
    def kappa(self, new_val):
        self._kappa = new_val
        self.jump_ops = _jump_op_lst(self)

    @property
    def gamma_1(self):
        return self._gamma_1
    @gamma_1.setter #FIXME: Setter can result in unchecked bad dimensions
    def gamma_1(self, new_val):
        self._gamma_1 = new_val
        self.jump_ops = _jump_op_lst(self)

    @property
    def gamma_phi(self):
        return self._gamma_phi

    @gamma_phi.setter #FIXME: Setter can result in unchecked bad dimensions
    def gamma_phi(self, new_val):
        self._gamma_phi = new_val
        self.jump_ops = _jump_op_lst(self)

    @property
    def purcell(self):
        return self._purcell

    @purcell.setter #FIXME: Setter can result in unchecked bad dimensions
    def purcell(self, new_val):
        self._purcell = new_val
        self.jump_ops = _jump_op_lst(self)

    def cavity_lti(self, sys_type='real', reg_idx=None):
        """
        Produces a set of matrices (A, B, C, D) which are familiar
        to people who use scipy.signal.lti. They have all real entries,
        so they're twice as large as the corresponding complex matrices.
        """
        sys_type = sys_type.lower()
        sys_t_lst = ['real', 'complex']
        if sys_type not in sys_t_lst:
            raise ValueError("sys_type is not in {}".format(sys_t_lst))

        nq, nm, ns = self.sizes()
        delta, kappa, chi = self.cav_params()
        if reg_idx is None:
            if sys_type == 'real':
                vec_l = 2 * nm * ns
                A = np.zeros((vec_l, vec_l), dtype=ut.flt)
                B = np.zeros((vec_l, 1), dtype=ut.flt)
                C = np.eye(vec_l, dtype=ut.flt)
                D = np.zeros((vec_l, 1), dtype=ut.flt)
                #A gets a damping term wherever i = i'
                
                for i in range(ns):
                    for k, kp in it.product(range(nm), repeat=2):
                        rdx = ns * k + i
                        cdx = ns * kp + i

                        dmp_term = -0.5 * np.sqrt(kappa[k] * kappa[kp]).real
                        rot_term = delta[k] + sum([ ut.bt_sn(i, l, nq) * chi[k, l]
                                                    for l in range(nq) ])
                        rot_term = rot_term.real
                        A[2 * rdx,     2 * cdx]     += dmp_term 
                        A[2 * rdx + 1, 2 * cdx + 1] += dmp_term
                        #The height of sloth
                        if k == kp:
                            A[2 * rdx, 2 * cdx + 1] += rot_term
                            A[2 * rdx + 1, 2 * cdx] -= rot_term
                
                for k, i in it.product(range(nm), range(ns)):
                    idx = ns * k + i
                    B[2 * idx + 1] = -np.sqrt(kappa[k]).real
            
            elif sys_type == 'complex':
                vec_l = nm * ns
                A = np.zeros((vec_l, vec_l), dtype=ut.cpx)
                B = np.zeros((vec_l, 1), dtype=ut.cpx)
                C = np.eye(vec_l, dtype=ut.cpx)
                D = np.zeros((vec_l, 1), dtype=ut.cpx)
                #A gets a damping term wherever i = i'
                
                for i in range(ns):
                    for k, kp in it.product(range(nm), repeat=2):
                        rdx = ns * k + i
                        cdx = ns * kp + i
                        A[rdx, cdx] -= 0.5 * np.sqrt(kappa[k] * kappa[kp])
                        #The height of sloth
                        if k == kp:
                            rot_term = -1j * delta[k]
                            rot_term -= 1j * sum([ ut.bt_sn(i, l, nq) * chi[k, l]
                                                     for l in range(nq) ])
                            A[rdx, cdx] += rot_term
                
                for k, i in it.product(range(nm), range(ns)):
                    idx = ns * k + i
                    B[idx] = -1j * np.sqrt(kappa[k])
        else:
            i = reg_idx
            if sys_type == 'real':
                vec_l = 2 * nm
                A = np.zeros((vec_l, vec_l), dtype=ut.flt)
                B = np.zeros((vec_l, 1), dtype=ut.flt)
                C = np.eye(vec_l, dtype=ut.flt)
                D = np.zeros((vec_l, 1), dtype=ut.flt)
                
                for k, kp in it.product(range(nm), repeat=2):
                    rdx = k
                    cdx = kp

                    dmp_term = -0.5 * np.sqrt(kappa[k] * kappa[kp]).real
                    rot_term = delta[k] + sum([ ut.bt_sn(i, l, nq) * chi[k, l]
                                                for l in range(nq) ])
                    rot_term = rot_term.real
                    A[2 * rdx,     2 * cdx]     += dmp_term 
                    A[2 * rdx + 1, 2 * cdx + 1] += dmp_term
                    #The height of sloth
                    if k == kp:
                        A[2 * rdx, 2 * cdx + 1] += rot_term
                        A[2 * rdx + 1, 2 * cdx] -= rot_term
                
                for k in range(nm):
                    idx = k
                    B[2 * idx + 1] = -np.sqrt(kappa[k]).real
            
            elif sys_type == 'complex':
                vec_l = nm
                A = np.zeros((vec_l, vec_l), dtype=ut.cpx)
                B = np.zeros((vec_l, 1), dtype=ut.cpx)
                C = np.eye(vec_l, dtype=ut.cpx)
                D = np.zeros((vec_l, 1), dtype=ut.cpx)
                #A gets a damping term wherever i = i'
                
                for k, kp in it.product(range(nm), repeat=2):
                    rdx = k 
                    cdx = kp
                    A[rdx, cdx] -= 0.5 * np.sqrt(kappa[k] * kappa[kp])
                    #The height of sloth
                    if k == kp:
                        rot_term = -1j * delta[k]
                        rot_term -= 1j * sum([ ut.bt_sn(i, l, nq) * chi[k, l]
                                                 for l in range(nq) ])
                        A[rdx, cdx] += rot_term
                
                for k in range(nm):
                    idx = k
                    B[idx] = -1j * np.sqrt(kappa[k])

        return A, B, C, D
    
    def steady_states(self, e_ss=1.):
        """
        Returns an array of cavity mode amplitudes corresponding to the
        steady states under a constant driving function. As usual, a
        two-dimensional array is used, with the zeroth index 
        representing the cavity mode, and the first representing the 
        register value. 
        """
        A, B, C, _ = self.cavity_lti(sys_type='complex')
        return reduce(np.dot, [C, -inv(A), B]) * e_ss
        
def _drift_h(app):
    """
    returns the time-independent Hamiltonian.
    """
    hamiltonian = np.zeros((app.ns, app.ns), ut.cpx)
    # In the rotating frame, this is always 0.
    # for l in range(app.nq):
    #     lamb_shift = sum([app.chi[k, l] for k in range(app.nm)])
    #     sigma_z_l = ut.single_op(ut.sigma_z, l, app.nq)
    #     hamiltonian += 0.5 * (app.omega[l] + lamb_shift) * sigma_z_l
    return hamiltonian 

def _jump_op_lst(app):
    """
    In the rotating frame, the Purcell rate just adds to the qubit 
    damping.
    """
    return tuple(it.chain.from_iterable([_qubit_damping_ops(app),
                                        _qubit_dephasing_ops(app)]))
def _qubit_damping_ops(app):
    """
    returns a tuple of the time-independent qubit amplitude damping
    operators.
    """
    op_lst = []
    for l in range(app.nq):
        sigma_m_l = ut.single_op(ut.sigma_m, l, app.nq)
        damp_rate = app.gamma_1[l]
        #purcell contribution in rotating frame
        damp_rate += np.dot(np.sqrt(app.kappa), app.purcell[:, l]) ** 2
        op_lst.append(np.sqrt(damp_rate) * sigma_m_l)
    return op_lst

def _qubit_dephasing_ops(app):
    """
    returns a tuple of the time-independent qubit dephasing operators.
    """
    op_lst = []
    for l in range(app.nq):
        sigma_z_l = ut.single_op(ut.sigma_z, l, app.nq)
        op_lst.append(np.sqrt(0.5 * app.gamma_phi[l]) * sigma_z_l)
    return op_lst

def _purcell_ops(app):
    """
    Returns the Purcell damping operators (different from the regular 
    damping operators). All qubits damp out into common modes.
    
    Non-rotating frame; deprecated.
    """
    op_lst = []
    # for k in range(app.nm):
    #     op = np.zeros((app.ns, app.ns), ut.cpx)
    #     for l in range(app.nq):
    #         sigma_m_l = ut.single_op(ut.sigma_m, l, app.nq)
    #         op += app.purcell[k, l] * sigma_m_l
    #     op *= np.sqrt(app.kappa[k])
    #     op_lst.append(op)
    return op_lst

