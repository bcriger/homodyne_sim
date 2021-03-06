# import pylab as pl 
import numpy as np
from numpy.random import rand
from numpy.linalg import eig, eigvalsh
import itertools as it
from scipy.linalg import sqrtm
from mpmath import gammainc
from math import factorial as fctrl
from operator import mul
from glob import glob as glob #glob
import pickle as pkl
import scipy as sp
import scipy.stats

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

__all__ = ['cpx', 'id_2', 'YY', 'sigma_z', 'sigma_m' , 'vec2mat', 
            'mat2vec', 'state2vec', 'single_op', 'arctan_updown',
            'arctan_up', 'overlap', 'op_trace', 'op_herm_dev', 'op_purity',
            'fidelity', 'photocurrent', 'concurrence', 'check_cb', 
            'all_zs', 'herm_dev', 'herm_dev_vec', 'min_max_eig', 
            'vec_purity', 'vec_tr', 'bt_sn', 'm_c_rho', 'm_c_rho_op',
            'com_mat', 'diss_mat', 'lin_meas_mat', 'gamma_1_lind', 
            'gamma_2_lind',  'z_ham', 'interquartile_range', 
            'f_d_bin_width', 'fd_bins', 'colour_hist', 'tanh_updown',
            'tanh_up', 'sigma_x', 'sigma_y', 'cnst_pulse', 
            'alt_photocurrent', 're_herm', 'hat_pulse', 'flt', 
            'rand_mat', 'rand_herm_mat', 'rand_dens_mat', 
            'rand_super_vec', 'rand_pure_state', 'pq_updown', 
            '_stepper_list', 'amplitudes_1', 'def_poly_exp_int', 
            'avg_data', 'sum_step_data', 'post_selected', 'pq_up',
            'exp_up', 'exp_updown', 'add_arrow_to_line2D', 
            'mean_confidence_interval', 'unified_photocurrent', 
            '_pq_updown']

#cpx = np.complex64
cpx = np.complex128
flt = np.float64
# cpx = np.complex256 #UNSUPPORTED IN LINALG

_stepper_list = ['ip15', 'p15', 'mem', 'mil', 'imil', 'irk1', 'its1', 
                    'its15', 'em']

"""
HINWEIS: We adopt the convention throughout that the excited state is 
the 0 state, and the ground state is the 1 state. This keeps the 
Hamiltonian proportional to Pauli Z, but implies that the other Pauli
matrices are transposed wrt the standard definition.
"""

def prod(iterable):
    return reduce(mul, iterable, 1.)

id_2 = np.eye(2, dtype=cpx)

#Computational Convention
# sigma_m = np.array([[0., 1.], [
                     # 0., 0.]], dtype=cpx)

#Physical Convention
sigma_m = np.array([[0., 0.], [
                     1., 0.]], dtype=cpx)

sigma_x = np.array([[0., 1.], [
                     1., 0.]], dtype=cpx)

#Physical Convention
sigma_y = np.array([[ 0.,  1j], [
                     -1j,  0.]], dtype=cpx)

#Computational Convention
# sigma_y = np.array([[ 0.,  -1j], [
#                      1j,  0.]], dtype=cpx)

sigma_z = np.array([[1.,  0.], [
                     0., -1.]], dtype=cpx)

YY = np.array([ [ 0., 0., 0., -1.],
                [ 0., 0., 1.,  0.],
                [ 0., 1., 0.,  0.],
                [-1., 0., 0.,  0.]], dtype=cpx)

def vec2mat(vec):
    sqrt_len = int(np.sqrt(len(vec)))
    return np.reshape(vec, (sqrt_len, sqrt_len)).transpose()

def mat2vec(mat):
    sq_len = int(mat.shape[0]**2)
    return np.reshape(mat.transpose(), (sq_len,))

def state2vec(lil_vec):
    return mat2vec(np.outer(lil_vec.conj(), lil_vec))

def single_op(mat, q, nq):
    return reduce(np.kron, [mat if l==q else id_2 for l in range(nq)])

cnst_pulse = np.vectorize(lambda t, cnst: cnst)

hat_pulse = np.vectorize(lambda t, cnst, t_on, t_off:
                            cnst if t_on < t < t_off else 0.)

def _updown_sigmas(sigma):
    assert not isinstance(sigma, basestring)
    if hasattr(sigma, '__iter__'):
        sigma_tpl = sigma
    else:
        sigma_tpl = (sigma, sigma)
    return sigma_tpl

def arctan_updown(t, e_ss, sigma, t_on, t_off):
    sigma_up, sigma_down = _updown_sigmas(sigma)
    return e_ss / np.pi * (np.arctan((t - t_on) / sigma_up) - 
                            np.arctan((t - t_off) / sigma_down ))

def arctan_up(t, e_ss, sigma, t_on):
    
    return e_ss / np.pi * (np.arctan((t - t_on) / sigma) + np.pi/2.)


def tanh_updown(t, e_ss, sigma, t_on, t_off):
    sigma_up, sigma_down = _updown_sigmas(sigma)
    return e_ss / 2. * (np.tanh((t - t_on) / sigma_up) - 
                            np.tanh((t - t_off) / sigma_down))

def tanh_up(t, e_ss, sigma, t_on):
    
    return e_ss / 2. * (np.tanh((t - t_on) / sigma) + 1.)

def _pq_updown(t, e_ss, sigma, t_on, t_off):
    """
    Evaluates a piecewise quadratic function which mimics basic pulse 
    behaviour; rising to e_ss with a switching time sigma, then 
    descending back to 0.
    """
    sigma_up, sigma_down = _updown_sigmas(sigma)
    if 0 <= t < t_on - 0.5 * sigma_up:
        eps = 0.
    elif t_on - 0.5 * sigma_up <= t < t_on:
        eps = 2. * e_ss / sigma_up**2 * (t - t_on + 0.5 * sigma_up) ** 2
    elif t_on <= t < t_on + 0.5 * sigma_up:
        eps = -2. * e_ss / sigma_up**2 * (t - t_on - 0.5 * sigma_up) ** 2 + e_ss
    elif t_on + 0.5 * sigma_up <= t < t_off - 0.5 * sigma_down:
        eps = e_ss
    elif t_off - 0.5 * sigma_down <= t < t_off:
        eps = -2. * e_ss / sigma_down**2 * (t - t_off + 0.5 * sigma_down) ** 2 + e_ss
    elif t_off <= t < t_off + 0.5 * sigma_down:
        eps = 2. * e_ss / sigma_down**2 * (t - t_off - 0.5 * sigma_down) ** 2
    elif t_off + 0.5 * sigma_down <= t:
        eps = 0.
    else:
        raise ValueError("Some kind of float gap?")

    return eps

pq_updown = np.vectorize(_pq_updown)

def _pq_up(t, e_ss, sigma, t_on):
    """
    Evaluates a piecewise quadratic function which mimics basic pulse 
    behaviour; rising to e_ss with a switching time sigma, then 
    staying at e_ss indefinitely.
    """
    if 0 <= t < t_on - 0.5 * sigma:
        eps = 0.
    elif t_on - 0.5 * sigma <= t < t_on:
        eps = 2. * e_ss / sigma**2 * (t - t_on + 0.5 * sigma) ** 2
    elif t_on <= t < t_on + 0.5 * sigma:
        eps = -2. * e_ss / sigma**2 * (t - t_on - 0.5 * sigma) ** 2 + e_ss
    elif t_on + 0.5 * sigma <= t:
        eps = e_ss
    else:
        raise ValueError("Equality test failed.")

    return eps

pq_up = np.vectorize(_pq_up)

def _exp_up(t, e_ss, sigma):
    """
    Mimics a low-pass-filtered step function by approaching e_ss 
    exponentially.
    """
    return e_ss * (1. - np.exp(-t / sigma))
   
exp_up = np.vectorize(_exp_up)

def _exp_updown(t, e_ss, sigma, t_off):
    """
    Mimics a low-pass-filtered hat function by approaching e_ss 
    exponentially, then coming back down. Poor selection of parameters
    may result in discontinuity.
    """
    sigma_up, sigma_down = _updown_sigmas(sigma)
    if t < t_off:
        return e_ss * (1. - np.exp(-t / sigma_up))
    else:
        return e_ss * (np.exp(-(t - t_off) / sigma_down))

exp_updown = np.vectorize(_exp_updown)

def overlap(a, b, nq):
    """
    Determines the overlap between two `nq`-qubit quantum 
    states/operators. Requires `nq` to determine whether the 
    states/operators are vectors, supervectors or matrices of the 
    appropriate size.
    """
    ns = 2**nq
    #Produce equivalent supervectors
    if a.shape == (ns, ):
        super_a = np.kron(a.conj(), a)
    elif a.shape == (ns, ns):
        super_a = mat2vec(a)
    else:
        super_a = a
    
    if b.shape == (ns, ):
        super_b = np.kron(b.conj(), b)
    elif b.shape == (ns, ns):
        super_b = mat2vec(b)
    else:
        super_b = b
    
    try:
        prod = np.dot(super_a.conj(), super_b)
    except ValueError as e:
        print("super_a.shape: {}".format(super_a.shape))
        print("super_b.shape: {}".format(super_b.shape))
        raise e
    return prod

def op_trace(op):
    """
    Calculates the trace of a quantum state, assuming that state is 
    either in the form of a matrix (in which case, we take the trace
    normally), or column-stacked (in which case, we iterate over the 
    big vector)
    """
    if len(op.shape) == 2:
        return np.trace(op)
    elif len(op.shape) == 1:
        sqrt_sz = int(np.sqrt(op.shape[0]))
        return sum([op[(sqrt_sz + 1) * ddx] for ddx in range(sqrt_sz)])
    else:
        raise ValueError("Shape of array must be 1d or 2d")

def op_herm_dev(op):
    """
    Calculates the deviation from hermiticity of a quantum state, 
    assuming that the state is either a density matrix or supervector.
    """
    if len(op.shape) == 2:
        return np.amax(abs(op - op.conj().transpose()))
    elif len(op.shape) == 1:
        sz = int(np.sqrt(op.shape[0]))
        return max([abs(op[sz * rdx + cdx] - op[sz * cdx + rdx].conj())
                     for rdx, cdx in it.product(range(sz), repeat=2)])
    else:
        raise ValueError("Shape of array must be 1d or 2d")


def op_purity(op):
    """
    Calculates the purity of a quantum state, 
    """
    if len(op.shape) == 2:
        return np.trace(np.dot(op.conj().transpose(), op))
    elif len(op.shape) == 1:
        return np.dot(op.conj(), op)
    else:
        raise ValueError("Shape of array must be 1d or 2d")

def fidelity(rho, sigma):
    """
    Determines the state fidelity between two density 
    operators.
    """
    sqrt_rho = sqrtm(rho)
    return np.trace(np.sqrt(reduce(np.dot, [sqrt_rho, sigma, sqrt_rho])))

def photocurrent(t, rho, dW, sim):
    """
    Given a simulation object, finds sqrt(\eta) <c + c^+> + dW by 
    translating the time t into an array index (forgive me).
    """
    tdx = np.argmin(np.abs(sim.times - t))
    dt = sim.times[1] - sim.times[0]
    c = sim.measurement[tdx, :, :]
    if len(rho.shape) == 1:
        rho_c = vec2mat(rho)
    else:
        rho_c = rho
    return np.trace(np.dot(c + c.conj().transpose(), rho_c)) * dt + dW

def alt_photocurrent(t, rho, dW, sim):
    """
    Given a simulation object, finds sqrt(\eta) <c + c^+> + dW by 
    translating the time t into an array index (forgive me).
    """
    tdx = np.argmin(np.abs(sim.times - t))
    dt = sim.times[1] - sim.times[0]
    c = sim.lin_meas_spr[tdx, :, :]
    if len(rho.shape) == 2:
        rho_c = mat2vec(rho)
    else:
        rho_c = rho
    return op_trace(np.dot(c, rho_c)) * dt + dW

def unified_photocurrent(t, rho, dW, sim, c, inc_dW=True):
    """
    Calculates the photocurrent without needing a giant array for 
    sim.measurement
    """
    dt = sim.times[1] - sim.times[0]
    if len(rho.shape) == 1:
        rho_c = vec2mat(rho)
    else:
        rho_c = rho
    meas_val = np.trace(np.dot(c + c.conj().transpose(), rho_c))
    return meas_val * dt + dW if inc_dW else meas_val * dt
         

def concurrence(rho):
    r"""
    wikipedia.org/wiki/Concurrence_%28quantum_computing%29#Definition
    """
    if rho.shape not in [(4, 4), (16, )]:
        raise ValueError("Concurrence only works for two-qubit states")
    rho_c = vec2mat(np.copy(rho)) if rho.shape == (16,) else rho
    test_mat = reduce(np.dot, [rho_c, YY, rho_c.conj(), YY])
    lmbds = list(reversed(sorted(map(np.sqrt, eig(test_mat)[0]))))
    return max(0., lmbds[0] - lmbds[1] - lmbds[2] - lmbds[3])

def check_cb(t, rho, dW):
    hm = op_herm_dev(rho)
    tr = op_trace(rho)
    pr = op_purity(rho) 
    min_e, max_e = min_max_eig(rho)
    return np.array([hm, tr, pr, min_e, max_e])

def all_zs(nq):
    """
    returns a matrix of Z^(\otimes nq) for some nq
    """
    return reduce(np.kron, [sigma_z for _ in range(nq)])

int_cat = lambda i, j, nq: (i << nq) + j

def herm_dev(mat):
    return np.amax(np.abs( mat - mat.conj().transpose() ))

def herm_dev_vec(vec):
    ns = int(np.sqrt(len(vec)))
    nq = int(np.log2(ns))
    max_abs_dev = 0.
    
    for j, k in it.product(range(ns), repeat=2):
        jk, kj = int_cat(j, k, nq), int_cat(k, j, nq)
        test_val = abs(vec[jk] - vec[kj].conj())
        if test_val > max_abs_dev:
            max_abs_dev = test_val
    
    return max_abs_dev

def min_max_eig(op, copy=True):
    """
    Calculates the min and max eigenvalues of a quantum state (either 
    density matrix or supervector).
    """
    op_c = op.copy() if copy else op
    if len(op_c.shape) == 1:
        op_c = vec2mat(op_c)
    
    eigs = np.linalg.eig(op_c)[0]
    return min(eigs), max(eigs)

def vec_purity(vec):
    return overlap(vec, vec)

def vec_tr(vec):
    """
    Assuming vec is a vectorised matrix, returns the trace
    """
    sqrt_sz = int(np.sqrt(len(vec)))
    return sum([vec[(sqrt_sz + 1) * j] for j in range(sqrt_sz)])

def bt_sn(i, l, nq):
    """
    (-1)^(i_l)
    """
    bits = bin(i)[2:].zfill(nq)
    return -1 if int(bits[l]) else 1

def m_c_rho(c_diag, rho_vec):
    temp_vec = np.multiply(c_diag, rho_vec)
    return temp_vec - (vec_tr(temp_vec) * rho_vec)

def m_c_rho_op(c_op, rho_vec):
    rho_mat = vec2mat(rho_vec)
    temp_mat = np.dot(c_op, rho_mat) + np.dot(rho_mat, c_op.conj().transpose())
    temp_mat -= np.trace(temp_mat) * rho_mat
    return mat2vec(temp_mat)

def com_mat(lil_ham):
    """
    Takes a 2**nq - by 2**nq sized Hamiltonian and returns the 
    supermatrix corresponding to -i[H, rho].
    """
    id_mat = np.eye(lil_ham.shape[0], dtype=lil_ham.dtype)
    return -1j * (np.kron(id_mat, lil_ham) - np.kron(lil_ham, id_mat))

def diss_mat(a):
    """
    Takes a 2**nq - by - 2**nq sized jump operator and returns the 
    supermatrix corresponding to D[A](rho) = A rho A^+ - {A^+ A, rho}/2
    """
    id_mat = np.eye(a.shape[0], dtype=a.dtype)
    out_val = np.kron(a.conj(), a)
    out_val -= 0.5 * np.kron(id_mat, np.dot(a.conj().transpose(), a))
    out_val -= 0.5 * np.kron(id_mat, np.dot(a.transpose(), a.conj()))
    return out_val
    
def lin_meas_mat(a):
    """
    Takes a 2**nq - by - 2**nq sized measurement operator and returns 
    the 
    supermatrix corresponding to m[A](rho) = A rho + rho A^+
    """
    id_mat = np.eye(a.shape[0], dtype=a.dtype)
    return (np.kron(id_mat, a) + np.kron(a.conj(), id_mat))
    
def gamma_1_lind(gamma_1, q, nq):
    """
    Lindbladian contribution from amplitude damping on the qth qubit of
    nq total.
    """
    a = reduce(np.kron, [sigma_m if k == q else id_2
                         for k in range(nq)])
    return gamma_1 * diss_mat(a)

def gamma_2_lind(gamma_2, q, nq):
    """
    Lindbladian contribution from dephasing on the qth qubit of nq 
    total.
    """
    a = reduce(np.kron, [sigma_z if k == q else id_2
                         for k in range(nq)])
    return gamma_2 * diss_mat(a)

def z_ham(omega, q, nq):
    """
    Lindbladian contribution from z-axis rotation on the qth qubit of
    nq total.
    """
    id_2 = np.eye(2, dtype=cpx)
    sigma_z = np.array([[1.,  0.], [
                         0., -1.]], dtype=cpx)
    a = reduce(np.kron, [sigma_z if k == q else id_2
                         for k in range(nq)])
    return omega * com_mat(a)

def re_herm(rho):
    #                                 filth
    return 0.5 * (rho + mat2vec(vec2mat(rho).conj().transpose()))

def amplitudes_1(app, times, e_ss, sigma, t_on, t_off):
    """
    Assuming a piecewise quadratic pulse, returns analytic solutions 
    to single-mode amplitudes. Something similar can be accomplished
    for pulses of higher polynomial order and larger numbers of modes
    (4 is definitely okay, more maybe), but we restrict ourselves for 
    now.
    """
    delta, kappa, chi = app.cav_params()
    
    if any([delta.shape != (1, ),
            kappa.shape != (1, ),
            chi.shape[0] != 1]):
        raise ValueError("Only works for one mode")
    
    nq = chi.shape[1]
    nt, ns = len(times), 2**nq
    
    delta = delta[0]
    kappa = kappa[0]
    chi = chi[0,:]
    
    amps = np.zeros((nt, 1, ns), dtype=cpx)
    for i in range(ns):
        eff_chi = sum([chi[l] * bt_sn(i, l, nq) for l in range(nq)])
        amps[:, 0, i] = _amplitude_integral( times, e_ss, sigma, t_on,
                                             t_off, delta, kappa, eff_chi )
    return amps

def _amplitude_integral(times, e_ss, sigma, t_on, t_off, delta, kappa,
                        chi):
    """
    Evaluates the definite integral of a single-mode coherent amplitude
    for all times in an array `times`. Stores a piecewise quadratic 
    model of the pulse internally.
    """
    t_f = times[-1]
    bnds = [0, t_on - sigma/2., t_on, t_on + sigma/2., 
                  t_off - sigma/2., t_off, t_off + sigma/2., t_f]
    poly_a = 2. * e_ss / sigma**2
    c_lsts = [
                [0],
                [ poly_a * bnds[1]**2, -2. * poly_a * bnds[1],  poly_a],
                [e_ss - poly_a * bnds[3]**2,  2. * poly_a * bnds[3], -poly_a],
                [e_ss],
                [e_ss - poly_a * bnds[4]**2,  2. * poly_a * bnds[4], -poly_a],
                [ poly_a * bnds[6]**2, -2. * poly_a * bnds[6],  poly_a],
                [0]
            ]

    scale = -0.5 * kappa - 1j * (delta + chi)

    amp_array = np.zeros((len(times), ), dtype=cpx)
    for tdx, t in enumerate(times[1:]):
        for idx, bnd in enumerate(bnds[:-1]):
            if bnd <= t < bnds[idx + 1]:
                
                inc = -1j * np.sqrt(kappa) * def_poly_exp_int(c_lsts[idx], bnd, t, scale, t)
                
                if np.isnan(inc):
                    raise ValueError("integration up to t results in nan for "
                        "inputs: {}, {}, {}, {}".format(c_lsts[idx], bnd, scale, t))
                
                amp_array[tdx + 1] += inc
                
                break

            else:
                
                inc = -1j * np.sqrt(kappa) * def_poly_exp_int(c_lsts[idx], bnd, bnds[idx + 1], scale, t)
                
                if np.isnan(inc):
                    raise ValueError("integration results in nan for "
                        "inputs: {}, {}, {}, {}, {}".format(
                            c_lsts[idx], bnd, bnds[idx + 1], scale, t))
                
                amp_array[tdx + 1] += inc
    
    return amp_array

def def_poly_exp_int(c_list, x_start, x_end, scale, t):
    """
    Uses indef_poly_exp_int to evaluate the corresponding definite
    integral.
    """
    return sum(c * def_mon_exp_int(x_start, x_end, j, scale, t)
                for j, c in enumerate(c_list))
    
def indef_poly_exp_int(c_list, x, scale, t):
    """
    Uses indef_mon_exp_int to evaluate the integral of 
    p(x) exp(scale * (t - x)) dx where p(x) is a polynomial represented
    by a list of coefficients: p(x) = sum_j c_list[j] x**j.
    """
    return sum(c * indef_mon_exp_int(x, j, scale, t)
                for j, c in enumerate(c_list))

def def_mon_exp_int(x_start, x_end, order, scale, t):
    """
    Evaluates the integral from `x_start` to `x_end` of 
    x^order * exp(scale*(t - x)).
    """
    if x_start == 0.0:
        return prod([x_end**order / scale, (scale * x_end)**(-order),
                    np.exp(scale * t), 
                    fctrl(order) - complex(gammainc(order + 1, scale * x_end))])
    else:
        return indef_mon_exp_int(x_end, order, scale, t) -\
                indef_mon_exp_int(x_start, order, scale, t) 

def indef_mon_exp_int(x, order, scale, t):
    """
    Evaluates the indefinite integral of x^order * exp(scale*(t - x)))
    with the constant of integration arbitrarily set to 0.
    """
    return prod([-np.exp(scale * t), x**(order + 1), 
                (scale * x)**(-1 - order),
                complex(gammainc(1 + order, scale * x))])

def rand_mat(sz, tp=cpx):
    return (rand(sz, sz) + 1j * rand(sz, sz)).astype(tp)

def rand_herm_mat(sz):
    rnd_mat = rand_mat(sz)    
    return rnd_mat + rnd_mat.conj().transpose()

def rand_dens_mat(sz):
    id_mat = np.eye(sz, dtype=cpx)
    rnd_mat = rand_herm_mat(sz) 
    large_mat = np.dot(rnd_mat, rnd_mat.conj().transpose())
    return large_mat / np.trace(large_mat)

def rand_super_vec(sz):
    rnd_mat = rand_dens_mat(sz)    
    return mat2vec(rnd_mat)

def rand_pure_state(sz):
    rand_vec = rand(sz) + 1j * rand(sz)
    return rand_vec/np.dot(rand_vec.conj(), rand_vec)


#stackoverflow.com/questions/23228244/
def interquartile_range(x):
    q75, q25 = np.percentile(x, [75 ,25])
    return q75 - q25

def f_d_bin_width(x):
    return 2. * interquartile_range(x) * len(x)**(-1./3.)

def fd_bins(x):
    return (max(x) - min(x))/f_d_bin_width(x)

def colour_hist(pc, f_p, f_m):
    """
    produces a histogram of photocurrent results where the colour of 
    the bar is given by the average fidelity of trials within the bin
    (with the appropriate post-measurement state).
    pc is the summed photocurrent
    f_p is the  
    """
    pass

def avg_data(fl_out):
    """
    Collects all files matching *.pkl in the current directory, 
    unpickles assuming a pickled dict, and averages the values 
    corresponding to the keys 'step_results' and 'final_results', if
    they exist.
    """
    
    potential_keys = ['step_results', 'final_results', 'apparatus', 
                        'pulse_shape', 'times']
    
    avg_dict = dict.fromkeys(potential_keys)
    
    flist = sorted(glob('*.pkl'))
    n_files = len(flist)

    with open(flist[0], 'r') as params_file:
        params_dict = pkl.load(params_file)

    #Things that don't depend on the individual trajectory are copied
    #over first
    for key in potential_keys[2:]:
        avg_dict[key] = params_dict[key]

    steps_exist = params_dict['step_results'] is not None
    finals_exist = params_dict['final_results'] is not None

    #Get sizes so we can make arrays
    if steps_exist:
        n_traj, n_times, n_quants = params_dict['step_results'].shape
        fl_stp_means = np.empty((n_files, n_times, n_quants), cpx)

    if finals_exist:
        rslt_shp = params_dict['final_results'].shape[1:]
        fl_fnl_means = np.empty((n_files, ) + rslt_shp, cpx)


    for fdx, phil in enumerate(flist):
        with open(phil, 'r') as fill:
            curr_dict = pkl.load(fill)
        
        stp_rslts = curr_dict['step_results']
        fnl_rslts = curr_dict['final_results']
        
        if steps_exist:
            fl_stp_means[fdx, ...] = np.mean(stp_rslts, axis=0)
        if finals_exist:
            fl_fnl_means[fdx, ...] = np.mean(fnl_rslts, axis=0)

    if steps_exist:
        avg_dict['step_results'] = np.mean(fl_stp_means, axis=0)
    if finals_exist:
        avg_dict['final_results'] = np.mean(fl_fnl_means, axis=0)

    with open(fl_out, 'w') as out_file:
        pkl.dump(avg_dict, out_file)

    pass


def sum_step_data(fl_out, filter_kernel=None):
    """
    Collects all files matching *.pkl in the current directory, 
    unpickles assuming a pickled dict, and sums the values from the 
    key 'step_results', if it exists. Assumes all files contain the 
    same number of trajectories.
    """
    
    potential_keys = ['step_results', 'final_results', 'apparatus', 
                        'pulse_shape', 'times']
    
    short_dict = dict.fromkeys(potential_keys)
    
    flist = sorted(glob('*.pkl'))
    n_files = len(flist)

    with open(flist[0], 'r') as params_file:
        params_dict = pkl.load(params_file)

    #Things that don't depend on the individual trajectory are copied
    #over first
    for key in potential_keys[2:]:
        short_dict[key] = params_dict[key]

    steps_exist = params_dict['step_results'] is not None
    finals_exist = params_dict['final_results'] is not None

    #Get sizes so we can make arrays
    if steps_exist:
        n_traj, n_times, n_quants = params_dict['step_results'].shape
        fl_stp_ints = np.empty((n_files * n_traj, n_quants), cpx)

    if finals_exist:
        rslt_shp = params_dict['final_results'].shape[1:]
        fl_fnls = np.empty((n_files * n_traj, ) + rslt_shp, cpx)


    for fdx, phil in enumerate(flist):
        with open(phil, 'r') as fill:
            try:
                curr_dict = pkl.load(fill)
            except ValueError:
                continue
        
        stp_rslts = curr_dict['step_results']
        fnl_rslts = curr_dict['final_results']
        
        if steps_exist:
            for trdx in range(n_traj):
                if filter_kernel is not None:
                    fl_stp_ints[fdx * n_traj + trdx, ...] = np.average(
                        stp_rslts[trdx, :, 0], axis=0, weights=filter_kernel
                        ) * len(stp_rslts[trdx, ...])
                else:
                    fl_stp_ints[fdx * n_traj + trdx, ...] = np.sum(stp_rslts[trdx, ...], axis=0)
        if finals_exist:
            for trdx in range(n_traj):
                fl_fnls[fdx * n_traj + trdx, ...] = fnl_rslts[trdx, ...]

        if steps_exist:
            short_dict['step_results'] = fl_stp_ints
        if finals_exist:
            short_dict['final_results'] = fl_fnls
    
    with open(fl_out, 'w') as out_file:
        pkl.dump(short_dict, out_file)

    pass

def post_selected(data_array, cond_array, pred):
    """
    Splits an iterable of data into two lists, with membership of an 
    element depending on whether the corresponding value of another
    iterable satisfies a predicate. 

    e.g.:
    post_selected([1, 2, 3, 4, 5], [0, 0, 1, 1, 1], lambda x: x > 0) = 
    ([3, 4, 5], [1, 2])
    """
    sat_list = []
    unsat_list = []
    
    for idx, elem in enumerate(data_array):
        if pred(cond_array[idx]):
            sat_list.append(elem)
        else:
            unsat_list.append(elem)
    
    return sat_list, unsat_list

def add_arrow_to_line2D(
    axes, line, arrow_locs=[0.2, 0.4, 0.6, 0.8],
    arrowstyle='-|>', arrowsize=1, transform=None):
    """

    (Copied from stackoverflow.com/questions/26911898)

    Add arrows to a matplotlib.lines.Line2D at selected locations.

    Parameters:
    -----------
    axes: 
    line: list of 1 Line2D obbject as returned by plot command
    arrow_locs: list of locations where to insert arrows, % of total length
    arrowstyle: style of the arrow
    arrowsize: size of the arrow
    transform: a matplotlib transform instance, default to data coordinates

    Returns:
    --------
    arrows: list of arrows
    """
    if (not(isinstance(line, list)) or not(isinstance(line[0], 
                                           mlines.Line2D))):
        raise ValueError("expected a matplotlib.lines.Line2D object")
    x, y = line[0].get_xdata(), line[0].get_ydata()

    arrow_kw = dict(arrowstyle=arrowstyle, mutation_scale=10 * arrowsize)

    color = line[0].get_color()
    use_multicolor_lines = isinstance(color, np.ndarray)
    if use_multicolor_lines:
        raise NotImplementedError("multicolor lines not supported")
    else:
        arrow_kw['color'] = color

    linewidth = line[0].get_linewidth()
    if isinstance(linewidth, np.ndarray):
        raise NotImplementedError("multiwidth lines not supported")
    else:
        arrow_kw['linewidth'] = linewidth

    if transform is None:
        transform = axes.transData

    arrows = []
    for loc in arrow_locs:
        s = np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))
        n = np.searchsorted(s, s[-1] * loc)
        arrow_tail = (x[n], y[n])
        arrow_head = (np.mean(x[n:n + 2]), np.mean(y[n:n + 2]))
        p = mpatches.FancyArrowPatch(
            arrow_tail, arrow_head, transform=transform,
            **arrow_kw)
        axes.add_patch(p)
        arrows.append(p)
    return arrows

#stackoverflow.com/q/15033511
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1 + confidence) / 2., n-1)
    return m, m - h, m + h

