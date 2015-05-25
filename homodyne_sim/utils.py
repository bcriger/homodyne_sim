import pylab as pl 
import numpy as np
from numpy.linalg import eig
import itertools as it
from scipy.linalg import sqrtm

__all__ = ['cpx', 'id_2', 'YY', 'sigma_z', 'sigma_m' , 'vec2mat', 
            'mat2vec', 'state2vec', 'single_op', 'arctan_updown',
            'arctan_up', 'overlap', 'op_trace', 'op_herm_dev', 'op_purity',
            'fidelity', 'photocurrent', 'concurrence', 'check_cb', 
            'all_zs', 'herm_dev', 'herm_dev_vec', 'min_max_eig', 
            'vec_purity', 'vec_tr', 'bt_sn', 'm_c_rho', 'm_c_rho_op',
            'com_mat', 'diss_mat', 'lin_meas_mat', 'gamma_1_lind', 
            'gamma_2_lind',  'z_ham', 'interquartile_range', 
            'f_d_bin_width', 'fd_bins', 'colour_hist', 'tanh_updown',
            'tanh_up', 'sigma_x', 'sigma_y']

#cpx = np.complex64
cpx = np.complex128
# cpx = np.complex256 #UNSUPPORTED IN LINALG

"""
HINWEIS: We adopt the convention throughout that the excited state is 
the 0 state, and the ground state is the 1 state. This keeps the 
Hamiltonian proportional to Pauli Z, but implies that the other Pauli
matrices are transposed wrt the standard definition.
"""

id_2 = np.eye(2, dtype=cpx)

# sigma_m = np.array([[0., 1.], [
#                      0., 0.]], dtype=cpx)
sigma_m = np.array([[0., 0.], [
                     1., 0.]], dtype=cpx)

sigma_x = np.array([[0., 1.], [
                     1., 0.]], dtype=cpx)

sigma_y = np.array([[ 0.,  1j], [
                     -1j,  0.]], dtype=cpx)

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

cnst_pulse = lambda t, cnst: cnst

def arctan_updown(t, e_ss, sigma, t_on, t_off):
    
    return e_ss / np.pi * (np.arctan((t - t_on) / sigma) - 
                            np.arctan((t - t_off) / sigma ))

def arctan_up(t, e_ss, sigma, t_on):
    
    return e_ss / np.pi * (np.arctan((t - t_on) / sigma) + np.pi/2.)


def tanh_updown(t, e_ss, sigma, t_on, t_off):
    
    return e_ss / 2. * (np.tanh((t - t_on) / sigma) - 
                            np.tanh((t - t_off) / sigma))

def tanh_up(t, e_ss, sigma, t_on):
    
    return e_ss / 2. * (np.tanh((t - t_on) / sigma) + 1.)

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
    except ValueError, e:
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
    return np.sqrt(reduce(np.dot, [sqrt_rho, sigma, sqrt_rho]))

def photocurrent(t, rho, dW, sim):
    """
    Given a simulation object, finds sqrt(\eta) <c + c^+> + dW by 
    translating the time t into an array index (forgive me).
    """
    tdx = np.argmin(np.abs(sim.times - t))
    c = sim.measurement[tdx, :, :]
    if len(rho.shape) == 1:
        rho_c = vec2mat(rho)
    else:
        rho_c = rho
    return np.trace(np.dot(c + c.conj().transpose(), rho_c)) + dW

def concurrence(rho):
    r"""
    wikipedia.org/wiki/Concurrence_%28quantum_computing%29#Definition
    """
    if rho.shape not in [(4,4), (16,)]:
        raise ValueError("Concurrence only works for two-qubit states")
    rho_c = vec2mat(np.copy(rho)) if rho.shape == (16,) else rho
    test_mat = reduce(np.dot, [rho_c, YY, rho_c.conj(), YY])
    lmbds = sorted(map(np.sqrt, eig(test_mat)[0]))
    return lmbds[0] - lmbds[1] - lmbds[2] - lmbds[3]

def check_cb(t, rho, dW):
    hm = op_herm_dev(rho)
    tr = op_trace(rho)
    pr = op_purity(rho) 
    min_e, max_e = min_max_eig(rho)
    return hm, tr, pr, min_e, max_e

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