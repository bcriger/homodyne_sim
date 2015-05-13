import pylab as pl 
import numpy as np 
import itertools as it

cpx = np.complex128

id_2 = np.eye(2, dtype=cpx)

sigma_m = np.array([[0., 1.], [
                     0., 0.]], dtype=cpx)

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

cnst_pulse = lambda t: 1.

def arctan_input(t, e_ss, sigma, t_on, t_off):
    
    return e_ss / np.pi * (np.arctan(sigma * (t - t_on)) - 
                            np.arctan(sigma * (t - t_off)))


def overlap(rho_vec, test_vec):
    """
    Determines the overlap (tr(rho * sigma)) between two 
    column-stacked density operators.
    """
    return np.dot(rho_vec.conj().transpose(), test_vec)

def fidelity(rho_vec, test_vec):
    """
    Determines the state fidelity between two column-stacked density 
    operators.
    """
    return np.sqrt(overlap(rho_vec, test_vec))

def photocurrent(t, rho, dW, sim):
    """
    Given a simulation object, finds sqrt(\eta) <c + c^+> + dW by 
    translating the time t into an array index (forgive me).
    """
    tdx = np.argmin(np.abs(sim.times - t))
    c = sim.measurement[:,tdx]
    return overlap(c, rho) + dW

def concurrence(rho_vec):
    r"""
    wikipedia.org/wiki/Concurrence_%28quantum_computing%29#Definition
    """
    if rho_vec.shape != (16,):
        raise ValueError("Concurrence only works for two-qubit states")
    rho_mat = vec2mat(rho_vec)
    test_mat = reduce(np.dot, [rho_mat, YY, rho_mat.conj(), YY])
    lmbds = sorted(map(np.sqrt, np.eigs(test_mat)[0]))
    return lmbds[0] - lmbds[1] - lmbds[2] - lmbds[3]


def check_cb(t, rho, dW):
    hm = herm_dev(rho)
    tr = vec_tr(rho)
    pr = vec_purity(rho)
    min_e, max_e = min_max_eig(rho)
    return hm, tr, pr, min_e, max_e

def all_zs(nq):
    """
    returns a column-stacked matrix of Z^(\otimes nq) for some nq
    """
    biterator = it.product(range(2), repeat=nq)
    output = np.zeros(4**nq)
    count = 0
    for bits in biterator:
        output[(2**nq + 1) * count] = np.product([(-1)**b for b in bits])
        count += 1
    return output

int_cat = lambda i, j, nq: (i << nq) + j

def herm_dev(vec):
    ns = int(np.sqrt(len(vec)))
    nq = int(np.log2(ns))
    max_abs_dev = 0.
    
    for j, k in it.product(range(ns), repeat=2):
        jk, kj = int_cat(j, k, nq), int_cat(k, j, nq)
        test_val = abs(vec[jk] - vec[kj].conj())
        if test_val > max_abs_dev:
            max_abs_dev = test_val
    
    return max_abs_dev

def min_max_eig(vec):
    """
    Assuming that vec is a column-stacked matrix, returns the min and 
    max eigenvalues.
    """
    mat = vec2mat(vec)
    eigs = np.linalg.eig(mat)[0]
    return min(eigs), max(eigs)

def vec_purity(vec):
    return overlap(vec, vec)

def vec_tr(vec):
    """
    Assuming vec is a vectorised matrix, returns the trace
    """
    sqrt_sz = int(np.sqrt(len(vec)))
    return sum([vec[(sqrt_sz + 1) * j] for j in range(sqrt_sz)])

def bt_sn(i, j):
    """
    (-1)^(i_j)
    """
    return -1 if (i & (1 << j)) else 1

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