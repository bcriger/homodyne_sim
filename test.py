import numpy as np
from numpy.linalg import eigvalsh
from numpy.random import rand
import homodyne_sim as hs
import itertools as it

"""
Meant to be run with nosetests
"""

#Lots of matrices used below, we make them all size 10 and use double 
#precision, so we can imagine that everything's correct up to 10**-14:
dtp = hs.cpx
num_tol = 10**-12

nq = 4
size = 2**nq

id_mat = np.eye(size, dtype=dtp)

#--------------------------convenience functions----------------------#

def rand_mat (sz, tp=dtp):
    return (rand(sz, sz) + 1j * rand(sz, sz)).astype(tp)

def rand_herm_mat(sz):
    rnd_mat = rand_mat(sz)    
    return rnd_mat + rnd_mat.conj().transpose()

def rand_dens_mat(sz):
    rnd_mat = rand_herm_mat(sz) 
    min_eig = min(eigvalsh(rnd_mat))
    rnd_mat += min_eig * id_mat
    rnd_mat /= np.trace(rnd_mat)
    return rnd_mat

def rand_super_vec(sz):
    rnd_mat = rand_dens_mat(sz)    
    return rnd_mat.transpose().reshape((sz**2,))

def rand_pure_state(sz):
    rand_vec = rand(sz) + 1j * rand(sz)
    return rand_vec/np.dot(rand_vec.conj(), rand_vec)

#-----------------------------test functions--------------------------#

def mat2vec2mat_test():
    rnd_vec = rand_super_vec(size)
    assert (hs.mat2vec(hs.vec2mat(rnd_vec)) == rnd_vec).all()

def vec2mat2vec_test():
    rnd_mat = rand_dens_mat(size)
    assert (hs.vec2mat(hs.mat2vec(rnd_mat)) == rnd_mat).all()

def op_trace_test():
    rnd_mat = rand_mat(size)
    assert abs(hs.op_trace(rnd_mat) - hs.op_trace(hs.mat2vec(rnd_mat))) < num_tol

def op_purity_test():
    rnd_mat = rand_mat(size)
    assert abs(hs.op_purity(rnd_mat) - hs.op_purity(hs.mat2vec(rnd_mat))) < num_tol

def purity_1_bit_test():
    r = rand()
    diag_state = np.matrix([[r, 0.], [0., 1. - r]], dtype=dtp)
    test_purity = r**2 + (1. - r)**2
    assert abs(hs.op_purity(diag_state) - test_purity) < num_tol
    assert abs(hs.op_purity(hs.mat2vec(diag_state)) - test_purity) < num_tol

def overlap_test():
    """
    The overlap function is a little complicated, we test it with nine 
    cases, where each of the two inputs can be:
     + a pure state, represented by a unit-norm complex vector
     + a mixed state, represented by a density matrix
     + a mixed state, represented by a column-stacked matrix.
    """
    pure_a = rand_pure_state(size)
    pure_b = rand_pure_state(size)
    proj_a = np.outer(pure_a, pure_a.conj())
    proj_b = np.outer(pure_b, pure_b.conj())
    vec_a = proj_a.transpose().reshape((size**2,))
    vec_b = proj_b.transpose().reshape((size**2,))
    
    pure_overlap = hs.overlap(pure_a, pure_b, nq)
    
    for a, b in it.product([pure_a, proj_a, vec_a],
                            [pure_b, proj_b, vec_b]):
        assert abs(hs.overlap(a, b, nq) - pure_overlap) < num_tol
    
    mix_a = rand_dens_mat(size)
    mix_b = rand_dens_mat(size)
    mix_v_a = mix_a.transpose().reshape((size**2,))
    mix_v_b = mix_b.transpose().reshape((size**2,))
    
    mix_overlap = hs.overlap(mix_a, mix_b, nq)
    for a, b in it.product([mix_a, mix_v_a], [mix_b, mix_v_b]):
        assert abs(hs.overlap(a, b, nq) - mix_overlap) < num_tol

def min_max_eig_test():
    rho = rand_dens_mat(size)
    real_eigs = min(eigvalsh(rho)), max(eigvalsh(rho))
    test_eigs = hs.min_max_eig(rho)
    assert abs(real_eigs[0] - test_eigs[0]) < num_tol
    assert abs(real_eigs[1] - test_eigs[1]) < num_tol

def op_herm_dev_test():
    #Generate random hermitian matrix
    hm_mat = rand_herm_mat(10)
    anti_hm_mat = 1j * hm_mat
    assert hs.op_herm_dev(hm_mat) < 10**-12
    assert hs.op_herm_dev(hs.mat2vec(hm_mat)) < 10**-12
    assert hs.op_herm_dev(anti_hm_mat) > 10**-12
    assert hs.op_herm_dev(hs.mat2vec(anti_hm_mat)) > 10**-12