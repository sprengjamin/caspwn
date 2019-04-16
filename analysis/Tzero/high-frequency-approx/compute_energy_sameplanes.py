"""Calculate TM contribution when A=1, B=C=D=0

"""
import numpy as np
import sys, os
import time
from numba import njit
from scipy.sparse import coo_matrix
from sksparse.cholmod import cholesky
sys.path.append("../../../plane-sphere")
sys.path.append("../../../sphere")
sys.path.append("../../../material")
from mie import mie_cache
from energy import energy_zero, mArray_sparse_mp
import energy
from kernel import kernel_polar
import kernel

@njit
def S1S2_high_frequency(x, z, mie):
    return -0.5*x, 0.

@njit
def ABCD_same_plane(xi, k1, k2, phi):
    return 1., 0., 0., 0. 

def LogDet(R, L, materials, Kvac, N, M, pts, wts, nproc):
    r"""
    Computes the sum of the logdets the m-matrices.

    Parameters
    ----------
    R: float
        positive, radius of the sphere
    L: float
        positive, surface-to-surface distance
    materials : tuple of strings
        names of material in the order (plane, medium, sphere) 
    Kvac: float
        positive, vacuum wave number multiplied by :math:`L`.
    N: int
        positive, quadrature order of k-integration
    M: int
        positive, quadrature order of phi-integration
    pts, wts: np.ndarray
        quadrature points and weights of the k-integration before rescaling
    nproc: int
        number of processes

    Returns
    -------
    logdet: float
        the sum of logdets of the m-matrices

    Dependencies
    ------------
    mArray_sparse_mp

    """
    n_plane = np.inf 
    n_medium = 1. 
    n_sphere = np.inf
    
    n = n_sphere/n_medium
    # aspect ratio
    rho = R/L
    # size parameter
    x = n_medium*Kvac*rho
    # dummy mie coefficients
    mie = mie_cache(1, x, n)

    row, col, data = mArray_sparse_mp(nproc, rho, Kvac*n_medium, N, M, pts, wts, mie)
    
    # m=0
    sprsmat = coo_matrix((data[:, 0], (row, col)), shape=(2*N,2*N))
    factor = cholesky(-sprsmat.tocsc(), beta=1.)
    logdet = factor.logdet()

    # m>0    
    for m in range(1, M//2):
        sprsmat = coo_matrix((data[:, m], (row, col)), shape=(2*N,2*N))
        factor = cholesky(-sprsmat.tocsc(), beta=1.)
        logdet += 2*factor.logdet()

    # last m
    sprsmat = coo_matrix((data[:, M//2], (row, col)), shape=(2*N,2*N))
    factor = cholesky(-sprsmat.tocsc(), beta=1.)
    if M%2==0:
        logdet += factor.logdet()
    else:
        logdet += 2*factor.logdet()
    print(Kvac, logdet)
    return logdet

kernel.ABCD = ABCD_same_plane
# replace S1S2 by the high frequency asmpytotics
kernel.S1S2 = S1S2_high_frequency
kernel_polar.recompile()
energy.phi_array.recompile()
energy.LogDet = LogDet

R = 1.
Lvals = np.logspace(-1, -4, 61)
materials = ("PR", "Vacuum", "PR")

eta = 20.
nproc = 4

filename = "approx_energy_sameplanes_eta20.dat"

if not os.path.isfile(filename):
    f=open(filename, "a")
    for L in Lvals:
        print("computing L=",L)
        rho = R/L
        N = int(eta*np.sqrt(rho))
        M = N
        start = time.time()
        en = energy_zero(R, L, materials, N, M, nproc)
        end = time.time()
        t = end-start
        f=open(filename, "ab")
        np.savetxt(f, [[L, en, t]])
        f.close()
else:
    print("File already exists!")




