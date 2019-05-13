"""Calculate 1st, 2nd and 3rd round trip for WKB approximation of scattering
amplitude

"""
import numpy as np
from numpy.linalg import eigvalsh
import math
from numba import njit
from numba import float64, int64
from numba.types import UniTuple
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import sys, os
import time
from numba import njit
sys.path.append("../../../../plane-sphere")
sys.path.append("../../../../sphere")
sys.path.append("../../../../ufuncs")
import energy
import kernel
from scattering_amplitude import S1S2
from mie import mie_cache
from energy import mArray_sparse_mp
from kernel import kernel_polar
from integration import quadrature

RMAX = 40

def Roundtrips(R, L, materials, Kvac, N, M, pts, wts, nproc):
    r"""
    Computes the first three round-trips.

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
    # maximal round trip
    rmax = RMAX
    traces = np.empty(rmax)

    n_plane = np.inf
    n_medium = 1.
    n_sphere = np.inf 
    
    n = n_sphere/n_medium
    # aspect ratio
    rho = R/L
    # size parameter
    x = n_medium*Kvac*rho
    
    # precompute mie coefficients
    if x > 5e3:
        mie = mie_cache(1, x, n)
    else:
        mie = mie_cache(int(2*x)+1000, x, n)    # initial lmax arbitrary

    row, col, data = mArray_sparse_mp(nproc, rho, Kvac*n_medium, N, M, pts, wts, mie)
    
    # m=0
    sprsmat = coo_matrix((data[:, 0], (row, col)), shape=(2*N,2*N)).tocsc()
    ev = eigvalsh(sprsmat.todense())
    for i in range(rmax):
        traces[i] = np.sum(ev**(i+1))
    
    # m>0    
    for m in range(1, M//2):
        sprsmat = coo_matrix((data[:, m], (row, col)), shape=(2*N,2*N)).tocsc()
        ev = eigvalsh(sprsmat.todense())
        for i in range(rmax):
            traces[i] += 2*np.sum(ev**(i+1))

    # last m
    sprsmat = coo_matrix((data[:, M//2], (row, col)), shape=(2*N,2*N)).tocsc()
    ev = eigvalsh(sprsmat.todense())
    for i in range(rmax):
        if M%2==0:
            traces[i] += np.sum(ev**(i+1))
        else:
            traces[i] += 2*np.sum(ev**(i+1))
    print(Kvac, traces)
    return traces

def xi_integration(R, L, N, M, X, nproc):
    materials = ("PR", "Vacuum", "PR")
    pts, wts = quadrature(N)
    xi_pts, xi_wts = quadrature(X)
    traces = np.zeros(RMAX)
    for i in range(X):
        traces += xi_wts[i]*Roundtrips(R, L, materials, xi_pts[i], N, M, pts, wts, nproc)
    return traces/(2*np.pi) 

R = 1.
Lvals = np.logspace(-1, -4, 61)

eta = 12.
nproc = 4
X = 250

filename = "em_roundtrips_eta12_X250.dat"

if not os.path.isfile(filename):
    f=open(filename, "a")
    for L in Lvals:
        print("computing L=",L)
        rho = R/L
        N = int(eta*np.sqrt(rho))
        M = N
        start = time.time()
        traces = xi_integration(R, L, N, M, X, nproc)
        end = time.time()
        t = end-start
        f=open(filename, "ab")
        data = np.insert(traces, 0, L)
        data = np.append(data, t)
        np.savetxt(f, data[np.newaxis,:])
        f.close()
else:
    print("File already exists!")
