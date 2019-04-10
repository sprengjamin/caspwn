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
sys.path.append("../../../plane-sphere")
sys.path.append("../../../sphere")
sys.path.append("../../../ufuncs")
import energy
from mie import mie_cache
from energy import mArray_sparse_mp
from kernel import phase
from integration import quadrature


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
    rmax = 10
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
    
    # dummy mie
    mie = mie_cache(1, x, n)

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

@njit(UniTuple(float64, 4)(float64, float64, float64, float64, float64, float64, float64, mie_cache.class_type.instance_type))
def kernel_WKB(rho, r, sign, K, k1, k2, phi, mie):
    r"""
    Returns the kernel of the reflection operator on a sphere in polar
    coordinates in symmetrized from.

    Parameters
    ----------
    rho: float
        positive, aspect ratio R/L
    r: float
        positive, relative effective radius R_eff/R
    sign: +1/-1
        sign, differs for the two spheres
    K: float
        positive, wave number in the medium multiplied by L, :math:`n\xi L/c`
    k1, k2: float
        positive, rescaled wave numbers
    phi: float
        between 0 and 2pi
    mie: class instance
        cache for the exponentially scaled mie coefficient
        
    Returns
    -------
    tuple
        tuple of length 4 of kernels for the polarization contributions
        TMTM, TETE, TMTE, TETM

    """
    kappa1 = math.sqrt(k1*k1+K*K)
    kappa2 = math.sqrt(k2*k2+K*K)
    z = (kappa1*kappa2+k1*k2*math.cos(phi))/K**2
    exponent = phase(rho, r, K, k1, k2, phi)
    if exponent < -37:
        return 0., 0., 0., 0.
    e = math.exp(exponent)
    norm = math.sqrt(k1*k2)/(2*math.pi*K*math.sqrt(kappa1*kappa2))
    S2 = 0.5*K*rho
    TMTM = norm*S2*e
    TETE = 0.
    TMTE = 0.
    TETM = 0.
    return TMTM, TETE, TMTE, TETM

# replace S1S2 by the high frequency asmpytotics
energy.kernel = kernel_WKB
energy.phi_array.recompile()

def xi_integration(R, L, N, M, X, nproc):
    materials = ("PR", "Vacuum", "PR")
    pts, wts = quadrature(N)
    xi_pts, xi_wts = quadrature(X)
    traces = np.zeros(10)
    for i in range(X):
        traces += xi_wts[i]*Roundtrips(R, L, materials, xi_pts[i], N, M, pts, wts, nproc)
    return traces/(2*np.pi) 

R = 1.
Lvals = np.logspace(-1, -4, 61)

eta = 12.
nproc = 4
X = 300

filename = "all_roundtrips.dat"

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
        np.savetxt(f, [[L, traces[0], traces[1], traces[2], traces[3], traces[4], traces[5], traces[6], traces[7], traces[8], traces[9], t]])
        f.close()
else:
    print("File already exists!")
