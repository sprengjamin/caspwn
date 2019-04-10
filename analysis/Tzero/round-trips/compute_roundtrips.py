"""Calculate 1st, 2nd and 3rd round trip for WKB approximation of scattering
amplitude

"""
import numpy as np
import math
from numba import njit
from numba import float64, int64
from numba.types import UniTuple
from scipy.sparse import spdiags, coo_matrix
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

@njit("float64(float64[:], float64[:], float64[:])")
def supersonic_trsquare(row, col, data):
    n = len(row)
    trace = 0.
    for i in range(n):
        if row[i] == col[i]:
            trace += data[i]**2
        else:
            trace += 2*data[i]**2
    return trace

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
    diag = spdiags(sprsmat.diagonal(), 0, 2*N, 2*N, format="csc")
    mat = sprsmat + sprsmat.T - diag
    trM = np.sum(diag.diagonal())
    trM2 = supersonic_trsquare(row, col, data[:, 0])
    mat2 = mat.dot(mat)
    trM3 = (mat2.multiply(mat)).sum()
    
    # m>0    
    for m in range(1, M//2):
        sprsmat = coo_matrix((data[:, m], (row, col)), shape=(2*N,2*N)).tocsc()
        diag = spdiags(sprsmat.diagonal(), 0, 2*N, 2*N, format="csc")
        mat = sprsmat + sprsmat.T - diag
        trM += 2*np.sum(diag.diagonal())
        trM2 += 2*supersonic_trsquare(row, col, data[:, m])
        mat2 = mat.dot(mat)
        trM3 += 2*(mat2.multiply(mat)).sum()

    # last m
    sprsmat = coo_matrix((data[:, M//2], (row, col)), shape=(2*N,2*N)).tocsc()
    diag = spdiags(sprsmat.diagonal(), 0, 2*N, 2*N, format="csc")
    mat = sprsmat + sprsmat.T - diag
    mat2 = mat.dot(mat)
    if M%2==0:
        trM += np.sum(diag.diagonal())
        trM2 += supersonic_trsquare(row, col, data[:, M//2])
        trM3 += (mat2.multiply(mat)).sum()
    else:
        trM += 2*np.sum(diag.diagonal())
        trM2 += 2*supersonic_trsquare(row, col, data[:, M//2])
        trM3 += 2*(mat2.multiply(mat)).sum()
    print(Kvac, trM, trM2, trM3)
    return trM, trM2, trM3

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
    tr1, tr2, tr3 = 0., 0., 0.
    for i in range(X):
        r1, r2, r3 = Roundtrips(R, L, materials, xi_pts[i], N, M, pts, wts, nproc)
        tr1 += xi_wts[i]*r1
        tr2 += xi_wts[i]*r2
        tr3 += xi_wts[i]*r3
    return tr1/(2*np.pi), tr2/(2*np.pi), tr3/(2*np.pi)

R = 1.
Lvals = np.logspace(-1, -4, 61)

eta = 10.
nproc = 4
X = 250

filename = "roundtrips.dat"

if not os.path.isfile(filename):
    f=open(filename, "a")
    for L in Lvals:
        print("computing L=",L)
        rho = R/L
        N = int(eta*np.sqrt(rho))
        M = N
        start = time.time()
        tr1, tr2, tr3 = xi_integration(R, L, N, M, X, nproc)
        end = time.time()
        t = end-start
        f=open(filename, "ab")
        np.savetxt(f, [[L, tr1, tr2, tr3, t]])
        f.close()
else:
    print("File already exists!")
