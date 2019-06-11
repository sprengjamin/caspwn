r""" Casimir energy for the plane-sphere geometry.

.. todo::
    * implementation of force

"""
import numpy as np
import multiprocessing as mp
from numba import njit
from numba import float64, int64
from numba.types import UniTuple
from sksparse.cholmod import cholesky
from scipy.sparse import coo_matrix
from scipy.integrate import quad
from scipy.constants import Boltzmann, hbar, c

from index import itt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../sphere/"))
from mie import mie_cache
import scattering_amplitude
from kernel import kernel_polar as kernel
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../plane/"))
from fresnel import rTM, rTE
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../ufuncs/"))
from integration import quadrature, auto_integration
from psd import psd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../material/"))
import material


@njit(float64[:,:](float64, float64, int64, float64, float64, float64, float64, mie_cache.class_type.instance_type))
def phi_array(rho, K, M, k1, k2, w1, w2, mie):
    r"""
    Returns the phi array for the kernel function for each polarization block.

    Parameters
    ----------
    rho: float
        positive, aspect ratio :math:`R/L`
    K: float
        positive, rescaled frequency
    M: int
        positive, length of sequence
    k1, k2: float
        positive, rescaled wave numbers
    w1, w2: float
        positive, quadrature weights corresponding to k1 and k2, respectively.
    mie: class instance
        cache for the exponentially scaled mie coefficients

    Returns
    -------
    np.ndarray
        array of length 4 of the phi array for the polarization contributions
        TMTM, TETE, TMTE, TETM

    """
    phiarr = np.empty((4, M))

    # phi = 0.
    phiarr[0, 0], phiarr[1, 0], phiarr[2, 0], phiarr[3, 0] =  kernel(rho, 1., 1., K, k1, k2, 0., mie)
    
    if M%2==0:
        # phi = np.pi is assumed
        phiarr[0, M//2], phiarr[1, M//2], phiarr[2, M//2], phiarr[3, M//2] =  kernel(rho, 1., 1., K, k1, k2, np.pi, mie)
        imax = M//2-1
        phi = 2*np.pi*np.arange(M//2)/M
    else:
        # phi = np.pi is not assumed
        imax = M//2
        phi = 2*np.pi*np.arange(M//2+1)/M
   
    # 0 < phi < np.pi
    for i in range(1, imax+1):
        phiarr[0, i], phiarr[1, i], phiarr[2, i], phiarr[3, i] =  kernel(rho, 1., 1., K, k1, k2, phi[i], mie)
        phiarr[0, M-i] = phiarr[0, i]
        phiarr[1, M-i] = phiarr[1, i]
        phiarr[2, M-i] = -phiarr[2, i]
        phiarr[3, M-i] = -phiarr[3, i]

    # for now support only perfect reflectors
    rTE1 = -1.#rTE(K, k1, np.inf)
    rTM1 = +1.#rTM(K, k1, np.inf)
    rTE2 = -1.#rTE(K, k2, np.inf)
    rTM2 = +1.#rTM(K, k2, np.inf)

    phiarr[0, :] = w1*w2*np.sqrt(rTM1*rTM2)*phiarr[0, :]
    phiarr[1, :] = -w1*w2*np.sqrt(rTE1*rTE2)*phiarr[1, :]
    phiarr[2, :] = w1*w2*np.sqrt(-rTM1*rTE2)*phiarr[2, :]
    phiarr[3, :] = w1*w2*np.sqrt(-rTE1*rTM2)*phiarr[3, :]
    return phiarr
        

def m_array(rho, K, M, k1, k2, w1, w2, mie):
    r"""
    Computes the m array by means of a FFT of the computed phi array.

    Parameters
    ----------
    rho: float
        positive, aspect ratio :math:`R/L`
    K: float
        positive, rescaled frequency
    M: int
        positive, length of sequence
    k1, k2: float
        positive, rescaled wave numbers
    w1, w2: float
        positive, quadrature weights corresponding to k1 and k2, respectively.
    mie: class instance
        cache for the exponentially scaled mie coefficients

    Dependencies
    ------------
    phi_array

    """
    phiarr = phi_array(rho, K, M, k1, k2, w1, w2, mie)
    marr = np.fft.rfft(phiarr)
    return np.array([marr[0,:].real, marr[1,:].real, marr[2,:].imag, marr[3,:].imag])


def compute_mElement_diag(i, rho, K, N, M, k, w, mie):
    r"""
    Computes the m-sequence of diagonal elements.

    Parameters
    ----------
    i: int
        non-negative, row or column index of the diagonal element
    rho: float
        positive, aspect ratio :math:`R/L`
    K: float
        positive, rescaled frequency
    N: int
        positive, quadrature order of k-integration
    M: int
        positive, quadrature order of phi-integration
    k, w: np.ndarray
        quadrature points and weights of the k-integration after rescaling
    mie: class instance
        cache for the exponentially scaled mie coefficients

    Returns
    -------
    row: np.ndarray
        array of row indices
    col: np.ndarray
        array of column indices
    data: np.ndarray
        array containing the m-sequence

    Dependencies
    ------------
    m_array

    """
    row = [i, N+i, N+i]
    col = [i, N+i, i]
    data = (m_array(rho, K, M, k[i], k[i], w[i], w[i], mie))[:-1,:]
    return row, col, data
   
    
def compute_mElement_offdiag(i, j, rho, K, N, M, k, w, mie):
    r"""
    Computes the m-sequence of off-diagonal elements.

    Parameters
    ----------
    i: int
        non-negative, row index of the off-diagonal element
    j: int
        non-negative, column index of the off-diagonal element
    rho: float
        positive, aspect ratio :math:`R/L`
    K: float
        positive, rescaled frequency
    N: int
        positive, quadrature order of k-integration
    M: int
        positive, quadrature order of phi-integration
    k, w: np.ndarray
        quadrature points and weights of the k-integration after rescaling
    mie: class instance
        cache for the exponentially scaled mie coefficients

    Returns
    -------
    row: np.ndarray
        array of row indices
    col: np.ndarray
        array of column indices
    data: np.ndarray
        array containing the m-sequence

    Dependencies
    ------------
    m_array

    """
    row = [i, N+i, N+j, N+i] 
    col = [j, N+j, i, j] 
    data = m_array(rho, K, M, k[i], k[j], w[i], w[j], mie)
    return row, col, data


def mArray_sparse_part(dindices, oindices, rho, K, N, M, k, w, mie):
    r"""
    Computes the m-array.

    Parameters
    ----------
    dindices: np.ndarray
        array of diagonal indices
    oindices: np.ndarray
        array of off-diagonal indices
    rho: float
        positive, aspect ratio :math:`R/L`
    K: float
        positive, rescaled frequency
    N: int
        positive, quadrature order of k-integration
    M: int
        positive, quadrature order of phi-integration
    k, w: np.ndarray
        quadrature points and weights of the k-integration after rescaling
    mie: class instance
        cache for the exponentially scaled mie coefficients

    Returns
    -------
    row: np.ndarray
        array of row indices
    col: np.ndarray
        array of column indices
    data: np.ndarray
        array containing the kernel data

    Dependencies
    ------------
    isFinite, compute_mElement_diag, itt, compute_mElement_offdiag

    """
    ###
    # 16 is just arbitrary here
    row = np.empty(16*N)
    col = np.empty(16*N)
    data = np.empty((16*N, M//2+1))
    
    ind = 0
    for i in dindices:
        if isFinite(rho, K, k[i], k[i]):
            row[ind:ind+3], col[ind:ind+3], data[ind:ind+3, :] = compute_mElement_diag(i, rho, K, N, M, k, w, mie)
            ind += 3

    for oindex in oindices:
        i, j = itt(oindex)
        if isFinite(rho, K, k[i], k[j]):
            if ind+4 >= len(row):
                row = np.hstack((row, np.empty(len(row))))
                col = np.hstack((col, np.empty(len(row))))
                data = np.vstack((data, np.empty((len(row), M//2+1))))
            row[ind:ind+4], col[ind:ind+4], data[ind:ind+4, :] = compute_mElement_offdiag(i, j, rho, K, N, M, k, w, mie)
            ind += 4
                
    row = row[:ind] 
    col = col[:ind] 
    data = data[:ind, :] 
    return row, col, data


def mArray_sparse_mp(nproc, rho, K, N, M, pts, wts, mie):
    r"""
    Computes the m-array in parallel using the multiprocessing module.

    Parameters
    ----------
    nproc: int
        number of processes
    rho: float
        positive, aspect ratio :math:`R/L`
    K: float
        positive, rescaled frequency
    N: int
        positive, quadrature order of k-integration
    M: int
        positive, quadrature order of phi-integration
    pts, wts: np.ndarray
        quadrature points and weights of the k-integration before rescaling
    mie: class instance
        cache for the exponentially scaled mie coefficients

    Returns
    -------
    row: np.ndarray
        array of row indices
    col: np.ndarray
        array of column indices
    data: np.ndarray
        array containing the kernel data

    Dependencies
    ------------
    get_b, mArray_sparse_part

    """
    if type(nproc) != int:
        raise TypeError("nproc must be an integer!")

    def worker(dindices, oindices, rho, K, N, M, k, w, mie, out):
        out.put(mArray_sparse_part(dindices, oindices, rho, K, N, M, k, w, 
mie))

    b = 0.5 # kinda arbitrary
    k = b*pts
    w = np.sqrt(b*wts*2*np.pi/M)
    
    dindices = np.array_split(np.random.permutation(N), nproc)
    oindices = np.array_split(np.random.permutation(N*(N-1)//2), nproc)
   
    
    os.environ["OMP_NUM_THREADS"] = "1" 
    out = mp.Queue()
    procs = []
    for i in range(nproc):
        p = mp.Process(
                target = worker,
                args = (dindices[i], oindices[i], rho, K, N, M, k, w, mie, 
out))
        procs.append(p)
        p.start()
    
    results = np.empty(nproc, dtype=object)
    for i in range(nproc):
        results[i] = out.get()
    
    for p in procs:
        p.join()
    
    os.environ["OMP_NUM_THREADS"] = str(nproc) 
    row = results[0][0]
    col = results[0][1]
    data = results[0][2]
    for i in range(1, nproc):
        row = np.hstack((row, results[i][0]))
        col = np.hstack((col, results[i][1]))
        data = np.vstack((data, results[i][2]))
        
    return row, col, data


def isFinite(rho, K, k1, k2):
    r"""
    Estimates by means of the asymptotics of the scattering amplitudes if the given
    matrix element can be numerically set to zero.

    Parameters
    ----------
    rho: float
        positive, aspect ratio :math:`R/L`
    K: float
        positive, rescaled frequency
    k1, k2: float
        positive, rescaled wave numbers

    Returns
    -------
    boolean
        True if the matrix element must not be neglected

    """
    if K == 0.:
        exponent = 2*rho*np.sqrt(k1*k2) - (k1+k2)*(rho+1)
    else:
        kappa1 = np.sqrt(k1*k1+K*K)
        kappa2 = np.sqrt(k2*k2+K*K)
        # copied from kernel with phi=0 > make a function! (which can be tested)
        exponent = -((k1 - k2)**2)/(np.sqrt(2*(kappa1*kappa2 + k1*k2 + K**2)) + kappa1 + kappa2)*rho - kappa1 - kappa2
    if exponent < -37:
        return False
    else:
        return True
            

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
    n_plane = eval("material."+materials[0]+".n(Kvac/L)")
    n_medium = eval("material."+materials[1]+".n(Kvac/L)")
    n_sphere = eval("material."+materials[2]+".n(Kvac/L)")
    
    n = n_sphere/n_medium
    # aspect ratio
    rho = R/L
    # size parameter
    x = n_medium*Kvac*rho
    # precompute mie coefficients
    if x == 0.:
        mie = mie_cache(1, 1., n, eval("material."+materials[2]+".materialclass"))
    elif x > 5e3:
        mie = mie_cache(1, x, n, eval("material."+materials[2]+".materialclass"))
    else:
        mie = mie_cache(int(2*x)+1000, x, n, eval("material."+materials[2]+".materialclass"))    # initial lmax arbitrary

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


def energy_zero(R, L, materials, N, M, nproc, X=None):
    r"""
    Computes the Casimir at zero temperature.

    Parameters
    ----------
    R: float
        positive, radius of the sphere
    L: float
        positive, surface-to-surface distance
    materials : string
        name of material 
    N: int
        positive, quadrature order of k-integration
    M: int
        positive, quadrature order of phi-integration
    nproc: int
        number of processes spawned by multiprocessing module

    Returns
    -------
    energy: float
        Casimir energy in units of :math:`\hbar c/L`.

    
    Dependencies
    ------------
    quadrature, get_mie, LogDet_sparse_mp

    """
    pts, wts = quadrature(N)
    if X == None:
        logdet = lambda Kvac : LogDet(R, L, materials, Kvac, N, M, pts, wts, nproc)
        energy = auto_integration(logdet)
    else:
        K_pts, K_wts = quadrature(X)
        
        energy = 0.
        for i in range(X):
            result = LogDet(R, L, materials, K_pts[i], N, M, pts, wts, nproc)
            print("K=", K_pts[i], ", val=", result)
            energy += K_wts[i]*result
    return energy/(2*np.pi)


def energy_quad(R, L, materials, N, M, nproc):
    r"""
    Computes the Casimir at zero temperature.

    Parameters
    ----------
    R: float
        positive, radius of the sphere
    L: float
        positive, surface-to-surface distance
    materials : string
        name of material 
    N: int
        positive, quadrature order of k-integration
    M: int
        positive, quadrature order of phi-integration
    nproc: int
        number of processes spawned by multiprocessing module

    Returns
    -------
    energy: float
        Casimir energy in Joule

    
    Dependencies
    ------------
    quadrature, get_mie, LogDet_sparse_mp

    """
    pts, wts = quadrature(N)
    logdet = lambda Kvac : LogDet(R, L, materials, Kvac, N, M, pts, wts, nproc)
    energy = quad(logdet, 0, np.inf)[0]
    return energy/(2*np.pi)*hbar*c/L


def energy_finite(R, L, T, materials, N, M, epsrel, nproc):
    """
    Computes the Casimir free energy at equilibrium temperature :math:`T`.

    Parameters
    ----------
    R: float
        positive, radius of the sphere
    L: float
        positive, surface-to-surface distance between plane and sphere
    materials: tuple
        contains the materials in the form (material of plane, medium, material
        of sphere)
    N: int
        positive, quadrature order of k-integration
    M: int
        positive, quadrature order of phi-integration
    epsrel: float
        positive, desired relative error for the matsubara sum
    nproc: int
        number of processes spawned by multiprocessing module

    Returns
    -------
    energy: float
        Casimir free energy in Joule
    
    Dependencies
    ------------
    quadrature, get_mie, LogDet_sparse_mp

    """
    pts, wts = quadrature(N)
    
    K_matsubara = Boltzmann*T*L/(hbar*c)

    energy0 = LogDet(R, L, materials, 0., N, M, pts, wts, nproc)
    energy = 0.
    Teff = 4*np.pi*Boltzmann/hbar/c*T*L
    order = int(max(np.ceil((1-1.5*np.log10(np.abs(epsrel)))/np.sqrt(Teff)), 5))
    xi, eta = psd(order)
    for n in range(order):
        term = 2*eta[n]*LogDet(R, L, materials, K_matsubara*xi[n], N, M, pts, wts, nproc)
        energy += term
    return 0.5*T*Boltzmann*(energy+energy0), 0.5*T*Boltzmann*energy


if __name__ == "__main__":
    np.random.seed(0)
    R = 1.
    L = 1.e-3
    T = 2.289885278703585880e-01
    #T = 1.e-03
    #T = 293.15
    rho = R/L
    N = int(10*np.sqrt(rho))
    print("N", N)
    M = N
    nproc = 4
    
    #mie = mie_e_array(1e4, 1.*rho)
    #print(phiSequence(rho, 1., M, 2.3, 2.3, 1., 1., mie))
    mat = ("PR", "Vacuum", "PR") 
    import time
    start = time.time()
    en = energy_finite(R, L, T, mat, N, M, 1e-8, nproc) 
    end = time.time()
    print("energy")
    print(en)
    print("PFA")
    print(-np.pi**3*rho/720)
    print("time")
    print(end-start)
