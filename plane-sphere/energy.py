r""" Casimir energy for the plane-sphere geometry.

.. todo::
    * implementation of force and pressure
    * add support for different material on plane (add n_plane, matclass_plane as arguments)
    * replace marray by just fft at the appropriate place
    * make dense implementation?

"""

import mkl
mkl.domain_set_num_threads(1, "fft")
import numpy as np

from math import sqrt
import concurrent.futures as futures
from numba import njit
from sksparse.cholmod import cholesky
from scipy.sparse import coo_matrix
from scipy.integrate import quad
from scipy.constants import Boltzmann, hbar, c
import time

from index import itt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../sphere/"))
from mie import mie_e_array
from kernel import kernel_polar as kernel
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../plane/"))
from fresnel import rTM, rTE
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../ufuncs/"))
from integration import quadrature, auto_integration
from psd import psd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../material/"))
import material


@njit("float64[:,:](float64, float64, int64, float64, float64, float64, float64, float64, string, float64, string, int64, float64[:], float64[:])", cache=True)
def phi_array(rho, K, M, k1, k2, w1, w2, n_plane, materialclass_plane, n_sphere, materialclass_sphere, lmax, mie_a, mie_b):
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
    n : float
        positive, refractive index
    lmax : int
        positive, cut-off angular momentum
    materialclass: string
        the material class (currently supports: drude, dielectric, PR)
    mie_a : list
        list of mie coefficients for electric polarizations
    mie_b : list
        list of mie coefficients for magnetic polarizations

    Returns
    -------
    np.ndarray
        array of length 4 of the phi array for the polarization contributions
        TMTM, TETE, TMTE, TETM

    """
    phiarr = np.empty((4, M))

    # phi = 0.
    phiarr[0, 0], phiarr[1, 0], phiarr[2, 0], phiarr[3, 0] = kernel(rho, 1., 1., K, k1, k2, 0., n_sphere, materialclass_sphere, lmax, mie_a, mie_b)
    
    if M%2==0:
        # phi = np.pi is assumed
        phiarr[0, M//2], phiarr[1, M//2], phiarr[2, M//2], phiarr[3, M//2] = kernel(rho, 1., 1., K, k1, k2, np.pi, n_sphere, materialclass_sphere, lmax, mie_a, mie_b)
        imax = M//2-1
        phi = 2*np.pi*np.arange(M//2)/M
    else:
        # phi = np.pi is not assumed
        imax = M//2
        phi = 2*np.pi*np.arange(M//2+1)/M
   
    # 0 < phi < np.pi
    for i in range(1, imax+1):
        phiarr[0, i], phiarr[1, i], phiarr[2, i], phiarr[3, i] = kernel(rho, 1., 1., K, k1, k2, phi[i], n_sphere, materialclass_sphere, lmax, mie_a, mie_b)
        phiarr[0, M-i] = phiarr[0, i]
        phiarr[1, M-i] = phiarr[1, i]
        phiarr[2, M-i] = -phiarr[2, i]
        phiarr[3, M-i] = -phiarr[3, i]

    rTE1 = rTE(K, k1, n_plane**2, materialclass_plane)
    rTM1 = rTM(K, k1, n_plane**2, materialclass_plane)
    rTE2 = rTE(K, k2, n_plane**2, materialclass_plane)
    rTM2 = rTM(K, k2, n_plane**2, materialclass_plane)

    phiarr[0, :] = w1*w2*sqrt(rTM1*rTM2)*phiarr[0, :]
    phiarr[1, :] = -w1*w2*sqrt(rTE1*rTE2)*phiarr[1, :]
    phiarr[2, :] = w1*w2*sqrt(-rTM1*rTE2)*phiarr[2, :]
    phiarr[3, :] = w1*w2*sqrt(-rTE1*rTM2)*phiarr[3, :]
    return phiarr
        

def m_array(rho, K, M, k1, k2, w1, w2, n_plane, materialclass_plane, n_sphere,  materialclass_sphere, lmax, mie_a, mie_b):
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
    n : float
        positive, refractive index
    lmax : int
        positive, cut-off angular momentum
    materialclass: string
        the material class (currently supports: drude, dielectric, PR)
    mie_a : list
        list of mie coefficients for electric polarizations
    mie_b : list
        list of mie coefficients for magnetic polarizations

    Dependencies
    ------------
    phi_array

    """
    phiarr = phi_array(rho, K, M, k1, k2, w1, w2, n_plane, materialclass_plane, n_sphere, materialclass_sphere, lmax, mie_a, mie_b)
    marr = np.fft.rfft(phiarr)
    return np.array([marr[0,:].real, marr[1,:].real, marr[2,:].imag, marr[3,:].imag])


def compute_mElement_diag(i, rho, K, N, M, k, w, n_plane, materialclass_plane, n_sphere, materialclass_sphere, lmax, mie_a, mie_b):
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
    n : float
        positive, refractive index
    lmax : int
        positive, cut-off angular momentum
    materialclass: string
        the material class (currently supports: drude, dielectric, PR)
    mie_a : list
        list of mie coefficients for electric polarizations
    mie_b : list
        list of mie coefficients for magnetic polarizations

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
    data = (m_array(rho, K, M, k[i], k[i], w[i], w[i], n_plane, materialclass_plane, n_sphere, materialclass_sphere, lmax, mie_a, mie_b))[:-1,:]
    return row, col, data


def compute_mElement_offdiag(i, j, rho, K, N, M, k, w, n_plane, materialclass_plane, n_sphere, materialclass_sphere, lmax, mie_a, mie_b):
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
    n : float
        positive, refractive index
    lmax : int
        positive, cut-off angular momentum
    materialclass: string
        the material class (currently supports: drude, dielectric, PR)
    mie_a : list
        list of mie coefficients for electric polarizations
    mie_b : list
        list of mie coefficients for magnetic polarizations

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
    data = m_array(rho, K, M, k[i], k[j], w[i], w[j], n_plane, materialclass_plane, n_sphere, materialclass_sphere, lmax, mie_a, mie_b)
    return row, col, data


def mArray_sparse_part(dindices, oindices, rho, K, N, M, k, w, n_plane, materialclass_plane, n_sphere, materialclass_sphere, lmax, mie_a, mie_b):
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
    n : float
        positive, refractive index
    lmax : int
        positive, cut-off angular momentum
    materialclass: string
        the material class (currently supports: drude, dielectric, PR)
    mie_a : list
        list of mie coefficients for electric polarizations
    mie_b : list
        list of mie coefficients for magnetic polarizations

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
    # 16 is just arbitrary here
    row = np.empty(16*N)
    col = np.empty(16*N)
    data = np.empty((16*N, M//2+1))
    
    ind = 0
    for i in dindices:
        if isFinite(rho, K, k[i], k[i]):
            row[ind:ind+3], col[ind:ind+3], data[ind:ind+3, :] = compute_mElement_diag(i, rho, K, N, M, k, w, n_plane, materialclass_plane, n_sphere, materialclass_sphere, lmax, mie_a, mie_b)
            ind += 3

    for oindex in oindices:
        i, j = itt(oindex)
        if isFinite(rho, K, k[i], k[j]):
            if ind+4 >= len(row):
                row = np.hstack((row, np.empty(len(row))))
                col = np.hstack((col, np.empty(len(row))))
                data = np.vstack((data, np.empty((len(row), M//2+1))))
            row[ind:ind+4], col[ind:ind+4], data[ind:ind+4, :] = compute_mElement_offdiag(i, j, rho, K, N, M, k, w, n_plane, materialclass_plane, n_sphere, materialclass_sphere, lmax, mie_a, mie_b)
            ind += 4
                
    row = row[:ind] 
    col = col[:ind] 
    data = data[:ind, :] 
    return row, col, data


def mArray_sparse_mp(nproc, rho, K, N, M, pts, wts, n_plane, materialclass_plane, n_sphere, materialclass_sphere, lmax, mie_a, mie_b):
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
    n : float
        positive, refractive index
    lmax : int
        positive, cut-off angular momentum
    materialclass: string
        the material class (currently supports: drude, dielectric, PR)
    mie_a : list
        list of mie coefficients for electric polarizations
    mie_b : list
        list of mie coefficients for magnetic polarizations

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
    b = 1. # has best convergence rate
    k = b * pts
    w = np.sqrt(b * wts * 2 * np.pi / M)
    
    ndiv = nproc*8 # factor is arbitrary, but can be chosen optimally

    dindices = np.array_split(np.random.permutation(N), ndiv)
    oindices = np.array_split(np.random.permutation(N * (N - 1) // 2), ndiv)

    with futures.ProcessPoolExecutor(max_workers=nproc) as executors:
        wait_for = [executors.submit(mArray_sparse_part, dindices[i], oindices[i], rho, K, N, M, k, w, n_plane, materialclass_plane, n_sphere, materialclass_sphere, lmax, mie_a, mie_b)
                    for i in range(ndiv)]
        results = [f.result() for f in futures.as_completed(wait_for)]
    
    # gather results into 3 arrays
    length = 0
    for i in range(ndiv):
        length += len(results[i][0])
    row = np.empty(length)
    col = np.empty(length)
    data = np.empty((length, results[0][2].shape[1]))
    ini = 0
    for i in range(ndiv):
        fin = ini + len(results[i][0])
        row[ini:fin] = results[i][0]
        col[ini:fin] = results[i][1]
        data[ini:fin] = results[i][2]
        ini = fin
    return row, col, data


@njit("boolean(float64, float64, float64, float64)", cache=True)
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


def LogDet(R, L, materials, Kvac, N, M, pts, wts, lmax, nproc):
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
    lmax : int
        positive, cut-off angular momentum
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
    start_matrix = time.time()
    n_plane = eval("material."+materials[0]+".n(Kvac/L)")
    materialclass_plane = eval("material." + materials[0] + ".materialclass")
    n_medium = eval("material."+materials[1]+".n(Kvac/L)")
    n_sphere = eval("material."+materials[2]+".n(Kvac/L)")
    materialclass_sphere = eval("material." + materials[2] + ".materialclass")

    nr_plane = n_plane / n_medium
    nr_sphere = n_sphere / n_medium
    # aspect ratio
    rho = R/L
    # size parameter
    x = n_medium*Kvac*rho
    # precompute mie coefficients
    if x == 0.:
        mie_a, mie_b = mie_e_array(2, 1., nr_sphere)
    elif x > 5e3:
        mie_a, mie_b = mie_e_array(2, x, nr_sphere)
    else:
        mie_a, mie_b = mie_e_array(lmax, x, nr_sphere)    # initial lmax arbitrary

    row, col, data = mArray_sparse_mp(nproc, rho, Kvac*n_medium, N, M, pts, wts, nr_plane, materialclass_plane, nr_sphere, materialclass_sphere, lmax, mie_a, mie_b)

    end_matrix = time.time()
    timing_matrix = end_matrix-start_matrix
    start_logdet = end_matrix
    
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
    end_logdet = time.time()
    timing_logdet = end_logdet-start_logdet
    print("# ", end="")
    print(Kvac, logdet, timing_matrix, timing_logdet, sep=", ")
    return logdet


def energy_zero(R, L, materials, N, M, lmax, nproc, X=None):
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
    lmax : int
        positive, cut-off angular momentum
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
        logdet = lambda Kvac : LogDet(R, L, materials, Kvac, N, M, pts, wts, lmax, nproc)
        energy = auto_integration(logdet)
    else:
        K_pts, K_wts = quadrature(X)
        
        energy = 0.
        for i in range(X):
            result = LogDet(R, L, materials, K_pts[i], N, M, pts, wts, lmax, nproc)
            energy += K_wts[i]*result
    return energy/(2*np.pi)


def energy_quad(R, L, materials, N, M, lmax, nproc):
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
    lmax : int
        positive, cut-off angular momentum
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
    logdet = lambda Kvac : LogDet(R, L, materials, Kvac, N, M, pts, wts, lmax, nproc)
    out = quad(logdet, 0, np.inf)
    return out[0]/(2*np.pi)


def energy_finite(R, L, T, materials, N, M, lmax, mode, epsrel, nproc):
    r"""
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
    lmax : int
        positive, cut-off angular momentum
    mode: str
        Matsubara spectrum decompostion (msd) or Pade spectrum decomposition (psd)
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
    energy0 = LogDet(R, L, materials, 0., N, M, pts, wts, lmax, nproc)

    if mode == "psd":
        energy = 0.
        Teff = 4*np.pi*Boltzmann/hbar/c*T*L
        order = int(max(np.ceil((1-1.5*np.log10(np.abs(epsrel)))/np.sqrt(Teff)), 5))
        xi, eta = psd(order)
        for n in range(order):
            term = 2*eta[n]*LogDet(R, L, materials, K_matsubara*xi[n], N, M, pts, wts, lmax, nproc)
            energy += term
    elif mode == "msd":
        energy = 0.
        n = 1
        while(True):
            term = LogDet(R, L, materials, 2*np.pi*K_matsubara*n, N, M, pts, wts, lmax, nproc)
            energy += 2*term
            if abs(term/energy0) < epsrel:
                break
            n += 1
    else:
        raise ValueError("mode can either be 'psd' or 'msd'")
     
    return 0.5*T*Boltzmann*(energy+energy0), 0.5*T*Boltzmann*energy


if __name__ == "__main__":
    np.random.seed(0)
    R = 50e-6
    L = 50e-6/100
    T = 300
    lmax = int(10*R/L)
    lmax = 2000
    #T = 1.e-03
    rho = R/L
    N = int(10*np.sqrt(rho))
    print("N", N)
    M = N
    nproc = 4
    
    #mie = mie_e_array(1e4, 1.*rho)
    #print(phiSequence(rho, 1., M, 2.3, 2.3, 1., 1., mie))
    mat = ("Gold", "Water", "Gold")
    start = time.time()
    #en = energy_finite(R, L, T, mat, N, M, "msd", 1e-8, nproc) 
    #print("msd", en)
    en = energy_finite(R, L, T, mat, N, M, lmax, "psd", 1e-8, nproc)
    #en = energy_quad(R, L, mat, N, M, nproc) 
    end = time.time()
    print("time")
    print(end-start)
    print("energy")
    print(en)
    print("PFA")
    print(-np.pi**3*rho/720)
