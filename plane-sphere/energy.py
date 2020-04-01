r""" Casimir energy for the plane-sphere geometry.

.. todo::
    * replace marray by just fft at the appropriate place

"""

import mkl
#mkl.domain_set_num_threads(1, "fft")
import numpy as np

from math import sqrt
import concurrent.futures as futures
from numba import njit
from scipy.linalg import cho_factor, cho_solve
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

    phiarr[0, :] = np.sign(rTM1)*w1*w2*sqrt(rTM1*rTM2)*phiarr[0, :]
    phiarr[1, :] = np.sign(rTE1)*w1*w2*sqrt(rTE1*rTE2)*phiarr[1, :]
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
    mkl.set_num_threads(1)
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


def mArray_sparse_mp(nproc, rho, K, N, M, pts, wts, n_plane, n_sphere, materialclass_plane, materialclass_sphere, lmax, mie_a, mie_b):
    r"""
    Computes the m-array in parallel using the multiprocessing module.

    Parameters
    ----------
    nproc: int
        number of processes
    rho: float
        positive, aspect ratio :math:`R/L`
    K: float
        positive, rescaled waven number
    N, M: int
        positive, quadrature order of k-integration and phi-integration
    pts, wts: np.ndarray
        quadrature points and weights of the k-integration before rescaling
    n_plane, n_sphere : float
        positive, relative refractive index of plane and sphere
    materialclass_plane, materialclass_sphere : string
        material class of plane and sphere
    lmax : int
        positive, cut-off angular momentum
    materialclass: string
        the material class (currently supports: drude, dielectric, PR)
    mie_a, mie_b : ndarray
        list of mie coefficients for electric and magnetic polarizations

    Returns
    -------
    (ndarray, ndarray, ndarray)
        array of row indices, column indices and matrix elements

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
    row = np.empty(length, dtype=np.int)
    col = np.empty(length, dtype=np.int)
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


@njit("UniTuple(float64[:,:,:], 3)(int64[:], int64[:], float64[:,:], int64, float64[:])", cache=True)
def construct_matrices(row, col, data, N, kappa):
    r"""Construct round-trip matrices and its first two derivatives for each m.

        Parameters
        ----------
        row, col : ndarray
            1D arrays of ints, row and column index of the matrix elements, respectively.
        data : ndarray
            2D arrays of floats, matrix elements for each :math:`m`
        N : int
            matrix dimension of each polarization block
        kappa : ndarray
            1D array of floats, imaginary z-component of the wave vectors

        Returns
        -------
        (ndarray, ndarray, ndarray)

    """
    data_len, m_len = data.shape
    rndtrp_matrix = np.zeros((2 * N, 2 * N, m_len))
    dL_rndtrp_matrix = np.zeros((2 * N, 2 * N, m_len))
    d2L_rndtrp_matrix = np.zeros((2 * N, 2 * N, m_len))
    for i in range(data_len):
        rndtrp_matrix[row[i], col[i], :] = data[i, :]
        rndtrp_matrix[col[i], row[i], :] = data[i, :]
        dL_rndtrp_matrix[row[i], col[i], :] = -(kappa[row[i] % N] + kappa[col[i] % N]) * data[i, :]
        dL_rndtrp_matrix[col[i], row[i], :] = dL_rndtrp_matrix[row[i], col[i], :]
        d2L_rndtrp_matrix[row[i], col[i], :] = (kappa[row[i] % N] + kappa[col[i] % N]) ** 2 * data[i, :]
        d2L_rndtrp_matrix[col[i], row[i], :] = d2L_rndtrp_matrix[row[i], col[i], :]
    return rndtrp_matrix, dL_rndtrp_matrix, d2L_rndtrp_matrix


def compute_matrix_operations(mat, dL_mat, d2L_mat, observable):
    r"""Computes a 3-tuple containing the quantities

        .. math::
            \begin{aligned}
            \log\det(1-\mathcal{M})\,,\\
            \mathrm{tr}\left[\frac{\partial_L\mathcal{M}}{1-\mathcal{M}}\right]\,,\\
            \mathrm{tr}\left[\frac{\partial_L^2\mathcal{M}}{1-\mathcal{M}} + \left(\frac{\partial_L\mathcal{M}}{1-\mathcal{M}}\right)^2\right]\,.
            \end{aligned}

        where :math:`\mathtt{mat}=\mathcal{M}`, :math:`\mathtt{dL_mat}=\partial_L\mathcal{M}` and :math:`\mathtt{d2L_mat}=\partial^2_L\mathcal{M}`.

        When observable="energy" only the first quantity is computed and the other two returned as zero.
        For observable="force" only the first two quantities are computed and the last one returned as zero.
        When observable="pressure" all quantities are computed.

        Parameters
        ----------
        mat : ndarray
            2D array, round-trip matrix
        dL_mat : ndarray
            2D array, first derivative of round-trip matrix
        d2L_mat : ndarray
            2D array, second derivative of round-trip matrix
        observable : string
            specification of which observables are to be computed, allowed values are "energy", "force", "pressure"

        Returns
        -------
        (float, float, float)


    """
    c, lower = cho_factor(np.eye(mat.shape[0]) - mat)
    logdet = 2*np.sum(np.log(np.diag(c)))
    if observable == "energy":
        return logdet, 0., 0.
    matA = cho_solve((c, lower), dL_mat)
    dL_logdet = np.trace(matA)
    if observable == "force":
        return logdet, dL_logdet, 0.
    matB = cho_solve((c, lower), d2L_mat)
    d2L_logdet = np.trace(matB) + np.sum(matA**2)
    if observable == "pressure":
        return logdet, dL_logdet, d2L_logdet
    else:
        raise ValueError


def LogDet(R, L, materials, Kvac, N, M, pts, wts, lmax, nproc, observable):
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

    row, col, data = mArray_sparse_mp(nproc, rho, Kvac*n_medium, N, M, pts, wts, nr_plane, nr_sphere, materialclass_plane, materialclass_sphere, lmax, mie_a, mie_b)

    end_matrix = time.time()
    timing_matrix = end_matrix-start_matrix
    start_logdet = end_matrix

    kappa = np.sqrt((n_medium*Kvac) ** 2 + pts ** 2)
    mat, dL_mat, d2L_mat = construct_matrices(row, col, data, N, kappa)

    # m=0
    logdet, dL_logdet, d2L_logdet = compute_matrix_operations(mat[:,:,0], dL_mat[:,:,0], d2L_mat[:,:,0], observable)

    # m>0    
    for m in range(1, M//2):
        term1, term2, term3 = compute_matrix_operations(mat[:,:,m], dL_mat[:,:,m], d2L_mat[:,:,m], observable)
        logdet += 2 * term1
        dL_logdet += 2 * term2
        d2L_logdet += 2 * term3

    # last m
    term1, term2, term3 = compute_matrix_operations(mat[:,:,M//2], dL_mat[:,:,M//2], d2L_mat[:,:,M//2], observable)
    if M%2==0:
        logdet += term1
        dL_logdet += term2
        d2L_logdet += term3
    else:
        logdet += 2 * term1
        dL_logdet += 2 * term2
        d2L_logdet += 2 * term3
    end_logdet = time.time()
    timing_logdet = end_logdet-start_logdet
    print("# ", end="")
    print(Kvac, logdet, timing_matrix, timing_logdet, sep=", ")
    return np.array([logdet, dL_logdet/L, d2L_logdet/L**2])


def casimir_zero(R, L, materials, N, M, lmax, nproc, observable, X=None):
    r"""
    Computes the Casimir interaction at zero temperature.

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
    observable : string
        specification of which observable is to be computed

    Returns
    -------
    ndarray
        1D array of length 3 containing the three Casimir observables in each entry:
        Casimir energy [J], force [N], force gradient [N/m]

    
    Dependencies
    ------------
    quadrature, get_mie, LogDet_sparse_mp

    """
    pts, wts = quadrature(N)
    if X == None:
        raise NotImplementedError("This part of the code might be broken.")
        logdet = lambda Kvac : LogDet(R, L, materials, Kvac, N, M, pts, wts, lmax, nproc, observable)
        casimir = auto_integration(logdet)
    else:
        K_pts, K_wts = quadrature(X)
        
        casimir = np.zeros(3)
        for i in range(X):
            result = LogDet(R, L, materials, K_pts[i], N, M, pts, wts, lmax, nproc, observable)
            casimir += K_wts[i]*result

    return casimir/(2*np.pi)*hbar*c/L


def casimir_quad(R, L, materials, N, M, lmax, nproc, observable):
    r"""
    Computes the Casimir interaction at zero temperature. The frequency integration is performed using the QUADPACK routine.

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
    observable : string
        specification of which observable is to be computed

    Returns
    -------
    ndarray
        1D array of length 3 containing the three Casimir observables in each entry:
        Casimir energy [J], force [N], force gradient [N/m]

    
    Dependencies
    ------------
    quadrature, get_mie, LogDet_sparse_mp

    """
    if observable != "energy":
        raise NotImplementedError("The T=0 quad-routine does not support force or force gradient at the moment.")
    pts, wts = quadrature(N)
    logdet = lambda Kvac : LogDet(R, L, materials, Kvac, N, M, pts, wts, lmax, nproc, "energy")[0]
    out = quad(logdet, 0, np.inf)
    return out[0]/(2*np.pi)*hbar*c/L


def casimir_finite(R, L, T, materials, N, M, lmax, mode, epsrel, nproc, observable, X=None):
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
    observable : string
        specification of which observable is to be computed

    Returns
    -------
    ndarray
        1D array of length 3 containing the three Casimir observables in each entry:
        Casimir energy [J], force [N], force gradient [N/m]
    
    Dependencies
    ------------
    quadrature, get_mie, LogDet_sparse_mp

    """
    pts, wts = quadrature(N)
    
    K_matsubara = Boltzmann*T*L/(hbar*c)
    energy0 = LogDet(R, L, materials, 0., N, M, pts, wts, lmax, nproc, observable)

    if mode == "psd":
        energy = np.zeros(3)
        Teff = 4*np.pi*Boltzmann/hbar/c*T*L
        if X == None:
            order = int(max(np.ceil((1-1.5*np.log10(np.abs(epsrel)))/np.sqrt(Teff)), 5))
        else:
            order = X
        xi, eta = psd(order)
        for n in range(order):
            term = 2*eta[n]*LogDet(R, L, materials, K_matsubara*xi[n], N, M, pts, wts, lmax, nproc, observable)
            energy += term
    elif mode == "msd":
        energy = np.zeros(3)
        n = 1
        if X == None:
            nmax = np.inf
        else:
            nmax = X

        while(n <= nmax):
            term = LogDet(R, L, materials, 2*np.pi*K_matsubara*n, N, M, pts, wts, lmax, nproc, observable)
            energy += 2*term
            if abs(term[0]/energy0[0]) < epsrel:
                break
            n += 1
    else:
        raise ValueError("mode can either be 'psd' or 'msd'")
     
    return 0.5*T*Boltzmann*(energy+energy0), 0.5*T*Boltzmann*energy


if __name__ == "__main__":
    # import program and disable asymptotics of S1S2
    """
    #sys.path.append(NYSTROM_PATH+"/sphere")
    from scattering_amplitude import S1S2
    import kernel as ker

    @njit
    def S1S2_no_asymptotics(x, z, n, lmax, mie_a, mie_b, use_asymptotics):
        return S1S2(x, z, n, lmax, mie_a, mie_b, False)

    ker.S1S2 = S1S2_no_asymptotics
    kernel.recompile()
    phi_array.recompile()
    """
    np.random.seed(0)
    R = 1.e-6
    L = 1.e-6/1000
    T = 293
    mat = ("PR", "Vacuum", "PR")
    lmax = int(14.*R/L)
    #T = 1.e-03
    rho = R/L
    N = int(14*np.sqrt(rho))
    print("N", N)
    M = N
    nproc = 4
    
    K_matsubara = Boltzmann*T*L/(hbar*c)
    Teff = 4*np.pi*Boltzmann/hbar/c*T*L
    epsrel = 1.e-10
    order = int(max(np.ceil((1-1.5*np.log10(np.abs(epsrel)))/np.sqrt(Teff)), 5))
    xi, eta = psd(order)
    Kvac = (xi*K_matsubara)[-5]
    
    pts, wts = quadrature(N)
    start = time.time()
    term = 2*eta[-5]*LogDet(R, L, mat, Kvac, N, M, pts, wts, lmax, nproc, "energy")
    end = time.time()
    
    #mie = mie_e_array(1e4, 1.*rho)
    #print(phiSequence(rho, 1., M, 2.3, 2.3, 1., 1., mie))
    #mat = ("Gold", "Water", "Gold")
    #en = energy_finite(R, L, T, mat, N, M, "msd", 1e-8, nproc) 
    #print("msd", en)
    #en = casimir_finite(R, L, T, mat, N, M, lmax, "psd", 1e-8, nproc, "energy")
    #en = energy_quad(R, L, mat, N, M, nproc) 
    print("time")
    print(end-start)
    print("energy")
    #print(en)
    print("PFA")
    print(-np.pi**3*rho/720)
