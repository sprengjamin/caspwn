r""" Casimir energy for the plane-sphere geometry.

.. todo::
    * replace dft_matrix_elements by a FFT after the whole round-trip matrix has been computed?
    * construct matrices for each m (to save memory)

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

from index import unpack
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../sphere/"))
from mie import mie_e_array
from kernels import kernel_polar_Kfinite as kernel
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../plane/"))
from fresnel import rTM_Kfinite as rTM
from fresnel import rTE_Kfinite as rTE
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../ufuncs/"))
from integration import quadrature, auto_integration
from psd import psd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../material/"))
import material


@njit("float64[:,:](float64, float64, int64, float64, float64, float64, float64, float64, float64, int64, float64[:], float64[:])", cache=True)
def angular_matrix_elements(rho, K, M, k1, k2, w1, w2, n_plane, n_sphere, lmax, mie_a, mie_b):
    r"""
    Computes the matrix elements of the round-trip operator for all angular
    transverse momenta with discretization order :math:`M` at fixed radial
    transverse momenta k1 and k2.

    Parameters
    ----------
    rho: float
        positive, aspect ratio :math:`R/L`
    K: float
        positive, rescaled wavenumber in medium
    M: int
        positive, angular discretizaton order
    k1, k2: float
        positive, rescaled transverse wave numbers
    w1, w2: float
        positive, total quadrature weights corresponding to k1 and k2
    n_plane, n_sphere : float
        positive, refractive index of plane and sphere
    lmax : int
        positive, cut-off angular momentum
    mie_a, mie_b : list
        list of mie coefficients for electric and magnetic polarizations

    Returns
    -------
    np.ndarray
        array of shape (4, M) for the polarization contributions
        TMTM, TETE, TMTE, TETM

    """
    phiarr = np.empty((4, M))

    # phi = 0.
    phiarr[0, 0], phiarr[1, 0], phiarr[2, 0], phiarr[3, 0] = kernel(rho, 1., 1., K, k1, k2, 0., n_sphere, lmax, mie_a, mie_b)
    
    if M%2==0:
        # phi = np.pi is assumed
        phiarr[0, M//2], phiarr[1, M//2], phiarr[2, M//2], phiarr[3, M//2] = kernel(rho, 1., 1., K, k1, k2, np.pi, n_sphere, lmax, mie_a, mie_b)
        imax = M//2-1
        phi = 2*np.pi*np.arange(M//2)/M
    else:
        # phi = np.pi is not assumed
        imax = M//2
        phi = 2*np.pi*np.arange(M//2+1)/M
   
    # 0 < phi < np.pi
    for i in range(1, imax+1):
        phiarr[0, i], phiarr[1, i], phiarr[2, i], phiarr[3, i] = kernel(rho, 1., 1., K, k1, k2, phi[i], n_sphere, lmax, mie_a, mie_b)
        phiarr[0, M-i] = phiarr[0, i]
        phiarr[1, M-i] = phiarr[1, i]
        phiarr[2, M-i] = -phiarr[2, i]
        phiarr[3, M-i] = -phiarr[3, i]

    rTE1 = rTE(K, k1, n_plane**2)
    rTM1 = rTM(K, k1, n_plane**2)
    rTE2 = rTE(K, k2, n_plane**2)
    rTM2 = rTM(K, k2, n_plane**2)

    phiarr[0, :] = np.sign(rTM1)*w1*w2*sqrt(rTM1*rTM2)*phiarr[0, :]
    phiarr[1, :] = np.sign(rTE1)*w1*w2*sqrt(rTE1*rTE2)*phiarr[1, :]
    phiarr[2, :] = w1*w2*sqrt(-rTM1*rTE2)*phiarr[2, :]
    phiarr[3, :] = w1*w2*sqrt(-rTE1*rTM2)*phiarr[3, :]
    return phiarr
        

def dft_matrix_elements(rho, K, M, k1, k2, w1, w2, n_plane, n_sphere,  lmax, mie_a, mie_b):
    r"""
    Computes the discrete Fourier transformed matrix elements of the round-trip operator at angular discretization order :math:`M` at fixed radial
    transverse momenta k1 and k2.

    Parameters
    ----------
    rho: float
        positive, aspect ratio :math:`R/L`
    K: float
        positive, rescaled wavenumber in medium
    M: int
        positive, angular discretization order
    k1, k2: float
        positive, rescaled transverse wave numbers
    w1, w2: float
        positive, total quadrature weights corresponding to k1 and k2
    n_plane, n_sphere : float
        positive, refractive index of plane and sphere
    lmax : int
        positive, cut-off angular momentum
    mie_a, mie_b : list
        list of mie coefficients for electric and magnetic polarizations
    
    Returns
    -------
    np.ndarray
        array of shape (4, M//2) for the polarization contributions
        TMTM, TETE, TMTE, TETM

    Dependencies
    ------------
    angular_matrix_elements

    """
    phiarr = angular_matrix_elements(rho, K, M, k1, k2, w1, w2, n_plane, n_sphere, lmax, mie_a, mie_b)
    marr = np.fft.rfft(phiarr)
    return np.array([marr[0,:].real, marr[1,:].real, marr[2,:].imag, marr[3,:].imag])


def dftME_diag(i, rho, K, N, M, k, w, n_plane, n_sphere, lmax, mie_a, mie_b):
    r"""
    Computes diagonal dft matrix elements with respect to the
    transverse momenta specified by index i.

    Parameters
    ----------
    i: int
        non-negative, row or column index of the diagonal element
    rho: float
        positive, aspect ratio :math:`R/L`
    K: float
        positive, rescaled wavenumber in medium
    N, M: int
        positive, radial and angular discretization order
    k: np.ndarray
        nodes of the radial quadrature rule
    w: np.ndarray
        symmetrized total weights
    n_plane, n_sphere : float
        positive, refractive index of plane and sphere
    lmax : int
        positive, cut-off angular momentum
    mie_a, mie_b : list
        list of mie coefficients for electric and magnetic polarizations

    Returns
    -------
    row: np.ndarray
        row indices
    col: np.ndarray
        column indices
    data: np.ndarray
        dft matrix elements

    Dependencies
    ------------
    dft_matrix_elements

    """
    row = [i, N+i, N+i]
    col = [i, N+i, i]
    data = (dft_matrix_elements(rho, K, M, k[i], k[i], w[i], w[i], n_plane, n_sphere, lmax, mie_a, mie_b))[:-1,:]
    return row, col, data


def dftME_offdiag(i, j, rho, K, N, M, k, w, n_plane, n_sphere, lmax, mie_a, mie_b):
    r"""
    Computes off-diagonal dft matrix elements with respect to the
    transverse momenta specified by index i and j.

    Parameters
    ----------
    i: int
        non-negative, row index of the off-diagonal element
    j: int
        non-negative, column index of the off-diagonal element
    rho: float
        positive, aspect ratio :math:`R/L`
    K: float
        positive, rescaled wavenumber in medium
    N, M: int
        positive, radial and angular discretization order
    k: np.ndarray
        nodes of the radial quadrature rule
    w: np.ndarray
        symmetrized total weights
    n_plane, n_sphere : float
        positive, refractive index of plane and sphere
    lmax : int
        positive, cut-off angular momentum
    mie_a, mie_b : list
        list of mie coefficients for electric and magnetic polarizations

    Returns
    -------
    row: np.ndarray
        row indices
    col: np.ndarray
        column indices
    data: np.ndarray
        dft matrix elements

    Dependencies
    ------------
    dft_matrix_elements

    """
    row = [i, N+i, N+j, N+i] 
    col = [j, N+j, i, j] 
    data = dft_matrix_elements(rho, K, M, k[i], k[j], w[i], w[j], n_plane, n_sphere, lmax, mie_a, mie_b)
    return row, col, data


def dftME_partial(dindices, oindices, rho, K, N, M, k, w, n_plane, n_sphere, lmax, mie_a, mie_b):
    r"""
    Computes dft matrix elements with respect to the transverse momenta
    specified by the arrays dindices and oindices specifying indices on the
    diagonal and off the diagonal, respectively.

    Parameters
    ----------
    dindices, oindices: np.ndarray
        array of diagonal and off-diagonal indices
    rho: float
        positive, aspect ratio :math:`R/L`
    K: float
        positive, rescaled wavenumber in medium
    N, M: int
        positive, radial and angular discretization order
    k: np.ndarray
        nodes of the radial quadrature rule
    w: np.ndarray
        symmetrized total weights
    n_plane, n_sphere : float
        positive, refractive index of plane and sphere
    lmax: int
        positive, cut-off angular momentum
    mie_a, mie_b: list
        list of mie coefficients for electric and magnetic polarizations

    Returns
    -------
    row: np.ndarray
        row indices
    col: np.ndarray
        column indices
    data: np.ndarray
        dft matrix elements

    Dependencies
    ------------
    isFinite, dftME_diag, itt, dftME_offdiag

    """
    # 16 is just arbitrary here
    mkl.set_num_threads(1)
    row = np.empty(16*N)
    col = np.empty(16*N)
    data = np.empty((16*N, M//2+1))
    
    ind = 0
    for i in dindices:
        if isFinite(rho, K, k[i], k[i]):
            row[ind:ind+3], col[ind:ind+3], data[ind:ind+3, :] = dftME_diag(i, rho, K, N, M, k, w, n_plane, n_sphere, lmax, mie_a, mie_b)
            ind += 3

    for oindex in oindices:
        i, j = unpack(oindex)
        if isFinite(rho, K, k[i], k[j]):
            if ind+4 >= len(row):
                row = np.hstack((row, np.empty(len(row))))
                col = np.hstack((col, np.empty(len(row))))
                data = np.vstack((data, np.empty((len(row), M//2+1))))
            row[ind:ind+4], col[ind:ind+4], data[ind:ind+4, :] = dftME_offdiag(i, j, rho, K, N, M, k, w, n_plane, n_sphere, lmax, mie_a, mie_b)
            ind += 4
                
    row = row[:ind] 
    col = col[:ind] 
    data = data[:ind, :] 
    return row, col, data


def dftME_full(nproc, rho, K, N, M, nds, wts, n_plane, n_sphere, lmax, mie_a, mie_b):
    r"""
    Computes all dft matrix elements. The calculation is parellelized among
    nproc processes.

    Parameters
    ----------
    nproc: int
        number of processes
    rho: float
        positive, aspect ratio :math:`R/L`
    K: float
        positive, rescaled wavenumber in medium
    N, M: int
        positive, radial and angular discretization order
    nds, wts: np.ndarray
        nodes and weights of the radial quadrature rule
    n_plane, n_sphere : float
        positive, refractive index of plane and sphere
    lmax : int
        positive, cut-off angular momentum
    mie_a, mie_b : list
        list of mie coefficients for electric and magnetic polarizations

    Returns
    -------
    (ndarray, ndarray, ndarray)
        array of row indices, column indices and matrix elements

    Dependencies
    ------------
    dftME_partial

    """
    k = nds
    w = np.sqrt(wts * 2 * np.pi / M)
    
    ndiv = nproc*8 # factor is arbitrary, but can be chosen optimally

    dindices = np.array_split(np.random.permutation(N), ndiv)
    oindices = np.array_split(np.random.permutation(N * (N - 1) // 2), ndiv)

    with futures.ProcessPoolExecutor(max_workers=nproc) as executors:
        wait_for = [executors.submit(dftME_partial, dindices[i], oindices[i], rho, K, N, M, k, w, n_plane, n_sphere, lmax, mie_a, mie_b)
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
    Estimator for matrix elements.

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
        True if the matrix element cannot be neglected

    """
    if K == 0.:
        exponent = 2*rho*sqrt(k1*k2) - (k1+k2)*(rho+1)
    else:
        kappa1 = sqrt(k1*k1+K*K)
        kappa2 = sqrt(k2*k2+K*K)
        exponent = -((k1 - k2)**2)/(sqrt(2*(kappa1*kappa2 + k1*k2 + K**2)) + kappa1 + kappa2)*rho - kappa1 - kappa2
    if exponent < -37:
        return False
    else:
        return True


@njit("UniTuple(float64[:,:], 3)(int64[:], int64[:], float64[:], int64, float64[:])", cache=True)
def construct_matrices(row, col, data, N, kappa):
    r"""Construct round-trip matrices and its first two derivatives for each m.

        Parameters
        ----------
        row, col : ndarray
            1D arrays of ints, row and column index of the matrix elements, respectively.
        data : ndarray
            1D array of floats, matrix elements
        N : int
            matrix dimension of each polarization block
        kappa : ndarray
            1D array of floats, imaginary z-component of the wave vectors

        Returns
        -------
        (ndarray, ndarray, ndarray)

    """
    rndtrp_matrix = np.zeros((2 * N, 2 * N))
    dL_rndtrp_matrix = np.zeros((2 * N, 2 * N))
    d2L_rndtrp_matrix = np.zeros((2 * N, 2 * N))
    for i in range(len(data)):
        rndtrp_matrix[row[i], col[i]] = data[i]
        rndtrp_matrix[col[i], row[i]] = data[i]
        dL_rndtrp_matrix[row[i], col[i]] = -(kappa[row[i] % N] + kappa[col[i] % N]) * data[i]
        dL_rndtrp_matrix[col[i], row[i]] = dL_rndtrp_matrix[row[i], col[i]]
        d2L_rndtrp_matrix[row[i], col[i]] = (kappa[row[i] % N] + kappa[col[i] % N]) ** 2 * data[i]
        d2L_rndtrp_matrix[col[i], row[i]] = d2L_rndtrp_matrix[row[i], col[i]]
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


def K_contribution(R, L, K, nr_plane, nr_sphere, N, M, nds, wts, lmax, nproc, observable):
    r"""
    Computes the contribution to the observable depending on the wave number K.

    Parameters
    ----------
    R: float
        positive, radius of the sphere
    L: float
        positive, surface-to-surface distance
    K: float
        positive, rescaled wavenumber in medium
    nr_plane, nr_sphere: float
        positive, refractive index of plane and sphere (relative to medium)
    N, M: int
        positive, radial and angular discretization order
    nds, wts: np.ndarray
        nodes and weights of the radial quadrature rule
    n_plane, n_sphere : float
        positive, refractive index of plane and sphere
    lmax : int
        positive, cut-off angular momentum
    nproc: int
        number of processes
    observable : string
        observable to be computed

    Returns
    -------
    logdet: float
        the sum of logdets of the m-matrices

    Dependencies
    ------------
    dftME_full, mie_e_array

    """
    start_matrix = time.time()

    # aspect ratio
    rho = R/L

    # size parameter
    x = K*rho

    # precompute mie coefficients
    if x > 5e3:
        # dummy variables
        mie_a, mie_b = mie_e_array(2, x, nr_sphere)
    else:
        mie_a, mie_b = mie_e_array(lmax, x, nr_sphere)

    row, col, data = dftME_full(nproc, rho, K, N, M, nds, wts, nr_plane, nr_sphere, lmax, mie_a, mie_b)

    end_matrix = time.time()
    timing_matrix = end_matrix-start_matrix
    start_logdet = end_matrix

    kappa = np.sqrt(K ** 2 + nds ** 2)

    # m=0
    mat, dL_mat, d2L_mat = construct_matrices(row, col, data[:,0], N, kappa)
    logdet, dL_logdet, d2L_logdet = compute_matrix_operations(mat, dL_mat, d2L_mat, observable)

    # m>0    
    for m in range(1, M//2):
        mat, dL_mat, d2L_mat = construct_matrices(row, col, data[:,m], N, kappa)
        term1, term2, term3 = compute_matrix_operations(mat, dL_mat, d2L_mat, observable)
        logdet += 2 * term1
        dL_logdet += 2 * term2
        d2L_logdet += 2 * term3

    # last m
    mat, dL_mat, d2L_mat = construct_matrices(row, col, data[:,M//2], N, kappa)
    term1, term2, term3 = compute_matrix_operations(mat, dL_mat, d2L_mat, observable)
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
    print(K, logdet, timing_matrix, timing_logdet, sep=", ")
    return np.array([logdet, dL_logdet/L, d2L_logdet/L**2])


def casimir_T0(R, L, nfunc_plane, nfunc_medium, nfunc_sphere, N, M, lmax, nproc, observable, X=None):
    r"""
    Computes the Casimir interaction at zero temperature.

    Parameters
    ----------
    R: float
        positive, radius of the sphere
    L: float
        positive, surface-to-surface distance
    nfunc_plane, nfunc_medium, nfunc_sphere : function
        refractive index as a function of the wavenumber for the plane, medium
        and sphere
    N, M: int
        positive, radial and angular discretization order
    lmax : int
        positive, cut-off angular momentum
    nproc: int
        number of processes spawned by multiprocessing module
    observable : string
        observable to be computed

    Returns
    -------
    ndarray
        1D array of length 3 containing the three Casimir observables in each entry:
        Casimir energy [J], force [N], force gradient [N/m]

    
    Dependencies
    ------------
    quadrature, K_contribution

    """
    pts, wts = quadrature(N)
    if X == None:
        raise NotImplementedError("This part of the code broke.")
    else:
        K_pts, K_wts = quadrature(X)
        
        casimir = np.zeros(3)
        for i in range(X):
            Kvac = K_pts[i]/L
            n_plane = nfunc_plane(Kvac)
            n_medium = nfunc_medium(Kvac)
            n_sphere = nfunc_sphere(Kvac)
            nr_plane = n_plane/n_medium
            nr_sphere = n_sphere/n_medium
            Kmed = n_medium*Kvac
            result = K_contribution(R, L, Kmed*L, nr_plane, nr_sphere, N, M, pts, wts, lmax, nproc, observable)
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
    N, M: int
        positive, radial and angular discretization order
    lmax : int
        positive, cut-off angular momentum
    nproc: int
        number of processes spawned by multiprocessing module
    observable : string
        observable is to be computed

    Returns
    -------
    ndarray
        1D array of length 3 containing the three Casimir observables in each entry:
        Casimir energy [J], force [N], force gradient [N/m]

    
    Dependencies
    ------------
    quadrature, get_mie, LogDet_sparse_mp

    """
    raise NotImplementedError("Broken.")
    if observable != "energy":
        raise NotImplementedError("The T=0 quad-routine does not support force or force gradient at the moment.")
    pts, wts = quadrature(N)
    logdet = lambda Kvac : LogDet(R, L, materials, Kvac, N, M, pts, wts, lmax, nproc, "energy")[0]
    out = quad(logdet, 0, np.inf)
    return out[0]/(2*np.pi)*hbar*c/L


def casimir_Tfinite(R, L, T, nfunc_plane, nfunc_medium, nfunc_sphere, N, M, lmax, mode, epsrel, nproc, observable, X=None):
    r"""
    Computes the Casimir free energy at equilibrium temperature :math:`T`.

    Parameters
    ----------
    R: float
        positive, radius of the sphere
    L: float
        positive, surface-to-surface distance between plane and sphere
    nfunc_plane, nfunc_medium, nfunc_sphere : function
        refractive index as a function of the wavenumber for the plane, medium
        and sphere
    N, M: int
        positive, radial and angular discretization order
    lmax : int
        positive, cut-off angular momentum
    mode: str
        Matsubara spectrum decompostion (msd) or Pade spectrum decomposition (psd)
    epsrel: float
        positive, desired relative error for the matsubara sum
    nproc: int
        number of processes
    observable : string
        observable is to be computed

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
    
    K_matsubara1 = 2*np.pi*Boltzmann*T/(hbar*c)
    
    energy = np.zeros(3)
    if mode == "psd":
        Teff = 4*np.pi*Boltzmann/hbar/c*T*L
        if X == None:
            order = int(max(np.ceil((1-1.5*np.log10(np.abs(epsrel)))/np.sqrt(Teff)), 5))
        else:
            order = X
        xi, eta = psd(order)
        for n in range(order):
            Kvac = K_matsubara1*xi[n]/2/np.pi
            n_plane = nfunc_plane(Kvac)
            n_medium = nfunc_medium(Kvac)
            n_sphere = nfunc_sphere(Kvac)
            nr_plane = n_plane/n_medium
            nr_sphere = n_sphere/n_medium
            term = 2*eta[n]*K_contribution(R, L, n_medium*Kvac*L, nr_plane, nr_sphere, N, M, pts, wts, lmax, nproc, observable)
            energy += term
    elif mode == "msd":
        n = 1
        if X == None:
            nmax = np.inf
        else:
            nmax = X

        while(n <= nmax):
            Kvac = K_matsubara1*n
            n_plane = nfunc_plane(Kvac)
            n_medium = nfunc_medium(Kvac)
            n_sphere = nfunc_sphere(Kvac)
            nr_plane = n_plane/n_medium
            nr_sphere = n_sphere/n_medium
            term = K_contribution(R, L, n_medium*Kvac*L, nr_plane, nr_sphere, N, M, pts, wts, lmax, nproc, observable)
            energy += 2*term
            if abs(term[0]/energy[0]) < epsrel:
                break
            n += 1
    else:
        raise ValueError("mode can either be 'psd' or 'msd'")
     
    return 0.5*T*Boltzmann*energy


if __name__ == "__main__":
    R = 1.e-6
    L = R/100
    T = 293
    
    nfunc_plane = material.PS1.n    
    nfunc_medium = material.Vacuum.n    
    nfunc_sphere = material.PS1.n    

    rho = max(R/L, 50)
    N = int(8.*np.sqrt(rho))
    M = int(6.*np.sqrt(rho))
    lmax = int(12*rho)

    mode = "psd"
    epsrel = 1.e-6
    nproc = 4
    observable = "energy"
        
    #en = casimir_T0(R, L, nfunc_plane, nfunc_medium, nfunc_sphere, N, M, lmax, nproc, observable, X=40)
    en = casimir_Tfinite(R, L, T, nfunc_plane, nfunc_medium, nfunc_sphere, N, M, lmax, mode, epsrel, nproc, observable, X=None)
    print(en)
