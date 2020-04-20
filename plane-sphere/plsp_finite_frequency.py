r""" Casimir energy for the plane-sphere geometry for finite frequency/wavenumber.

.. todo::
    * replace dft_matrix_elements by a FFT after the whole round-trip matrix has been computed?
    * construct matrices for each m (to save memory)

"""

import numpy as np

from math import sqrt
import concurrent.futures as futures
from numba import njit
from scipy.linalg import cho_factor, cho_solve
from scipy.integrate import quad
from scipy.constants import Boltzmann, hbar, c
import time

from index import itt_scalar
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../sphere/"))
from mie import mie_e_array
from kernels import kernel_polar_finite as kernel
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../plane/"))
from fresnel import rTM_finite as rTM
from fresnel import rTE_finite as rTE
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../ufuncs/"))
from integration import quadrature, auto_integration
from psd import psd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../material/"))
import material


@njit("UniTuple(float64[:], 4)(float64, float64, int64, float64, float64, float64, float64, float64, float64, int64, float64[:], float64[:])", cache=True)
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
    4-tuple
        with arrays of length M for the polarization contributions
        TMTM, TETE, TMTE, TETM

    """
    phiarrTMTM = np.empty(M)
    phiarrTETE = np.empty(M)
    phiarrTMTE = np.empty(M)
    phiarrTETM = np.empty(M)
    
    rTE1 = rTE(K, k1, n_plane**2)
    rTM1 = rTM(K, k1, n_plane**2)
    rTE2 = rTE(K, k2, n_plane**2)
    rTM2 = rTM(K, k2, n_plane**2)
    
    factorTMTM = np.sign(rTM1)*w1*w2*sqrt(rTM1*rTM2)
    factorTETE = np.sign(rTE1)*w1*w2*sqrt(rTE1*rTE2)
    factorTMTE = w1*w2*sqrt(-rTM1*rTE2)
    factorTETM = w1*w2*sqrt(-rTE1*rTM2)

    # phi = 0.
    phiarrTMTM[0], phiarrTETE[0], phiarrTMTE[0], phiarrTETM[0] = kernel(rho, 1., 1., K, k1, k2, 0., n_sphere, lmax, mie_a, mie_b)
    phiarrTMTM[0] *= factorTMTM
    phiarrTETE[0] *= factorTETE
    phiarrTMTE[0] *= factorTMTE
    phiarrTETM[0] *= factorTETM
    
    if M%2==0:
        # phi = np.pi is assumed
        phiarrTMTM[M//2], phiarrTETE[M//2], phiarrTMTE[M//2], phiarrTETM[M//2] = kernel(rho, 1., 1., K, k1, k2, np.pi, n_sphere, lmax, mie_a, mie_b)
        phiarrTMTM[M//2] *= factorTMTM
        phiarrTETE[M//2] *= factorTETE
        phiarrTMTE[M//2] *= factorTMTE
        phiarrTETM[M//2] *= factorTETM
        imax = M//2-1
        phi = 2*np.pi*np.arange(M//2)/M
    else:
        # phi = np.pi is not assumed
        imax = M//2
        phi = 2*np.pi*np.arange(M//2+1)/M
   
    # 0 < phi < np.pi
    for i in range(1, imax+1):
        phiarrTMTM[i], phiarrTETE[i], phiarrTMTE[i], phiarrTETM[i] = kernel(rho, 1., 1., K, k1, k2, phi[i], n_sphere, lmax, mie_a, mie_b)
        phiarrTMTM[i] *= factorTMTM
        phiarrTETE[i] *= factorTETE
        phiarrTMTE[i] *= factorTMTE
        phiarrTETM[i] *= factorTETM
        phiarrTMTM[M-i] = phiarrTMTM[i]
        phiarrTETE[M-i] = phiarrTETE[i]
        phiarrTMTE[M-i] = -phiarrTMTE[i]
        phiarrTETM[M-i] = -phiarrTETM[i]
    return phiarrTMTM, phiarrTETE, phiarrTMTE, phiarrTETM
        

def ME_partial(indices, rho, K, N, M, k, w, n_plane, n_sphere, lmax, mie_a, mie_b):
    r"""
    Computes dft matrix elements with respect to the transverse momenta
    specified by the arrays indices specifying indices on the
    diagonal and off the diagonal, respectively.

    Parameters
    ----------
    indices : np.ndarray
        array containing indices
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
    dataTMTM, dataTETE, dataTMTE, dataTETM : np.ndarray
        angular matrix elements for TMTM, TETE, TMTE, TETM polarization

    Dependencies
    ------------
    isFinite, angular_matrix_elements

    """
    row = np.empty(N)
    col = np.empty(N)
    dataTMTM = np.empty((N, M))
    dataTETE = np.empty((N, M))
    dataTMTE = np.empty((N, M))
    dataTETM = np.empty((N, M))

    ind = 0
    for index in indices:
        i, j = itt_scalar(index)
        if isFinite(rho, K, k[i], k[j]):
            if ind+1 >= len(row):
                row = np.hstack((row, np.empty(len(row))))
                col = np.hstack((col, np.empty(len(row))))
                dataTMTM = np.vstack((dataTMTM, np.empty((len(row), M))))
                dataTETE = np.vstack((dataTETE, np.empty((len(row), M))))
                dataTMTE = np.vstack((dataTMTE, np.empty((len(row), M))))
                dataTETM = np.vstack((dataTETM, np.empty((len(row), M))))
            row[ind] = i
            col[ind] = j
            dataTMTM[ind, :], dataTETE[ind, :], dataTMTE[ind, :], dataTETM[ind, :] = angular_matrix_elements(rho, K, M, k[i], k[j], w[i], w[j], n_plane, n_sphere, lmax, mie_a, mie_b)
            ind += 1
                
    row = row[:ind] 
    col = col[:ind] 
    dataTMTM = dataTMTM[:ind, :] 
    dataTETE = dataTETE[:ind, :] 
    dataTMTE = dataTMTE[:ind, :] 
    dataTETM = dataTETM[:ind, :] 
    return row, col, dataTMTM, dataTETE, dataTMTE, dataTETM


def ME_full(nproc, rho, K, N, M, nds, wts, n_plane, n_sphere, lmax, mie_a, mie_b):
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
    ME_partial

    """
    k = nds
    w = np.sqrt(wts * 2 * np.pi / M)
    
    ndiv = nproc*8 # factor is arbitrary, but can be chosen optimally

    indices = np.array_split(np.random.permutation(N * (N + 1) // 2), ndiv)

    with futures.ProcessPoolExecutor(max_workers=nproc) as executors:
        wait_for = [executors.submit(ME_partial, indices[i], rho, K, N, M, k, w, n_plane, n_sphere, lmax, mie_a, mie_b)
                    for i in range(ndiv)]
        results = [f.result() for f in futures.as_completed(wait_for)]
    
    # gather results into 3 arrays
    length = 0
    for i in range(ndiv):
        length += len(results[i][0])
    row = np.empty(length, dtype=np.int)
    col = np.empty(length, dtype=np.int)
    dataTMTM = np.empty((length, M))
    dataTETE = np.empty((length, M))
    dataTMTE = np.empty((length, M))
    dataTETM = np.empty((length, M))
    ini = 0
    for i in range(ndiv):
        fin = ini + len(results[i][0])
        row[ini:fin] = results[i][0]
        col[ini:fin] = results[i][1]
        dataTMTM[ini:fin] = results[i][2]
        dataTETE[ini:fin] = results[i][3]
        dataTMTE[ini:fin] = results[i][4]
        dataTETM[ini:fin] = results[i][5]
        ini = fin
    return row, col, dataTMTM, dataTETE, dataTMTE, dataTETM


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


@njit("UniTuple(float64[:,:], 3)(int64[:], int64[:], float64[:], float64[:], float64[:], float64[:], int64, float64[:])", cache=True)
def construct_matrices(row, col, dataTMTM, dataTETE, dataTMTE, dataTETM, N, kappa):
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
    mat = np.zeros((2 * N, 2 * N))
    dL_mat = np.zeros((2 * N, 2 * N))
    d2L_mat = np.zeros((2 * N, 2 * N))
    for i in range(len(row)):
        factor = kappa[row[i]] + kappa[col[i]]
        mat[row[i], col[i]] = dataTMTM[i]
        mat[col[i], row[i]] = dataTMTM[i]
        mat[row[i]+N, col[i]+N] = dataTETE[i]
        mat[col[i]+N, row[i]+N] = dataTETE[i]
        mat[row[i], col[i]+N] = dataTMTE[i]
        mat[col[i]+N, row[i]] = dataTMTE[i]
        mat[row[i]+N, col[i]] = dataTETM[i]
        mat[col[i], row[i]+N] = dataTETM[i]
        
        dL_mat[row[i], col[i]] = -factor*dataTMTM[i]
        dL_mat[col[i], row[i]] = dL_mat[row[i], col[i]]
        dL_mat[row[i]+N, col[i]+N] = -factor*dataTETE[i]
        dL_mat[col[i]+N, row[i]+N] = dL_mat[row[i]+N, col[i]+N]
        dL_mat[row[i], col[i]+N] = -factor*dataTMTE[i]
        dL_mat[col[i]+N, row[i]] = dL_mat[row[i], col[i]+N]
        dL_mat[row[i]+N, col[i]] = -factor*dataTETM[i]
        dL_mat[col[i], row[i]+N] = dL_mat[row[i]+N, col[i]]

        d2L_mat[row[i], col[i]] = factor ** 2 * dataTMTM[i]
        d2L_mat[col[i], row[i]] = d2L_mat[row[i], col[i]]
        d2L_mat[row[i]+N, col[i]+N] = factor ** 2 * dataTETE[i]
        d2L_mat[col[i]+N, row[i]+N] = d2L_mat[row[i]+N, col[i]+N]
        d2L_mat[row[i], col[i]+N] = factor ** 2 * dataTMTE[i]
        d2L_mat[col[i]+N, row[i]] = d2L_mat[row[i], col[i]+N]
        d2L_mat[row[i]+N, col[i]] = factor ** 2 * dataTETM[i]
        d2L_mat[col[i], row[i]+N] = d2L_mat[row[i]+N, col[i]]
    return mat, dL_mat, d2L_mat

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
    ME_full, mie_e_array

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

    row, col, dataTMTM, dataTETE, dataTMTE, dataTETM = ME_full(nproc, rho, K, N, M, nds, wts, nr_plane, nr_sphere, lmax, mie_a, mie_b)

    end_matrix = time.time()
    timing_matrix = end_matrix-start_matrix
    start_dft = end_matrix
    if len(row) != 0.:
        dataTMTM = np.fft.rfft(dataTMTM).real
        dataTETE = np.fft.rfft(dataTETE).real
        dataTMTE = np.fft.rfft(dataTMTE).imag
        dataTETM = np.fft.rfft(dataTETM).imag
    end_dft = time.time()
    timing_dft = end_dft-start_dft
    start_logdet = end_dft

    kappa = np.sqrt(K ** 2 + nds ** 2)

    # m=0
    mat, dL_mat, d2L_mat = construct_matrices(row, col, dataTMTM[:,0], dataTETE[:,0], dataTMTE[:,0], dataTETM[:,0], N, kappa)
    logdet, dL_logdet, d2L_logdet = compute_matrix_operations(mat, dL_mat, d2L_mat, observable)

    # m>0    
    for m in range(1, M//2):
        mat, dL_mat, d2L_mat = construct_matrices(row, col, dataTMTM[:,m], dataTETE[:,m], dataTMTE[:,m], dataTETM[:,m], N, kappa)
        term1, term2, term3 = compute_matrix_operations(mat, dL_mat, d2L_mat, observable)
        logdet += 2 * term1
        dL_logdet += 2 * term2
        d2L_logdet += 2 * term3

    # last m
    mat, dL_mat, d2L_mat = construct_matrices(row, col, dataTMTM[:,M//2], dataTETE[:,M//2], dataTMTE[:,M//2], dataTETM[:,M//2], N, kappa)
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
    print(K, logdet, timing_matrix, timing_dft, timing_logdet, sep=", ")
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
    L = R/10
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

    nr_plane = nfunc_plane(1.)/nfunc_medium(1.)
    nr_sphere = nfunc_sphere(1.)/nfunc_medium(1.)
    nds, wts = quadrature(N)
    K = nfunc_medium(1.)*1
    print('K',K)
    print('np',nr_plane)
    print('ns',nr_sphere)

    c = K_contribution(R, L, K, nr_plane, nr_sphere, N, M, nds, wts, lmax, nproc, observable)


    #en = casimir_T0(R, L, nfunc_plane, nfunc_medium, nfunc_sphere, N, M, lmax, nproc, observable, X=40)
    #en = casimir_Tfinite(R, L, T, nfunc_plane, nfunc_medium, nfunc_sphere, N, M, lmax, mode, epsrel, nproc, observable, X=None)
    #print(en)
