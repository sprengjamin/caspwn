r""" Casimir interaction for the plane-sphere geometry at vanishing frequency/wavenumber

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
from kernels import kernel_polar_Kzero as kernel
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../plane/"))
from fresnel import rTM_zero as rTM
from fresnel import rTE_zero as rTE
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../ufuncs/"))
from integration import quadrature, auto_integration
from psd import psd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../material/"))
import material


@njit("UniTuple(float64[:], 2)(float64, int64, float64, float64, float64, float64, float64, float64, string, string, int64)", cache=True)
def angular_matrix_elements(rho, M, k1, k2, w1, w2, alpha_plane, alpha_sphere, materialclass_plane, materialclass_sphere, lmax):
    r"""
    Computes the matrix elements of the round-trip operator for all angular
    transverse momenta with discretization order :math:`M` at fixed radial
    transverse momenta k1 and k2.

    Parameters
    ----------
    rho: float
        positive, aspect ratio :math:`R/L`
    M: int
        positive, angular discretizaton order
    k1, k2: float
        positive, rescaled transverse wave numbers
    w1, w2: float
        positive, total quadrature weights corresponding to k1 and k2
    alpha_plane, alpha_sphere : float
        positive, parameter of plane and sphere (meaning depends on
        materialclass)
    materialclass_plane, materialclass_sphere : string
        materialclass of plane and sphere
    lmax : int
        positive, cut-off angular momentum

    Returns
    -------
    (np.ndarray, np.ndarray)
        tuple of arrays of length M for the polarization contributions TMTM,
        TETE

    """
    phiarrTM = np.empty(M)
    phiarrTE = np.empty(M)
    
    rTE1 = rTE(k1, alpha_plane, materialclass_plane)
    rTM1 = rTM(k1, alpha_plane, materialclass_plane)
    rTE2 = rTE(k2, alpha_plane, materialclass_plane)
    rTM2 = rTM(k2, alpha_plane, materialclass_plane)
    factorTM = np.sign(rTM1)*w1*w2*sqrt(rTM1*rTM2)
    factorTE = np.sign(rTE1)*w1*w2*sqrt(rTE1*rTE2)

    # phi = 0.
    phiarrTM[0], phiarrTE[0] = kernel(rho, 1., k1, k2, 0., alpha_sphere, materialclass_sphere, lmax)
    phiarrTM[0] *= factorTM
    phiarrTE[0] *= factorTE
    
    if M%2==0:
        # phi = np.pi is assumed
        phiarrTM[M//2], phiarrTE[M//2] = kernel(rho, 1., k1, k2, np.pi, alpha_sphere, materialclass_sphere, lmax)
        phiarrTM[M//2] *= factorTM
        phiarrTE[M//2] *= factorTE
        imax = M//2-1
        phi = 2*np.pi*np.arange(M//2)/M
    else:
        # phi = np.pi is not assumed
        imax = M//2
        phi = 2*np.pi*np.arange(M//2+1)/M
   
    # 0 < phi < np.pi
    for i in range(1, imax+1):
        phiarrTM[i], phiarrTE[i] = kernel(rho, 1., k1, k2, phi[i], alpha_sphere, materialclass_sphere, lmax)
        phiarrTM[i] *= factorTM
        phiarrTE[i] *= factorTE
        phiarrTM[M-i] = phiarrTM[i]
        phiarrTE[M-i] = phiarrTE[i]

    return phiarrTM, phiarrTE


def ME_partial(indices, rho, N, M, k, w, alpha_plane, alpha_sphere, materialclass_plane, materialclass_sphere, lmax):
    r"""
    Computes matrix elements with respect to the transverse momenta
    specified by the array indices.

    Parameters
    ----------
    indices: np.ndarray
        array of indices
    rho: float
        positive, aspect ratio :math:`R/L`
    N, M: int
        positive, radial and angular discretization order
    k: np.ndarray
        nodes of the radial quadrature rule
    w: np.ndarray
        symmetrized total weights
    alpha_plane, alpha_sphere : float
        positive, parameter of plane and sphere (meaning depends on
        materialclass)
    materialclass_plane, materialclass_sphere : string
        materialclass of plane and sphere
    lmax: int
        positive, cut-off angular momentum

    Returns
    -------
    row: np.ndarray
        row indices
    col: np.ndarray
        column indices
    dataTM, dataTE: np.ndarray
        dft matrix elements

    Dependencies
    ------------
    isFinite, angular_matrix_elements, itt_scalar

    """
    # size N is initial guess
    row = np.empty(N)
    col = np.empty(N)
    dataTM = np.empty((N, M))
    dataTE = np.empty((N, M))

    ind = 0
    for index in indices:
        i, j = itt_scalar(index)
        if isFinite(rho, k[i], k[j]):
            if ind+1 >= len(row):
                row = np.hstack((row, np.empty(len(row))))
                col = np.hstack((col, np.empty(len(row))))
                dataTM = np.vstack((dataTM, np.empty((len(row), M))))
                dataTE = np.vstack((dataTE, np.empty((len(row), M))))
            row[ind] = i
            col[ind] = j
            dataTM[ind, :], dataTE[ind, :] = angular_matrix_elements(rho, M, k[i], k[j], w[i], w[j], alpha_plane, alpha_sphere, materialclass_plane, materialclass_sphere, lmax)
            ind += 1
                
    row = row[:ind] 
    col = col[:ind] 
    dataTM = dataTM[:ind, :] 
    dataTE = dataTE[:ind, :] 
    return row, col, dataTM, dataTE


def ME_full(nproc, rho, N, M, nds, wts, alpha_plane, alpha_sphere, materialclass_plane, materialclass_sphere, lmax):
    r"""
    Computes all matrix elements. The calculation is parellelized among
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
    alpha_plane, alpha_sphere : float
        positive, parameter of plane and sphere (meaning depends on
        materialclass)
    materialclass_plane, materialclass_sphere : string
        materialclass of plane and sphere
    lmax : int
        positive, cut-off angular momentum

    Returns
    -------
    (ndarray, ndarray, ndarray)
        tuple containing row indices, column indices and TM and TE matrix elements

    Dependencies
    ------------
    dftME_partial

    """
    k = nds
    w = np.sqrt(wts * 2 * np.pi / M)
    
    ndiv = nproc*8 # factor is arbitrary, but can be chosen optimally

    indices = np.array_split(np.random.permutation(N * (N + 1) // 2), ndiv)

    with futures.ProcessPoolExecutor(max_workers=nproc) as executors:
        wait_for = [executors.submit(ME_partial, indices[i], rho, N, M, k, w, alpha_plane, alpha_sphere, materialclass_plane, materialclass_sphere, lmax)
                    for i in range(ndiv)]
        results = [f.result() for f in futures.as_completed(wait_for)]
    
    # gather results into 3 arrays
    length = 0
    for i in range(ndiv):
        length += len(results[i][0])
    row = np.empty(length, dtype=np.int)
    col = np.empty(length, dtype=np.int)
    dataTM = np.empty((length, M))
    dataTE = np.empty((length, M))
    ini = 0
    for i in range(ndiv):
        fin = ini + len(results[i][0])
        row[ini:fin] = results[i][0]
        col[ini:fin] = results[i][1]
        dataTM[ini:fin] = results[i][2]
        dataTE[ini:fin] = results[i][3]
        ini = fin
    return row, col, dataTM, dataTE


@njit("boolean(float64, float64, float64)", cache=True)
def isFinite(rho, k1, k2):
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
    exponent = 2*rho*sqrt(k1*k2) - (k1+k2)*(rho+1)
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
    rndtrp_matrix = np.zeros((N, N))
    dL_rndtrp_matrix = np.zeros((N, N))
    d2L_rndtrp_matrix = np.zeros((N, N))
    for i in range(len(data)):
        rndtrp_matrix[row[i], col[i]] = data[i]
        rndtrp_matrix[col[i], row[i]] = data[i]
        dL_rndtrp_matrix[row[i], col[i]] = -(kappa[row[i]] + kappa[col[i]]) * data[i]
        dL_rndtrp_matrix[col[i], row[i]] = dL_rndtrp_matrix[row[i], col[i]]
        d2L_rndtrp_matrix[row[i], col[i]] = (kappa[row[i]] + kappa[col[i]]) ** 2 * data[i]
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


def Kzero_contribution(R, L, alpha_plane, alpha_sphere, materialclass_plane, materialclass_sphere, N, M, nds, wts, lmax, nproc, observable):
    r"""
    Computes the contribution to the observable depending on the wave number K.

    Parameters
    ----------
    R: float
        positive, radius of the sphere
    L: float
        positive, surface-to-surface distance
    alpha_plane, alpha_sphere : float
        positive, parameter of plane and sphere (meaning depends on
        materialclass)
    materialclass_plane, materialclass_sphere : string
        materialclass of plane and sphere
    N, M: int
        positive, radial and angular discretization order
    nds, wts: np.ndarray
        nodes and weights of the radial quadrature rule
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

    row, col, dataTM, dataTE = ME_full(nproc, rho, N, M, nds, wts, alpha_plane, alpha_sphere, materialclass_plane, materialclass_sphere, lmax)
    end_matrix = time.time()
    timing_matrix = end_matrix-start_matrix

    ## discrete Fourier transform
    start_fft = end_matrix
    if len(row) != 0.:
        dataTM = np.fft.rfft(dataTM).real
        dataTE = np.fft.rfft(dataTE).real
    end_fft = time.time()
    timing_fft = end_fft-start_fft

    start_logdet = end_fft
    k = nds
    ## TM contribution
    # m=0
    mat, dL_mat, d2L_mat = construct_matrices(row, col, dataTM[:,0], N, k)
    logdet, dL_logdet, d2L_logdet = compute_matrix_operations(mat, dL_mat, d2L_mat, observable)

    # m>0    
    for m in range(1, M//2):
        mat, dL_mat, d2L_mat = construct_matrices(row, col, dataTM[:,m], N, k)
        term1, term2, term3 = compute_matrix_operations(mat, dL_mat, d2L_mat, observable)
        logdet += 2 * term1
        dL_logdet += 2 * term2
        d2L_logdet += 2 * term3

    # last m
    mat, dL_mat, d2L_mat = construct_matrices(row, col, dataTM[:,M//2], N, k)
    term1, term2, term3 = compute_matrix_operations(mat, dL_mat, d2L_mat, observable)
    if M%2==0:
        logdet += term1
        dL_logdet += term2
        d2L_logdet += term3
    else:
        logdet += 2 * term1
        dL_logdet += 2 * term2
        d2L_logdet += 2 * term3
    
    logdet = 0.
    ## TE contribution
    if materialclass_plane != "dielectric" and materialclass_plane != "drude" and materialclass_sphere != "dielectric" and materialclass_sphere != "drude":
        # m=0
        mat, dL_mat, d2L_mat = construct_matrices(row, col, dataTE[:,0], N, k)
        term1, term2, term3 = compute_matrix_operations(mat, dL_mat, d2L_mat, observable)
        logdet += term1
        dL_logdet += term2
        d2L_logdet += term3

        # m>0    
        for m in range(1, M//2):
            mat, dL_mat, d2L_mat = construct_matrices(row, col, dataTE[:,m], N, k)
            term1, term2, term3 = compute_matrix_operations(mat, dL_mat, d2L_mat, observable)
            logdet += 2 * term1
            dL_logdet += 2 * term2
            d2L_logdet += 2 * term3

        # last m
        mat, dL_mat, d2L_mat = construct_matrices(row, col, dataTE[:,M//2], N, k)
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
    print(0., logdet, timing_matrix, timing_fft, timing_logdet, sep=", ")
    return np.array([logdet, dL_logdet/L, d2L_logdet/L**2])


def casimir(R, L, T, alpha_plane, alpha_sphere, materialclass_plane, materialclass_sphere, N, M, lmax, nproc, observable):
    r"""
    Computes the Casimir free energy at equilibrium temperature :math:`T`.

    Parameters
    ----------
    R: float
        positive, radius of the sphere
    L: float
        positive, surface-to-surface distance between plane and sphere
    N, M: int
        positive, radial and angular discretization order
    lmax : int
        positive, cut-off angular momentum
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
    nds, wts = quadrature(N)
    
    res = Kzero_contribution(R, L, alpha_plane, alpha_sphere, materialclass_plane, materialclass_sphere, N, M, nds, wts, lmax, nproc, observable)
    return 0.5*T*Boltzmann*res


if __name__ == "__main__":
    from scipy.constants import hbar, c, e
    R = 50.e-6
    L = 500.e-9
    T = 300
    
    wp = 9*e/hbar
    alpha_sphere = wp*R/c
    alpha_plane = wp*L/c
    materialclass_plane = "plasma"
    materialclass_sphere = "plasma"
    nproc = 4
    observable = "energy"
    rho = max(R/L, 50)
    N = int(9.*np.sqrt(rho))
    M = int(8.*np.sqrt(rho))
    lmax = int(12*rho)
    en = casimir(R, L, T, alpha_plane, alpha_sphere, materialclass_plane, materialclass_sphere, N, M, lmax, nproc, observable)
    print(en)
