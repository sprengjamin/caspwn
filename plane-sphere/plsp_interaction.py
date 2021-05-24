r""" Casimir interaction for the plane-sphere geometry at given frequency/wavenumber

"""
import numpy as np
from math import sqrt
import concurrent.futures as futures
from numba import njit
from scipy.linalg import cho_factor, cho_solve
from scipy.linalg import lu_factor, lu_solve
from scipy.integrate import quad
from scipy.constants import Boltzmann, hbar, c
from time import perf_counter

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../sphere/"))
from mie import mie_e_array
from reflection_matrix import arm_full_zero, arm_full_finite


@njit("UniTuple(float64[:,:], 3)(int64[:], int64[:], float64[:], float64[:], float64[:], float64[:], int64, int64, float64[:], float64[:], float64[:], float64[:])", cache=True)
def construct_roundtrip_finite(row, col, dataTMTM, dataTETE, dataTMTE, dataTETM, N, M, kappa, w, rTM, rTE):
    r"""Construct round-trip matrices and its first two derivatives.

        Parameters
        ----------
        row, col : list
            1D arrays of ints, row and column index of the matrix elements, respectively.
        data : list
            1D array of floats, matrix elements
        N, M : int
            positive, radial and angular discretization order
        kappa : list
            1D array of floats, imaginary z-component of the wave vectors
        w : list
            weights of radial quadrature
        rTM, rTE : list
            Fresnel reflection coefficients for TM and TE polarization

        Returns
        -------
        (ndarray, ndarray, ndarray)

    """
    mat = np.zeros((2 * N, 2 * N))
    dL_mat = np.zeros((2 * N, 2 * N))
    d2L_mat = np.zeros((2 * N, 2 * N))
    if np.all(np.sign(rTM)==np.sign(rTM[0])) and np.all(np.sign(rTE)==np.sign(rTE[0])):
        # symmetrize matrices
        for l in range(len(row)):
            i, j = row[l], col[l]
            weight = 1/M/2/np.pi*sqrt(w[i]*w[j])
            factorTMTM = np.sign(rTM[i])*sqrt(rTM[i]*rTM[j])*weight
            factorTETE = np.sign(rTE[i])*sqrt(rTE[i]*rTE[j])*weight
            factorTMTE = sqrt(-rTM[i]*rTE[j])*weight
            factorTETM = sqrt(-rTE[i]*rTM[j])*weight
            dL_factor = -(kappa[i] + kappa[j])
            d2L_factor = dL_factor**2
            mat[i, j] = factorTMTM*dataTMTM[l]
            mat[j, i] = mat[i, j]
            mat[i+N, j+N] = factorTETE*dataTETE[l]
            mat[j+N, i+N] = mat[i+N, j+N]
            mat[i, j+N] = factorTMTE*dataTMTE[l]
            mat[j+N, i] = mat[i, j+N]
            mat[i+N, j] = factorTETM*dataTETM[l]
            mat[j, i+N] = mat[i+N, j]

            dL_mat[i, j] = dL_factor*mat[i, j]
            dL_mat[j, i] = dL_mat[i, j]
            dL_mat[i+N, j+N] = dL_factor*mat[i+N, j+N]
            dL_mat[j+N, i+N] = dL_mat[i+N, j+N]
            dL_mat[i, j+N] = dL_factor*mat[i, j+N]
            dL_mat[j+N, i] = dL_mat[i, j+N]
            dL_mat[i+N, j] = dL_factor*mat[i+N, j]
            dL_mat[j, i+N] = dL_mat[i+N, j]

            d2L_mat[i, j] = d2L_factor*mat[i, j]
            d2L_mat[j, i] = d2L_mat[i, j]
            d2L_mat[i+N, j+N] = d2L_factor*mat[i+N, j+N]
            d2L_mat[j+N, i+N] = d2L_mat[i+N, j+N]
            d2L_mat[i, j+N] = d2L_factor*mat[i, j+N]
            d2L_mat[j+N, i] = d2L_mat[i, j+N]
            d2L_mat[i+N, j] = d2L_factor*mat[i+N, j]
            d2L_mat[j, i+N] = d2L_mat[i+N, j]
    else:
        # do not symmetrize matrices
        for l in range(len(row)):
            i, j = row[l], col[l]
            weight = 1/M/2/np.pi*sqrt(w[i]*w[j])
            dL_factor = -(kappa[i] + kappa[j])
            d2L_factor = dL_factor**2
            mat[i, j] = weight*rTM[i]*dataTMTM[l]
            mat[j, i] = weight*rTM[j]*dataTMTM[l]
            mat[i+N, j+N] = weight*rTE[i]*dataTETE[l]
            mat[j+N, i+N] = weight*rTE[j]*dataTETE[l]
            mat[i, j+N] = weight*rTM[i]*dataTMTE[l]
            mat[j+N, i] = weight*rTE[j]*dataTETM[l]
            mat[i+N, j] = weight*rTE[i]*dataTETM[l]
            mat[j, i+N] = weight*rTM[j]*dataTMTE[l]

            dL_mat[i, j] = dL_factor*mat[i, j]
            dL_mat[j, i] = dL_factor*mat[j, i]
            dL_mat[i+N, j+N] = dL_factor*mat[i+N, j+N]
            dL_mat[j+N, i+N] = dL_factor*mat[j+N, i+N]
            dL_mat[i, j+N] = dL_factor*mat[i, j+N]
            dL_mat[j+N, i] = dL_factor*mat[j+N, i]
            dL_mat[i+N, j] = dL_factor*mat[i+N, j]
            dL_mat[j, i+N] = dL_factor*mat[j, i+N]

            d2L_mat[i, j] = d2L_factor*mat[i, j]
            d2L_mat[j, i] = d2L_factor*mat[j, i]
            d2L_mat[i+N, j+N] = d2L_factor*mat[i+N, j+N]
            d2L_mat[j+N, i+N] = d2L_factor*mat[j+N, i+N]
            d2L_mat[i, j+N] = d2L_factor*mat[i, j+N]
            d2L_mat[j+N, i] = d2L_factor*mat[j+N, i]
            d2L_mat[i+N, j] = d2L_factor*mat[i+N, j]
            d2L_mat[j, i+N] = d2L_factor*mat[j, i+N]

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
    norm_mat = np.linalg.norm(mat)
    tr_mat = np.trace(mat)
    if tr_mat == 0.:
        assert(norm_mat == 0.)
        return 0., 0., 0.

    if abs(norm_mat**2/(1-norm_mat)/tr_mat) < 1.e-10:
        # make trace approximation
        if observable == "energy":
            return -tr_mat, 0., 0.
        tr_dL_mat = np.trace(dL_mat)
        if observable == "force":
            return -tr_mat, tr_dL_mat, 0.
        tr_d2L_mat = np.trace(d2L_mat)
        if observable == "pressure":
            return -tr_mat, tr_dL_mat, tr_d2L_mat
        else: raise ValueError
    else:
        # compute logdet etc. exactly
        if np.all(mat == mat.T):
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
        else:
            lu, piv = lu_factor(np.eye(mat.shape[0]) - mat)
            logdet = np.sum(np.log(np.diag(lu)))
            if observable == "energy":
                return logdet, 0., 0.

            matA = lu_solve((lu, piv), dL_mat)
            dL_logdet = np.trace(matA)
            if observable == "force":
                return logdet, dL_logdet, 0.
            matB = lu_solve((lu, piv), d2L_mat)
            d2L_logdet = np.trace(matB) + np.sum(matA**2)
            if observable == "pressure":
                return logdet, dL_logdet, d2L_logdet
            else:
                raise ValueError


def contribution_finite(R, L, k0, K, n_plane, n_sphere, rTM_finite, rTE_finite, N, M, nds, wts, lmax, nproc, observable):
    r"""
    Computes the contribution to the observable depending on the wave number K.

    Parameters
    ----------
    R: float
        positive, radius of the sphere
    L: float
        positive, surface-to-surface distance
    k0: float
        positive, rescaled vacuum wave number
    K: float
        positive, rescaled wavenumber in medium
    n_plane, n_sphere: float
        positive, refractive index of plane and sphere (relative to medium)
    rTM_finite, rTE_finite: function
        the reflection coefficients on the plane for TM and TE polarization
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
    arm_full_finite, mie_e_array

    """
    start_matrix = perf_counter()

    rho = R/L # aspect ratio
    x = K*rho # size parameter

    # precompute mie coefficients
    if x > 5e3:
        # dummy variables
        mie_a, mie_b = np.empty(1), np.empty(1)
    else:
        mie_a, mie_b = mie_e_array(lmax, x, n_sphere)

    row, col, dataTMTM, dataTETE, dataTMTE, dataTETM = arm_full_finite(nproc, rho, 1., 1., K, N, N, M, nds, nds, n_sphere, lmax, mie_a, mie_b)

    end_matrix = perf_counter()
    timing_matrix = end_matrix-start_matrix
    start_dft = end_matrix
    if len(row) != 0.:
        dataTMTM = np.fft.rfft(dataTMTM).real
        dataTETE = np.fft.rfft(dataTETE).real
        dataTMTE = np.fft.rfft(dataTMTE).imag
        dataTETM = np.fft.rfft(dataTETM).imag
    end_dft = perf_counter()
    timing_dft = end_dft-start_dft
    start_logdet = end_dft

    kappa = np.sqrt(K ** 2 + nds ** 2)
    rTM = np.array([rTM_finite(k0, k) for k in nds])
    rTE = np.array([rTE_finite(k0, k) for k in nds])

    # m=0
    mat, dL_mat, d2L_mat = construct_roundtrip_finite(row, col, dataTMTM[:,0], dataTETE[:,0], dataTMTE[:,0], dataTETM[:,0], N, M, kappa, wts, rTM, rTE)
    logdet, dL_logdet, d2L_logdet = compute_matrix_operations(mat, dL_mat, d2L_mat, observable)

    # m>0
    for m in range(1, M//2):
        mat, dL_mat, d2L_mat = construct_roundtrip_finite(row, col, dataTMTM[:,m], dataTETE[:,m], dataTMTE[:,m], dataTETM[:,m], N, M, kappa, wts, rTM, rTE)
        term1, term2, term3 = compute_matrix_operations(mat, dL_mat, d2L_mat, observable)
        logdet += 2 * term1
        dL_logdet += 2 * term2
        d2L_logdet += 2 * term3

    # last m
    mat, dL_mat, d2L_mat = construct_roundtrip_finite(row, col, dataTMTM[:,M//2], dataTETE[:,M//2], dataTMTE[:,M//2], dataTETM[:,M//2], N, M, kappa, wts, rTM, rTE)
    term1, term2, term3 = compute_matrix_operations(mat, dL_mat, d2L_mat, observable)
    if M%2==0:
        logdet += term1
        dL_logdet += term2
        d2L_logdet += term3
    else:
        logdet += 2 * term1
        dL_logdet += 2 * term2
        d2L_logdet += 2 * term3
    end_logdet = perf_counter()
    timing_logdet = end_logdet-start_logdet
    print("# ", end="")
    print(K, logdet, "%.3f"%timing_matrix, "%.3f"%timing_dft, "%.3f"%timing_logdet, sep=", ")
    return np.array([logdet, dL_logdet/L, d2L_logdet/L**2])

@njit("UniTuple(float64[:,:], 3)(int64[:], int64[:], float64[:], int64, int64, float64[:], float64[:], float64[:])", cache=True)
def construct_roundtrip_zero(row, col, data, N, M, k, w, rp):
    r"""Construct round-trip matrices and its first two derivatives for each m.

        Parameters
        ----------
        row, col : ndarray
            1D arrays of ints, row and column index of the matrix elements, respectively.
        data : ndarray
            1D array of floats, matrix elements
        N, M : int
            matrix dimension of each polarization block
        k, w: list
            nodes and weights of radial quadrature
        rp : list
            Fresnel reflection coefficients for a given polarization

        Returns
        -------
        (ndarray, ndarray, ndarray)

    """
    mat = np.zeros((N, N))
    dL_mat = np.zeros((N, N))
    d2L_mat = np.zeros((N, N))
    if np.all(np.sign(rp)==np.sign(rp[0])):
        # symmetrize matrices
        for l in range(len(data)):
            i, j = row[l], col[l]
            weight = 1/M/2/np.pi*sqrt(w[i]*w[j])
            factor = np.sign(rp[i])*sqrt(rp[i]*rp[j])*weight
            dL_factor = -(k[i] + k[j])
            d2L_factor = dL_factor**2
            mat[i, j] = factor*data[l]
            mat[j, i] = factor*data[l]
            dL_mat[i, j] = dL_factor*mat[i, j]
            dL_mat[j, i] = dL_mat[i, j]
            d2L_mat[i, j] = d2L_factor*mat[i, j]
            d2L_mat[j, i] = d2L_mat[i, j]
    else:
        # do not symmetrize matrices
        for l in range(len(data)):
                i, j = row[l], col[l]
                weight = 1/M/2/np.pi*sqrt(w[i]*w[j])
                dL_factor = -(k[i] + k[j])
                d2L_factor = dL_factor**2
                mat[i, j] = weight*data[l]*rp[i]
                mat[j, i] = weight*data[l]*rp[j]
                dL_mat[i, j] = dL_factor*mat[i, j]
                dL_mat[j, i] = dL_factor*mat[j, i]
                d2L_mat[i, j] = d2L_factor*mat[i, j]
                d2L_mat[j, i] = d2L_factor*mat[j, i]
    return mat, dL_mat, d2L_mat


def contribution_zero(R, L, alpha_sphere, materialclass_plane, materialclass_sphere, rTM_zero, rTE_zero, N, M, nds, wts, lmax, nproc, observable):
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
    rTM_zero, rTE_zero: function
        the reflection coefficients on the plane for TM and TE polarization
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
    (float, float, float)

    Dependencies
    ------------
    arm_full_zero

    """
    start_matrix = perf_counter()

    # aspect ratio
    rho = R/L

    row, col, dataTM, dataTE = arm_full_zero(nproc, rho, 1., N, N, M, nds, nds, alpha_sphere, materialclass_sphere, lmax)
    end_matrix = perf_counter()
    timing_matrix = end_matrix-start_matrix

    ## discrete Fourier transform
    start_fft = end_matrix
    if len(row) != 0.:
        dataTM = np.fft.rfft(dataTM).real
        dataTE = np.fft.rfft(dataTE).real
    end_fft = perf_counter()
    timing_fft = end_fft-start_fft

    start_logdet = end_fft
    k = nds
    rTM = np.array([rTM_zero(k) for k in nds])
    ## TM contribution
    # m=0
    mat, dL_mat, d2L_mat = construct_roundtrip_zero(row, col, dataTM[:,0], N, M, k, wts, rTM)
    logdet, dL_logdet, d2L_logdet = compute_matrix_operations(mat, dL_mat, d2L_mat, observable)

    # m>0
    for m in range(1, M//2):
        mat, dL_mat, d2L_mat = construct_roundtrip_zero(row, col, dataTM[:,m], N, M, k, wts, rTM)
        term1, term2, term3 = compute_matrix_operations(mat, dL_mat, d2L_mat, observable)
        logdet += 2 * term1
        dL_logdet += 2 * term2
        d2L_logdet += 2 * term3

    # last m
    mat, dL_mat, d2L_mat = construct_roundtrip_zero(row, col, dataTM[:,M//2], N, M, k, wts, rTM)
    term1, term2, term3 = compute_matrix_operations(mat, dL_mat, d2L_mat, observable)
    if M%2==0:
        logdet += term1
        dL_logdet += term2
        d2L_logdet += term3
    else:
        logdet += 2 * term1
        dL_logdet += 2 * term2
        d2L_logdet += 2 * term3

    ## TE contribution
    rTE = np.array([rTE_zero(k) for k in nds])
    if materialclass_plane != "dielectric" and materialclass_plane != "drude" and materialclass_sphere != "dielectric" and materialclass_sphere != "drude":
        # m=0
        mat, dL_mat, d2L_mat = construct_roundtrip_zero(row, col, dataTE[:,0], N, M, k, wts, rTE)
        term1, term2, term3 = compute_matrix_operations(mat, dL_mat, d2L_mat, observable)
        logdet += term1
        dL_logdet += term2
        d2L_logdet += term3

        # m>0
        for m in range(1, M//2):
            mat, dL_mat, d2L_mat = construct_roundtrip_zero(row, col, dataTE[:,m], N, M, k, wts, rTE)
            term1, term2, term3 = compute_matrix_operations(mat, dL_mat, d2L_mat, observable)
            logdet += 2 * term1
            dL_logdet += 2 * term2
            d2L_logdet += 2 * term3

        # last m
        mat, dL_mat, d2L_mat = construct_roundtrip_zero(row, col, dataTE[:,M//2], N, M, k, wts, rTE)
        term1, term2, term3 = compute_matrix_operations(mat, dL_mat, d2L_mat, observable)
        if M%2==0:
            logdet += term1
            dL_logdet += term2
            d2L_logdet += term3
        else:
            logdet += 2 * term1
            dL_logdet += 2 * term2
            d2L_logdet += 2 * term3

    end_logdet = perf_counter()
    timing_logdet = end_logdet-start_logdet
    print("# ", end="")
    print(0., logdet, "%.3f"%timing_matrix, "%.3f"%timing_fft, "%.3f"%timing_logdet, sep=", ")
    return np.array([logdet, dL_logdet/L, d2L_logdet/L**2])

if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../ufuncs/"))
    from integration import quadrature

    R = 1.e-6
    L = 1.e-9

    K = 1.
    n_plane = np.inf
    n_sphere = np.inf

    rho = max(50, R/L)
    N = int(8.*np.sqrt(rho))
    M = int(6.*np.sqrt(rho))
    lmax = int(12.*rho)

    nproc = 4
    observable = "energy"

    nds, wts = quadrature(N)

    c = contribution_finite(R, L, K, n_plane, n_sphere, N, M, nds, wts, lmax, nproc, observable)

    alpha_plane = 4.
    alpha_sphere = 4.
    materialclass_plane = 'dielectric'
    materialclass_sphere = 'dielectric'

    #c0 = contribution_zero(R, L, alpha_plane, alpha_sphere, materialclass_plane, materialclass_sphere, N, M, nds, wts, lmax, nproc, observable)


