r"""sphere reflection matrix in angular representation
.. to do:
    * why does jit slow down arm_partial_finite?
"""

import numpy as np
from math import sqrt
import numba
from numba import njit, prange
from .kernels import kernel_polar_finite, kernel_polar_zero

@njit("int64[:,:](int64, int64)", cache=True)
def generate_tuples(N1, N2):
    """
    generates a list of
    
    Parameters
    ----------
    N1, N2: integer
        non-negative, index

    Returns
    -------
    array

    """
    if N1 == N2:
        tuples = np.empty((N1 * (N1 + 1) // 2, 2), dtype=np.int64)
        counter = 0
        for i in range(N1):
            for j in range(i+1):
                tuples[counter, 0] = i
                tuples[counter, 1] = j
                counter += 1
    else:
        tuples = np.empty((N1 * N2, 2), dtype=np.int64)
        counter = 0
        for i in range(N1):
            for j in range(N2):
                tuples[counter, 0] = i
                tuples[counter, 1] = j
                counter += 1
    return tuples


@njit("boolean(float64, float64, float64, float64, float64, float64)", cache=True)
def isFinite(R, L, r, K, k1, k2):
    """
    Estimates by means of the asymptotics of the scattering amplitudes if the given
    matrix element can be numerically set to zero.

    Parameters
    ----------
    R: float
        positive, sphere radius
    L: float
        positive, surface-to-surface distance
    r: float
        positive, relative amount of surface-to-surface translation
    K: float
        positive, rescaled wavenumber
    k1, k2: float
        positive, rescaled wave numbers

    Returns
    -------
    boolean
        True if the matrix element must not be neglected

    """
    if K == 0.:
        exponent = 2*R*sqrt(k1*k2) - (k1+k2)*(R+r*L)
    else:
        kappa1 = sqrt(k1*k1+K*K)
        kappa2 = sqrt(k2*k2+K*K)
        exponent = R*sqrt(2*(K*K + kappa1*kappa2 + k1*k2)) - (kappa1+kappa2)*(R+r*L)
    if exponent < -37.:
        return False
    else:
        return True

@njit("UniTuple(float64[:], 4)(float64, float64, float64, float64, float64, int64, float64, float64, float64, int64, float64[:], float64[:])", cache=True)
def arm_elements_finite(R, L, r, sign, K, M, k1, k2, n_sphere, lmax, mie_a, mie_b):
    r"""Angular Reflection Matrix elements for FINITE frequencies/wavenumbers.

    Computes the matrix elements of the reflection operator at the sphere for
    all angular transverse momenta with discretization order :math:`M` at fixed
    radial transverse momenta k1 and k2 for each polarization block.


    Parameters
    ----------
    R: float
        positive, sphere radius
    L: float
        positive, surface-to-surface distance
    r: float
        positive, relative amount of surface-to-surface translation
    sign: float
        sign, +/-1, differs for the two spheres
    K: float
        positive, rescaled wavenumber in medium
    M: int
        positive, angular discretizaton order
    k1, k2: float
        positive, rescaled transverse wave numbers
    n_sphere : float
        positive, refractive index of sphere
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
    TMTM = np.empty(M)
    TETE = np.empty(M)
    TMTE = np.empty(M)
    TETM = np.empty(M)

    # phi = 0.
    TMTM[0], TETE[0], TMTE[0], TETM[0] = kernel_polar_finite(R, L, r, sign, K, k1, k2, 0., n_sphere, lmax, mie_a, mie_b)
    
    if M%2==0:
        # phi = np.pi is assumed
        TMTM[M//2], TETE[M//2], TMTE[M//2], TETM[M//2] = kernel_polar_finite(R, L, r, sign, K, k1, k2, np.pi, n_sphere, lmax, mie_a, mie_b)
        imax = M//2-1
        phi = 2*np.pi*np.arange(M//2)/M
    else:
        # phi = np.pi is not assumed
        imax = M//2
        phi = 2*np.pi*np.arange(M//2+1)/M
   
    # 0 < phi < np.pi
    for i in range(1, imax+1):
        TMTM[i], TETE[i], TMTE[i], TETM[i] = kernel_polar_finite(R, L, r, sign, K, k1, k2, phi[i], n_sphere, lmax, mie_a, mie_b)
        TMTM[M-i] = TMTM[i]
        TETE[M-i] = TETE[i]
        TMTE[M-i] = -TMTE[i]
        TETM[M-i] = -TETM[i]
    return TMTM, TETE, TMTE, TETM
        
def arm_partial_finite(tuples, R, L, r, sign, K, len_init, M, k1, k2, n_sphere, lmax, mie_a, mie_b):
    r"""
    PARTIAL Angular Reflection Matrix for FINITE frequencies/wavenumbers.
    The matrix elements are computed for the given indices.

    Parameters
    ----------
    tuples : np.ndarray
        2D array containing row and column indices
    R: float
        positive, sphere radius
    L: float
        positive, surface-to-surface distance
    r: float
        positive, relative amount of surface-to-surface translation
    sign: float
        sign, +/-1, differs for the two spheres
    K: float
        positive, rescaled wavenumber in medium
    len_init : int
        initial size of arrays
    M: int
        positive, radial and angular discretization order
    k1, k2: np.ndarray
        nodes of the radial quadrature rule
    n_sphere : float
        positive, refractive index of sphere
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
    isFinite, angular_reflection_matrix_elements

    """
    row = np.empty(len_init)
    col = np.empty(len_init)
    dataTMTM = np.empty((len_init, M))
    dataTETE = np.empty((len_init, M))
    dataTMTE = np.empty((len_init, M))
    dataTETM = np.empty((len_init, M))

    ind = 0
    for tuple in tuples:
        i = tuple[0]
        j = tuple[1]
        if isFinite(R, L, r, K, k1[i], k2[j]):
            if ind+1 >= len(row):
                row = np.hstack((row, np.empty(len(row))))
                col = np.hstack((col, np.empty(len(row))))
                dataTMTM = np.vstack((dataTMTM, np.empty((len(row), M))))
                dataTETE = np.vstack((dataTETE, np.empty((len(row), M))))
                dataTMTE = np.vstack((dataTMTE, np.empty((len(row), M))))
                dataTETM = np.vstack((dataTETM, np.empty((len(row), M))))
            row[ind] = i
            col[ind] = j
            dataTMTM[ind, :], dataTETE[ind, :], dataTMTE[ind, :], dataTETM[ind, :] = arm_elements_finite(R, L, r, sign, K, M, k1[i], k2[j], n_sphere, lmax, mie_a, mie_b)
            ind += 1
                
    row = row[:ind]
    col = col[:ind]
    dataTMTM = dataTMTM[:ind, :]
    dataTETE = dataTETE[:ind, :]
    dataTMTE = dataTMTE[:ind, :]
    dataTETM = dataTETM[:ind, :]
    return row, col, dataTMTM, dataTETE, dataTMTE, dataTETM


@njit("boolean[:](int64[:,:], float64, float64, float64, float64, float64[:], float64[:])", parallel=True, cache=True)
def finite_mask(tuples, R, L, r, K, k1, k2):
    r"""Evaluates ``isFinite`` for all index tuples in parallel."""
    mask = np.empty(len(tuples), dtype=np.bool_)
    for t in prange(len(tuples)):
        mask[t] = isFinite(R, L, r, K, k1[tuples[t, 0]], k2[tuples[t, 1]])
    return mask


def finite_tuples(nproc, R, L, r, K, N1, N2, k1, k2):
    r"""
    Index tuples of the matrix elements that must not be neglected.

    Parameters
    ----------
    nproc: int
        number of threads
    R, L, r, K: float
        see ``isFinite``
    N1, N2: int
        positive, radial discretization order
    k1, k2 : np.ndarray
        nodes of the radial quadrature rule

    Returns
    -------
    np.ndarray
        2D array containing row and column indices

    """
    numba.set_num_threads(max(1, min(nproc, numba.config.NUMBA_NUM_THREADS)))
    tuples = generate_tuples(N1, N2)
    tuples = tuples[finite_mask(tuples, R, L, r, K, k1, k2)]
    # random order balances the load among the threads
    return tuples[np.random.permutation(len(tuples))]


@njit("UniTuple(float64[:,:], 4)(int64[:,:], float64, float64, float64, float64, float64, int64, float64[:], float64[:], float64, int64, float64[:], float64[:])", parallel=True, cache=True)
def arm_parallel_finite(tuples, R, L, r, sign, K, M, k1, k2, n_sphere, lmax, mie_a, mie_b):
    r"""
    Angular Reflection Matrix elements for FINITE frequencies/wavenumbers for the given indices,
    computed in parallel threads.

    Parameters
    ----------
    tuples : np.ndarray
        2D array containing row and column indices
    R, L, r, sign, K, M, k1, k2, n_sphere, lmax, mie_a, mie_b
        see ``arm_full_finite``

    Returns
    -------
    dataTMTM, dataTETE, dataTMTE, dataTETM : np.ndarray
        angular matrix elements for TMTM, TETE, TMTE, TETM polarization

    """
    n = len(tuples)
    dataTMTM = np.empty((n, M))
    dataTETE = np.empty((n, M))
    dataTMTE = np.empty((n, M))
    dataTETM = np.empty((n, M))
    for t in prange(n):
        TMTM, TETE, TMTE, TETM = arm_elements_finite(R, L, r, sign, K, M, k1[tuples[t, 0]], k2[tuples[t, 1]], n_sphere, lmax, mie_a, mie_b)
        dataTMTM[t, :] = TMTM
        dataTETE[t, :] = TETE
        dataTMTE[t, :] = TMTE
        dataTETM[t, :] = TETM
    return dataTMTM, dataTETE, dataTMTE, dataTETM


def arm_full_finite(nproc, R, L, r, sign, K, N1, N2, M, k1, k2, n_sphere, lmax, mie_a, mie_b):
    r"""
    FULL Angular Reflection Matrix for a FINITE frequency/wavenumber.
    The calculation is parellelized among nproc threads.

    Parameters
    ----------
    nproc: int
        number of threads
    R: float
        positive, sphere radius
    L: float
        positive, surface-to-surface distance
    r: float
        positive, relative amount of surface-to-surface translation
    sign: float
        sign, +/-1, differs for the two spheres
    K: float
        positive, rescaled wavenumber in medium
    N1, N2, M: int
        positive, radial and angular discretization order
    k1, k2 : np.ndarray
        nodes of the radial quadrature rule
    n_sphere : float
        positive, refractive index sphere
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
    dataTMTM, dataTETE, dataTMTE, dataTETM : np.ndarray
        angular matrix elements for TMTM, TETE, TMTE, TETM polarization

    Dependencies
    ------------
    finite_tuples, arm_parallel_finite

    """
    tuples = finite_tuples(nproc, R, L, r, K, N1, N2, k1, k2)
    dataTMTM, dataTETE, dataTMTE, dataTETM = arm_parallel_finite(tuples, R, L, r, sign, K, M, k1, k2, n_sphere, lmax, mie_a, mie_b)
    row = np.ascontiguousarray(tuples[:, 0])
    col = np.ascontiguousarray(tuples[:, 1])
    return row, col, dataTMTM, dataTETE, dataTMTE, dataTETM


@njit(
    "UniTuple(float64[:], 2)(float64, float64, float64, int64, float64, float64, float64, string, int64)",
    cache=True)
def arm_elements_zero(R, L, r, M, k1, k2, alpha_sphere, materialclass_sphere, lmax):
    r"""Angular Reflection Matrix elements the ZERO-frequency/wavenumber contribution.

    Computes the matrix elements of the reflection operator at the sphere for
    all angular transverse momenta with discretization order :math:`M` at fixed
    radial transverse momenta k1 and k2 for each polarization block.


    Parameters
    ----------
    R: float
        positive, sphere radius
    L: float
        positive, surface-to-surface distance
    r: float
        positive, relative amount of surface-to-surface translation
    K: float
        positive, rescaled wavenumber in medium
    M: int
        positive, angular discretizaton order
    k1, k2: float
        positive, rescaled transverse wave numbers
    alpha_sphere : float
        positive, parameter
    materialclass_sphere : string
        materialclass
    lmax : int
        positive, cut-off angular momentum

    Returns
    -------
    2-tuple
        with arrays of length M for the polarization contributions TMTM, TETE

    """
    TMTM = np.empty(M)
    TETE = np.empty(M)

    # phi = 0.
    TMTM[0], TETE[0] = kernel_polar_zero(R, L, r, k1, k2, 0., alpha_sphere, materialclass_sphere, lmax)

    if M % 2 == 0:
        # phi = np.pi is assumed
        TMTM[M // 2], TETE[M // 2] = kernel_polar_zero(R, L, r, k1, k2, np.pi, alpha_sphere, materialclass_sphere, lmax)
        imax = M // 2 - 1
        phi = 2 * np.pi * np.arange(M // 2) / M
    else:
        # phi = np.pi is not assumed
        imax = M // 2
        phi = 2 * np.pi * np.arange(M // 2 + 1) / M

    # 0 < phi < np.pi
    for i in range(1, imax + 1):
        TMTM[i], TETE[i] = kernel_polar_zero(R, L, r, k1, k2, phi[i], alpha_sphere, materialclass_sphere, lmax)
        TMTM[M - i] = TMTM[i]
        TETE[M - i] = TETE[i]
    return TMTM, TETE


#@njit(cache=True)
def arm_partial_zero(tuples, R, L, r, len_init, M, k1, k2, alpha_sphere, materialclass_sphere, lmax):
    r"""
    PARTIAL Angular Reflection Matrix for the ZERO-frequency/wavenumber contribution.
    The matrix elements are computed for the given indices.

    Parameters
    ----------
    tuples : np.ndarray
        2D array containing row and column indices
    R: float
        positive, sphere radius
    L: float
        positive, surface-to-surface distance
    r: float
        positive, relative amount of surface-to-surface translation
    len_init : int
        initial size of arrays
    M: int
        positive, angular discretization order
    k1, k2: np.ndarray
        nodes of the radial quadrature rule
    alpha_sphere : float
        positive, parameter
    materialclass_sphere : string
        materialclass
    lmax: int
        positive, cut-off angular momentum

    Returns
    -------
    row: np.ndarray
        row indices
    col: np.ndarray
        column indices
    dataTMTM, dataTETE : np.ndarray
        angular matrix elements for TMTM, TETE polarization

    Dependencies
    ------------
    isFinite, ARM_elements_zero

    """
    row = np.empty(len_init)
    col = np.empty(len_init)
    dataTMTM = np.empty((len_init, M))
    dataTETE = np.empty((len_init, M))

    ind = 0
    for tuple in tuples:
        i = tuple[0]
        j = tuple[1]
        if isFinite(R, L, r, 0., k1[i], k2[j]):
            if ind + 1 >= len(row):
                row = np.hstack((row, np.empty(len(row))))
                col = np.hstack((col, np.empty(len(row))))
                dataTMTM = np.vstack((dataTMTM, np.empty((len(row), M))))
                dataTETE = np.vstack((dataTETE, np.empty((len(row), M))))
            row[ind] = i
            col[ind] = j
            dataTMTM[ind, :], dataTETE[ind, :] = arm_elements_zero(R, L, r, M, k1[i], k2[j], alpha_sphere, materialclass_sphere, lmax)
            ind += 1

    row = row[:ind]
    col = col[:ind]
    dataTMTM = dataTMTM[:ind, :]
    dataTETE = dataTETE[:ind, :]
    return row, col, dataTMTM, dataTETE


@njit("UniTuple(float64[:,:], 2)(int64[:,:], float64, float64, float64, int64, float64[:], float64[:], float64, unicode_type, int64)", parallel=True, cache=True)
def arm_parallel_zero(tuples, R, L, r, M, k1, k2, alpha_sphere, materialclass_sphere, lmax):
    r"""
    Angular Reflection Matrix elements for the ZERO-frequency/wavenumber contribution for the given indices,
    computed in parallel threads.

    Parameters
    ----------
    tuples : np.ndarray
        2D array containing row and column indices
    R, L, r, M, k1, k2, alpha_sphere, materialclass_sphere, lmax
        see ``arm_full_zero``

    Returns
    -------
    dataTMTM, dataTETE : np.ndarray
        angular matrix elements for TMTM, TETE polarization

    """
    n = len(tuples)
    dataTMTM = np.empty((n, M))
    dataTETE = np.empty((n, M))
    for t in prange(n):
        TMTM, TETE = arm_elements_zero(R, L, r, M, k1[tuples[t, 0]], k2[tuples[t, 1]], alpha_sphere, materialclass_sphere, lmax)
        dataTMTM[t, :] = TMTM
        dataTETE[t, :] = TETE
    return dataTMTM, dataTETE


def arm_full_zero(nproc, R, L, r, N1, N2, M, k1, k2, alpha_sphere, materialclass_sphere, lmax):
    r"""
    FULL Angular Reflection Matrix for the ZERO-frequency/wavenumber contribution.
    The calculation is parellelized among nproc threads.

    Parameters
    ----------
    nproc: int
        number of threads
    R: float
        positive, sphere radius
    L: float
        positive, surface-to-surface distance
    r: float
        positive, relative amount of surface-to-surface translation
    N1, N2, M: int
        positive, radial and angular discretization order
    k1, k2 : np.ndarray
        nodes of the radial quadrature rule
    alpha_sphere : float
        positive, parameter
    materialclass_sphere : string
        materialclass
    lmax : int
        positive, cut-off angular momentum

    Returns
    -------
    row: np.ndarray
        row indices
    col: np.ndarray
        column indices
    dataTMTM, dataTETE : np.ndarray
        angular matrix elements for TMTM, TETE polarization

    Dependencies
    ------------
    finite_tuples, arm_parallel_zero

    """
    tuples = finite_tuples(nproc, R, L, r, 0., N1, N2, k1, k2)
    dataTMTM, dataTETE = arm_parallel_zero(tuples, R, L, r, M, k1, k2, alpha_sphere, materialclass_sphere, lmax)
    row = np.ascontiguousarray(tuples[:, 0])
    col = np.ascontiguousarray(tuples[:, 1])
    return row, col, dataTMTM, dataTETE
