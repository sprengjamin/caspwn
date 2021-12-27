r""" Casimir interaction for the plane-sphere geometry at given frequency/wavenumber

"""
import numpy as np
from math import sqrt
from numba import njit
from scipy.linalg import lu_factor, lu_solve
from time import perf_counter

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../sphere/"))
from mie import mie_e_array
from reflection_matrix import arm_full_zero, arm_full_finite


@njit("UniTuple(float64[:,:], 3)(int64[:], int64[:], float64[:], float64[:], float64[:], float64[:], float64, int64, int64, int64, float64[:], float64[:], float64[:], float64[:])", cache=True)
def construct_matrix_finite(row, col, dataTMTM, dataTETE, dataTMTE, dataTETM, r, N1, N2, M, kappa1, kappa2, w1, w2):
    r"""Construct matrix and its first two derivatives.

        Parameters
        ----------
        row, col : list
            1D arrays of ints, row and column index of the matrix elements, respectively.
        data : list
            1D array of floats, matrix elements
        r : float
            relative translation distance contained in the reflection matrix, between 0 and 1
        N1, N2, M : int
            positive, radial and angular discretization order
        kappa1, kappa2 : list
            1D array of floats, imaginary z-component of the wave vectors
        w1, w2 : list
            weights of radial quadrature

        Returns
        -------
        (ndarray, ndarray, ndarray)

    """
    mat = np.zeros((2 * N1, 2 * N2))
    dL_mat = np.zeros((2 * N1, 2 * N2))
    d2L_mat = np.zeros((2 * N1, 2 * N2))
    for l in range(len(row)):
        i, j = row[l], col[l]
        weight = 1/M/2/np.pi*sqrt(w1[i]*w2[j])
        dL_factor = -(kappa1[i] + kappa2[j])*r
        d2L_factor = dL_factor**2
        mat[i, j] = weight*dataTMTM[l]
        mat[i+N1, j+N2] = weight*dataTETE[l]
        mat[i, j+N2] = weight*dataTMTE[l]
        mat[i+N1, j] = weight*dataTETM[l]

        dL_mat[i, j] = dL_factor*mat[i, j]
        dL_mat[i+N1, j+N2] = dL_factor*mat[i+N1, j+N2]
        dL_mat[i, j+N2] = dL_factor*mat[i, j+N2]
        dL_mat[i+N1, j] = dL_factor*mat[i+N1, j]

        d2L_mat[i, j] = d2L_factor*mat[i, j]
        d2L_mat[i+N1, j+N2] = d2L_factor*mat[i+N1, j+N2]
        d2L_mat[i, j+N2] = d2L_factor*mat[i, j+N2]
        d2L_mat[i+N1, j] = d2L_factor*mat[i+N1, j]
        if N1 == N2:
            mat[j, i] = mat[i, j]
            mat[j+N2, i+N1] = mat[i+N1, j+N2]
            mat[j+N2, i] = mat[i, j+N2]
            mat[j, i+N1] = mat[i+N1, j]

            dL_mat[j, i] = dL_mat[i, j]
            dL_mat[j + N2, i + N1] = dL_mat[i + N1, j + N2]
            dL_mat[j + N2, i] = dL_mat[i, j + N2]
            dL_mat[j, i + N1] = dL_mat[i + N1, j]

            d2L_mat[j, i] = d2L_mat[i, j]
            d2L_mat[j + N2, i + N1] = d2L_mat[i + N1, j + N2]
            d2L_mat[j + N2, i] = d2L_mat[i, j + N2]
            d2L_mat[j, i + N1] = d2L_mat[i + N1, j]
    return mat, dL_mat, d2L_mat


def compute_matrix_operations(mat1, dL_mat1, d2L_mat1, mat2, dL_mat2, d2L_mat2, observable):
    r"""Computes a 3-tuple containing the quantities

        .. math::
            \begin{aligned}
            \log\det(1-\mathcal{M})\,,\\
            \mathrm{tr}\left[\frac{\partial_L\mathcal{M}}{1-\mathcal{M}}\right]\,,\\
            \mathrm{tr}\left[\frac{\partial_L^2\mathcal{M}}{1-\mathcal{M}} + \left(\frac{\partial_L\mathcal{M}}{1-\mathcal{M}}\right)^2\right]\,.
            \end{aligned}

        where
        .. math::
            \begin{aligned}
            \mathcal{M}&=\mathcal{M}_1\mathcal{M}_2\,,\\
            \partial_L\mathcal{M}&= (\partial_L\mathcal{M}_1)\mathcal{M}_2 + \mathcal{M}_1(\partial_L\mathcal{M}_2)\,,\\
            \partial^2_L\mathcal{M}&= (\partial^2_L\mathcal{M}_1)\mathcal{M}_2 + 2(\partial_L\mathcal{M}_1)(\partial_L\mathcal{M}_2) + \mathcal{M}_1(\partial^2_L\mathcal{M}_2)\,.
            \end{aligned}

        with :math:`\mathtt{mat1}=\mathcal{M}_2`, :math:`\mathtt{dL_mat1}=\partial_L\mathcal{M}_1` and :math:`\mathtt{d2L_mat1}=\partial^2_L\mathcal{M}_1` and the same for the index 2.

        When observable="energy" only the first quantity is computed and the other two returned as zero.
        For observable="force" only the first two quantities are computed and the last one returned as zero.
        When observable="forcegradient" all quantities are computed.

        Parameters
        ----------
        mat : ndarray
            2D array, round-trip matrix
        dL_mat : ndarray
            2D array, first derivative of round-trip matrix
        d2L_mat : ndarray
            2D array, second derivative of round-trip matrix
        observable : string
            specification of which observables are to be computed, allowed values are "energy", "force", "forcegradient"

        Returns
        -------
        (float, float, float)


    """
    mat = mat1.dot(mat2)

    norm_mat = np.linalg.norm(mat)
    tr_mat = np.trace(mat)
    if tr_mat == 0.:
        assert (norm_mat == 0.)
        return 0., 0., 0.

    if abs(norm_mat ** 2 / (1 - norm_mat) / tr_mat) < 1.e-10:
        # make trace approximation
        if observable == "energy":
            return -tr_mat, 0., 0.
        dL_mat = dL_mat1.dot(mat2) + mat1.dot(dL_mat2)
        tr_dL_mat = np.trace(dL_mat)
        if observable == "force":
            return -tr_mat, tr_dL_mat, 0.
        d2L_mat = d2L_mat1.dot(mat2) + 2 * dL_mat1.dot(dL_mat2) + mat1.dot(d2L_mat2)
        tr_d2L_mat = np.trace(d2L_mat)
        if observable == "forcegradient":
            return -tr_mat, tr_dL_mat, tr_d2L_mat
        else:
            raise ValueError
    else:
        # compute logdet etc. exactly
        lu, piv = lu_factor(np.eye(mat.shape[0]) - mat)
        logdet = np.sum(np.log(np.diag(lu)))
        if observable == "energy":
            return logdet, 0., 0.

        dL_mat = dL_mat1.dot(mat2) + mat1.dot(dL_mat2)
        matA = lu_solve((lu, piv), dL_mat)
        dL_logdet = np.trace(matA)
        if observable == "force":
            return logdet, dL_logdet, 0.
        d2L_mat = d2L_mat1.dot(mat2) + 2 * dL_mat1.dot(dL_mat2) + mat1.dot(d2L_mat2)
        matB = lu_solve((lu, piv), d2L_mat)
        d2L_logdet = np.trace(matB) + np.sum(matA**2)
        if observable == "forcegradient":
            return logdet, dL_logdet, d2L_logdet
        else:
            raise ValueError

def contribution_finite(R1, R2, L, K, n_sphere1, n_sphere2, Nouter, Ninner, M, nds_outer, wts_outer, nds_inner, wts_inner, lmax1, lmax2, nproc, observable):
    r"""
    Computes the contribution to the observable depending on the wave number K.

    Parameters
    ----------
    R1, R2: float
        positive, radius of the sphere 1 and 2
    L: float
        positive, surface-to-surface distance
    K: float
        positive, rescaled wavenumber in medium
    n_sphere1, n_sphere2: float
        positive, refractive index of sphere1 and sphere2 (relative to medium)
    Nouter, Ninner, M: int
        positive, outer and inner radial and angular discretization order
    nds_outer, wts_outer: np.ndarray
        nodes and weights of the outer radial quadrature rule
    nds_inner, wts_inner: np.ndarray
        nodes and weights of the inner radial quadrature rule
    lmax1, lmax2 : int
        positive, cut-off angular momentum for sphere 1 and 2
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

    ### sphere 1
    rho1 = R1/L # aspect ratio
    x1 = K*rho1
    # precompute mie coefficients
    if x1 > 5e3:
        # dummy variables
        mie_a, mie_b = np.empty(1), np.empty(1)
    else:
        mie_a, mie_b = mie_e_array(lmax1, x1, n_sphere1)
    row1, col1, TMTM1, TETE1, TMTE1, TETM1 = arm_full_finite(nproc, rho1, 0.5, 1., K, Nouter, Ninner, M, nds_outer,
                                                             nds_inner, n_sphere1, lmax1, mie_a, mie_b)

    ### sphere 2
    rho2 = R2 / L  # aspect ratio
    x2 = K * rho2
    # precompute mie coefficients
    if x2 > 5e3:
        # dummy variables
        mie_a, mie_b = np.empty(1), np.empty(1)
    else:
        mie_a, mie_b = mie_e_array(lmax2, x2, n_sphere2)
    row2, col2, TMTM2, TETE2, TMTE2, TETM2 = arm_full_finite(nproc, rho2, 0.5, -1., K, Ninner, Nouter, M, nds_inner,
                                                             nds_outer, n_sphere2, lmax2, mie_a, mie_b)

    end_matrix = perf_counter()
    timing_matrix = end_matrix-start_matrix
    start_dft = end_matrix
    if len(row1) != 0.:
        TMTM1 = np.fft.rfft(TMTM1).real
        TETE1 = np.fft.rfft(TETE1).real
        TMTE1 = np.fft.rfft(TMTE1).imag
        TETM1 = -np.fft.rfft(TETM1).imag
    if len(row2) != 0.:
        TMTM2 = np.fft.rfft(TMTM2).real
        TETE2 = np.fft.rfft(TETE2).real
        TMTE2 = np.fft.rfft(TMTE2).imag
        TETM2 = -np.fft.rfft(TETM2).imag
    end_dft = perf_counter()
    timing_dft = end_dft-start_dft
    start_logdet = end_dft

    kappa_outer = np.sqrt(K ** 2 + nds_outer ** 2)
    kappa_inner = np.sqrt(K ** 2 + nds_inner ** 2)

    # m=0
    mat1, dL_mat1, d2L_mat1 = construct_matrix_finite(row1, col1, TMTM1[:,0], TETE1[:,0], TMTE1[:,0], TETM1[:,0], 0.5, Nouter, Ninner, M, kappa_outer, kappa_inner, wts_outer, wts_inner)
    mat2, dL_mat2, d2L_mat2 = construct_matrix_finite(row2, col2, TMTM2[:,0], TETE2[:,0], TMTE2[:,0], TETM2[:,0], 0.5, Ninner, Nouter, M, kappa_inner, kappa_outer, wts_inner, wts_outer)

    logdet, dL_logdet, d2L_logdet = compute_matrix_operations(mat1, dL_mat1, d2L_mat1, mat2, dL_mat2, d2L_mat2, observable)
    # m>0
    for m in range(1, M//2):
        mat1, dL_mat1, d2L_mat1 = construct_matrix_finite(row1, col1, TMTM1[:, m], TETE1[:, m], TMTE1[:, m],
                                                          TETM1[:, m], 0.5, Nouter, Ninner, M, kappa_outer, kappa_inner,
                                                          wts_outer, wts_inner)
        mat2, dL_mat2, d2L_mat2 = construct_matrix_finite(row2, col2, TMTM2[:, m], TETE2[:, m], TMTE2[:, m],
                                                          TETM2[:, m], 0.5, Ninner, Nouter, M, kappa_inner, kappa_outer,
                                                          wts_inner, wts_outer)
        term1, term2, term3 = compute_matrix_operations(mat1, dL_mat1, d2L_mat1, mat2, dL_mat2, d2L_mat2,
                                                                  observable)
        logdet += 2 * term1
        dL_logdet += 2 * term2
        d2L_logdet += 2 * term3

    # last m
    mat1, dL_mat1, d2L_mat1 = construct_matrix_finite(row1, col1, TMTM1[:, M//2], TETE1[:, M//2], TMTE1[:, M//2],
                                                      TETM1[:, M//2], 0.5, Nouter, Ninner, M, kappa_outer, kappa_inner,
                                                      wts_outer, wts_inner)
    mat2, dL_mat2, d2L_mat2 = construct_matrix_finite(row2, col2, TMTM2[:, M//2], TETE2[:, M//2], TMTE2[:, M//2],
                                                      TETM2[:, M//2], 0.5, Ninner, Nouter, M, kappa_inner, kappa_outer,
                                                      wts_inner, wts_outer)
    term1, term2, term3 = compute_matrix_operations(mat1, dL_mat1, d2L_mat1, mat2, dL_mat2, d2L_mat2,
                                                    observable)
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

@njit("UniTuple(float64[:,:], 3)(int64[:], int64[:], float64[:], float64, int64, int64, int64, float64[:], float64[:], float64[:], float64[:])", cache=True)
def construct_roundtrip_zero(row, col, data, r, N1, N2, M, k1, k2, w1, w2):
    r"""Construct round-trip matrices and its first two derivatives for each m.

        Parameters
        ----------
        row, col : ndarray
            1D arrays of ints, row and column index of the matrix elements, respectively.
        data : ndarray
            1D array of floats, matrix elements
        r : float
            relative translation distance contained in the reflection matrix, between 0 and 1
        N1, N2, M : int
            matrix dimension of each polarization block
        k1, k2, w1, w2: list
            nodes and weights of radial quadrature

        Returns
        -------
        (ndarray, ndarray, ndarray)

    """
    mat = np.zeros((N1, N2))
    dL_mat = np.zeros((N1, N2))
    d2L_mat = np.zeros((N1, N2))
    for l in range(len(data)):
        i, j = row[l], col[l]
        weight = 1/M/2/np.pi*sqrt(w1[i]*w2[j])
        dL_factor = -(k1[i] + k2[j])*r
        d2L_factor = dL_factor**2
        mat[i, j] = weight*data[l]
        dL_mat[i, j] = dL_factor*mat[i, j]
        d2L_mat[i, j] = d2L_factor*mat[i, j]
        if N1 == N2:
            mat[j, i] = mat[i, j]
            dL_mat[j, i] = dL_mat[i, j]
            d2L_mat[j, i] = d2L_mat[i, j]
    return mat, dL_mat, d2L_mat


def contribution_zero(R1, R2, L, alpha_sphere1, alpha_sphere2, materialclass_sphere1, materialclass_sphere2, N_outer, N_inner, M, nds_outer, wts_outer, nds_inner, wts_inner, lmax1, lmax2, nproc, observable):
    r"""
    Computes the contribution to the observable for a vanishing frequency/wavenumber.

    Parameters
    ----------
    R1, R2: float
        positive, radius of the sphere 1 and 2
    L: float
        positive, surface-to-surface distance
    alpha_sphere1, alpha_sphere2 : float
        positive, parameter for sphere 1 and 2 (meaning depends on
        materialclass)
    materialclass_sphere1, materialclass_sphere2 : string
        materialclass of sphere 1 and sphere 2
    N_outer, N_inner, M: int
        positive, radial and angular discretization order
    nds_outer, wts_outer: np.ndarray
        nodes and weights of the outer radial quadrature rule
    nds_inner, wts_inner: np.ndarray
        nodes and weights of the inner radial quadrature rule
    lmax1, lmax2 : int
        positive, cut-off angular momentum for sphere 1 and 2
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
    rho1 = R1/L
    row1, col1, TM1, TE1 = arm_full_zero(nproc, rho1, 0.5, N_outer, N_inner, M, nds_outer, nds_inner, alpha_sphere1, materialclass_sphere1, lmax1)
    rho2 = R2/L
    row2, col2, TM2, TE2 = arm_full_zero(nproc, rho2, 0.5, N_inner, N_outer, M, nds_inner, nds_outer, alpha_sphere2, materialclass_sphere2, lmax2)
    end_matrix = perf_counter()
    timing_matrix = end_matrix-start_matrix

    ## discrete Fourier transform
    start_fft = end_matrix
    if len(row1) != 0.:
        TM1 = np.fft.rfft(TM1).real
        TE1 = np.fft.rfft(TE1).real
    if len(row2) != 0.:
        TM2 = np.fft.rfft(TM2).real
        TE2 = np.fft.rfft(TE2).real
    end_fft = perf_counter()
    timing_fft = end_fft-start_fft

    start_logdet = end_fft

    ## TM contribution
    # m=0
    mat1, dL_mat1, d2L_mat1 = construct_roundtrip_zero(row1, col1, TM1[:,0], 0.5, N_outer, N_inner, M, nds_outer, nds_inner, wts_outer, wts_inner)
    mat2, dL_mat2, d2L_mat2 = construct_roundtrip_zero(row2, col2, TM2[:,0], 0.5, N_inner, N_outer, M, nds_inner, nds_outer, wts_inner, wts_outer)
    logdet_TM, dL_logdet_TM, d2L_logdet_TM = compute_matrix_operations(mat1, dL_mat1, d2L_mat1, mat2, dL_mat2, d2L_mat2, observable)

    # m>0
    for m in range(1, M//2):
        mat1, dL_mat1, d2L_mat1 = construct_roundtrip_zero(row1, col1, TM1[:, m], 0.5, N_outer, N_inner, M, nds_outer,
                                                           nds_inner, wts_outer, wts_inner)
        mat2, dL_mat2, d2L_mat2 = construct_roundtrip_zero(row2, col2, TM2[:, m], 0.5, N_inner, N_outer, M, nds_inner,
                                                           nds_outer, wts_inner, wts_outer)
        term1, term2, term3 = compute_matrix_operations(mat1, dL_mat1, d2L_mat1, mat2, dL_mat2, d2L_mat2,
                                                                  observable)
        logdet_TM += 2 * term1
        dL_logdet_TM += 2 * term2
        d2L_logdet_TM += 2 * term3

    # last m
    mat1, dL_mat1, d2L_mat1 = construct_roundtrip_zero(row1, col1, TM1[:, M//2], 0.5, N_outer, N_inner, M, nds_outer,
                                                       nds_inner, wts_outer, wts_inner)
    mat2, dL_mat2, d2L_mat2 = construct_roundtrip_zero(row2, col2, TM2[:, M//2], 0.5, N_inner, N_outer, M, nds_inner,
                                                       nds_outer, wts_inner, wts_outer)
    term1, term2, term3 = compute_matrix_operations(mat1, dL_mat1, d2L_mat1, mat2, dL_mat2, d2L_mat2,
                                                    observable)
    if M%2==0:
        logdet_TM += term1
        dL_logdet_TM += term2
        d2L_logdet_TM += term3
    else:
        logdet_TM += 2 * term1
        dL_logdet_TM += 2 * term2
        d2L_logdet_TM += 2 * term3

    ## TE contribution
    if materialclass_sphere1 != "dielectric" and materialclass_sphere1 != "drude" and materialclass_sphere2 != "dielectric" and materialclass_sphere2 != "drude":
        # m=0
        mat1, dL_mat1, d2L_mat1 = construct_roundtrip_zero(row1, col1, TE1[:, 0], 0.5, N_outer, N_inner, M, nds_outer,
                                                           nds_inner, wts_outer, wts_inner)
        mat2, dL_mat2, d2L_mat2 = construct_roundtrip_zero(row2, col2, TE2[:, 0], 0.5, N_inner, N_outer, M, nds_inner,
                                                           nds_outer, wts_inner, wts_outer)
        logdet_TE, dL_logdet_TE,d2L_logdet_TE = compute_matrix_operations(mat1, dL_mat1, d2L_mat1, mat2, dL_mat2, d2L_mat2,
                                                                  observable)

        # m>0
        for m in range(1, M//2):
            mat1, dL_mat1, d2L_mat1 = construct_roundtrip_zero(row1, col1, TE1[:, m], 0.5, N_outer, N_inner, M, nds_outer,
                                                               nds_inner, wts_outer, wts_inner)
            mat2, dL_mat2, d2L_mat2 = construct_roundtrip_zero(row2, col2, TE2[:, m], 0.5, N_inner, N_outer, M, nds_inner,
                                                               nds_outer, wts_inner, wts_outer)
            term1, term2, term3 = compute_matrix_operations(mat1, dL_mat1, d2L_mat1, mat2, dL_mat2, d2L_mat2,
                                                            observable)
            logdet_TE += 2 * term1
            dL_logdet_TE += 2 * term2
            d2L_logdet_TE += 2 * term3

        # last m
        mat1, dL_mat1, d2L_mat1 = construct_roundtrip_zero(row1, col1, TE1[:, M//2], 0.5, N_outer, N_inner, M, nds_outer,
                                                           nds_inner, wts_outer, wts_inner)
        mat2, dL_mat2, d2L_mat2 = construct_roundtrip_zero(row2, col2, TE2[:, M//2], 0.5, N_inner, N_outer, M, nds_inner,
                                                           nds_outer, wts_inner, wts_outer)
        term1, term2, term3 = compute_matrix_operations(mat1, dL_mat1, d2L_mat1, mat2, dL_mat2, d2L_mat2,
                                                                  observable)
        if M%2==0:
            logdet_TE += term1
            dL_logdet_TE += term2
            d2L_logdet_TE += term3
        else:
            logdet_TE += 2 * term1
            dL_logdet_TE += 2 * term2
            d2L_logdet_TE += 2 * term3
    else:
        logdet_TE = 0.
        dL_logdet_TE = 0.
        d2L_logdet_TE = 0.

    end_logdet = perf_counter()
    timing_logdet = end_logdet-start_logdet
    print("# ", end="")
    print(0., logdet_TM+logdet_TE, "%.3f"%timing_matrix, "%.3f"%timing_fft, "%.3f"%timing_logdet, sep=", ")
    return np.array([logdet_TM, dL_logdet_TM/L, d2L_logdet_TM/L**2]), np.array([logdet_TE, dL_logdet_TE/L, d2L_logdet_TE/L**2])

if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../ufuncs/"))
    from integration import fc_quadrature

    R1 = 1#1.e-6
    R2 = 2#1.e-6
    L = 0.01#10.e-9

    K = 1.
    n_sphere1 = 2.3
    n_sphere2 = 4.5

    rho1 = max(50, R1/L)
    rho2 = max(50, R2/L)
    rhoeff = max(50, R1*R2/(R1+R2)/L)
    N_outer = int(8.*np.sqrt(rhoeff))
    N_inner = int(8.*np.sqrt(rho1+rho2))
    print(N_outer)
    print(N_inner)
    M = int(6.*np.sqrt(rho1+rho2))
    lmax1 = int(12.*rho1)
    lmax2 = int(12.*rho2)


    nproc = 4
    observable = "energy"

    nds_outer, wts_outer = fc_quadrature(N_outer)
    nds_inner, wts_inner = fc_quadrature(N_inner)

    #c = contribution_finite(R1, R2, L, K, n_sphere1, n_sphere2, N_outer, N_inner, M, nds_outer, wts_outer, nds_inner, wts_inner, lmax1, lmax2, nproc, observable)

    alpha_sphere1 = 4.
    alpha_sphere2 = 4.
    materialclass_sphere2 = 'drude'
    materialclass_sphere1 = 'drude'

    c0 = contribution_zero(R1, R2, L, alpha_sphere1, alpha_sphere2, materialclass_sphere1, materialclass_sphere2, N_outer, N_inner, M, nds_outer, wts_outer, nds_inner, wts_inner, lmax1, lmax2, nproc, observable)


