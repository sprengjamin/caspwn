r""" Casimir energy for the sphere-sphere geometry.

"""
import mkl
#mkl.domain_set_num_threads(1, "fft")
import numpy as np
import time
import concurrent.futures as futures
from numba import njit

from scipy.sparse.linalg import splu
from scipy.sparse import eye
from scipy.sparse import coo_matrix

from scipy.constants import Boltzmann, hbar, c

from scipy.linalg import lu_factor, lu_solve
from scipy.integrate import quad

from index import itt, itt_nosquare
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../sphere/"))
from mie import mie_e_array
from kernel import kernel_polar as kernel
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../ufuncs/"))
from integration import quadrature
from psd import psd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../material/"))
import material


def logdet_sparse(mat):
    r"""This module computes

    .. math::
        \log\det(1-\mathcal{M})

    where :math:`\mathcal{M}` is given by the sparse matrix mat.

    Parameters
    ----------
    mat: sparse coo_matrix
        round-trip matrix

    Returns
    -------
    float
    
    """
    dim = mat.shape[0]
    lu = splu(eye(dim, format="csc")-mat)
    return np.sum(np.log(lu.U.diagonal()))


@njit("float64[:,:](float64, float64, float64, float64, int64, float64, float64, float64, float64, float64, string, int64, float64[:], float64[:])", cache=True)
def phi_array(rho, r, sign, K, M, k1, k2, w1, w2, n, materialclass, lmax, mie_a, mie_b):
    """
    Returns the a phi sqeuence for the kernel function for each polarization block.

    Parameters
    ----------
    rho: float
        positive, aspect ratio R/L
    r: float
        positive, relative effective radius R_eff/R
    sign: +1/-1
        sign, differs for the two spheres
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
        array of length 4 of the phi sequence for the polarization contributions
        TMTM, TETE, TMTE, TETM

    """
    phiarr = np.empty((4, M))

    # phi = 0.
    kernelTMTM, kernelTETE, kernelTMTE, kernelTETM =  kernel(rho, r, sign, K, k1, k2, 0., n, materialclass, lmax, mie_a, mie_b)
    phiarr[0, 0] = w1*w2*kernelTMTM
    phiarr[1, 0] = w1*w2*kernelTETE
    phiarr[2, 0] = w1*w2*kernelTMTE
    phiarr[3, 0] = w1*w2*kernelTETM
    
    if M%2==0:
        # phi = np.pi
        kernelTMTM, kernelTETE, kernelTMTE, kernelTETM =  kernel(rho, r, sign, K, k1, k2, np.pi, n, materialclass, lmax, mie_a, mie_b)
        phiarr[0, M//2] = w1*w2*kernelTMTM
        phiarr[1, M//2] = w1*w2*kernelTETE
        phiarr[2, M//2] = w1*w2*kernelTMTE
        phiarr[3, M//2] = w1*w2*kernelTETM
        imax = M//2-1
        phi = 2*np.pi*np.arange(M//2)/M
    else:
        imax = M//2
        phi = 2*np.pi*np.arange(M//2+1)/M
    
    for i in range(1, imax+1):
        kernelTMTM, kernelTETE, kernelTMTE, kernelTETM =  kernel(rho, r, sign, K, k1, k2, phi[i], n, materialclass, lmax, mie_a, mie_b)
        phiarr[0, i] = w1*w2*kernelTMTM
        phiarr[1, i] = w1*w2*kernelTETE
        phiarr[2, i] = w1*w2*kernelTMTE
        phiarr[3, i] = w1*w2*kernelTETM
        phiarr[0, M-i] = phiarr[0, i]
        phiarr[1, M-i] = phiarr[1, i]
        phiarr[2, M-i] = -phiarr[2, i]
        phiarr[3, M-i] = -phiarr[3, i]
    return phiarr


def m_array(rho, r, sign, K, M, k1, k2, w1, w2, n, materialclass, lmax, mie_a, mie_b):
    """
    Computes the m_array by means of a FFT of the computed phi_array.

    Parameters
    ----------
    rho: float
        positive, aspect ratio R/L
    r: float
        positive, relative effective radius R_eff/R
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
        array of length 4 of the phi sequence for the polarization contributions
        TMTM, TETE, TMTE, TETM

    Dependencies
    ------------
    phi_array

    """
    phiarr = phi_array(rho, r, sign, K, M, k1, k2, w1, w2, n, materialclass, lmax, mie_a, mie_b)
    marr = np.fft.rfft(phiarr)
    return np.array([marr[0,:].real, marr[1,:].real, -marr[2,:].imag, marr[3,:].imag])


def mArray_sparse_part(indices, rho, r, sign, K, Nrow, Ncol, M, krow, wrow, kcol, wcol, n, materialclass, lmax, mie_a, mie_b):
    """
    Computes the m-array.

    Parameters
    ----------
    dindices: np.ndarray
        array of diagonal indices
    oindices: np.ndarray
        array of off-diagonal indices
    rho: float
        positive, aspect ratio R/L
    r: float
        positive, relative effective radius R_eff/R
    sign: +1/-1
        sign, differs for the two spheres
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
    N = max(Nrow, Ncol)
    row = np.empty(16*N, dtype=np.int32)
    col = np.empty(16*N, dtype=np.int32)
    data = np.empty((16*N, M//2+1))
    
    ind = 0
    for index in indices:
        i, j = itt_nosquare(index, Nrow, Ncol)
        if isFinite(rho, r, K, krow[i], kcol[j]):
            row[ind] = i
            col[ind] = j
            row[ind+1] = i+Nrow
            col[ind+1] = j+Ncol
            row[ind+2] = i
            col[ind+2] = j+Ncol
            row[ind+3] = i+Nrow
            col[ind+3] = j
            data[ind:ind+4, :] = m_array(rho, r, sign, K, M, krow[i], kcol[j], wrow[i], wcol[j], n, materialclass, lmax, mie_a, mie_b)
            ind += 4
            if ind >= len(row):
                row = np.hstack((row, np.empty(len(row), dtype=np.int32)))
                col = np.hstack((col, np.empty(len(row), dtype=np.int32)))
                data = np.vstack((data, np.empty((len(row), M//2+1))))
    
    row = row[:ind] 
    col = col[:ind] 
    data = data[:ind, :] 
    return row, col, data


def mArray_sparse_mp(nproc, rho, r, sign, K, Nrow, Ncol, M, pts_row, wts_row, pts_col, wts_col, n, materialclass, lmax, mie_a, mie_b):
    """
    Computes the m-array in parallel using the multiprocessing module.

    Parameters
    ----------
    nproc: int
        number of processes
    rho: float
        positive, aspect ratio R/L
    r: float
        positive, relative effective radius R_eff/R
    sign: +1/-1
        sign, differs for the two spheres
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
    b = 1.
    krow = b * pts_row
    kcol = b * pts_col
    wrow = np.sqrt(b * wts_row * 2 * np.pi / M)
    wcol = np.sqrt(b * wts_col * 2 * np.pi / M)

    ndiv = nproc * 8  # factor is arbitrary, but can be chosen optimally

    indices = np.array_split(np.random.permutation(Nrow*Ncol), ndiv)

    with futures.ProcessPoolExecutor(max_workers=nproc) as executors:
        wait_for = [
            executors.submit(mArray_sparse_part, indices[i], rho, r, sign, K, Nrow, Ncol, M, krow, wrow, kcol, wcol, n, materialclass, lmax, mie_a, mie_b)
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


def isFinite(rho, r, K, k1, k2):
    """
    Estimates by means of the asymptotics of the scattering amplitudes if the given
    matrix element can be numerically set to zero.

    Parameters
    ----------
    rho: float
        positive, aspect ratio R/L
    r: float
        positive, relative effective radius R_eff/R
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
        exponent = 2*rho*np.sqrt(k1*k2) - (k1+k2)*(rho+r)
    else:
        kappa1 = np.sqrt(k1*k1+K*K)
        kappa2 = np.sqrt(k2*k2+K*K)
        exponent = rho*np.sqrt(2*(K*K + kappa1*kappa2 + k1*k2)) - (kappa1+kappa2)*(rho+r)
    if exponent < -37:
        return False
    else:
        return True

@njit("UniTuple(float64[:,:,:], 3)(int64[:], int64[:], float64[:,:], int64, int64, float64[:], float64[:])", cache=True)
def construct_matrices(row, col, data, Nrow, Ncol, kappa_row, kappa_col):
    r"""Construct round-trip matrices and its first two derivatives for each m.

        Parameters
        ----------
        row, col : ndarray
            1D arrays of ints, row and column index of the matrix elements, respectively.
        data : ndarray
            2D arrays of floats, matrix elements for each :math:`m`
        Nrow, Ncol : int
            row and column dimension of each polarization block
        kappa_row, kappa_col : ndarray
            1D arrays of floats, imaginary z-component of the wave vectors corresponding to row and column discretization

        Returns
        -------
        (ndarray, ndarray, ndarray)

        """
    data_len, m_len = data.shape
    M = np.zeros((2*Nrow, 2*Ncol, m_len))
    dL_M = np.zeros((2*Nrow, 2*Ncol, m_len))
    d2L_M = np.zeros((2 * Nrow, 2 * Ncol, m_len))
    for i in range(data_len):
        M[row[i], col[i],:] = data[i,:]
        dL_M[row[i], col[i],:] = -0.5 * (kappa_row[row[i] % Nrow] + kappa_col[col[i] % Ncol]) * data[i,:]
        d2L_M[row[i], col[i],:] = 0.25 * (kappa_row[row[i] % Nrow] + kappa_col[col[i] % Ncol]) ** 2 * data[i,:]
    return M, dL_M, d2L_M

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
    mat = mat1.dot(mat2)
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
    if observable == "pressure":
        return logdet, dL_logdet, d2L_logdet
    else:
        raise ValueError


def LogDet(R1, R2, L, materials, Kvac, Nin, Nout, M, pts_in, wts_in, pts_out, wts_out, lmax1, lmax2, nproc, observable):
    """
    Computes the sum of the logdets of the m-matrices.

    Parameters
    ----------
    R1, R2: float
        positive, radii of the spheres
    L: float
        positive, surface-to-surface distance
    materials : tuple of strings
        names of material in the order (sphere1, medium, sphere2) 
    K: float
        positive, rescaled frequency
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
    start_matrix = time.time()
    n_sphere1 = eval("material."+materials[0]+".n(Kvac/L)")
    n_medium = eval("material."+materials[1]+".n(Kvac/L)")
    n_sphere2 = eval("material."+materials[2]+".n(Kvac/L)")
    materialclass_sphere1 = eval("material."+materials[0]+".materialclass")
    materialclass_sphere2 = eval("material." + materials[2] + ".materialclass")
    
    #r1 = 1/(1+rho1/rho2)
    r1 = 0.5
    n1 = n_sphere1/n_medium
    # aspect ratio
    rho1 = R1/L
    
    # precompute mie coefficients for sphere 1
    x1 = n_medium * Kvac * rho1
    if x1 == 0.:
        mie_a, mie_b = mie_e_array(2, 1., n1)
    elif x1 > 5e3:
        mie_a, mie_b = mie_e_array(2, x1, n1)
    else:
        mie_a, mie_b = mie_e_array(lmax1, x1, n1)

    row1, col1, data1 = mArray_sparse_mp(nproc, rho1, r1, +1., Kvac*n_medium, Nout, Nin, M, pts_out, wts_out, pts_in, wts_in, n1, materialclass_sphere1, lmax1, mie_a, mie_b)

    #r2 = 1/(1+rho2/rho1)
    r2 = 0.5
    n2 = n_sphere2/n_medium
    # aspect ratio
    rho2 = R2/L
    
    # precompute mie coefficients for sphere 2
    x2 = n_medium * Kvac * rho2
    if x2 == 0.:
        mie_a, mie_b = mie_e_array(2, 1., n2)
    elif x2 > 5e3:
        mie_a, mie_b = mie_e_array(2, x2, n2)
    else:
        mie_a, mie_b = mie_e_array(lmax2, x2, n2)

    row2, col2, data2 = mArray_sparse_mp(nproc, rho2, r2, -1., Kvac*n_medium, Nin, Nout, M, pts_in, wts_in, pts_out, wts_out, n2, materialclass_sphere2, lmax2, mie_a, mie_b)
    
    end_matrix = time.time()
    timing_matrix = end_matrix-start_matrix
    start_logdet = end_matrix

    kappa_in = np.sqrt((n_medium*Kvac)**2 + pts_in**2)
    kappa_out = np.sqrt((n_medium*Kvac)**2 + pts_out**2)
    M1, dL_M1, d2L_M1 = construct_matrices(row1, col1, data1, Nout, Nin, kappa_out, kappa_in)
    M2, dL_M2, d2L_M2 = construct_matrices(row2, col2, data2, Nin, Nout, kappa_in, kappa_out)

    # m=0
    logdet, dL_logdet, d2L_logdet = compute_matrix_operations(M1[:,:,0], dL_M1[:,:,0], d2L_M1[:,:,0],
                                                              M2[:,:,0], dL_M2[:,:,0], d2L_M2[:,:,0], observable)
    
    # m>0    
    for m in range(1, M//2):
        term1, term2, term3 = compute_matrix_operations(M1[:, :, m], dL_M1[:, :, m], d2L_M1[:, :, m],
                                                        M2[:, :, m], dL_M2[:, :, m], d2L_M2[:, :, m], observable)
        logdet += 2 * term1
        dL_logdet += 2 * term2
        d2L_logdet += 2 * term3
    
    # last m
    term1, term2, term3 = compute_matrix_operations(M1[:, :, M//2], dL_M1[:, :, M//2], d2L_M1[:, :, M//2],
                                                    M2[:, :, M//2], dL_M2[:, :, M//2], d2L_M2[:, :, M//2], observable)
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

    
def casimir_zero(R1, R2, L, materials, Nin, Nout, M, X, lmax1, lmax2, nproc, observable):
    """
    Computes the Casimir energy at zero temperature.

    Parameters
    ----------
    R1, R2: float
        positive, radii of the spheres
    L: float
        positive, surface-to-surface distance
    materials: tuple
        contains the materials in the form (material of sphere1, medium,
        material of sphere2)
    Nin, Nout: int
        positive, quadrature order of inner and outer k-integration, respectively
    M: int
        positive, quadrature order of phi-integration
    X: int
        positive, quadrature order of K-integration
    nproc: int
        number of processes spawned by multiprocessing module

    Returns
    -------
    energy: float
        Casimir energy in Joule

    """
    p_in, w_in = quadrature(Nin)
    p_out, w_out = quadrature(Nout)
    K_pts, K_wts = quadrature(X)
    
    energy = 0.
    for i in range(X):
        result = LogDet(R1, R2, L, materials, K_pts[i], Nin, Nout, M, p_in, w_in, p_out, w_out, lmax1, lmax2, nproc)
        energy += K_wts[i]*result
    return energy/(2*np.pi)*hbar*c/L


def casimir_finite(R1, R2, L, T, materials, Nin, Nout, M, lmax1, lmax2, mode, epsrel, nproc, observable):
    """
    Computes the Casimir free energy at equilibrium temperature :math:`T`.

    Parameters
    ----------
    R1, R2: float
        positive, radii of the spheres
    L: float
        positive, surface-to-surface distance
    materials: tuple
        contains the materials in the form (material of sphere1, medium,
        material of sphere2)
    Nin, Nout: int
        positive, quadrature order of inner and outer k-integration, respectively
    M: int
        positive, quadrature order of phi-integration
    mode: str
        Matsubara spectrum decompostion (msd) or Pade spectrum decomposition (psd)
    epsrel: float
        positive, desired relative error for the Matsubara sum
    nproc: int
        number of processes spawned by multiprocessing module

    Returns
    -------
    energy: float
        Casimir free energy in Joule
    
    """
    p_in, w_in = quadrature(Nin)
    p_out, w_out = quadrature(Nout)
    
    K_matsubara = Boltzmann*T*L/(hbar*c)
    energy0 = LogDet(R1, R2, L, materials, 0., Nin, Nout, M, p_in, w_in, p_out, w_out, lmax1, lmax2, nproc, observable)

    if mode == "psd":
        energy = 0.
        Teff = 4*np.pi*Boltzmann/hbar/c*T*L
        order = int(max(np.ceil((1-1.5*np.log10(np.abs(epsrel)))/np.sqrt(Teff)), 5))
        xi, eta = psd(order)
        for n in range(order):
            term = 2*eta[n]*LogDet(R1, R2, L, materials, K_matsubara*xi[n], Nin, Nout, M, p_in, w_in, p_out, w_out, lmax1, lmax2, nproc, observable)
            energy += term
    elif mode == "msd":
        energy = np.zeros(3)
        n = 1
        while(True):
            term = LogDet(R1, R2, L, materials, 2*np.pi*K_matsubara*n, Nin, Nout, M, p_in, w_in, p_out, w_out, lmax1, lmax2, nproc, observable)
            energy += 2*term
            if abs(term[0]/energy0[0]) < epsrel:
                break
            n += 1
    else:
        raise ValueError("mode can either be 'psd' or 'msd'")
     
    return 0.5*T*Boltzmann*(energy+energy0), 0.5*T*Boltzmann*energy


if __name__ == "__main__":
    np.random.seed(0)
    R2 = 1.25e-05
    R1 = 2.5e-06
    L = 1.e-06
    T = 293.
    materials = ("Silica1", "Water", "Silica1")
    observable = "energy"
    #materials = ("PR", "Vacuum", "PR")
    lmax1 = int(10*R1/L)
    lmax2 = int(10 * R2 / L)

    rho1 = R1/L
    rho2 = R2/L
    rhoeff = rho1*rho2/(rho1+rho2)
    eta = 10

    nproc = 1
    Nin = 32#int(eta*np.sqrt(rho1+rho2))
    Nout = 7#int(eta*np.sqrt(rhoeff))
    M = 24#Nin

    n0, n1 = casimir_finite(R1, R2, L, T, materials, Nin, Nout, M, lmax1, lmax2, "psd", 1.e-08, nproc, observable)
    print("energy")
    print(n0, n1)
