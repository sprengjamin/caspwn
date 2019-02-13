import numpy as np

from numba import jit
from numba import float64, int64
from numba.types import UniTuple

import multiprocessing as mp

from scipy.sparse.linalg import splu
from scipy.sparse import eye
from scipy.sparse import coo_matrix

from scipy.constants import Boltzmann, hbar, c

from index import itt, itt_nosquare
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../sphere/"))
from mie import mie_cache
import scattering_amplitude
from kernel import kernel_polar
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


def make_phiSequence(kernel):
    """
    Makes phiSequence function for specified kernel.

    Parameters
    ----------
    kernel: function
        specified kernel function

    Returns
    -------
    phiSequence

    """
    @jit(float64[:,:](float64, float64, float64, float64, int64, float64, float64, float64, float64, mie_cache.class_type.instance_type), nopython=True)
    def phiSequence(rho, r, sign, K, M, k1, k2, w1, w2, mie):
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
        mie: class instance
            cache for the exponentially scaled mie coefficients

        Returns
        -------
        np.ndarray
            array of length 4 of the phi sequence for the polarization contributions
            TMTM, TETE, TMTE, TETM

        """
        phiarr = np.empty((4, M))

        # phi = 0.
        kernelTMTM, kernelTETE, kernelTMTE, kernelTETM =  kernel(rho, r, sign, K, k1, k2, 0., mie)
        phiarr[0, 0] = w1*w2*kernelTMTM
        phiarr[1, 0] = w1*w2*kernelTETE
        phiarr[2, 0] = w1*w2*kernelTMTE
        phiarr[3, 0] = w1*w2*kernelTETM
        
        if M%2==0:
            # phi = np.pi
            kernelTMTM, kernelTETE, kernelTMTE, kernelTETM =  kernel(rho, r, sign, K, k1, k2, np.pi, mie)
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
            kernelTMTM, kernelTETE, kernelTMTE, kernelTETM =  kernel(rho, r, sign, K, k1, k2, phi[i], mie)
            phiarr[0, i] = w1*w2*kernelTMTM
            phiarr[1, i] = w1*w2*kernelTETE
            phiarr[2, i] = w1*w2*kernelTMTE
            phiarr[3, i] = w1*w2*kernelTETM
            phiarr[0, M-i] = phiarr[0, i]
            phiarr[1, M-i] = phiarr[1, i]
            phiarr[2, M-i] = -phiarr[2, i]
            phiarr[3, M-i] = -phiarr[3, i]
        return phiarr
    return phiSequence


def mSequence(rho, r, sign, K, M, k1, k2, w1, w2, mie):
    """
    Computes mSqeuence by means of a FFT of the computed phiSequence.

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
    mie: class instance
        cache for the exponentially scaled mie coefficients

    Returns
    -------
    np.ndarray
        array of length 4 of the phi sequence for the polarization contributions
        TMTM, TETE, TMTE, TETM

    Dependencies
    ------------
    phiSequence

    """
    phiarr = phiSequence(rho, r, sign, K, M, k1, k2, w1, w2, mie)
    marr = np.fft.rfft(phiarr)
    return np.array([marr[0,:].real, marr[1,:].real, -marr[2,:].imag, marr[3,:].imag])


def mArray_sparse_part(indices, rho, r, sign, K, Nrow, Ncol, M, krow, wrow, kcol, wcol, mie):
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
    # 16 is just arbitrary here
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
            data[ind:ind+4, :] = mSequence(rho, r, sign, K, M, krow[i], kcol[j], wrow[i], wcol[j], mie)
            ind += 4
            if ind >= len(row):
                row = np.hstack((row, np.empty(len(row), dtype=np.int32)))
                col = np.hstack((col, np.empty(len(row), dtype=np.int32)))
                data = np.vstack((data, np.empty((len(row), M//2+1))))
    
    row = row[:ind] 
    col = col[:ind] 
    data = data[:ind, :] 
    return row, col, data


def mArray_sparse_mp(nproc, rho, r, sign, K, Nrow, Ncol, M, pts_row, wts_row, pts_col, wts_col, mie):
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
    
    def worker(indices, rho, r, sign, K, Nrow, Ncol, M, krow, wrow, kcol, wcol, mie, out):
        out.put(mArray_sparse_part(indices, rho, r, sign, K, Nrow, Ncol, M, krow, wrow, kcol, wcol, mie))

    b = 0.5 ### for now
    krow = b*pts_row
    kcol = b*pts_col
    wrow = np.sqrt(b*wts_row*2*np.pi/M)
    wcol = np.sqrt(b*wts_col*2*np.pi/M)
    
    indices = np.array_split(np.random.permutation(Nrow*Ncol), nproc)
    out = mp.Queue()
    procs = []
    for i in range(nproc):
        p = mp.Process(
                target = worker,
                args = (indices[i], rho, r, sign, K, Nrow, Ncol, M, krow, wrow, kcol, wcol, mie, out))
        procs.append(p)
        p.start()
    
    results = np.empty(nproc, dtype=object)
    for i in range(nproc):
        results[i] = out.get()
    
    for p in procs:
        p.join()
    
    row = results[0][0]
    col = results[0][1]
    data = results[0][2]
    for i in range(1, nproc):
        row = np.hstack((row, results[i][0]))
        col = np.hstack((col, results[i][1]))
        data = np.vstack((data, results[i][2]))
        
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
            

def LogDet(R1, R2, L, materials, Kvac, Nin, Nout, M, pts_in, wts_in, pts_out, wts_out, nproc): 
    """
    Computes the sum of the logdets the m-matrices.

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
    n_sphere1 = eval("material."+materials[0]+".n(Kvac/L)")
    n_medium = eval("material."+materials[1]+".n(Kvac/L)")
    n_sphere2 = eval("material."+materials[2]+".n(Kvac/L)")
    
    #r1 = 1/(1+rho1/rho2)
    r1 = 0.5
    n1 = n_sphere1/n_medium
    # aspect ratio
    rho1 = R1/L
    
    # precompute mie coefficients
    if Kvac == 0.:
        mie = mie_cache(3, 1., n1)              # dummy cache
    else:
        x1 = n_medium*Kvac*rho1                 # size parameter
        mie = mie_cache(int(2*x1)+1, x1, n1)    # initial lmax arbitrary
    
    row1, col1, data1 = mArray_sparse_mp(nproc, rho1, r1, +1., Kvac*n_medium, Nout, Nin, M, pts_out, wts_out, pts_in, wts_in, mie)
    
    #r2 = 1/(1+rho2/rho1)
    r2 = 0.5
    n2 = n_sphere2/n_medium
    # aspect ratio
    rho2 = R2/L
    
    # precompute mie coefficients
    if Kvac == 0.:
        mie = mie_cache(3, 1., n2)              # dummy cache
    else:
        x2 = n_medium*Kvac*rho2                 # size parameter
        mie = mie_cache(int(2*x2)+1, x2, n2)    # initial lmax arbitrary
    
    row2, col2, data2 = mArray_sparse_mp(nproc, rho2, r2, -1., Kvac*n_medium, Nin, Nout, M, pts_in, wts_in, pts_out, wts_out, mie)
    
    # m=0
    sprsmat1 = coo_matrix((data1[:, 0], (row1, col1)), shape=(2*Nout,2*Nin)).tocsc()
    sprsmat2 = coo_matrix((data2[:, 0], (row2, col2)), shape=(2*Nin,2*Nout)).tocsc()
    mat = sprsmat1.dot(sprsmat2)
    logdet = logdet_sparse(mat) 
    
    # m>0    
    for m in range(1, M//2):
        sprsmat1 = coo_matrix((data1[:, m], (row1, col1)), shape=(2*Nout,2*Nin)).tocsc()
        sprsmat2 = coo_matrix((data2[:, m], (row2, col2)), shape=(2*Nin,2*Nout)).tocsc()
        mat = sprsmat1.dot(sprsmat2)
        logdet += 2*logdet_sparse(mat) 
    
    # last m
    sprsmat1 = coo_matrix((data1[:, M//2], (row1, col1)), shape=(2*Nout,2*Nin)).tocsc()
    sprsmat2 = coo_matrix((data2[:, M//2], (row2, col2)), shape=(2*Nin,2*Nout)).tocsc()
    mat = (sprsmat1).dot(sprsmat2)
    if M%2==0:
        logdet += logdet_sparse(mat)
    else:
        logdet += 2*logdet_sparse(mat)
    return logdet

    
def energy_zero(R1, R2, L, materials, Nin, Nout, M, X, nproc):
    """
    Computes the energy. (add formula?)

    Parameters
    ----------
    eps: float
        positive, ratio L/R
    N: int
        positive, quadrature order of k-integration
    M: int
        positive, quadrature order of phi-integration
    X: int
        positive, quadrature order of K-integration
    nproc: int
        number of processes spawned by multiprocessing module

    Returns
    -------
    energy: float
        Casimir energy

    
    Dependencies
    ------------
    quadrature, get_mie, LogDet_sparse_mp

    """
    p_in, w_in = quadrature(Nin)
    p_out, w_out = quadrature(Nout)
    K_pts, K_wts = quadrature(X)
    
    energy = 0.
    for i in range(X):
        result = LogDet(R1, R2, L, materials, K_pts[i], Nin, Nout, M, p_in, w_in, p_out, w_out, nproc)
        print("K=", K_pts[i], ", val=", result)
        energy += K_wts[i]*result
    return energy/(2*np.pi)


def energy_finite(R1, R2, L, T, materials, Nin, Nout, M, nproc):
    """
    Computes the energy. (add formula?)

    Parameters
    ----------
    eps: float
        positive, ratio L/R
    N: int
        positive, quadrature order of k-integration
    M: int
        positive, quadrature order of phi-integration
    X: int
        positive, quadrature order of K-integration
    nproc: int
        number of processes spawned by multiprocessing module

    Returns
    -------
    energy: float
        Casimir energy

    
    Dependencies
    ------------
    quadrature, get_mie, LogDet_sparse_mp

    """
    p_in, w_in = quadrature(Nin)
    p_out, w_out = quadrature(Nout)
    
    K_matsubara = 2*np.pi*Boltzmann*T/(hbar*c)*L
    n = 0
    energy0 = LogDet(R1, R2, L, materials, 0., Nin, Nout, M, p_in, w_in, p_out, w_out, nproc)
    energy = 0.
    n += 1
    while(True):
        term = 2*LogDet(R1, R2, L, materials, K_matsubara*n, Nin, Nout, M, p_in, w_in, p_out, w_out, nproc)
        energy += term
        print(K_matsubara*n, term)
        if abs(term/energy) < 1.e-12:
            break
        n += 1
        
    return 0.5*T*(energy+energy0), 0.5*T*energy
    #return energy

def energy_faster(R1, R2, L, T, materials, Nin, Nout, M, nproc):
    """
    Computes the energy. (add formula?)

    Parameters
    ----------
    eps: float
        positive, ratio L/R
    N: int
        positive, quadrature order of k-integration
    M: int
        positive, quadrature order of phi-integration
    X: int
        positive, quadrature order of K-integration
    nproc: int
        number of processes spawned by multiprocessing module

    Returns
    -------
    energy: float
        Casimir energy

    
    Dependencies
    ------------
    quadrature, get_mie, LogDet_sparse_mp

    """
    p_in, w_in = quadrature(Nin)
    p_out, w_out = quadrature(Nout)
    
    K_matsubara = Boltzmann*T*L/(hbar*c)
    energy0 = LogDet(R1, R2, L, materials, 0., Nin, Nout, M, p_in, w_in, p_out, w_out, nproc)

    energy = 0.
    Teff = 4*np.pi*Boltzmann/hbar/c*T*L
    order = int(max(np.ceil(10/np.sqrt(Teff)), 5))
    xi, eta = psd(order)
    for n in range(order):
        term = 2*eta[n]*LogDet(R1, R2, L, materials, K_matsubara*xi[n], Nin, Nout, M, p_in, w_in, p_out, w_out, nproc)
        print(K_matsubara*xi[n], term)
        energy += term
    
    return 0.5*T*(energy+energy0), 0.5*T*energy


if __name__ == "__main__":
    R1 = 8e-06
    R2 = 16.5e-06
    L = 0.8e-06
    T = 293.015
    materials = ("PS1", "Water", "Silica1")
    
    rho1 = R1/L
    rho2 = R2/L
    rhoeff = rho1*rho2/(rho1+rho2)
    eta = 10

    nproc = 4
    Nin = int(eta*np.sqrt(rho1+rho2))
    Nout = int(eta*np.sqrt(rhoeff))
    M = Nin
    X = 20
    phiSequence = make_phiSequence(kernel_polar)

    #print(energy_zero(R1, R2, L, materials, N, M, X, nproc))
    print(energy_zero(R1, R2, L, materials, Nin, Nout, M, X, nproc))
    print(energy_finite(R1, R2, L, T, materials, Nin, Nout, M, nproc))
    print(energy_faster(R1, R2, L, T, materials, Nin, Nout, M, nproc))
