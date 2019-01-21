r""" Casimir energy for the plane-sphere geometry.

.. todo::
    * implement for finite temperature
    * restructure folders into sphere and ufuncs

"""
import numpy as np
import multiprocessing as mp
from numba import jit
from sksparse.cholmod import cholesky
from scipy.sparse import coo_matrix

import sys
sys.path.append("../sphere/")
sys.path.append("../ufuncs/")
from mie import mie_e_array
from integration import quadrature, auto_integration
from index import itt


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
    @jit("float64[:,:](float64, float64, int64, float64, float64, float64, float64, float64[:], float64[:])", nopython=True)
    def phiSequence(rho, xi, M, k1, k2, w1, w2, ale, ble):
        r"""
        Returns the a phi sqeuence for the kernel function for each polarization block.

        Parameters
        ----------
        rho: float
            positive, aspect ratio :math:`R/L`
        xi: float
            positive, rescaled frequency
        M: int
            positive, length of sequence
        k1, k2: float
            positive, rescaled wave numbers
        w1, w2: float
            positive, quadrature weights corresponding to k1 and k2, respectively.
        ale, ble: np.ndarray
            array containing the exponentially scaled mie coefficients :math:`\tilde{a}_\ell` and :math:`\tilde{b}_\ell`.

        Returns
        -------
        np.ndarray
            array of length 4 of the phi sequence for the polarization contributions
            TMTM, TETE, TMTE, TETM

        """
        phiarr = np.empty((4, M))

        # phi = 0.
        phiarr[:, 0] = -w1*w2*kernel(rho, xi, k1, k2, 0., ale, ble)
        
        if M%2==0:
            # phi = np.pi
            phiarr[:, M//2] = -w1*w2*kernel(rho, xi, k1, k2, np.pi, ale, ble)
            imax = M//2-1
            phi = 2*np.pi*np.arange(M//2)/M
        else:
            imax = M//2
            phi = 2*np.pi*np.arange(M//2+1)/M
        
        for i in range(1, imax+1):
            phiarr[:, i] = -w1*w2*kernel(rho, xi, k1, k2, phi[i], ale, ble)
            phiarr[0, M-i] = phiarr[0, i]
            phiarr[1, M-i] = phiarr[1, i]
            phiarr[2, M-i] = -phiarr[2, i]
            phiarr[3, M-i] = -phiarr[3, i]
        return phiarr
    return phiSequence

def mSequence(rho, xi, M, k1, k2, w1, w2, ale, ble):
    r"""
    Computes mSqeuence by means of a FFT of the computed phiSequence.

    Parameters
    ----------
    rho: float
        positive, aspect ratio :math:`R/L`
    xi: float
        positive, rescaled frequency
    M: int
        positive, length of sequence
    k1, k2: float
        positive, rescaled wave numbers
    w1, w2: float
        positive, quadrature weights corresponding to k1 and k2, respectively.
    ale, ble: np.ndarray
        array containing the exponentially scaled mie coefficients :math:`\tilde{a}_\ell` and :math:`\tilde{b}_\ell`.

    Dependencies
    ------------
    phiSequence

    """
    phiarr = phiSequence(rho, xi, M, k1, k2, w1, w2, ale, ble)
    marr = np.fft.rfft(phiarr)
    return np.array([marr[0,:].real, marr[1,:].real, -marr[2,:].imag, marr[3,:].imag])


def compute_mElement_diag(i, rho, xi, N, M, k, w, ale, ble):
    r"""
    Computes the m-sequence of diagonal elements.

    Parameters
    ----------
    i: int
        non-negative, row or column index of the diagonal element
    rho: float
        positive, aspect ratio :math:`R/L`
    xi: float
        positive, rescaled frequency
    N: int
        positive, quadrature order of k-integration
    M: int
        positive, quadrature order of phi-integration
    k, w: np.ndarray
        quadrature points and weights of the k-integration after rescaling
    ale, ble: np.ndarray
        array containing the exponentially scaled mie coefficients :math:`\tilde{a}_\ell` and :math:`\tilde{b}_\ell`.

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
    mSequence

    """
    row = [i, N+i, N+i]
    col = [i, N+i, i]
    data = (mSequence(rho, xi, M, k[i], k[i], w[i], w[i], ale, ble))[:-1,:]
    return row, col, data
   
    
def compute_mElement_offdiag(i, j, rho, xi, N, M, k, w, ale, ble):
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
    xi: float
        positive, rescaled frequency
    N: int
        positive, quadrature order of k-integration
    M: int
        positive, quadrature order of phi-integration
    k, w: np.ndarray
        quadrature points and weights of the k-integration after rescaling
    ale, ble: np.ndarray

        array containing the exponentially scaled mie coefficients :math:`\tilde{a}_\ell` and :math:`\tilde{b}_\ell`.

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
    mSequence

    """
    row = [i, N+i, N+j, N+i] 
    col = [j, N+j, i, j] 
    data = mSequence(rho, xi, M, k[i], k[j], w[i], w[j], ale, ble)
    return row, col, data


def mArray_sparse_part(dindices, oindices, rho, xi, N, M, k, w, ale, ble):
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
    xi: float
        positive, rescaled frequency
    N: int
        positive, quadrature order of k-integration
    M: int
        positive, quadrature order of phi-integration
    k, w: np.ndarray
        quadrature points and weights of the k-integration after rescaling
    ale, ble: np.ndarray
        array containing the exponentially scaled mie coefficients :math:`\tilde{a}_\ell` and :math:`\tilde{b}_\ell`.

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
    ###
    # 16 is just arbitrary here
    row = np.empty(16*N)
    col = np.empty(16*N)
    data = np.empty((16*N, M//2+1))
    
    ind = 0
    for i in dindices:
        if isFinite(rho, xi, k[i], k[i]):
            row[ind:ind+3], col[ind:ind+3], data[ind:ind+3, :] = compute_mElement_diag(i, rho, xi, N, M, k, w, ale, ble)
            ind += 3

    for oindex in oindices:
        i, j = itt(oindex)
        if isFinite(rho, xi, k[i], k[j]):
            if ind+4 >= len(row):
                row = np.hstack((row, np.empty(len(row))))
                col = np.hstack((col, np.empty(len(row))))
                data = np.vstack((data, np.empty((len(row), M//2+1))))
            row[ind:ind+4], col[ind:ind+4], data[ind:ind+4, :] = compute_mElement_offdiag(i, j, rho, xi, N, M, k, w, ale, ble)
            ind += 4
                
    row = row[:ind] 
    col = col[:ind] 
    data = data[:ind, :] 
    return row, col, data


def mArray_sparse_mp(nproc, rho, xi, N, M, pts, wts, ale, ble):
    r"""
    Computes the m-array in parallel using the multiprocessing module.

    Parameters
    ----------
    nproc: int
        number of processes
    rho: float
        positive, aspect ratio :math:`R/L`
    xi: float
        positive, rescaled frequency
    N: int
        positive, quadrature order of k-integration
    M: int
        positive, quadrature order of phi-integration
    pts, wts: np.ndarray
        quadrature points and weights of the k-integration before rescaling
    ale, ble: np.ndarray
        array containing the exponentially scaled mie coefficients :math:`\tilde{a}_\ell` and :math:`\tilde{b}_\ell`.

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
    
    def worker(dindices, oindices, rho, xi, N, M, k, w, ale, ble, out):
        out.put(mArray_sparse_part(dindices, oindices, rho, xi, N, M, k, w, ale, ble))

    b = 0.5 # kinda arbitrary
    k = b*pts
    w = np.sqrt(b*wts*2*np.pi/M)
    
    dindices = np.array_split(np.random.permutation(N), nproc)
    oindices = np.array_split(np.random.permutation(N*(N-1)//2), nproc)
    
    out = mp.Queue()
    procs = []
    for i in range(nproc):
        p = mp.Process(
                target = worker,
                args = (dindices[i], oindices[i], rho, xi, N, M, k, w, ale, ble, out))
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


def isFinite(rho, xi, k1, k2):
    r"""
    Estimates by means of the asymptotics of the scattering amplitudes if the given
    matrix element can be numerically set to zero.

    Parameters
    ----------
    rho: float
        positive, aspect ratio :math:`R/L`
    xi: float
        positive, rescaled frequency
    k1, k2: float
        positive, rescaled wave numbers

    Returns
    -------
    boolean
        True if the matrix element must not be neglected

    """
    if xi == 0.:
        exponent = 2*np.sqrt(k1*k2) - (k1+k2)*(1+eps)
    else:
        kappa1 = np.sqrt(k1*k1+xi*xi)
        kappa2 = np.sqrt(k2*k2+xi*xi)
        # copied from kernel with phi=0 > make a function! (which can be tested)
        exponent = -((k1 - k2)**2)/(np.sqrt(2*(kappa1*kappa2 + k1*k2 + xi**2)) + kappa1 + kappa2)*rho - kappa1 - kappa2
    if exponent < -37:
        return False
    else:
        return True
            

def LogDet(nproc, rho, xi, N, M, pts, wts):
    r"""
    Computes the sum of the logdets the m-matrices.

    Parameters
    ----------
    nproc: int
        number of processes
    rho: float
        positive, aspect ratio :math:`R/L`
    xi: float
        positive, rescaled frequency
    N: int
        positive, quadrature order of k-integration
    M: int
        positive, quadrature order of phi-integration
    pts, wts: np.ndarray
        quadrature points and weights of the k-integration before rescaling

    Returns
    -------
    logdet: float
        the sum of logdets of the m-matrices

    Dependencies
    ------------
    mArray_sparse_mp

    """
    # precompute mie coefficients
    lmax = int(1e4) # for testing purpose
    ale, ble = mie_e_array(lmax, xi*rho)
    
    row, col, data = mArray_sparse_mp(nproc, rho, xi, N, M, pts, wts, ale, ble)
    
    # m=0
    sprsmat = coo_matrix((data[:, 0], (row, col)), shape=(2*N,2*N))
    factor = cholesky(sprsmat.tocsc(), beta=1.)
    logdet = factor.logdet()

    # m>0    
    for m in range(1, M//2):
        sprsmat = coo_matrix((data[:, m], (row, col)), shape=(2*N,2*N))
        factor = cholesky(sprsmat.tocsc(), beta=1.)
        logdet += 2*factor.logdet()

    # last m
    sprsmat = coo_matrix((data[:, M//2], (row, col)), shape=(2*N,2*N))
    factor = cholesky(sprsmat.tocsc(), beta=1.)
    if M%2==0:
        logdet += factor.logdet()
    else:
        logdet += 2*factor.logdet()
    return logdet

def energy_zero(rho, N, M, nproc):
    r"""
    Computes the Casimir at zero temperature.

    Parameters
    ----------
    rho: float
        positive, aspect ratio :math:`R/L`
    N: int
        positive, quadrature order of k-integration
    M: int
        positive, quadrature order of phi-integration
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
    k_pts, k_wts = quadrature(N)
    logdet = lambda xi : LogDet(nproc, rho, xi, N, M, k_pts, k_wts)
    energy = auto_integration(logdet)
    return energy


if __name__ == "__main__":
    rho = 40.
    N = int(5*np.sqrt(rho))*2+1
    print("N", N)
    M = N
    nproc = 4
    from kernel import phiKernel
    phiSequence = make_phiSequence(phiKernel)
    
    #ale, ble = mie_e_array(1e4, 1.*rho)
    #print(phiSequence(rho, 1., M, 2.3, 2.3, 1., 1., ale, ble))
    
    en = energy_zero(rho, N, M, nproc) 
    print("energy")
    print(en)
    print("PFA")
    print(-np.pi**3*rho/720)
