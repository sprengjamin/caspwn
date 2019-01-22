import numpy as np
from numba import jit
from numba import float64, int64
from numba.types import UniTuple
import multiprocessing as mp
from scipy.sparse.linalg import splu
from scipy.sparse import eye
from scipy.sparse import coo_matrix

from index import itt
from kernel import phiKernel
import sys
sys.path.append("../sphere/")
from mie import mie_cache
sys.path.append("../ufuncs/")
from integration import quadrature

def logdet_sparse(mat):
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
    def phiSequence(rho, r, sign, xi, M, k1, k2, w1, w2, mie):
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
        xi: float
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
        phiarr[:, 0] = w1*w2*kernel(rho, r, sign, xi, k1, k2, 0., mie)
        
        if M%2==0:
            # phi = np.pi
            phiarr[:, M//2] = w1*w2*kernel(rho, r, sign, xi, k1, k2, np.pi, mie)
            imax = M//2-1
            phi = 2*np.pi*np.arange(M//2)/M
        else:
            imax = M//2
            phi = 2*np.pi*np.arange(M//2+1)/M
        
        for i in range(1, imax+1):
            phiarr[:, i] = w1*w2*kernel(rho, r, sign, xi, k1, k2, phi[i], mie)
            phiarr[0, M-i] = phiarr[0, i]
            phiarr[1, M-i] = phiarr[1, i]
            phiarr[2, M-i] = -phiarr[2, i]
            phiarr[3, M-i] = -phiarr[3, i]
        return phiarr
    return phiSequence


def mSequence(rho, r, sign, xi, M, k1, k2, w1, w2, mie):
    """
    Computes mSqeuence by means of a FFT of the computed phiSequence.

    Parameters
    ----------
    rho: float
        positive, aspect ratio R/L
    r: float
        positive, relative effective radius R_eff/R
    xi: float
        positive, rescaled frequency
    M: int
        positive, length of sequence
    k1, k2: float
        positive, rescaled wave numbers
    w1, w2: float
        positive, quadrature weights corresponding to k1 and k2, respectively.
    mie: class instance
        cache for the exponentially scaled mie coefficients

    Dependencies
    ------------
    phiSequence

    """
    phiarr = phiSequence(rho, r, sign, xi, M, k1, k2, w1, w2, mie)
    marr = np.fft.rfft(phiarr)
    return np.array([marr[0,:].real, marr[1,:].real, -marr[2,:].imag, marr[3,:].imag])


def compute_mElement_diag(i, rho, r, sign, xi, N, M, k, w, mie):
    """
    Computes the m-sequence of diagonal elements.

    Parameters
    ----------
    i: int
        non-negative, row or column index of the diagonal element
    rho: float
        positive, aspect ratio R/L
    r: float
        positive, relative effective radius R_eff/R
    sign: +1/-1
        sign, differs for the two spheres
    xi: float
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
        array containing the m-sequence

    Dependencies
    ------------
    mSequence

    """
    row = [i, N+i, N+i]
    col = [i, N+i, i]
    data = (mSequence(rho, r, sign, xi, M, k[i], k[i], w[i], w[i], mie))[:-1,:]
    return row, col, data
   
    
def compute_mElement_offdiag(i, j, rho, r, sign, xi, N, M, k, w, mie):
    """
    Computes the m-sequence of off-diagonal elements.

    Parameters
    ----------
    i: int
        non-negative, row index of the off-diagonal element
    j: int
        non-negative, column index of the off-diagonal element
    rho: float
        positive, aspect ratio R/L
    r: float
        positive, relative effective radius R_eff/R
    sign: +1/-1
        sign, differs for the two spheres
    xi: float
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
        array containing the m-sequence

    Dependencies
    ------------
    mSequence

    """
    #row = [i, N+i, N+j, N+i] 
    #col = [j, N+j, i, j] 
    row = [i, N+i, N+j, j] 
    col = [j, N+j, i, N+i] 
    data = mSequence(rho, r, sign, xi, M, k[i], k[j], w[i], w[j], mie)
    return row, col, data


def mArray_sparse_part(dindices, oindices, rho, r, sign, xi, N, M, k, w, mie):
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
    xi: float
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
    row = np.empty(16*N, dtype=np.int32)
    col = np.empty(16*N, dtype=np.int32)
    data = np.empty((16*N, M//2+1))
    
    ind = 0
    for i in dindices:
        if isFinite(rho, r, xi, k[i], k[i]):
            row[ind:ind+3], col[ind:ind+3], data[ind:ind+3, :] = compute_mElement_diag(i, rho, r, sign, xi, N, M, k, w, mie)
            row[ind+3] = col[ind+2]
            col[ind+3] = row[ind+2]
            data[ind+3, :] = -data[ind+2, :]
            ind += 4

    for oindex in oindices:
        i, j = itt(oindex)
        if isFinite(rho, r, xi, k[i], k[j]):
            if ind+8 >= len(row):
                row = np.hstack((row, np.empty(len(row), dtype=np.int32)))
                col = np.hstack((col, np.empty(len(row), dtype=np.int32)))
                data = np.vstack((data, np.empty((len(row), M//2+1))))
            row[ind:ind+4], col[ind:ind+4], data[ind:ind+4, :] = compute_mElement_offdiag(i, j, rho, r, sign, xi, N, M, k, w, mie)
            row[ind+4:ind+8] = col[ind:ind+4]
            col[ind+4:ind+8] = row[ind:ind+4]
            data[ind+4:ind+6, :] = data[ind:ind+2, :]
            data[ind+6:ind+8, :] = -data[ind+2:ind+4, :]
            ind += 8
                
    row = row[:ind] 
    col = col[:ind] 
    data = data[:ind, :] 
    return row, col, data


def mArray_sparse_mp(nproc, rho, r, sign, xi, N, M, pts, wts, mie):
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
    xi: float
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
    
    def worker(dindices, oindices, rho, r, sign, xi, N, M, k, w, mie, out):
        out.put(mArray_sparse_part(dindices, oindices, rho, r, sign, xi, N, M, k, w, mie))

    b = 0.5 ### for now
    k = b*pts
    w = np.sqrt(b*wts*2*np.pi/M)
    
    dindices = np.array_split(np.random.permutation(N), nproc)
    oindices = np.array_split(np.random.permutation(N*(N-1)//2), nproc)
    
    out = mp.Queue()
    procs = []
    for i in range(nproc):
        p = mp.Process(
                target = worker,
                args = (dindices[i], oindices[i], rho, r, sign, xi, N, M, k, w, mie, out))
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


def isFinite(rho, r, xi, k1, k2):
    """
    Estimates by means of the asymptotics of the scattering amplitudes if the given
    matrix element can be numerically set to zero.

    Parameters
    ----------
    rho: float
        positive, aspect ratio R/L
    r: float
        positive, relative effective radius R_eff/R
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
        exponent = 2*rho*np.sqrt(k1*k2) - (k1+k2)*(rho+r)
    else:
        kappa1 = np.sqrt(k1*k1+xi*xi)
        kappa2 = np.sqrt(k2*k2+xi*xi)
        exponent = rho*np.sqrt(2*(xi*xi + kappa1*kappa2 + k1*k2)) - (kappa1+kappa2)*(rho+r)
    if exponent < -37:
        return False
    else:
        return True
            

def LogDet_sparse_mp(nproc, rho1, rho2, xi, N, M, pts, wts):
    """
    Computes the sum of the logdets the m-matrices.

    Parameters
    ----------
    nproc: int
        number of processes
    rho1, rho2: float
        positive, aspect ratios R1/L, R2/L
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
    #r1 = 1/(1+rho1/rho2)
    r1 = 0.5
    mie = mie_cache(int(xi*rho1)+1, xi*rho1)
    row1, col1, data1 = mArray_sparse_mp(nproc, rho1, r1, +1., xi, N, M, pts, wts, mie)
    
    #r2 = 1/(1+rho2/rho1)
    r2 = 0.5
    mie = mie_cache(int(xi*rho2)+1, xi*rho2)
    row2, col2, data2 = mArray_sparse_mp(nproc, rho2, r2, -1., xi, N, M, pts, wts, mie)
    
    # m=0
    sprsmat1 = coo_matrix((data1[:, 0], (row1, col1)), shape=(2*N,2*N)).tocsc()
    sprsmat2 = coo_matrix((data2[:, 0], (row2, col2)), shape=(2*N,2*N)).tocsc()
    mat = sprsmat1.dot(sprsmat2)
    logdet = logdet_sparse(mat) 
    
    # m>0    
    for m in range(1, M//2):
        sprsmat1 = coo_matrix((data1[:, m], (row1, col1)), shape=(2*N,2*N)).tocsc()
        sprsmat2 = coo_matrix((data2[:, m], (row2, col2)), shape=(2*N,2*N)).tocsc()
        mat = sprsmat1.dot(sprsmat2)
        logdet += 2*logdet_sparse(mat) 
    
    # last m
    sprsmat1 = coo_matrix((data1[:, M//2], (row1, col1)), shape=(2*N,2*N)).tocsc()
    sprsmat2 = coo_matrix((data2[:, M//2], (row2, col2)), shape=(2*N,2*N)).tocsc()
    mat = (sprsmat1).dot(sprsmat2)
    if M%2==0:
        logdet += logdet_sparse(mat)
    else:
        logdet += 2*logdet_sparse(mat)
    return logdet

    
def energy_zero(rho1, rho2, N, M, X, nproc):
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
        positive, quadrature order of xi-integration
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
    k_pts, k_wts = quadrature(N)
    xi_pts, xi_wts = quadrature(X)
    
    energy = 0.
    for i in range(X):
        result = LogDet_sparse_mp(nproc, rho1, rho2, xi_pts[i], N, M, k_pts, k_wts)
        print("xi=", xi_pts[i], ", val=", result)
        energy += xi_wts[i]*result
    return energy/(2*np.pi)


if __name__ == "__main__":
    R1 = 10
    R2 = 10
    L = 1 
    
    rho1 = R1/L
    rho2 = R2/L
    eta = 10

    nproc = 4
    N = int(eta*np.sqrt(max(rho1, rho2)))
    M = N
    X = 20
    
    phiSequence = make_phiSequence(phiKernel)
    #k_pts, k_wts = quadrature(N)
    #print(LogDet_sparse_mp(nproc, rho1, rho2, 0.00001, N, M, k_pts, k_wts))
    print(energy_zero(rho1, rho2, N, M, X, nproc))
    #from analytical import PFA_spheresphere, PFAcorr_spheresphere
    #print(PFA_spheresphere(rho1, rho2))
    #print(PFAcorr_spheresphere(rho1, rho2))
