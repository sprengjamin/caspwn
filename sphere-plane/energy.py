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
sys.path.append("../ufuncs-sphere/")
from mie import mie_e_array
from index import itt

@jit("UniTuple(float64[:],2)(int64)", nopython=True)
def quadrature(N):
    """points and weights divided by boostparameter
        boostparameter has to be multiplied later
        :type N: integer
    
    """
    i = np.arange(1, N+1, 1)
    t = np.pi/(N+1)*i
    points = 1./(np.tan(t/2))**2
    weights = np.zeros(N)
    for j in i:
        weights += np.sin(j*t)*(1-np.cos(j*np.pi))/j
    weights *= 2*np.sin(t)*(2/(N+1))/(1-np.cos(t))**2
    return points, weights

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
    def phiSequence(eps, xiR, M, k1, k2, w1, w2, ale, ble):
        r"""
        Returns the a phi sqeuence for the kernel function for each polarization block.

        Parameters
        ----------
        eps: float
            positive, ratio L/R
        xiR: float
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
        phiarr[:, 0] = -w1*w2*kernel(eps, xiR, k1, k2, 0., aob, logb)
        
        if M%2==0:
            # phi = np.pi
            phiarr[:, M//2] = -w1*w2*kernel(eps, xiR, k1, k2, np.pi, aob, logb)
            imax = M//2-1
            phi = 2*np.pi*np.arange(M//2)/M
        else:
            imax = M//2
            phi = 2*np.pi*np.arange(M//2+1)/M
        
        for i in range(1, imax+1):
            phiarr[:, i] = -w1*w2*kernel(eps, xiR, k1, k2, phi[i], aob, logb)
            phiarr[0, M-i] = phiarr[0, i]
            phiarr[1, M-i] = phiarr[1, i]
            phiarr[2, M-i] = -phiarr[2, i]
            phiarr[3, M-i] = -phiarr[3, i]
        return phiarr
    return phiSequence

def mSequence(eps, xiR, M, k1, k2, w1, w2, ale, ble):
    """
    Computes mSqeuence by means of a FFT of the computed phiSequence.

    Parameters
    ----------
    eps: float
        positive, ratio L/R
    xiR: float
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
    phiarr = phiSequence(eps, xiR, M, k1, k2, w1, w2, aob, logb)
    marr = np.fft.rfft(phiarr)
    return np.array([marr[0,:].real, marr[1,:].real, -marr[2,:].imag, marr[3,:].imag])


def compute_mElement_diag(i, eps, xiR, N, M, k, w, ale, ble):
    """
    Computes the m-sequence of diagonal elements.

    Parameters
    ----------
    i: int
        non-negative, row or column index of the diagonal element
    eps: float
        positive, ratio L/R
    xiR: float
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
    data = (mSequence(eps, xiR, M, k[i], k[i], w[i], w[i], ale, ble))[:-1,:]
    return row, col, data
   
    
def compute_mElement_offdiag(i, j, eps, xiR, N, M, k, w, aob, logb):
    """
    Computes the m-sequence of off-diagonal elements.

    Parameters
    ----------
    i: int
        non-negative, row index of the off-diagonal element
    j: int
        non-negative, column index of the off-diagonal element
    eps: float
        positive, ratio L/R
    xiR: float
        positive, size parameter
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
    data = mSequence(eps, xiR, M, k[i], k[j], w[i], w[j], ale, ble)
    return row, col, data


def mArray_sparse_part(dindices, oindices, eps, xiR, N, M, k, w, ale, ble):
    """
    Computes the m-array.

    Parameters
    ----------
    dindices: np.ndarray
        array of diagonal indices
    oindices: np.ndarray
        array of off-diagonal indices
    eps: float
        positive, ratio L/R
    xiR: float
        positive, size parameter
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
    # 16 is just arbitrary here
    row = np.empty(16*N)
    col = np.empty(16*N)
    data = np.empty((16*N, M//2+1))
    
    ind = 0
    for i in dindices:
        if isFinite(eps, xiR, k[i], k[i]):
            row[ind:ind+3], col[ind:ind+3], data[ind:ind+3, :] = compute_mElement_diag(i, eps, xiR, N, M, k, w, ale, ble)
            ind += 3

    for oindex in oindices:
        i, j = itt(oindex)
        if isFinite(eps, xiR, k[i], k[j]):
            if ind+4 >= len(row):
                row = np.hstack((row, np.empty(len(row))))
                col = np.hstack((col, np.empty(len(row))))
                data = np.vstack((data, np.empty((len(row), M//2+1))))
            row[ind:ind+4], col[ind:ind+4], data[ind:ind+4, :] = compute_mElement_offdiag(i, j, eps, xiR, N, M, k, w, ale, ble)
            ind += 4
                
    row = row[:ind] 
    col = col[:ind] 
    data = data[:ind, :] 
    return row, col, data


def mArray_sparse_mp(nproc, eps, xiR, N, M, pts, wts, aob, logb):
    """
    Computes the m-array in parallel using the multiprocessing module.

    Parameters
    ----------
    nproc: int
        number of processes
    eps: float
        positive, ratio L/R
    xiR: float
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
    
    def worker(dindices, oindices, eps, xiR, N, M, k, w, ale, ble, out):
        out.put(mArray_sparse_part(dindices, oindices, eps, xiR, N, M, k, w, ale, ble))

    b = get_b(eps, xiR)
    k = b*pts
    w = np.sqrt(b*wts*2*np.pi/M)
    
    dindices = np.array_split(np.random.permutation(N), nproc)
    oindices = np.array_split(np.random.permutation(N*(N-1)//2), nproc)
    
    out = mp.Queue()
    procs = []
    for i in range(nproc):
        p = mp.Process(
                target = worker,
                args = (dindices[i], oindices[i], eps, xiR, N, M, k, w, aob, logb, out))
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


def isFinite(eps, xiR, k1, k2):
    """
    Estimates by means of the asymptotics of the scattering amplitudes if the given
    matrix element can be numerically set to zero.

    Parameters
    ----------
    eps: float
        positive, ratio L/R
    xiR: float
        positive, rescaled frequency
    k1, k2: float
        positive, rescaled wave numbers

    Returns
    -------
    boolean
        True if the matrix element must not be neglected

    """
    if xiR == 0.:
        exponent = 2*np.sqrt(k1*k2) - (k1+k2)*(1+eps)
    else:
        kappa1 = np.sqrt(k1*k1+xiR*xiR)
        kappa2 = np.sqrt(k2*k2+xiR*xiR)
        exponent = np.sqrt(2*(xiR*xiR + kappa1*kappa2 + k1*k2)) - (kappa1+kappa2)*(1+eps)
    if exponent < -37:
        return False
    else:
        return True
            

def LogDet(nproc, eps, xiR, N, M, pts, wts, ale, ble):
    """
    Computes the sum of the logdets the m-matrices.

    Parameters
    ----------
    nproc: int
        number of processes
    eps: float
        positive, ratio L/R
    xiR: float
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
    logdet: float
        the sum of logdets of the m-matrices

    Dependencies
    ------------
    mArray_sparse_mp

    """
    row, col, data = mArray_sparse_mp(nproc, eps, xiR, N, M, pts, wts, aob, logb)
    
    # m=0
    sprsmat = coo_matrix((data[:, 0], (row, col)), shape=(2*N,2*N))
    factor = cholesky(sprsmat.tocsc(), beta=1.)
    logdet = factor.logdet()

    # m>0    
    for m in range(1, M//2+1):
        sprsmat = coo_matrix((data[:, m], (row, col)), shape=(2*N,2*N))
        factor = cholesky(sprsmat.tocsc(), beta=1.)
        logdet += 2*factor.logdet()
    ##### WATCH OUT ######
    ##### BUG 
    ##### if M is even, last value may not be multiplied by 2!!
    return logdet

def energy_zero(eps, N, M, X, nproc, computeMie):
    """
    Computes the Casimir at zero temperature.

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
    computeMie: boolean
        True if mie's shall be computed

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
    xi_pts = xi_pts
    xi_wts = xi_wts
    
    energy = 0.
    for i in range(X):
        lmax = ?
        ale, ble = mie_e_array(lmax, xi_pts[i])
        result = LogDet(nproc, eps, xi_pts[i], N, M, k_pts, k_wts, ale, ble)
        print("xi=", xi_pts[i], ", val=", result)
        energy += xi_wts[i]*result
    return energy


if __name__ == "__main__":
    eps = 0.1
    xi = 10.
    N = int(15/np.sqrt(eps))
    print("N", N)
    M = N
    X = 40
    nproc = 4
    #aob, logb = get_mie(xi, 10000)
    #print("got mies")
    from kernel import phiKernel_asymplead, phiKernel_asympcorr, phiKernel_Gauss
    phiSequence = make_phiSequence(phiKernel_asympcorr)
    en = energy_FC(eps, N, M, X, nproc, False) 
    print("energy")
    print(en)
    #pts, wts = quadrature(N)
     
    #row, col, data = mArray_sparse_mp(nproc, eps, xi, N, M, pts, wts, aob, logb)
    #print(LogDet(eps, xi, N, M, pts, wts, aob, logb)) 
    #print(LogDet_sparse_mp(nproc, eps, xi, N, M, pts, wts, aob, logb)) 
    #row, col, data = mArray_sparse(eps, xi, N, M, pts, wts, aob, logb) 
    #coo = coo_matrix((data[:,0], (row, col)), shape=(2*N, 2*N))
    #mat = coo.toarray()
    """ 
    n = 5
    import matplotlib.pyplot as plt
    f, axarr = plt.subplots(2,n)
    imlist = []
    #arr[0,:,:] -= np.eye(2*N)
    for i in range(n):
        coo = coo_matrix((data[:,i], (row, col)), shape=(2*N, 2*N))
        mat = coo.toarray()
        imlist.append(axarr[0,i].imshow(mat))
        imlist.append(axarr[1,i].imshow(np.log10(np.abs(mat))))
        #imlist.append(axarr[1,i].imshow(fftreal[i,:,:]))
    #for i in range(n):
    #    f.colorbar(imlist[i], ax=axarr[i])
    plt.show()
    """ 
