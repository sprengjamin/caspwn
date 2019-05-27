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
from energy import mArray_sparse_mp
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../sphere/"))
from mie import mie_cache
import scattering_amplitude
from kernel import kernel_polar as kernel
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../ufuncs/"))
from integration import quadrature
from psd import psd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../material/"))
import material


def dL_logdet_sparse(M1, M2, M1p, M2p):
    r"""This module computes

    .. math::
        \frac{\partial}{\partial L}\log\det(1-\mathcal{M})

    where :math:`\mathcal{M}` is given by the sparse matrix mat.

    Parameters
    ----------
    mat: sparse coo_matrix
        round-trip matrix

    Returns
    -------
    float
    
    """
    Mp = M1p.dot(M2) + M1.dot(M2p)
    M = M1.dot(M2)
    dim = M.shape[0]
    lu = splu(eye(dim, format="csc")-M)
    A = lu.solve(Mp.todense())
    return -np.trace(A)
            

def dL_LogDet(R1, R2, L, materials, Kvac, Nin, Nout, M, pts_in, wts_in, pts_out, wts_out, nproc): 
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
    
    K1 = Kvac*n_medium
    k_in = 0.5*pts_in   # this 0.5 is b
    k_out = 0.5*pts_out # this 0.5 is b
    kappa_in = np.sqrt(k_in*k_in + K1*K1)
    kappa_out = np.sqrt(k_out*k_out + K1*K1)
    data1prime = data1.copy()
    for i in range(len(data1prime)):
        data1prime[i,:] *= -0.5*(kappa_in[col1[i]%Nin]+kappa_out[row1[i]%Nout])
    data2prime = data2.copy()
    for i in range(len(data2prime)):
        data2prime[i,:] *= -0.5*(kappa_in[row2[i]%Nin]+kappa_out[col2[i]%Nout])

    # m=0
    M1 = coo_matrix((data1[:, 0], (row1, col1)), shape=(2*Nout,2*Nin)).tocsc()
    M1p = coo_matrix((data1prime[:, 0], (row1, col1)), shape=(2*Nout,2*Nin)).tocsc()
    M2 = coo_matrix((data2[:, 0], (row2, col2)), shape=(2*Nin,2*Nout)).tocsc()
    M2p = coo_matrix((data2prime[:, 0], (row2, col2)), shape=(2*Nin,2*Nout)).tocsc()
    dL_logdet = dL_logdet_sparse(M1, M2, M1p, M2p)

    # m>0    
    for m in range(1, M//2):
        M1 = coo_matrix((data1[:, m], (row1, col1)), shape=(2*Nout,2*Nin)).tocsc()
        M1p = coo_matrix((data1prime[:, m], (row1, col1)), shape=(2*Nout,2*Nin)).tocsc()
        M2 = coo_matrix((data2[:, m], (row2, col2)), shape=(2*Nin,2*Nout)).tocsc()
        M2p = coo_matrix((data2prime[:, m], (row2, col2)), shape=(2*Nin,2*Nout)).tocsc()
        dL_logdet += 2*dL_logdet_sparse(M1, M2, M1p, M2p)
    
    # last m
    M1 = coo_matrix((data1[:, M//2], (row1, col1)), shape=(2*Nout,2*Nin)).tocsc()
    M1p = coo_matrix((data1prime[:, M//2], (row1, col1)), shape=(2*Nout,2*Nin)).tocsc()
    M2 = coo_matrix((data2[:, M//2], (row2, col2)), shape=(2*Nin,2*Nout)).tocsc()
    M2p = coo_matrix((data2prime[:, M//2], (row2, col2)), shape=(2*Nin,2*Nout)).tocsc()
    if M%2==0:
        dL_logdet += dL_logdet_sparse(M1, M2, M1p, M2p)
    else:
        dL_logdet += 2*dL_logdet_sparse(M1, M2, M1p, M2p)
    return dL_logdet


def force_finite(R1, R2, L, T, materials, Nin, Nout, M, nproc):
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
    force0 = dL_LogDet(R1, R2, L, materials, 0., Nin, Nout, M, p_in, w_in, p_out, w_out, nproc)
    force = 0.
    n += 1
    while(True):
        term = 2*dL_LogDet(R1, R2, L, materials, K_matsubara*n, Nin, Nout, M, p_in, w_in, p_out, w_out, nproc)
        force += term
        print(K_matsubara*n, term)
        if abs(term/force) < 1.e-12:
            break
        n += 1
        
    return 0.5*T*(force+force0)/L, 0.5*T*force/L
    #return energy

def force_faster(R1, R2, L, T, materials, Nin, Nout, M, nproc):
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
    force0 = dL_LogDet(R1, R2, L, materials, 0., Nin, Nout, M, p_in, w_in, p_out, w_out, nproc)

    force = 0.
    Teff = 4*np.pi*Boltzmann/hbar/c*T*L
    order = int(max(np.ceil(10/np.sqrt(Teff)), 5))
    xi, eta = psd(order)
    for n in range(order):
        term = 2*eta[n]*dL_LogDet(R1, R2, L, materials, K_matsubara*xi[n], Nin, Nout, M, p_in, w_in, p_out, w_out, nproc)
        print(K_matsubara*xi[n], term)
        force += term
    
    return -0.5*T*Boltzmann*(force+force0)/L, -0.5*T*Boltzmann*force/L


if __name__ == "__main__":
    R1 = 8e-06
    R2 = 16.5e-06
    L = 0.799e-06
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

    #print(energy_zero(R1, R2, L, materials, N, M, X, nproc))
    #print(energy_zero(R1, R2, L, materials, Nin, Nout, M, X, nproc))
    print(force_finite(R1, R2, L, T, materials, Nin, Nout, M, nproc))
    print(force_faster(R1, R2, L, T, materials, Nin, Nout, M, nproc))
