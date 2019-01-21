r""" Kernel functions for the plane-sphere geometry.

.. todo::
    * test expression for phase difference

    * test expresion for :math:`z`

"""
import numpy as np
from numba import jit
import sys
sys.path.append("../sphere/")
sys.path.append("../ufuncs/")
from ABCD import ABCD
from scattering_amplitude import S1S2

@jit("float64(float64, float64, float64, float64, float64)", nopython=True)
def phase(rho, xi, k1, k2, phi):
    r"""The phase difference.

    Parameters
    ----------
    rho: float
        positive, aspect ratio :math:`R/L`
    xi: float
        positive, rescaled frequency
    k1, k2: float
        positive, rescaled wave numbers
    phi: float
        between 0 and 2pi
        
    Returns
    -------
    float    
        phase difference

    """
    kappa1 = np.sqrt(k1*k1+xi*xi)
    kappa2 = np.sqrt(k2*k2+xi*xi)
    return -((k1 - k2)**2 + 4*k1*k2*np.sin(phi/2)**2)/(np.sqrt(2*(kappa1*kappa2 + k1*k2*np.cos(phi) + xi**2)) + kappa1 + kappa2)*rho - kappa1 - kappa2

    
@jit("float64[:](float64, float64, float64, float64, float64, float64[:], float64[:])", nopython=True)
def phiKernel(rho, xi, k1, k2, phi, ale, ble):
    r"""
    Returns the phikernels.

    Parameters
    ----------
    rho: float
        positive, aspect ratio :math:`R/L`
    xi: float
        positive, rescaled frequency
    k1, k2: float
        positive, rescaled wave numbers
    phi: float
        between 0 and 2pi
    ale, ble: np.ndarray
        array containing the exponentially scaled mie coefficients :math:`\tilde{a}_\ell` and :math:`\tilde{b}_\ell`.
        
    Returns
    -------
    np.ndarray
        array of length 4 of phikernels for the polarization contributions
        TMTM, TETE, TMTE, TETM

    """
    kappa1 = np.sqrt(k1*k1+xi*xi)
    kappa2 = np.sqrt(k2*k2+xi*xi)
    z = (kappa1*kappa2+k1*k2*np.cos(phi))/xi**2
    #print("%16f" % z)
    #print(xi, k1, k2, phi)
    #assert(z>=1.)
    exponent = phase(rho, xi, k1, k2, phi)
    if phi == np.pi:
        z = (k1**2 + k2**2 + xi**2)/(kappa1*kappa2 + k1*k2)
    if exponent < -37:
        return np.array([0., 0., 0., 0.])
    e = np.exp(exponent)
    A, B, C, D = ABCD(xi, k1, k2, phi)
    prefactor = np.sqrt(k1*k2)/(2*np.pi*xi*np.sqrt(kappa1*kappa2))
    S1, S2 = S1S2(xi*rho, z, ale, ble)
    pkTMTM =  prefactor*(B*S1+A*S2)*e
    pkTETE =  -prefactor*(A*S1+B*S2)*e
    pkTMTE =  prefactor*(C*S1+D*S2)*e
    pkTETM =  prefactor*(D*S1+C*S2)*e
    return np.array([pkTMTM, pkTETE, pkTMTE, pkTETM])

if __name__ == "__main__":
    rho = 0.001
    xi = 1.0e-08
    k1 = 1.0e-08
    k2 = 1.0e-08
    phi = 0.
    print(phase(rho, xi, k1, k2, phi))
    #from mie import mie_e_array
    #ale, ble = mie_e_array(1e5, xi*rho)
    #print(phiKernel(rho, xi, k1, k2, phi, ale, ble))
