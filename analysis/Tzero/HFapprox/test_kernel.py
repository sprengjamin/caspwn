r""" Kernel functions for the plane-sphere geometry.

.. todo::
    * test expression for phase difference

    * test expresion for :math:`z`

"""
import numpy as np
from numba import jit
from numba import float64, int64
from numba.types import UniTuple
import sys
sys.path.append("../sphere/")
sys.path.append("../ufuncs/")
from ABCD import ABCD
from scattering_amplitude import S1S2
from mie import mie_cache


@jit("float64(float64, float64, float64, float64, float64)", nopython=True)
def phase(rho, K, k1, k2, phi):
    r"""The phase difference.

    Parameters
    ----------
    rho: float
        positive, aspect ratio :math:`R/L`
    K: float
        positive, wave number in the medium multiplied by L, :math:`n\xi L/c`
    k1, k2: float
        positive, rescaled parallel wave numbers :math:`k L`
    phi: float
        between math:`0` and math:`2\pi`.
        
    Returns
    -------
    float    
        phase difference

    """
    kappa1 = np.sqrt(k1*k1+K*K)
    kappa2 = np.sqrt(k2*k2+K*K)
    return -((k1 - k2)**2 + 4*k1*k2*np.sin(phi/2)**2)/(np.sqrt(2*(kappa1*kappa2 + k1*k2*np.cos(phi) + K**2)) + kappa1 + kappa2)*rho - kappa1 - kappa2

    
@jit(float64[:](float64, float64, float64, float64, float64, mie_cache.class_type.instance_type), nopython=True)
def phiKernel_approx(rho, K, k1, k2, phi, mie):
    r"""
    Returns the phikernels.
    
    Note that the fresnel coefficients for a perfectly reflecting plane are
    taken into account here.

    Parameters
    ----------
    rho: float
        positive, aspect ratio :math:`R/L`
    K: float
        positive, wave number in the medium multiplied by L, :math:`n\xi L/c`
    k1, k2: float
        positive, rescaled parallel wave numbers :math:`k L`
    phi: float
        between math:`0` and math:`2\pi`.
    mie: class instance
        cache for the exponentially scaled mie coefficients
        
    Returns
    -------
    np.ndarray
        array of length 4 of phikernels for the polarization contributions
        TMTM, TETE, TMTE, TETM

    """
    kappa1 = np.sqrt(k1*k1+K*K)
    kappa2 = np.sqrt(k2*k2+K*K)
    z = (kappa1*kappa2+k1*k2*np.cos(phi))/K**2
    #print("%16f" % z)
    #print(K, k1, k2, phi)
    #assert(z>=1.)
    exponent = phase(rho, K, k1, k2, phi)
    if phi == np.pi:
        z = (k1**2 + k2**2 + K**2)/(kappa1*kappa2 + k1*k2)
    if exponent < -37:
        return np.array([0., 0., 0., 0.])
    e = np.exp(exponent)
    A, B, C, D = ABCD(K, k1, k2, phi)
    norm = np.sqrt(k1*k2)/(2*np.pi*K*np.sqrt(kappa1*kappa2)) # factor of 2pi due to k integral, move this into weight?
    S1 = -0.5*K*rho
    S2 = 0.5*K*rho
    TMTM = norm*(B*S1+A*S2)*e
    TETE = -norm*(A*S1+B*S2)*e
    TMTE = norm*(C*S1+D*S2)*e
    TETM = norm*(D*S1+C*S2)*e
    return np.array([TMTM, TETE, TMTE, TETM])


if __name__ == "__main__":
    rho = 1.
    K = 1.3
    k1 = 1.2
    k2 = 0.8
    phi = 0.76
    #print(phase(rho, K, k1, k2, phi))
    from mie import mie_e_array
    mie = mie_e_array(1e4, K*rho)
    print(phiKernel(rho, K, k1, k2, phi, mie))
