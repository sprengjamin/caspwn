import numpy as np
from numba import jit
from numba import float64, int64
from numba.types import UniTuple
import sys
sys.path.append("../sphere")
from mie import mie_cache
from scattering_amplitude import S1S2, zero_frequency
sys.path.append("../ufuncs")
from ABCD import ABCD


@jit(float64[:](float64, float64, float64, float64, float64, float64, float64, mie_cache.class_type.instance_type), nopython=True)
def phiKernel(rho, r, sign, K, k1, k2, phi, mie):
    r"""
    Returns the phikernels.

    Parameters
    ----------
    rho: float
        positive, aspect ratio R/L
    r: float
        positive, relative effective radius R_eff/R
    sign: +1/-1
        sign, differs for the two spheres
    K: float
        positive, wave number in the medium multiplied by L, :math:`n\xi L/c`
    k1, k2: float
        positive, rescaled wave numbers
    phi: float
        between 0 and 2pi
    mie: class instance
        cache for the exponentially scaled mie coefficient
        
    Returns
    -------
    np.ndarray
        array of length 4 of phikernels for the polarization contributions
        TMTM, TETE, TMTE, TETM

    """
    if K == 0:
        x = 2*rho*np.sqrt(k1*k2)*np.cos(phi/2)
        exponent = x - (k1+k2)*(rho+r)
        if exponent < -37:
            return np.array([0., 0., 0., 0.])
        e = np.exp(exponent)
        norm = rho/(2*np.pi)
        S = zero_frequency(x, mie)
        TMTM = norm*S*e
        TETE = 0.
        TMTE = 0.
        TETM = 0.
        return np.array([TMTM, TETE, TMTE, TETM])
    else:
        kappa1 = np.sqrt(k1*k1+K*K)
        kappa2 = np.sqrt(k2*k2+K*K)
        z = (kappa1*kappa2+k1*k2*np.cos(phi))/K**2
        exponent = 2*rho*K*np.sqrt((1+z)/2) - (kappa1+kappa2)*(rho+r)
        if exponent < -37:
            return np.array([0., 0., 0., 0.])
        e = np.exp(exponent)
        A, B, C, D = ABCD(K, k1, k2, phi)
        norm = np.sqrt(k1*k2)/(2*np.pi*K*np.sqrt(kappa1*kappa2))
        S1, S2 = S1S2(K*rho, z, mie)
        pkTMTM =  norm*(B*S1+A*S2)*e
        pkTETE =  norm*(A*S1+B*S2)*e
        pkTMTE =  -norm*(C*S1+D*S2)*e*sign
        pkTETM =  norm*(D*S1+C*S2)*e*sign
        return np.array([pkTMTM, pkTETE, pkTMTE, pkTETM])


if __name__ == "__main__":
    rho = 1.
    K = 1.
    k1 = 1.
    k2 = 1.
    phi = 1.
    #print(phase(rho, K, k1, k2, phi))
    from mie import mie_e_array
    ale, ble = mie_e_array(1e5, K*rho)
    print(phiKernel(rho, 0.5, +1, K, k1, k2, phi, ale, ble))
    print(phiKernel(rho, 0.5, -1, K, k1, k2, phi, ale, ble))
