import numpy as np
from numba import jit
import sys
sys.path.append("../sphere")
from scattering_amplitude import S1S2
sys.path.append("../ufuncs")
from ABCD import ABCD


@jit("float64[:](float64, float64, float64, float64, float64, float64, float64, float64[:], float64[:])", nopython=True)
def phiKernel(rho, r, sign, xi, k1, k2, phi, ale, ble):
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
    exponent = 2*rho*xi*np.sqrt((1+z)/2) - (kappa1+kappa2)*(rho+r)
    if exponent < -37:
        return np.array([0., 0., 0., 0.])
    expFactor = np.exp(exponent)
    A, B, C, D = ABCD(xi, k1, k2, phi)
    prefactor = np.sqrt(k1*k2)/(2*np.pi*xi*np.sqrt(kappa1*kappa2))
    S1, S2 = S1S2(xi*rho, z, ale, ble)
    pkTMTM =  prefactor*(B*S1+A*S2)*expFactor
    pkTETE =  prefactor*(A*S1+B*S2)*expFactor
    pkTMTE =  -prefactor*(C*S1+D*S2)*expFactor*sign
    pkTETM =  prefactor*(D*S1+C*S2)*expFactor*sign
    return np.array([pkTMTM, pkTETE, pkTMTE, pkTETM])


if __name__ == "__main__":
    rho = 1.
    xi = 1.
    k1 = 1.
    k2 = 1.
    phi = 1.
    #print(phase(rho, xi, k1, k2, phi))
    from mie import mie_e_array
    ale, ble = mie_e_array(1e5, xi*rho)
    print(phiKernel(rho, 0.5, +1, xi, k1, k2, phi, ale, ble))
    print(phiKernel(rho, 0.5, -1, xi, k1, k2, phi, ale, ble))
