r"""Kernel of the reflection operator on a sphere.

"""
import numpy as np
import math
from numba import njit
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))
from scattering_amplitudes import S1S2_finite, S1S2_zero
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../ufuncs/"))
from ABCD import ABCD


@njit("float64(float64, float64, float64, float64, float64, float64)", cache=True)
def phase(rho, r, K, k1, k2, phi):
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
    kappa1 = math.sqrt(k1 * k1 + K * K)
    kappa2 = math.sqrt(k2 * k2 + K * K)
    return -((k1 - k2) ** 2 + 4 * k1 * k2 * math.sin(phi / 2) ** 2) / (
                math.sqrt(2 * (kappa1 * kappa2 + k1 * k2 * math.cos(phi) + K ** 2)) + kappa1 + kappa2) * rho - r * (
                       kappa1 + kappa2)


@njit("UniTuple(float64, 4)(float64, float64, float64, float64, float64, float64, float64, float64, int64, float64[:], float64[:])", cache=True)
def kernel_polar_finite(rho, r, sign, K, k1, k2, phi, n, lmax, mie_a, mie_b):
    r"""
    Returns the kernel of the reflection operator on a sphere in polar
    coordinates in symmetrized from.

    Parameters
    ----------
    rho: float
        positive, aspect ratio R/L
    r: float
        positive, relative amount of surface-to-surface translation
    sign: float
        sign, +/-1, differs for the two spheres
    K: float
        positive, wave number in the medium multiplied by L, :math:`n\xi L/c`
    k1, k2: float
        positive, rescaled wave numbers
    phi: float
        between 0 and 2pi
    n : float
        positive, refractive index
    mie_a : list
        list of mie coefficients for electric polarizations
    mie_b : list
        list of mie coefficients for magnetic polarizations
    materialclass: string
        the material class (currently supports: drude, dielectric, PR)
    lmax : int
        positive, cut-off angular momentum
        
    Returns
    -------
    tuple
        tuple of length 4 of kernels for the polarization contributions
        TMTM, TETE, TMTE, TETM

    """
    kappa1 = math.sqrt(k1 * k1 + K * K)
    kappa2 = math.sqrt(k2 * k2 + K * K)
    z = (kappa1 * kappa2 + k1 * k2 * math.cos(phi)) / K ** 2
    e = math.exp(phase(rho, r, K, k1, k2, phi))
    A, B, C, D = ABCD(K, k1, k2, phi)
    norm = 2 * math.pi * math.sqrt(k1 * k2) / (K * math.sqrt(kappa1 * kappa2))
    S1, S2 = S1S2_finite(K * rho, z, n, lmax, mie_a, mie_b, True)
    TMTM = norm * (B * S1 + A * S2) * e
    TETE = norm * (A * S1 + B * S2) * e
    TMTE = -sign * norm * (C * S1 + D * S2) * e
    TETM = sign * norm * (D * S1 + C * S2) * e
    return TMTM, TETE, TMTE, TETM


@njit("UniTuple(float64, 2)(float64, float64, float64, float64, float64, float64, string, int64)", cache=True)
def kernel_polar_zero(rho, r, k1, k2, phi, alpha, materialclass, lmax):
    r"""
    Returns the kernel of the reflection operator on a sphere in polar
    coordinates in symmetrized from.

    Parameters
    ----------
    rho: float
        positive, aspect ratio R/L
    r: float
        positive, relative effective radius R_eff/R
    k1, k2: float
        positive, rescaled wave numbers
    phi: float
        between 0 and 2pi
    alpha : float
        positive, parameter depending on materialclass
    materialclass: string
        the material class (currently supports: drude, dielectric, PR)
    lmax : int
        positive, cut-off angular momentum
        
    Returns
    -------
    tuple
        tuple of length 2 of kernels for the polarization contributions
        TMTM, TETE

    """
    if phi == np.pi:
        return 0., 0.
    x = 2 * rho * math.sqrt(k1 * k2) * abs(math.cos(phi / 2))
    e = math.exp(x - (k1 + k2) * (rho + r))
    norm = 2 * math.pi * rho
    S1, S2 = S1S2_zero(x, alpha, lmax, materialclass)
    TMTM = norm * S2 * e
    TETE = norm * S1 * e
    return TMTM, TETE

if __name__ == "__main__":
    rho = 1.
    K = 1.
    k1 = 1.
    k2 = 1.
    phi = 1.
    # print(phase(rho, K, k1, k2, phi))
    from mie import mie_cache

    mie = mie_cache(1e4, K, 1.7)
    print(kernel_polar(rho, 0.5, +1, K, k1, k2, phi, mie))
    print(kernel_polar(rho, 0.5, -1, K, k1, k2, phi, mie))
