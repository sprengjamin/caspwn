r"""Kernel of the reflection operator on a sphere.

"""

import numpy as np
import math

from numba import njit

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))
from scattering_amplitude import S1S2, zero_frequency

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


@njit("UniTuple(float64, 4)(float64, float64, float64, float64, float64, float64, float64, float64, int64, string, float64[:], float64[:])", cache=True)
def kernel_polar(rho, r, sign, K, k1, k2, phi, n, lmax, materialclass, mie_a, mie_b):
    r"""
    Returns the kernel of the reflection operator on a sphere in polar
    coordinates in symmetrized from.

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
    if K == 0:
        if phi == np.pi:
            return 0., 0., 0., 0.
        x = 2 * rho * math.sqrt(k1 * k2) * math.cos(phi / 2)
        e = math.exp(x - (k1 + k2) * (rho + r))
        norm = rho / (2 * math.pi)
        S1, S2 = zero_frequency(x, n, lmax, materialclass)
        TMTM = norm * S2 * e
        TETE = norm * S1 * e
        TMTE = 0.
        TETM = 0.
        return TMTM, TETE, TMTE, TETM
    else:
        kappa1 = math.sqrt(k1 * k1 + K * K)
        kappa2 = math.sqrt(k2 * k2 + K * K)
        z = (kappa1 * kappa2 + k1 * k2 * math.cos(phi)) / K ** 2
        e = math.exp(phase(rho, r, K, k1, k2, phi))
        A, B, C, D = ABCD(K, k1, k2, phi)
        norm = math.sqrt(k1 * k2) / (2 * math.pi * K * math.sqrt(kappa1 * kappa2))
        S1, S2 = S1S2(K * rho, z, n, lmax, mie_a, mie_b, True)
        TMTM = norm * (B * S1 + A * S2) * e
        TETE = norm * (A * S1 + B * S2) * e
        TMTE = -sign * norm * (C * S1 + D * S2) * e
        TETM = sign * norm * (D * S1 + C * S2) * e
        return TMTM, TETE, TMTE, TETM


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
