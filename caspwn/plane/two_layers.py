"""
TO DO: *change doc-strings
* what happens if layer 1 is plasma metal and layer 2 something else?
"""

from math import sqrt, exp
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))
from fresnel import rTM_zero, rTM_finite, rTE_zero, rTE_finite

def rTM_1slab_finite(k0, k, epsm, eps1, eps2, delta):
    """
    Fresnel reflection amplitude for TM polarization at a slab made of medium 1 followed by medium 2.

    :param k0: vacuum wavenumber (scaled by 1/L)
    :param k: in-plane wave vector (scaled by 1/L)
    :param epsm:  relative permittivity medium
    :param eps1: relative permittivity medium 1
    :param eps2: relative permittivity medium 2
    :param delta: ratio of thickness of slab to separation
    :return: float
    """
    rTM_m1 = rTM_finite(k0*sqrt(epsm), k, eps1/epsm)
    rTM_12 = rTM_finite(k0*sqrt(eps1), k, eps2/eps1)
    kappa1 = sqrt(eps1*k0**2 + k**2)
    return (rTM_m1 + rTM_12*exp(-2*kappa1*delta))/(1+rTM_m1*rTM_12*exp(-2*kappa1*delta))

def rTE_1slab_finite(k0, k, epsm, eps1, eps2, delta):
    """
    Fresnel reflection amplitude for TM polarization at a slab made of medium 1 followed by medium 2.

    :param k0: vacuum wavenumber (scaled by 1/L)
    :param k: in-plane wave vector (scaled by 1/L)
    :param epsm:  relative permittivity medium
    :param eps1: relative permittivity medium 1
    :param eps2: relative permittivity medium 2
    :param delta: ratio of thickness of slab to separation
    :return: float
    """
    rTE_m1 = rTE_finite(k0 * sqrt(epsm), k, eps1 / epsm)
    rTE_12 = rTE_finite(k0 * sqrt(eps1), k, eps2 / eps1)
    kappa1 = sqrt(eps1*k0**2 + k**2)
    return (rTE_m1 + rTE_12*exp(-2*kappa1*delta))/(1+rTE_m1*rTE_12*exp(-2*kappa1*delta))

def rTM_1slab_zero(k, eps1, materialclass1, eps2, materialclass2, delta):
    """
    Fresnel reflection amplitude for TM polarization at a slab made of medium 1 followed by medium 2.

    :param k: in-plane wave vector (scaled by 1/L)
    :param epsm:  relative permittivity medium
    :param eps1: relative permittivity medium 1
    :param eps2: relative permittivity medium 2
    :param delta: ratio of thickness of slab to separation
    :return: float
    """
    rTM_m1 = rTM_zero(k, eps1, materialclass1)
    rTM_12 = rTM_zero(k, eps2, materialclass2)
    return (rTM_m1 + rTM_12*exp(-2*k*delta))/(1+rTM_m1*rTM_12*exp(-2*k*delta))

def rTE_1slab_zero(k, Kp1, materialclass1, Kp2, materialclass2, delta):
    """
    Fresnel reflection amplitude for TM polarization at a slab made of medium 1 followed by medium 2.

    :param k: in-plane wave vector (scaled by 1/L)
    :param epsm:  relative permittivity medium
    :param eps1: relative permittivity medium 1
    :param eps2: relative permittivity medium 2
    :param delta: ratio of thickness of slab to separation
    :return: float
    """
    rTE_m1 = rTE_zero(k, Kp1, materialclass1)
    rTE_12 = rTE_zero(k, Kp2, materialclass2)
    return (rTE_m1 + rTE_12*exp(-2*k*delta))/(1+rTE_m1*rTE_12*exp(-2*k*delta))