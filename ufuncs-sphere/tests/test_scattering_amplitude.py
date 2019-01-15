import numpy as np
import sys
sys.path.append(".")
from scattering_amplitude import S1S2
from mie import mie_e_array
from mpmath import *
mpf.dps = 40
maxterms = 1e6

def mp_chi(l, x, z):
    # not needed at present
    l = mpf(l)
    x = mpf(x)
    z = mpf(z)
    nu = l+1/2
    nueta = sqrt(nu**2 + x**2) - nu*asinh(nu/x)
    return float(nu*acosh(z) + 2*nueta - 2*x*sqrt((1+z)/2))

def mp_pt(l, z):
    # implements angular functions p and t
    p = (2*l+1)/(l*(l+1))*legenp(l, 1, z, type=3, maxterms=maxterms)/sqrt(z**2-1)
    t = -z*p + (2*l+1)*legendre(l, z, maxterms=maxterms)
    return p, t

def mp_mie(l, x):
    # mie coefficients
    bim = besseli(l-1/2, x, maxterms=maxterms)
    bip = besseli(l+1/2, x, maxterms=maxterms)
    bkm = besselk(l-1/2, x, maxterms=maxterms)
    bkp = besselk(l+1/2, x, maxterms=maxterms)
    a = pi/2*(x*bim - l*bip)/(x*bkm + l*bkp)
    b = pi/2*bip/bkp
    return a, b

def mp_S1S2(x, z):
    r"""Mie scattering amplitudes for plane waves implemented in mpmath.

    Parameters
    ----------
    x : float
        positive, imaginary frequency
    z : float
        positive, :math:`z=-\cos \Theta`
    
    Returns
    -------
    (float, float)
        (:math:`\tilde S_1`, :math:`\tilde S_2`)
    """
    x = mpf(x)
    z = mpf(z)
    err = mpf("1.0e-20")
    arccoshz = acosh(z)
    lest = x*sqrt((z-1)/2)
    lest_int = floor(lest)+1
    print("lest", lest)

    p, t = mp_pt(lest_int, z)
    a, b = mp_mie(lest_int, x)
    e = exp(-2*x*sqrt((1+z)/2))
    S1 = (a*p + b*t)*e
    S2 = (a*t + b*p)*e
    
    # upwards
    l = lest_int+1
    while(True):
        p, t = mp_pt(l, z)
        a, b = mp_mie(l, x)
        S1term = (a*p + b*t)*e
        S2term = (a*t + b*p)*e
        if S1term/S1 < err:
            S1 += S1term
            S2 += S2term
            break
        S1 += S1term
        S2 += S2term
        l += 1
    print("dl+", l-lest_int)
    lmax = l
    if lest_int == 1:
        return float(S1), float(S2) 
    
    # downwards
    l = lest_int-1
    while(True):
        p, t = mp_pt(l, z)
        a, b = mp_mie(l, x)
        S1term = (a*p + b*t)*e
        S2term = (a*t + b*p)*e
        if S1term/S1 < err:
            S1 += S1term
            S2 += S2term
            break
        S1 += S1term
        S2 += S2term
        l -= 1
        if l == 0:
            break
    print("dl-", lest_int - l)
    return float(S1), float(S2), int(lmax)

if __name__ == "__main__":
    x = 1000.
    z = 2.3

    S1, S2, lmax = mp_S1S2(x, z)
    print(S1, S2)

    ale, ble = mie_e_array(lmax, x)
    print(S1S2(x, z, ale, ble))


