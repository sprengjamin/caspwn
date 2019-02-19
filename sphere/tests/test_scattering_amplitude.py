import numpy as np
import sys
sys.path.append(".")
from scattering_amplitude import S1S2, S1S2_asymptotics
from mie import mie_cache
from mpmath import *
mpf.dps = 80
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
        return float(S1), float(S2), int(lmax)
    
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

def test_scattering_amplitude():
    rtol = 1.e-10
    mp_data = np.loadtxt("tests/testdata/scattering_amplitude.dat")
    for data in mp_data:
        x = data[0]
        z = data[1]
        mpS1 = data[2]
        mpS2 = data[3]
        lmax = data[4]
        mie = mie_cache(lmax, x, np.inf)
        S1, S2 = S1S2(x, z, mie, False)
        print(x, z)
        print(abs(-S1/mpS1-1.))
        print(abs(S2/mpS2-1.))
        np.testing.assert_allclose(mpS1, -S1, rtol=rtol)
        np.testing.assert_allclose(mpS2, S2, rtol=rtol)

def test_asymptotics_low():
    N = [1.1, 10., np.inf]
    rtol = 1.1e-06
    for n in N:
        x = 5.e3
        Z = 1. + np.logspace(-3, 3, 5)
        mie = mie_cache(1e4, x, n)
        for z in Z:
            S1a, S2a = S1S2_asymptotics(x, z, n)
            S1, S2 = S1S2(x, z, mie, False)
            np.testing.assert_allclose(S1a, S1, rtol=rtol)
            np.testing.assert_allclose(S2a, S2, rtol=rtol)

def test_asymptotics_high():
    N = [1.1, 10., np.inf]
    rtol = 3.e-08
    for n in N:
        x = 6.e4
        Z = 1. + np.logspace(-3, 3, 5)
        mie = mie_cache(1e4, x, n)
        for z in Z:
            S1a, S2a = S1S2_asymptotics(x, z, n)
            S1, S2 = S1S2(x, z, mie, False)
            print(x, z, n)
            print(np.fabs(S1/S1a-1.))
            print(np.fabs(S2/S2a-1.))
            np.testing.assert_allclose(S1a, S1, rtol=rtol)
            np.testing.assert_allclose(S2a, S2, rtol=rtol)

if __name__ == "__main__":
    test_scattering_amplitude()
"""
    x = 100000.
    z = 1.000001

    mpS1, mpS2, lmax = mp_S1S2(x, z)
    #print(S1, S2)

    ale, ble = mie_e_array(lmax, x)
    S1, S2 = S1S2(x, z, ale, ble)
    
    print(fabs(S1/mpS1-1.))
    print(fabs(S2/mpS2-1.))
"""
