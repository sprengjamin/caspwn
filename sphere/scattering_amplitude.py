r"""Exponentially scaled Mie scattering amplitudes for plane waves.

This module provides functions for computing the exponentially scaled Mie scattering amplitudes for plane waves :math:`\tilde S_1` and :math:`\tilde S_2` defined by

.. math::
    \begin{aligned}
        \tilde S_1(x, z) &= \exp\left(-2 x \sqrt{(1+z)/2}\right) S_1(x, z)\,, \\
        \tilde S_2(x, z) &= \exp\left(-2 x \sqrt{(1+z)/2}\right) S_2(x, z)
    \end{aligned}

and thus in terms of the exponentially scaled functions

.. math::
    \begin{aligned}
    \tilde S_1(x, z) &= -\sum_{\ell=1}^\infty 
        \left[ \tilde a_\ell(ix) \tilde p_\ell(z) +  \tilde b_\ell(ix) \tilde t_\ell(z)\right]e^{\chi(\ell,x,z)}\\
    \tilde S_2(x, z) &= \phantom{-}\sum_{\ell=1}^\infty \left[\tilde a_\ell(i x)\tilde t_\ell(z) +\tilde b_\ell(i x)\tilde p_\ell(z)\right]e^{\chi(\ell,x,z)}\,,
    \end{aligned}

with

.. math::
    \chi(\ell,x,z) = (\ell+1/2)\mathrm{arccosh}z + \sqrt{(2\ell+1)^2 + (2 x)^2} + (2\ell+1) \log\frac{2x}{2\ell+1+\sqrt{(2\ell+1)^2+(2x)^2}}-2x\sqrt{(1+z)/2}\,.

.. todo::
    * Write tests (mpmath for low l, perhaps also for higher ones; test vs asymptotics). 
    
    * Understand behavior close to z=1. for large x,
      see analysis/scattering-amplitude/S1S2/plot_high.py

    * figure out when to use chi and chi_old: chi_old seems better on testdata

"""
import numpy as np
import math
from numba import njit
from numba import float64
from numba.types import UniTuple
from math import lgamma
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))
from angular import pte, pte_low, pte_asymptotics
from mie import mie_cache

"""
def make_mat_coeff(e):
    @njit
    def mat_coeff(l):
        return l/(l+1)*(e-1)/(e+(l+1)/l)
    return mat_coeff
"""

@njit
def zero_frequency(x, mie):
    err = 1.e-16
    l_init = int(0.5*x)+1
    logx = math.log(x)
    e = mie.n**2
    S =  (e-1)/(e+(l_init+1)/l_init)*math.exp(2*l_init*logx - lgamma(2*l_init+1)-x)
    
    # upward summation
    l = l_init + 1
    while True:
        term = (e-1)/(e+(l+1)/l)*math.exp(2*l*logx - lgamma(2*l+1)-x)
        S += term
        if term/S < err:
            break
        l += 1

    if l_init == 1:
        return S

    # downward summation
    l = l_init - 1
    while True:
        term = (e-1)/(e+(l+1)/l)*math.exp(2*l*logx - lgamma(2*l+1)-x)
        S += term
        if term/S < err:
            break
        l -= 1
        if l == 0:
            break

    return S


@njit("float64(float64, float64)")
def chi_back(nu, x):
    return nu**2/(math.sqrt(nu**2 + x**2) + x) + nu*math.log(x/(nu + math.sqrt(nu**2 + x**2)))


@njit(float64(float64, mie_cache.class_type.instance_type))
def S_back(x, mie):
    r"""Mie scattering amplitudes for plane waves in the backward scattering limit.

    Parameters
    ----------
    x : float
        positive, imaginary frequency
    ale : nd.array
        Mie coefficient cache
    ble : nd.array
        Mie coefficient cache
    
    Returns
    -------
    float
        (:math:`\tilde S`)
    """
    err = 1.0e-16

    l = 1
    exp = math.exp(2*chi_back(l+0.5, x))
    ale, ble = mie.read(l)
    S = (l+0.5)*(ale + ble)*exp
    
    l += 1
    while(True):
        exp = math.exp(2*chi_back(l+0.5, x))
        ale, ble = mie.read(l)
        Sterm = (l+0.5)*(ale + ble)*exp
        if Sterm/S < err:
            S += Sterm
            break
        S += Sterm
        l += 1
    return S


@njit("float64(int64, float64, float64)")
def chi(l, x, z):    
    r"""Implementation of :math:`\chi(\ell, x, z)` where cancellation errors are minimized
    by some algebraic manipulation of the expressions.

    Parameters
    ----------
    l : int
        positive, angular momentum number
    x : float
        positive, imaginary frequency
    z : float
        positive, :math:`z=-\cos \Theta`
    
    Returns
    -------
    float
        function value
    """
    # computes chi using some algebraic manipulation
    nu = l + 0.5
    y = nu/x
    delta = y - math.sqrt((z-1)/2) # this is the accuracy bottle neck
    t1 = delta*(y+math.sqrt((z-1)/2))/(math.sqrt(y**2+1) + math.sqrt((1+z)/2))
    W = math.sqrt((delta + math.sqrt((z-1)/2))**2 + 1)
    t21 = -2*delta**2 - 4*delta*math.sqrt((z-1)/2) - 2*delta*W - 2*(z-1)*delta*(delta+math.sqrt(2*(z-1)))/(math.sqrt(z**2-1) + math.sqrt(z**2-1 + 2*(z-1)*delta*(delta+math.sqrt(2*(z-1)))))
    t2 = math.log1p(t21/(y+math.sqrt(y**2+1))**2)
    return x*(y*t2 + 2*t1)


@njit("float64(int64, float64, float64)")
def chi_old(l, x, z):
    nu = l + 0.5
    return nu*math.acosh(z) + 2*math.sqrt(nu**2 + x**2) - 2*nu*math.asinh(nu/x) - 2*x*math.sqrt((1+z)/2)


@njit(UniTuple(float64, 2)(float64, float64, mie_cache.class_type.instance_type))
def S1S2(x, z, mie):
    r"""Mie scattering amplitudes for plane waves.

    Parameters
    ----------
    x : float
        positive, imaginary frequency
    z : float
        positive, :math:`z=-\cos \Theta`
    ale : nd.array
        Mie coefficient cache
    ble : nd.array
        Mie coefficient cache
    
    Returns
    -------
    (float, float)
        (:math:`\tilde S_1`, :math:`\tilde S_2`)
    
    
    """
    err = 1.0e-16
    chi = chi_old # for the moment

    if z <= 1.:
        S = S_back(x, mie)
        return -S, S

    acoshz = math.acosh(z)
    pte_cache = np.vstack(pte_low(1000, acoshz))
    lest = x*math.sqrt(math.fabs(z-1)/2)
    lest_int = int(lest)+1
    #print("lest", lest)

    pe, te = pte(lest_int, acoshz, pte_cache)
    ale, ble = mie.read(lest_int)
    exp = math.exp(chi(lest_int, x, z))
    S1 = (ale*pe + ble*te)*exp
    S2 = (ale*te + ble*pe)*exp
    
    # upwards summation
    l = lest_int+1
    while(True):
        pe, te = pte(l, acoshz, pte_cache)
        ale, ble = mie.read(l)
        exp = math.exp(chi(l, x, z))
        S1term = (ale*pe + ble*te)*exp
        S2term = (ale*te + ble*pe)*exp
        if S1term/S1 < err:
            S1 += S1term
            S2 += S2term
            break
        S1 += S1term
        S2 += S2term
        l += 1
    #print("dl+", l-lest_int)
    
    if lest_int == 1:
        return -S1, S2 
    
    # downwards summation
    l = lest_int-1
    while(True):
        pe, te = pte(l, acoshz, pte_cache)
        ale, ble = mie.read(l)
        exp = math.exp(chi(l, x, z))
        S1term = (ale*pe + ble*te)*exp
        S2term = (ale*te + ble*pe)*exp
        if S1term/S1 < err:
            S1 += S1term
            S2 += S2term
            break
        S1 += S1term
        S2 += S2term
        l -= 1
        if l == 0:
            break
    #print("dl-",lest_int-l)
    return -S1, S2


if __name__ == "__main__":
    x = 100
    z = 2.3
    n = 1.8
    #ale, ble = mie_e_array(1e5, x)
    mie = mie_cache(1e1, x, n)
    print(mie.lmax)
    S1, S2 = S1S2(x, z, mie)
    print(mie.lmax)
    S1, S2 = S1S2(1000, z, mie)
    print(mie.lmax)
    
    """
    sigma = math.sqrt((1+z)/2)
    S1a = -0.5*x*(1+((1-2*sigma**2)/(2*sigma**3))/x)
    S2a = 0.5*x*(1+(-1/(2*sigma**3))/x)
    print("compare to asymptotics")
    print(S1)
    print(S1a)
    print((S1-S1a)/S1a)
    print(S2)
    print(S2a)
    print((S2-S2a)/S2a)
    width = math.sqrt(x*math.sqrt((1+z)/2))
    print("width", width)
    print("6*width", 6*width)
    #jit()(S1S2).inspect_types()
    """
