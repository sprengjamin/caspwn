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
    \tilde S_1(x, z) &= \phantom{-}\sum_{\ell=1}^\infty 
        \left[ \tilde a_\ell(ix) \tilde p_\ell(z) +  \tilde b_\ell(ix) \tilde t_\ell(z)\right]e^{\chi(\ell,x,z)}\\
    \tilde S_2(x, z) &= -\sum_{\ell=1}^\infty \left[\tilde a_\ell(i x)\tilde t_\ell(z) +\tilde b_\ell(i x)\tilde p_\ell(z)\right]e^{\chi(\ell,x,z)}\,,
    \end{aligned}

with

.. math::
    \chi(\ell,x,z) = (\ell+1/2)\mathrm{arccosh}z + \sqrt{(2\ell+1)^2 + (2 x)^2} + (2\ell+1) \log\frac{2x}{2\ell+1+\sqrt{(2\ell+1)^2+(2x)^2}}-2x\sqrt{(1+z)/2}\,.

.. todo::
    * Write tests (mpmath for low l, perhaps also for higher ones; test vs asymptotics). 

"""
import numpy as np
from numpy import sqrt as Sqrt
from numba import jit
from angular import pte, pte_low, pte_asymptotics
from mie import mie_e_array

@jit("float64(int64, float64, float64)", nopython=True)
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
    delta = y - np.sqrt((z-1)/2) # this is the accuracy bottle neck
    t1 = delta*(y+np.sqrt((z-1)/2))/(np.sqrt(y**2+1) + np.sqrt((1+z)/2))
    W = np.sqrt((delta + np.sqrt((z-1)/2))**2 + 1)
    t21 = -2*delta**2 - 4*delta*np.sqrt((z-1)/2) - 2*delta*W - 2*(z-1)*delta*(delta+np.sqrt(2*(z-1)))/(np.sqrt(z**2-1) + np.sqrt(z**2-1 + 2*(z-1)*delta*(delta+np.sqrt(2*(z-1)))))
    t2 = np.log1p(t21/(y+np.sqrt(y**2+1))**2)
    return x*(y*t2 + 2*t1)

@jit("UniTuple(float64, 2)(float64, float64, float64[:], float64[:])", nopython=True)
def S1S2(x, z, ale, ble):
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
    arccoshz = np.arccosh(z)
    pte_cache = np.vstack(pte_low(1000, arccoshz))
    lest = x*np.sqrt(np.abs(z-1)/2)
    lest_int = int(lest)+1
    print("lest", lest)

    pe, te = pte(lest_int, arccoshz, pte_cache)
    #pe, te = pte_asymptotics(lest_int, arccoshz)
    #exp = np.exp(chi(lest_int, x, z, lest))
    exp = np.exp(chi(lest_int, x, z))
    S1 = (ale[lest_int-1]*pe + ble[lest_int-1]*te)*exp
    S2 = (ale[lest_int-1]*te + ble[lest_int-1]*pe)*exp
    
    # upwards
    l = lest_int+1
    while(True):
        pe, te = pte(l, arccoshz, pte_cache)
        #pe, te = pte_asymptotics(l, arccoshz)
        #exp = np.exp(chi(l, x, z, lest))
        exp = np.exp(chi(lest_int, x, z))
        S1term = (ale[l-1]*pe + ble[l-1]*te)*exp
        S2term = (ale[l-1]*te + ble[l-1]*pe)*exp
        if S1term/S1 < err:
            S1 += S1term
            S2 += S2term
            break
        S1 += S1term
        S2 += S2term
        l += 1
    #print("dl+", l-lest_int)
    
    if lest_int == 1:
        return S1, S2 
    # downwards
    l = lest_int-1
    while(True):
        pe, te = pte(l, arccoshz, pte_cache)
        #pe, te = pte_asymptotics(l, arccoshz)
        #exp = np.exp(chi(l, x, z, lest))
        exp = np.exp(chi(lest_int, x, z))
        S1term = (ale[l-1]*pe + ble[l-1]*te)*exp
        S2term = (ale[l-1]*te + ble[l-1]*pe)*exp
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
    return S1, S2

@jit("float64(float64, float64)", nopython=True)
def chi_back(nu, x):
    return nu**2/(np.sqrt(nu**2 + x**2) + x) + nu*np.log(x/(nu + np.sqrt(nu**2 + x**2)))

@jit("float64(float64, float64[:], float64[:])", nopython=True)
def S_back(x, ale, ble):
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
    exp = np.exp(2*chi_back(l+0.5, x))
    S = (l+0.5)*(ale[l-1] + ble[l-1])*exp
    
    l += 1
    while(True):
        exp = np.exp(2*chi_back(l+0.5, x))
        Sterm = (l+0.5)*(ale[l-1] + ble[l-1])*exp
        if Sterm/S < err:
            S += Sterm
            break
        S += Sterm
        l += 1
    return S

if __name__ == "__main__":
    x = 4000
    z = 30
    ale, ble = mie_e_array(1e5, x)
    """
    S1, S2 = S1S2(x, z, ale, ble)
    sigma = np.sqrt((1+z)/2)
    S1a = 0.5*x*(1+((1-2*sigma**2)/(2*sigma**3))/x)
    S2a = 0.5*x*(1+(-1/(2*sigma**3))/x)
    print(S1)
    print(S1a)
    print((S1-S1a)/S1a)
    print(S2)
    print(S2a)
    print((S2-S2a)/S2a)
    width = np.sqrt(x*np.sqrt((1+z)/2))
    print("width", width)
    print("6*width", 6*width)
    """
    import matplotlib.pyplot as plt
    n=30
    X = np.logspace(1, 3, n)
    s = np.empty(n)
    for i, x in enumerate(X):
        ale, ble = mie_e_array(1e5, x)
        s[i] = S_limit(x, ale, ble)
    f, ax = plt.subplots()
    #ax.loglog(X, np.abs(s), "b.")
    #ax.loglog(X, X/2, "k--")
    #ax.loglog(X, 1.5*X**3, "k--")
    ax.loglog(X, 0.5*X**-4, "k", label=r"$1/2 (R\xi/c)^{-4}$")
    ax.loglog(X, np.abs(s/(0.5*X)-(1-0.5/X)), "b.", label=r"$S_{1/2}$")
    #ax.loglog(X, X**-2, "k--")

    ax.set_title(r"$\cos\Theta=-1$")
    ax.set_xlabel(r"$R\xi/c$")
    ax.set_ylabel(r"$S_p/S_{p,\mathrm{WKB}}-(1+s_p/R)$")

    ax.legend()
    plt.savefig("fig5.pdf")
    #plt.show()
