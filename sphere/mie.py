r"""Exponentially scaled Mie coefficients.

This module provides functions for computing the exponentially scaled Mie coefficients

.. math::
    \begin{aligned}
    \tilde a_\ell(ix) &=  e^{-2\psi(\ell,x)} \left\vert a_\ell(ix)\right\vert\,,\\
    \tilde b_\ell(ix) &=  e^{-2\psi(\ell,x)} \left\vert b_\ell(ix)\right\vert
    \end{aligned}

with

.. math::
    \psi(\ell,x) = \sqrt{(\ell+1/2)^2 + x^2} + (\ell+1/2) \log\frac{x}{\ell+1/2+\sqrt{(\ell+1/2)^2+x^2}}\,.

For perfectly reflecting spheres the mie coefficients are

.. math::
    \begin{aligned}
        a_\ell(ix) &= (-1)^\ell \frac{\pi}{2}\frac{xI_{\ell-1/2} - \ell I_{\ell+1/2}(x)}{x K_{\ell-1/2}(x) + \ell K_{    \ell+1/2}(x)}\,, \\
        b_\ell(ix) &= (-1)^{\ell+1} \frac{\pi}{2} \frac{I_{\ell+1/2}(x)}{K_{\ell+1/2}(x)}\,.
    \end{aligned}

.. note::
    So far only limited to perfect reflecting spheres.

.. todo::
    * Extend to arbitrary materials.
    
    * Test for low :math:`\ell`.

"""

import numpy as np
from numba import jit, jitclass
from numba import int64, float64
import sys
sys.path.append("../ufuncs/")
from bessel import InuKnu_e
from bessel import fraction

@jit("float64(float64, float64)", nopython=True)
def _expdiff(l, x):
    r"""Computes the expression

    .. math::
        \exp(\psi(\ell-1, x) - \psi(\ell-1, x))

    in a way in which no cancellation errors occur.

    Parameters
    ----------
    l : int
        Order
    x : float
        Argument

    Returns
    -------
    float

    """
    t1 = l-0.5+np.sqrt((l-0.5)**2+x**2)
    t2 = l+0.5+np.sqrt((l+0.5)**2+x**2)
    delta = 2*l/(t1+t2-2*l)
    ans = -delta
    ans += l*np.log1p((1+delta)/t1) 
    return np.exp(ans)*np.sqrt(t1*t2)/x


@jit("UniTuple(float64, 2)(int64, float64, float64, float64, float64, float64)", nopython=True)
def mie_e(l, x, inum, knum, inup, knup):
    r"""
    Exponentially scaled Mie coefficients.

    Parameters
    ----------
    l : int
        Order
    x : float
        Argument
    inum: float
        :math:`\tilde I_{\ell-1/2}(x)`
    knum: float
        :math:`\tilde K_{\ell-1/2}(x)`
    inup : float
        :math:`\tilde I_{\ell+1/2}(x)`
    knup : float
        :math:`\tilde K_{\ell+1/2}(x)`
    
    Returns
    -------
    (float, float)
        (:math:`\tilde a_\ell(ix)`, :math:`\tilde b_\ell(ix))`
    
    """
    ble = np.pi/2*inup/knup
    epsi = _expdiff(l, x)
    ale = np.pi/2*(x*epsi*inum - l*inup)/(x/epsi*knum + l*knup)
    return ale, ble


@jit("UniTuple(float64, 2)(int64, float64, float64, float64, float64, float64, float64)", nopython=True)
def mie_e_mat(l, x, n, i_p_x, i_p_nx, k_m_x, k_p_x):
    r"""
    Exponentially scaled Mie coefficients for real materials.

    Implementation has problems for values of n close to 1.

    Parameters
    ----------
    l : int
        Order
    x : float
        Argument
    n : float
        refractive index
    i_p_x: float
        :math:`\tilde I_{\ell+1/2}(x)`
    i_p_nx: float
        :math:`\tilde I_{\ell+1/2}(nx)`
    k_m_x: float
        :math:`\tilde K_{\ell-1/2}(x)`
    k_p_x : float
        :math:`\tilde K_{\ell+1/2}(x)`
    
    Returns
    -------
    (float, float)
        (:math:`\tilde a_\ell(ix)`, :math:`\tilde b_\ell(ix))`
    
    """
    epsi = _expdiff(l, x)
    
    g_x = x*fraction(l-0.5, x)
    g_nx = n*x*fraction(l-0.5, n*x)
    
    sa = i_p_nx*i_p_x*(g_x - l)
    sb = i_p_nx*i_p_x*(g_nx - l)
    sc = i_p_nx*(x/epsi*k_m_x + l*k_p_x)
    sd = i_p_nx*k_p_x*(g_nx - l)
    
    ale = np.pi/2*(n**2*sa - sb)/(n**2*sc + sd)
    ble = np.pi/2*(sb-sa)/(sc + sd)
    return ale, ble


@jit("UniTuple(float64[:], 2)(int64, float64, float64)", nopython=True)
def mie_e_array_mat(lmax, x, n):
    r"""
    Array of exponentially scaled Mie coefficients.

    Parameters
    ----------
    lmax : int
        Maximal angular momentum :math:`\ell_\mathrm{max}`.
    x : float
        Argument
    n : float
        refractive index
    
    Returns
    -------
    (nd.array, nd.array)
        (:math:`\left[\tilde a_1(ix), \dots, \tilde a_{\ell_\mathrm{max}}(ix) \right]`, :math:`\left[\tilde b_1(ix), \dots, \tilde b_{\ell_\mathrm{max}}(ix) \right]`
    
    """
    i_x, k_x = InuKnu_e(lmax, x)
    i_nx, k_nx = InuKnu_e(lmax, x*n)
    ale = np.empty(lmax)
    ble = np.empty(lmax)
    for l in range(1, lmax+1):
        ale[l-1], ble[l-1] = mie_e_mat(l, x, n, i_x[l], i_nx[l], k_x[l-1], k_x[l])
    return ale, ble


@jit("UniTuple(float64[:], 2)(int64, float64)", nopython=True)
def mie_e_array_PR(lmax, x):
    r"""
    Array of exponentially scaled Mie coefficients.

    Parameters
    ----------
    lmax : int
        Maximal angular momentum :math:`\ell_\mathrm{max}`.
    x : float
        Argument
    
    Returns
    -------
    (nd.array, nd.array)
        (:math:`\left[\tilde a_1(ix), \dots, \tilde a_{\ell_\mathrm{max}}(ix) \right]`, :math:`\left[\tilde b_1(ix), \dots, \tilde b_{\ell_\mathrm{max}}(ix) \right]`
    
    """
    Inue, Knue = InuKnu_e(lmax, x)
    ale = np.empty(lmax)
    ble = np.empty(lmax)
    for l in range(1, lmax+1):
        ale[l-1], ble[l-1] = mie_e(l, x, Inue[l-1], Knue[l-1], Inue[l], Knue[l])
    return ale, ble


spec = [
    ("lmax", int64),
    ("x", float64),
    ("n", float64),
    ("ale", float64[:]),
    ("ble", float64[:]),
]

@jitclass(spec)
class mie_cache(object):
    def __init__(self, lmax, x, n):
        assert(lmax > 0)
        self.lmax = lmax
        self.x = x
        self.n = n
        self._make_mie_array()
    
    def _make_mie_array(self):
        if self.n == np.inf:
            self.ale, self.ble = mie_e_array_PR(self.lmax, self.x)
        else:
            self.ale, self.ble = mie_e_array_mat(self.lmax, self.x, self.n)

    def read(self, l):
        assert(l > 0)
        if l <= self.lmax:
            return self.ale[l-1], self.ble[l-1]
        else:
            # ensure that new array will be large enough
            # and mie_e_array will not be called to often
            # during upward summation in S1S2
            self.lmax = self.lmax + l
            self._make_mie_array()
            return self.ale[l-1], self.ble[l-1]


if __name__ == "__main__":
    lmax = 1e4
    x = 2.3
    n = np.inf
    cache = mie_cache(lmax, x, n)
    print(cache.read(int(1e4)+1))
    print(len(cache.ale))
    print(len(cache.ble))
    ale = cache.ale
    ble = cache.ble
    import matplotlib.pyplot as plt
    plt.loglog(ale)
    plt.loglog(ble)
    plt.show()
    print(ale)
    print(ble)
