r"""Exponentially scaled angular functions.

This module provides functions for computing the angular functions
:math:`\pi_\ell(z)` and :math:`\pi_\ell(z)`. However, for convenience, we compute
the functions

.. math::
    \begin{aligned}
    p_\ell(\cosh x) &\equiv \frac{2\ell+1}{\ell(\ell+1)} \pi_\ell(\cosh x) = \frac{2\ell+1}{\sinh x} P^{-1}_\ell (\cosh x) \,,\\
    t_\ell(\cosh x) &\equiv \frac{2\ell+1}{\ell(\ell+1)} \tau_\ell(\cosh x) = -\cosh x\, p_\ell(\cosh x) + (2\ell+1)P_\ell(\cosh x)
    \end{aligned}

and to prevent under/overflow it is important to consider the exponentially scaled functions

.. math::
    \begin{aligned}
    \tilde p_\ell(x) &= p_\ell(x) e^{-(\ell+1/2)\mathrm{arccosh}\vert x\vert}\,,\\
    \tilde t_\ell(x) &= t_\ell(x) e^{-(\ell+1/2)\mathrm{arccosh}\vert x\vert}\,.
    \end{aligned}

.. todo::
    * l237, pte_array, it seems that lmin<1000 is almost never used. Test this!

"""
import numpy as np
import math
from numba import njit
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../ufuncs/"))
from bessel import Ine
from legendre import Ple_asymptotics

@njit("float64(int64, float64)", cache=True)
def _cf(m, x):
    """
    2*n+1 is start value
    """
    x2 = x*x
    num = 2*m+3
    denom = 2*m+3 + x2/(2*m+1)
    ratio = 1/(2*m+1)*num/denom
    ratio_last = 0.
    n = m+2
    while(True):
        num = 1+2*n+x2/num
        denom = 1+2*n+x2/denom
        ratio *= num/denom
        if ratio == ratio_last:
            return ratio
        ratio_last = ratio
        n += 1

@njit("float64(float64)", cache=True)
def _c1(x):
    if x < 1.:
        y = _cf(1, x)
        return -y/8.
    else:
        return (1 - x/math.tanh(x))/(8.*x**2)

@njit("float64(float64)", cache=True)
def _c2(x):
    if x < 1.:
        y = _cf(2, x)
        return (-3 + 8*y*(3 + x**2*y))/(384.*(3 + x**2*y)**2)
    else:
        y = x/math.tanh(x)
        return (8*x**2 - 3*(-7 + 6*y + y**2))/(384.*x**4) 

@njit("float64(float64)", cache=True)
def _c3(x):
    if x < 1.:
        y = _cf(3, x)
        return (225*(1 - 8*y) - x**2*(-40 + 3*y*(75 + y*(240 + x**2*(23 + (24 + x**2)*y)))))/(3072.*(15 + x**2*(1 + 3*y))**3)
    else:
        y = x/math.tanh(x)
        return (40*x**2 - 3*(-33 + 27*y + 5*y**2 + y**3))/(3072.*x**6) 

@njit("float64(float64)", cache=True)
def _c4(x):
    if x < 1.:
        y = _cf(4, x)
        return (64*x**12*y**4 + 1157625*(-41 + 384*y) + 73500*x**2*(-181 + 9*y*(191 + 288*y)) + 80*x**10*y**2*(3 + y*(68 + 321*y)) + 350*x**4*(-5293 + 3*y*(10994 + 81*y*(641 + 320*y))) + 100*x**6*(-291 + y*(-1589 + 9*y*(5795 + 3*y*(2699 + 480*y)))) + 15*x**8*(-15 + y*(-124 + y*(3238 + y*(43364 + 21105*y)))))/(1.47456e6*(105 + x**4*y + 5*x**2*(2 + 3*y))**4) 
    else:
        y = x/math.tanh(x)
        return (64*x**4 + 240*x**2*(55 + y**2) - 45*(-715 + 572*y + 110*y**2 + 28*y**3 + 5*y**4))/(1.47456e6*x**8)

@njit("float64(float64)", cache=True)
def _c5(x):
    if x < 1.:
        y = _cf(5, x)
        return -(105*x**20*y**5 + 265831216875*(-23 + 256*y) + 5*x**18*y**3*(2 + 5*y)*(-16 + 575*y) + 843908625*x**2*(-4426 + 35*y*(1037 + 1024*y)) + 281302875*x**4*(-1640 + y*(17183 + 10*y*(4937 + 1792*y))) + 595350*x**6*(-87376 + 5*y*(159064 + 7*y*(114732 + 5*y*(21851 + 3584*y)))) + x**16*y*(64 + 5*y*(-1072 + y*(19416 + 25*y*(6467 + 9314*y)))) + 33075*x**8*(-72320 + y*(319544 + 5*y*(1376594 + y*(2351546 + 35*y*(28379 + 1792*y))))) + 315*x**10*(-159072 + 5*y*(-46832 + 5*y*(953804 + 35*y*(127002 + y*(98842 + 16051*y))))) + 6*x**14*(128 + 5*y*(-2368 + y*(15296 + 5*y*(107096 + 7*y*(58516 + 52555*y))))) + 35*x**12*(-11392 + y*(-169136 + 25*y*(126584 + y*(1247390 + y*(2623270 + 806183*y))))))/(3.93216e6*(945 + 105*x**2*(1 + y) + x**4*(1 + 10*y))**5)
    else:
        y = x/math.tanh(x)
        return (-64*x**4*(-3 + y) + 80*x**2*(325 + 9*y**2 + 2*y**3) - 15*(-4199 + 3315*y + 650*y**2 + 182*y**3 + 45*y**4 + 7*y**5))/(3.93216e6*x**10)

@njit("float64[:](float64)", cache=True)
def c_coefficients(x):
    r"""Coefficients :math:`c_k(x)`.

    The coefficients can be obtained from the series expansion

    .. math::
        \left(2x\frac{\cosh x - \cosh\left(\sqrt{x^2-y}\right)}{y \sinh(x)}\right)^{1/2} = \sum_{k=0}^\infty c_k(x) y^k\,.

    The first coefficients are

    .. math::
        \begin{aligned}
        c_0 &= 1\,, \\
        c_1 &= \frac{1-x \coth (x)}{8 x^2}\,, \\
        c_2 &= \frac{8 x^2-3 x^2 \coth ^2(x)-18 x \coth (x)+21}{384 x^4}\,, \\
        c_3 &= \frac{-3 x^3 \coth ^3(x)+40 x^2-15 x^2 \coth ^2(x)-81 x \coth (x)+99}{3072 x^6}\,.
        \end{aligned}

    .. todo::
        Explain cancellation problem and how to prevent it.

    Parameters
    ----------
    x : float
    
    Returns
    -------
    nd.array
        First six coefficients, :math:`\left[ c_0(x), \dots, c_5(x)\right]`.

    """
    c = np.empty(8)
    c[0] = 1. 
    c[1] = _c1(x)
    c[2] = _c2(x)
    c[3] = _c3(x)
    c[4] = _c4(x)
    c[5] = _c5(x)
    return c 

@njit("float64(float64, float64)", cache=True)
def pe_asymptotics(l, x):
    r"""Exponentially scaled angular function :math:`\tilde p_\ell\left(\cosh(x)\right)` for large order.

    The asymptotics of the angular function can be expressed through the
    asymptotics of the associated Legendre function :math:`P^{-1}_\ell(\cosh x)`
    and we find

    .. math::
        p_\ell\left(\cosh(x)\right) \simeq 2\left(\frac{x}{\sinh^3 x}\right)^{1/2} \sum_{k=0}^\infty c_k(x) \left(\frac{3}{2}\right)^{(k)} I_{k+1}(\nu x) \left(\frac{2x}{\nu}\right)^k
    
    with the coefficients :math:`c_k(x)` defined in :meth:`angular.c_coefficients` and
    the rising factorial defined by
    
    .. math::
        \left(\frac{3}{2}\right)^{(k)} = \frac{\Gamma(3/2+k)}{\Gamma(3/2)} 
    
    which is computed recursively. For the asymptotics :math:`\tilde p_\ell\left(\cosh(x)\right)`
    one simply needs to replace the Bessel functions by their exponentially
    scaled versions.
        
    Parameters
    ----------
    l : float
        Order
    x : float
        Argument

    Returns
    -------
    float
        Exponentially scaled angular function

    """
    nu = l+0.5
    kmax = 6
    bessels = Ine(kmax+1, nu*x)
    c = c_coefficients(x)

    ff = 1.
    ans = 0.
    for k in range(kmax):
        ans += c[k]*ff*bessels[k+1]*(2*x/nu)**k
        ff *= 1.5+k
    return 2.*math.sqrt(x/math.sinh(x)**3)*ans

@njit("UniTuple(float64, 2)(float64, float64)", cache=True)
def pte_high(l, x):
    r"""Exponentially scaled angular function :math:`\tilde p_\ell\left(\cosh(x)\right)`
    and :math:`\tilde t_\ell\left(\cosh(x)\right)` for large order.

    :math:`\tilde p_\ell\left(\cosh(x)\right)` is computed by calling :meth:`angular.pe_asymptotics`.
    :math:`\tilde t_\ell\left(\cosh(x)\right)` is computed using the relation

    .. math::
        \tilde t_\ell(\cosh x) = -\cosh x\, \tilde p_\ell(\cosh x) + (2\ell+1) \tilde P_\ell(\cosh x)
    
    and :math:`\tilde P_\ell(\cosh x)` by calling :meth:`legendre.Ple_asymptotics`.
    
    Parameters
    ----------
    l : float
        Order
    x : float
        Argument

    Returns
    -------
    (float, float)
        Exponentially scaled angular functions :math:`(\tilde p_\ell\left(\cosh(x)\right), \tilde t_\ell\left(\cosh(x)\right)`

    """
    pe = pe_asymptotics(l, x)
    te = -math.cosh(x)*pe + (2*l+1)*Ple_asymptotics(l, x)
    return pe, te

@njit("UniTuple(float64, 2)(int64, float64, float64, float64, float64, float64, float64)", cache=True)
def pte_next(l, x, z, emx, em2x, pe_new, pe_old):
    """Compute pe_l+1 and te_l+1 by means of the recurrence relation
    """
    pe_next = (2*l+3)/(l+2)*(z*emx*pe_new - (l-1)/(2*l-1)*em2x*pe_old)
    te_next = (l+1)*z*pe_next - (2*l+3)*l/(2*l+1)*emx*pe_new
    return pe_next, te_next

@njit("UniTuple(float64[:], 2)(int64, int64, float64)", cache=True)
def pte_array(lmin, lmax, x):
    """Compute array for pe, te from lmin to lmax
    """
    assert(lmax > lmin+1)
    z = math.cosh(x)
    emx = math.exp(-x)
    em2x = emx*emx
    if lmin < 1000:
        p = np.empty(lmax)
        t = np.empty(lmax)
        p[-1] = 0.
        p[0] = 3/2*math.exp(-1.5*x)
        t[0] = z*p[0]
    
        for l in range(1, lmax):
            p[l], t[l] = pte_next(l, x, z, emx, em2x, p[l-1], p[l-2])
    else:
        p = np.empty(lmax-lmin+1)
        t = np.empty(lmax-lmin+1)
        p[0], t[0] = pte_high(lmin, x)
        p[1], t[1] = pte_high(lmin+1, x)
        for l in range(lmin+2, lmax+1):
            p[l-lmin], t[l-lmin] = pte_next(l-1, x, z, emx, em2x, p[l-lmin-1], p[l-lmin-2])
    return p, t

@njit("UniTuple(float64[:], 2)(int64, float64)", cache=True)
def pte_low(l, x):
    r"""Exponentially scaled angular function :math:`\tilde p_\ell\left(\cosh(x)\right)`
    and :math:`\tilde t_\ell\left(\cosh(x)\right)` for small order.
    
    The angular functions are computed using the recurrence relation
    
    .. math::
        \begin{aligned}
        \tilde p_\ell(\cosh x) &= \frac{2\ell+1}{\ell+1}\left[\cosh x\, e^{-\vert x\vert} \tilde p_{\ell-1}(\cosh x) - \frac{\ell-2}{2\ell-3}e^{-2\vert x\vert}\tilde p_{\ell-2}(\cosh x)\right] \\
        \tilde t_\ell(\cosh x) &= \ell \cosh x\, \tilde p_\ell(\cosh x) - \frac{(2\ell+1)(\ell-1)}{2\ell-1} e^{-\vert x\vert} \tilde p_{\ell-1}(\cosh x)
        \end{aligned}

    with the initial values :math:`\tilde p_0(\cosh x) = 0` and :math:`\tilde p_1(\cosh x) = \frac{3}{2} e^{\vert x\vert}`.
    
    .. todo::
        Improve this.

    Parameters
    ----------
    l : float
        Order
    x : float
        Argument

    Returns
    -------
    (nd.array, nd.array)
        Exponentially scaled angular functions :math:`(\tilde p_\ell\left(\cosh(x)\right), \tilde t_\ell\left(\cosh(x)\right)`

    """
    p = np.empty(l)
    t = np.empty(l)
    z = math.cosh(x)
    emz = math.exp(-x)
    em2z = math.exp(-2*x)
    p[-1] = 0.
    p[0] = 3/2*math.exp(-1.5*x)
    t[0] = z*p[0]
    
    for i in range(1, l):
        p[i], t[i] = pte_next(i, x, z, emz, em2z, p[i-1], p[i-2])
    return p, t

@njit("UniTuple(float64, 2)(int64, float64, float64[:,:])", cache=True)
def pte(l, x, cache):
    if l < 1001:
        return cache[0, l-1], cache[1, l-1]
    else:
        return pte_high(l, x)

@njit("UniTuple(float64, 2)(int64, float64)", cache=True)
def pte_asymptotics(l, x):
    if l < 1001:
        pe, te = pte_low(l, x)
        return pe[-1], te[-1]
    else:
        return pte_high(l, x)

if __name__ == "__main__":
    import time
    lmin = 1200
    lmax = 3000
    x = 1.
    a1, a2 = (pte_array(lmin, lmax, x))
    start = time.time()
    a1, a2 = (pte_array(lmin, lmax, x))
    end = time.time()
    print(a1)
    print(a2)
    print("time", end-start)
    @njit
    def foo(lmin, lmax, x):
        pe = np.empty(lmax-lmin+1)
        te = np.empty(lmax-lmin+1)
        for l in range(lmin, lmax+1):
            pe[l-lmin], te[l-lmin] = pte_asymptotics(l, x)
        return pe, te
    b1, b2 = foo(lmin, lmax, x)
    start = time.time()
    b1, b2 = foo(lmin, lmax, x)
    end = time.time()
    print(b1)
    print(b2)
    print("time", end-start)

