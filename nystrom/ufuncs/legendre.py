r"""Exponentially scaled Legendre functions :math:`\tilde P_\ell(\cosh x)` for
large order.

This module provides functions for computing the exponentially scaled Legendre functions

.. math::
    \tilde P_\ell(\cosh x) = P_\ell(\cosh x) e^{-(\ell+1/2)\vert x\vert}

for large orders :math:`\ell\geq 1000` using asymptotics.

"""

import numpy as np
import math
from ..ufuncs.bessel import Ine
from numba import njit

@njit("float64(int64, float64)", cache=True)
def Ple_low(l, x):
    r"""Exponentially scaled Legendre function for small arguments.

    For small arguments, :math:`(\ell+1)\sinh(x) \leq 25`, the Legendre
    polynomials can be evaluated using the asymptotic expansion

    .. math::
        P_\ell(\cosh x) \simeq \sum_{n=0}^{12} \frac{f_n(x\nu)}{\nu^n}

    where :math:`\nu=\ell+1/2`. The functions :math:`f_n(x)` vanish for odd values of :math:`n`
    and for even values of :math:`n` they are given by

    .. math::
        \frac{f_n(x\nu)}{\nu^n} = (-1)^{n/2}\sum_{m=n/2}^{n} a_{n,m} x^m \nu^{m-n} I_m(x\nu)
    
    where the coefficients :math:`a_{n,m}` are given in some table.  By
    replacing the Bessel functions with their exponentially scaled versions we
    obtain the expression for the exponentially scaled Legendre function.

    .. todo::
        Include the table.
    
    Parameters
    ----------
    l : float
        Order
    x: float
        Argument

    Returns
    -------
    float
        Exponentially scaled Legendre function

    """
    nu = l+0.5
    
    # compute Bessels
    bessels = Ine(12, nu*x)
    
    # n = 0
    Pl = bessels[0]
    
    # n = 2
    Pl += -bessels[2]*x**2/12
    Pl += -bessels[1]*x/(8*nu)

    # n = 4
    Pl += bessels[4]*x**4/160
    Pl += bessels[3]*x**3/nu*7/160
    Pl += bessels[2]*(x/nu)**2*11/384

    # n = 6
    Pl += -bessels[6]*x**6*61/120960
    Pl += -bessels[5]*x**5/nu*671/80640
    Pl += -bessels[4]*x**4/nu**2*101/3583
    Pl += -bessels[3]*(x/nu)**3*173/15360
    """
    # n = 8
    Pl += bessels[8]*x**8*1261/29030400
    Pl += bessels[7]*x**7/nu*1261/967680
    Pl += bessels[6]*x**6/nu**2*217/20480
    Pl += bessels[5]*x**5/nu**3*90497/3870720
    Pl += bessels[4]*(x/nu)**4*22931/3440640
    
    # n = 10
    Pl += -bessels[10]*x**10*79/20275200
    Pl += -bessels[9]*x**9/nu*1501/8110080
    Pl += -bessels[8]*x**8/nu**2*7034857/2554675200
    Pl += -bessels[7]*x**7/nu**3*1676287/113541120
    Pl += -bessels[6]*x**6/nu**4*10918993/454164480
    Pl += -bessels[5]*(x/nu)**5*1319183/2477226080
    
    # n = 12
    Pl += bessels[12]*x**12*66643/185980354560
    Pl += bessels[11]*x**11/nu*1532789/61993451520
    Pl += bessels[10]*x**10/nu**2*3135577/5367398400
    Pl += bessels[9]*x**9/nu**3*72836747/12651724800
    Pl += bessels[8]*x**8/nu**4*2323237523/101213798400
    Pl += bessels[7]*x**7/nu**5*1396004969/47233105920
    Pl += bessels[6]*(x/nu)**6*233526463/43599790080
    """
    return Pl

@njit("float64(float64)", cache=True)
def gammafraction(l):
    r"""Coefficient :math:`C_{\ell, 0}` for large :math:`\ell`.

    The coefficient is computed by evaluating the asymptotic series of the :math:`\Gamma`-function:

    .. math::
        C_{\ell,0} \simeq \frac{1}{\sqrt{\ell}}\left(1-\frac{3}{8\,\ell}+\frac{25}{128\,\ell^2} - \frac{105}{1024\,\ell^3} + \frac{1659}{32768\,\ell^4} - \frac{6237}{262144\,\ell^5}+ \dots\right)
    
    .. note::
        Although :math:`\ell` is an integer, we need to specify the
        corresponding parameter as a float in order to prevent division by zero.

    Parameters
    ----------
    l : float
        Order

    Returns
    -------
    float
        Coefficient :math:`C_{\ell, 0}`

    """
    return  (1 - 3/(8*l) + 25/(128*l**2) - 105/(1024*l**3) + 1659/(32768*l**4 - 6237/262144*l**5))/math.sqrt(l)

@njit("float64(int64, float64)", cache=True)
def Ple_high(l, x):
    r"""Exponentially scaled Legendre function for large arguments.
    
    For large arguments, :math:`(\ell+1)\sinh x > 25`,
    we make use of the asymptotic expansion
    
    .. math::
        \tilde P_\ell(\cosh x) \simeq \left(\frac{1}{2\pi \sinh x}\right)^{1/2} \sum_{m=0}^{M-1} \tilde{C}_{\ell, m} \frac{1+e^{-(2m+2\ell+1)x}}{(1    -e^{-2x})^m}
         
    The coefficients :math:`C_{\ell,m}` are given in terms of the recurrence relation 

    .. math::
        \tilde C_{\ell,m+1} = \frac{(m+1/2)^2}{(m+1)(\ell+m+3/2)} \tilde C_{\ell,m}

    with the initial value :math:`\tilde C_{\ell,0} = C_{\ell,0}` given in :meth:`legendre.gammafraction`.
    
    Parameters
    ----------
    l : float
        Order
    x: float
        Argument

    Returns
    -------
    float
        Exponentially scaled Legendre function

    """
    res = 1.
    Clm = 1.
    for m in range(1, 17):
        Clm *= (m-0.5)**2/(m*(l+m+0.5))
        res += Clm*(1+math.exp(-(2*m+2*l+1)*x))/((-math.expm1(-2*x))**m)
    return math.sqrt(1/(2*math.pi*math.sinh(x)))*gammafraction(l)*res

@njit("float64(int64, float64)", cache=True)
def Ple_asymptotics(l, x):
    r"""Exponentially scaled Legendre function for large arguments.
    
    For large arguments, :math:`(\ell+1)\sinh x > 25`, :meth:`legendre.Ple_high` is called,
    while :meth:`legendre.Ple_low` is called else.

    Parameters
    ----------
    l : float
        Order
    x: float
        Argument

    Returns
    -------
    float
        Exponentially scaled Legendre function

    """
    if (l+1)*math.sinh(x) > 25:
        return Ple_high(l, x)
    else:
        return Ple_low(l, x)
