r"""Exponentially scaled modified Bessel functions.

This module provides functions for computing the exponentially scaled modified
Bessel functions :math:`\tilde{I}_\nu(x)` and :math:`\tilde{K}_\nu(x)` defined
by

.. math::
    \tilde{I}_\nu(x) = e^{-\psi} I_\nu(x)
    
    \tilde{K}_\nu(x) = e^{\psi} K_\nu(x)

with

.. math::
    \psi = (\nu^2 + x^2)^{1/2} + \nu \log\left(\frac{x}{\nu + (\nu^2 + x^2)^{1/2}}\right)

motivated by the `Debye asymptotic expansion`_.

.. _Debye asymptotic expansion:
    https://dlmf.nist.gov/10.41#ii

"""

import numpy as np
import math
from numba import jit

@jit("float64(int64, float64)", nopython=True)
def _U(k, p):
    r"""Helper function used in :meth:`bessel.InuKnu_e_asymptotics`.

    Returns the Polynomials :math:`U_k(p)` appearing in the `Debye asymptotic
    expansion`_ up to forth order.

    .. _Debye asymptotic expansion:
        https://dlmf.nist.gov/10.41#ii
    """
    if k == 0:
        return 1.
    if k == 1:
        return (p - (5*p**3)/3.)/8.
    if k == 2:
        return (p**2*(81 - 462*p**2 + 385*p**4))/1152.
    if k == 3:
        return (30375*p**3 - 369603*p**5 + 765765*p**7 - 425425*p**9)/414720.
    if k == 4:
        return (p**4*(4465125 - 94121676*p**2 + 349922430*p**4 - 446185740*p**6 + 185910725*p**8))/3.981312e7 
    return 0.

@jit("UniTuple(float64, 2)(float64, float64)", nopython=True) 
def InuKnu_e_asymptotics(nu, x):
    r"""Exponentially scaled modified Bessel functions using Debye asymptotics.
    
    The function have been tested to become exact within machine precision for
    :math:`\nu\geq 1000`.

    Parameters
    ----------
    nu : float
        Order of the modified Bessel functions.
    x : float
        Argument of the modified Bessel functions.

    Returns
    -------
    Inu_e : float
        Exponentially scaled modified Bessel function of the *first* kind.
    Knu_e : float
        Exponentially scaled modified Bessel function of the *second* kind.

    References
    ----------
    `https://dlmf.nist.gov/10.41#ii`_

    .. _https://dlmf.nist.gov/10.41#ii:
        https://dlmf.nist.gov/10.41#ii

    """
    z = x/nu
    pinv = math.sqrt(1+z*z)
    Inu_e = math.sqrt(1/((2*math.pi*nu)*pinv)) 
    Knu_e = math.sqrt(math.pi/((2*nu)*pinv))
    p = 1/pinv
    Inu_e *= 1+_U(1,p)/nu+_U(2,p)/(nu*nu)+_U(3,p)/(nu*nu*nu)+_U(4,p)/(nu**4)
    Knu_e *= 1-_U(1,p)/nu+_U(2,p)/(nu*nu)-_U(3,p)/(nu*nu*nu)+_U(4,p)/(nu**4)
    return Inu_e, Knu_e

@jit("float64(float64, float64)", nopython=True)
def _t(nu, x):
    return nu + math.sqrt(nu**2 + x**2)

@jit("float64(float64, float64)", nopython=True)
def _expPsi1(nu, x):
    delta11 = (2*nu + 1)/(math.sqrt((nu + 1)**2 + x**2) + math.sqrt(nu**2 + x**2))
    return x/_t(nu+1, x)*math.exp(delta11 - nu*math.log1p((1 + delta11)/_t(nu, x)))

@jit("float64(float64, float64)", nopython=True)
def _expPsi2(nu, x):
    delta21 = 4*nu/(math.sqrt((nu + 1)**2 + x**2) + math.sqrt((nu - 1)**2 + x**2))
    tnup1 = _t(nu+1, x)
    tnum1 = _t(nu-1, x)
    return x**2/(tnup1*tnum1)*math.exp(delta21 - nu*math.log1p((2 + delta21)/tnum1))

@jit("float64(float64, float64, float64, float64)", nopython=True)
def Knu_e_next(nu, x, new, old):
    r"""Recurrence relation for exponentially scaled modified Bessel function
    of second kind.

    Finds the next value of :math:`\tilde{K}_{\nu+1}(x)` in terms of the two
    previous values :math:`\tilde{K}_{\nu}(x)` and :math:`\tilde{K}_{\nu-1}(x)`
    using the recurrence relation.
    
    Parameters
    ----------
        nu: float
            Order
        x: float
            Argument
        new: float
            :math:`\tilde{K}_{\nu}(x)`
        old: float
            :math:`\tilde{K}_{\nu-1}(x)`
    
    Returns
    -------
        float
            :math:`\tilde{K}_{\nu+1}(x)`

    """
    return 2*nu/x*_expPsi1(nu, x)*new + _expPsi2(nu, x)*old

@jit("float64(float64, float64)", nopython=True)
def fraction(nu, x):
    r"""Returns the fraction

    .. math::
        \frac{I_\nu(x)}{I_{\nu+1}(x)}

    where :math:`I_\nu` is the modified Bessel function of the first kind of
    order :math:`\nu`.

    Parameters
    ----------
    nu : float
        Order
    x : float
        Argument

    Returns
    -------
    float

    """
    invx = 1/x

    a1 = 2*(nu+1)*invx
    a2 = 2*(nu+2)*invx

    num = a2+1/a1
    denom = a2
    ratio = a1*num/denom
    ratio_last = 0.
    l = 3
    while True:
        an = (2*nu+2*l)*invx
        num = an+1/num
        denom = an+1/denom
        ratio *= num/denom
        if(ratio == ratio_last):
            return ratio
        ratio_last = ratio
        l += 1

@jit("float64(float64, float64, float64, float64)", nopython=True)
def Inu_e_wronskian(nu, x, Knu0, Knu1):
    r"""Computes :math:`\tilde{I}_\nu(x)` using the Wronskian.

    Parameters
    ----------
        nu: float
            Order
        x: float
            Argument
        Knu0: float
            :math:`\tilde{K}_{\nu}(x)`
        Knu1: float
            :math:`\tilde{K}_{\nu+1}(x)`
    
    Returns
    -------
        float
            :math:`\tilde{I}_{\nu}(x)`

    """
    return 1/(x*(Knu1/_expPsi1(nu, x) + Knu0/fraction(nu, x)))

@jit("UniTuple(float64[:], 2)(int64, float64)", nopython=True)
def InuKnu_e(lmax, x):
    r"""Computes nd.array of exponentially scaled modified Bessel functions of
    first and second kind of half-integer order:

    .. math::
        \left[\tilde{I}_{1/2}(x), \dots, \tilde{I}_{\ell_\mathrm{max}+1/2}(x)\right],
        
        \left[\tilde{K}_{1/2}(x), \dots, \tilde{K}_{\ell_\mathrm{max}+1/2}(x)\right]

    Parameters
    ----------
        lmax: int
            :math:`\ell_\mathrm{max}`
        x: float
            Argument
    
    Returns
    -------
        (nd.array, nd.array)

    """
    tI = np.empty(lmax+1)
    tK = np.empty(lmax+2)
    tK[0] = math.sqrt(math.pi/(2*_t(0.5, x)))*math.exp(0.25/(math.sqrt(0.25 + x**2) + x))
    tK[1] = math.sqrt(math.pi/2)*(_t(1.5, x)**-1.5)*(1+x)*math.exp(2.25/(math.sqrt(2.25 + x**2) + x))
    tI[0] = Inu_e_wronskian(0.5, x, tK[0], tK[1])
    for l in range(1, min(lmax, 1000)+1):
        tK[l+1] = Knu_e_next(l+0.5, x, tK[l], tK[l-1])
        tI[l] = Inu_e_wronskian(l+0.5, x, tK[l], tK[l+1])
    if lmax > 1000:
        for l in range(1001, lmax+1):
            tI[l], tK[l] = InuKnu_e_asymptotics(l+0.5, x)
    return tI, tK[:-1]

A0 = np.array([    
    -4.41534164647933937950E-18,
     3.33079451882223809783E-17,
    -2.43127984654795469359E-16,
     1.71539128555513303061E-15,
    -1.16853328779934516808E-14,
     7.67618549860493561688E-14,
    -4.85644678311192946090E-13,
     2.95505266312963983461E-12,
    -1.72682629144155570723E-11,
     9.67580903537323691224E-11,
    -5.18979560163526290666E-10,
     2.65982372468238665035E-9,
    -1.30002500998624804212E-8,
     6.04699502254191894932E-8,
    -2.67079385394061173391E-7,
     1.11738753912010371815E-6,
    -4.41673835845875056359E-6,
     1.64484480707288970893E-5,
    -5.75419501008210370398E-5,
     1.88502885095841655729E-4,
    -5.76375574538582365885E-4,
     1.63947561694133579842E-3,
    -4.32430999505057594430E-3,
     1.05464603945949983183E-2,
    -2.37374148058994688156E-2,
     4.93052842396707084878E-2,
    -9.49010970480476444210E-2,
     1.71620901522208775349E-1,
    -3.04682672343198398683E-1,
     6.76795274409476084995E-1
     ], dtype=np.float64)

B0 = np.array([
    -7.23318048787475395456E-18,
    -4.83050448594418207126E-18,
     4.46562142029675999901E-17,
     3.46122286769746109310E-17,
    -2.82762398051658348494E-16,
    -3.42548561967721913462E-16,
     1.77256013305652638360E-15,
     3.81168066935262242075E-15,
    -9.55484669882830764870E-15,
    -4.15056934728722208663E-14,
     1.54008621752140982691E-14,
     3.85277838274214270114E-13,
     7.18012445138366623367E-13,
    -1.79417853150680611778E-12,
    -1.32158118404477131188E-11,
    -3.14991652796324136454E-11,
     1.18891471078464383424E-11,
     4.94060238822496958910E-10,
     3.39623202570838634515E-9,
     2.26666899049817806459E-8,
     2.04891858946906374183E-7,
     2.89137052083475648297E-6,
     6.88975834691682398426E-5,
     3.36911647825569408990E-3,
     8.04490411014108831608E-1
    ], dtype=np.float64)

@jit("float64(float64)", nopython=True)
def _chbevlA0(x):
    r"""Evaluation of Chebyshev series.

    Parameters
    ----------
    x: float

    Returns
    -------
    float
    
    References
    ----------
    Algorithm based on `www.netlib.org/cephes/cmath.tgz`_
    
    .. _www.netlib.org/cephes/cmath.tgz:
        www.netlib.org/cephes/cmath.tgz

    """
    global A0
    b1 = 0.0
    b0 = A0[0]
    for i in range(29):
        b2 = b1
        b1 = b0
        b0 = x*b1 - b2 + A0[i+1]
    return 0.5*(b0-b2)

@jit("float64(float64)", nopython=True)
def _chbevlB0(x):
    """
    Evaluation of Chebyshev series.
    
    Parameters
    ----------
    x: float

    Returns
    -------
    float
    
    References
    ----------
    Algorithm based on `www.netlib.org/cephes/cmath.tgz`_
    
    .. _www.netlib.org/cephes/cmath.tgz:
        www.netlib.org/cephes/cmath.tgz

    """
    global B0
    b1 = 0.0
    b0 = B0[0]
    for i in range(24):
        b2 = b1
        b1 = b0
        b0 = x*b1 - b2 + B0[i+1]
    return 0.5*(b0-b2)

@jit("float64(float64)", nopython=True)
def I0e(x):
    r"""Exponentially scaled modified Bessel function :math:`e^{-x}I_0(x)` of zero order.

    Parameters
    ----------
    x: float
    
    Returns
    -------
    I0e: float

    References
    ----------
    Algorithm based on `www.netlib.org/cephes/bessel.tgz`_
    
    .. _www.netlib.org/cephes/bessel.tgz:
        www.netlib.org/cephes/bessel.tgz
    
    """
    if x < 0:
        x = -x
    if x <= 8.0:
        y = x/2.0 - 2.0
        return _chbevlA0(y)
    else:
        return _chbevlB0(32.0/x - 2.0)/math.sqrt(x)

A1 = np.array([
     2.77791411276104639959E-18,
    -2.11142121435816608115E-17,
     1.55363195773620046921E-16,
    -1.10559694773538630805E-15,
     7.60068429473540693410E-15,
    -5.04218550472791168711E-14,
     3.22379336594557470981E-13,
    -1.98397439776494371520E-12,
     1.17361862988909016308E-11,
    -6.66348972350202774223E-11,
     3.62559028155211703701E-10,
    -1.88724975172282928790E-9,
     9.38153738649577178388E-9,
    -4.44505912879632808065E-8,
     2.00329475355213526229E-7,
    -8.56872026469545474066E-7,
     3.47025130813767847674E-6,
    -1.32731636560394358279E-5,
     4.78156510755005422638E-5,
    -1.61760815825896745588E-4,
     5.12285956168575772895E-4,
    -1.51357245063125314899E-3,
     4.15642294431288815669E-3,
    -1.05640848946261981558E-2,
     2.47264490306265168283E-2,
    -5.29459812080949914269E-2,
     1.02643658689847095384E-1,
    -1.76416518357834055153E-1,
     2.52587186443633654823E-1
     ])
     
B1 = np.array([     
     7.51729631084210481353E-18,
     4.41434832307170791151E-18,
    -4.65030536848935832153E-17,
    -3.20952592199342395980E-17,
     2.96262899764595013876E-16,
     3.30820231092092828324E-16,
    -1.88035477551078244854E-15,
    -3.81440307243700780478E-15,
     1.04202769841288027642E-14,
     4.27244001671195135429E-14,
    -2.10154184277266431302E-14,
    -4.08355111109219731823E-13,
    -7.19855177624590851209E-13,
     2.03562854414708950722E-12,
     1.41258074366137813316E-11,
     3.25260358301548823856E-11,
    -1.89749581235054123450E-11,
    -5.58974346219658380687E-10,
    -3.83538038596423702205E-9,
    -2.63146884688951950684E-8,
    -2.51223623787020892529E-7,
    -3.88256480887769039346E-6,
    -1.10588938762623716291E-4,
    -9.76109749136146840777E-3,
     7.78576235018280120474E-1
    ])

@jit("float64(float64)", nopython=True)
def _chbevlA1(x):
    """
    Evaluation of Chebyshev series.
    Algorithm based on www.netlib.org/cephes/cmath.tgz

    """
    global A1
    b1 = 0.0
    b0 = A1[0]
    for i in range(28):
        b2 = b1
        b1 = b0
        b0 = x*b1 - b2 + A1[i+1]
    return 0.5*(b0-b2)

@jit("float64(float64)", nopython=True)
def _chbevlB1(x):
    """
    Evaluation of Chebyshev series.
    Algorithm based on www.netlib.org/cephes/cmath.tgz

    """
    global B1
    b1 = 0.0
    b0 = B1[0]
    for i in range(24):
        b2 = b1
        b1 = b0
        b0 = x*b1 - b2 + B1[i+1]
    return 0.5*(b0-b2)

@jit("float64(float64)", nopython=True)
def I1e(x):
    r"""Exponentially scaled modified Bessel function :math:`e^{-x}I_1(x)` of first order.

    Parameters
    ----------
    x: float
    
    Returns
    -------
    I1e: float

    References
    ----------
    Algorithm based on `www.netlib.org/cephes/bessel.tgz`_
    
    .. _www.netlib.org/cephes/bessel.tgz:
        www.netlib.org/cephes/bessel.tgz
    
    """
    z = math.fabs(x)
    if z <= 8.0:
        y = z/2.0 - 2.0
        z = z*_chbevlA1(y)
    else:
        z = _chbevlB1(32.0/z - 2.0)/math.sqrt(z)
    if x < 0.:
        z = -z
    return z

ACC = 10000.
BIGNO = 1.0e10
BIGNI = 1.0e-10

@jit("float64[:](int64, float64)", nopython=True)
def Ine(n, x):
    r"""Compute array of exponentially scaled modified Bessel function of
    integer order using Miller's algorithm. Instead of rescaling all the array
    elements by only :math:`e^{-x}I_0(x)`, we rescale odd integer Bessel
    functions by :math:`e^{-x}I_1(x)` for improved accuracy.

    .. math::
        \left[e^{-x}I_0(x), \dots, e^{-x}I_n(x) \right]

    Parameters
    ----------
    n : int
        highest integer order
    x : float
        argument
    
    Returns
    -------
    nd.array
    
    """
    global BIGNO
    global BIGNI
    assert(n > 1)
    assert(x >= 0.)
    if x == 0.:
        bi = np.zeros(n+1)
        bi[0] = 1.
        return bi
    bi = np.empty(n+1, dtype=np.float64)
    bi[n] = 1.
    bi[n-1] = fraction(n-1, x)
    tox = 2.0/math.fabs(x)
    j = n-1
    while(j>0):
        bi[j-1] = bi[j+1]+j*tox*bi[j]
        if math.fabs(bi[j-1]) > BIGNO:
            bi *= BIGNI
        j -= 1
    bi[::2] *= I0e(x)/bi[0]
    bi[1::2] *= I1e(x)/bi[1]
    return bi
