import numpy as np
from numba import njit
from scipy.integrate import quad
from scipy.constants import hbar, c


@njit("float64[:](int64)", cache=True)
def weights(N):
    r"""Fourier-Chebyshev quadrature weights.
    
    """
    i = np.arange(1, N+1, 1)
    t = np.pi/(N+1)*i
    wts = np.zeros(N)
    for j in i:
        wts += np.sin(j*t)*(1-np.cos(j*np.pi))/j
    wts *= 2*np.sin(t)*(2/(N+1))/(1-np.cos(t))**2
    return wts


@njit("UniTuple(float64[:], 2)(int64)", cache=True)
def fc_quadrature(N):
    r"""Fourier-Chebyshev quadrature rule.

    Parameters
    ----------
    N : int
        quadrature order

    Returns
    -------
    points : nd.array
        quadrature points
    weights : nd.array
        quadrature weights
    
    """
    i = np.arange(1, N+1, 1)
    t = np.pi/(N+1)*i
    pts = 1./(np.tan(t/2))**2
    wts = weights(N)
    return pts, wts


def integral_fcq(L, func, X):
    """

    Parameters
    ----------
    L : float
        surface-to-surface distance
    func : function
        scalar valued integrand
    X : int
        FC quadrature order

    Returns
    -------
    float

    """
    K_pts, K_wts = fc_quadrature(X)
    res = 0.
    for i in range(X):
        res += K_wts[i] * func(K_pts[i])
    return res / (2 * np.pi) * hbar * c / L


def integral_quad(L, func, epsrel):
    """

    Parameters
    ----------
    L : float
        surface-to-surface distance
    func : function
        scalar valued integrand
    epsrel : float
        relative error passed to quad-routine

    Returns
    -------
    float

    """
    res = quad(func, 0, np.inf, epsrel=epsrel)[0]
    return res / (2 * np.pi) * hbar * c / L


def zipzap(a, b):
    """ Zip two arrays together.
    """
    c = np.empty(len(a)+len(b))
    c[::2] = a
    c[1::2] = b
    return c


def auto_integration(func, Ninit=10, rtol=1.0e-8):
    r"""Automatic integration based on Fourier-Chebyshev scheme.

    Determines the value of the integal
        
    .. math::
        \int_0^\infty \mathrm{d}x\, f(x)\,.

    
    Parameters
    ----------
    func : function :math:`f`
        integrand
    Ninit : int
        initial quadrature order
    rtol : float
        relative tolerance, precision of final result

    Returns
    -------
    integral : float
        integral
    
    """
    N = Ninit
    i = np.arange(1, N+1, 1)
    t = np.pi/(N+1)*i
    pts = 1./(np.tan(t/2))**2
    
    func_values = np.empty(N)
    for i, p in enumerate(pts):
        func_values[i] = func(p)
    
    wts = weights(N)
     
    integral = np.sum(func_values*wts)

    while(True):
        N = 2*N+1
        i = np.arange(1, N+1, 2)
        t = np.pi/(N+1)*i
        pts = 1./(np.tan(t/2))**2
    
        func_values_new = np.empty(len(i))
        for i, p in enumerate(pts):
            func_values_new[i] = func(p)
        func_values = zipzap(func_values_new, func_values)
        wts = weights(N)
        #print(N, integral)
        integral_new = np.sum(func_values*wts)
        if np.abs(integral/integral_new-1.) < rtol:
            return integral_new
        integral = integral_new
