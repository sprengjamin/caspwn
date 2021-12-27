import numpy as np
from scipy.constants import k, hbar, c
from .psd import psd

def psd_sum(T, L, func, epsrel, order=None):
    """Computes the PSD sum for the finite frequency/wavenumber contributions

    Parameters
    ----------
    T : float
        temperature in Kelvin
    L : float
        surface-to-surface distance in meters
    func : function
        (vector valued) function to be summed over
    epsrel : float
        relative error for the summation
    X : int
        PSD order

    Returns
    -------
    (float, float, float)

    """
    K_matsubara = 2 * np.pi * k * T / (hbar * c)
    res = np.zeros(3)
    Teff = 4 * np.pi * k / hbar / c * T * L
    if order == None:
        order = int(max(np.ceil((1 - 1.5 * np.log10(np.abs(epsrel))) / np.sqrt(Teff)), 5))
    xi, eta = psd(order)
    for n in range(order):
        res += 2 * eta[n] * func(K_matsubara * xi[n]/2/np.pi)
    return 0.5*k*T*res


def msd_sum(T, L, func, epsrel, nmax=None):
    """Computes the PSD sum for the finite frequency/wavenumber contributions

    Parameters
    ----------
    T : float
        temperature in Kelvin
    L : float
        surface-to-surface distance in meters
    func : function
        (vector valued) function to be summed over
    epsrel : float
        relative error for the summation
    X : int


    Returns
    -------
    (float, float, float)

    """
    K_matsubara = 2 * np.pi * k * T / (hbar * c)
    res = np.zeros(3)
    n = 1
    if nmax == None:
        nmax = np.inf

    while (n <= nmax):
        term = func(K_matsubara * n)
        res += 2 * term
        if abs(term[0] / res[0]) < epsrel:
            break
        n += 1
    return 0.5*k*T*res