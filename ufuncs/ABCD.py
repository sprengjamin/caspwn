r""" Coefficients describing the transformation between the two
polarization basis.

.. todo::
    * test (mpmath, corner cases)

"""

import numpy as np
import math
from numba import njit

@njit("UniTuple(float64, 4)(float64, float64, float64, float64)")
def ABCD(xi, k1, k2, phi):
    r"""Coefficients :math:`A`, :math:`B`, :math:`C` and :math:`D`.

    The coefficients describe the transformation from the polarization basis
    defined through the scattering plane into the TE, TM polarization basis.

    They are defined by

    .. math::
        \begin{aligned}
        A &=
        \frac{\xi^4\cos(\varphi)-c^4\big[\kappa_1\kappa_2+k_1k_2\cos(\varphi)\big]
        \big[k_1k_2+\kappa_1\kappa_2\cos(\varphi)\big]}
        {\xi^4-c^4\big[\kappa_1\kappa_2+k_1k_2\cos(\varphi)\big]^2} \\
        B & = -\frac{\xi^2 c^2 k_1 k_2\sin^2(\varphi)}
        {\xi^4-c^4\big[\kappa_1\kappa_2+k_1k_2\cos(\varphi)\big]^2} \\
        C & = +\xi\frac{c^3\kappa_2k_1^2+c^3\kappa_1k_1k_2\cos(\varphi)}
        {\xi^4-c^4\big[\kappa_1\kappa_2+k_1k_2\cos(\varphi)\big]^2}\sin(\varphi) \\
        D & = -\xi\frac{c^3\kappa_1k_2^2+c^3\kappa_2k_1k_2\cos(\varphi)}
        {\xi^4-c^4\big[\kappa_1\kappa_2+k_1k_2\cos(\varphi)\big]^2}\sin(\varphi)
        \end{aligned}

    with :math:`\kappa_1 = \sqrt{\xi^2/c^2+{k_1}^2}` and :math:`\kappa_2 = \sqrt{\xi^2/c^2+{k_2}^2}`.

    .. todo::
        Close to :math:`\varphi=\pi` the numerical error becomes large. Find
        suitable expressions!

        Write test!

    Parameters
    ----------
    xi : float
        frequency :math:`\xi/c`
    k1 : float
        radial momentum component :math:`k_1`
    k2 : float
        radial momentum component :math:`k_2`
    phi : float
        difference of anular components :math:`\varphi=\varphi_2-\varphi_1`

    Returns
    -------
    (A, B, C, D) : (float, float, float, float)
        Polarization transformation coefficients

    """
    if phi == 0.:
        return 1., 0., 0., 0.
    elif phi == math.pi:
        return -1., 0., 0., 0.
    else:
        kappa1 = math.sqrt(xi**2+k1**2)
        kappa2 = math.sqrt(xi**2+k2**2)
        denom = k1**2*k2**2*(1+math.cos(phi)**2) + xi**2*(k1**2+k2**2) + 2*k1*k2*kappa1*kappa2*math.cos(phi)
        numA = xi**2*(k1**2+k2**2)*math.cos(phi) + 2*k1**2*k2**2*math.cos(phi) + k1*k2*kappa1*kappa2*(1+math.cos(phi)**2)
        numB = xi**2*k1*k2*math.sin(phi)**2
        numC = -xi*math.sin(phi)*(kappa2*k1**2+kappa1*k1*k2*math.cos(phi))
        numD = xi*math.sin(phi)*(kappa1*k2**2+kappa2*k1*k2*math.cos(phi))
        return numA/denom, numB/denom, numC/denom, numD/denom
