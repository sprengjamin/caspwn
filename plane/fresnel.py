r"""Fresnel reflection coefficients.

They describe the reflection amplitudes of plane waves for a given polarization.

.. todo::
    * implement different material classes like drude, plasma

"""
from numba import njit
import math

@njit("float64(float64, float64, float64)", cache=True)
def rTE(K, k, epsilon):
    r"""Fresnel reflection coefficients for TE-polarized modes.

    .. math::
        r_\mathrm{TE}(iK, k) = \frac{\kappa - \sqrt{\kappa^2 + K^2[\epsilon(iK)-1]}}{\kappa + \sqrt{\kappa^2 + K^2[\epsilon(iK)-1]}

    Parameters
    ----------
    K : float
        positive, wavenumber in medium
    k : float
        positive, parallel wavenumber
    epsilon : float
        positive, relative permittivity

    Returns
    -------
    rTE : float
        TE fresnel reflection coefficient :math:`\in[-1,0]`

    """
    if epsilon == math.inf:
        return -1.
    kappa = math.sqrt(k**2 + K**2)
    #num = kappa - math.sqrt(kappa**2 + K**2*(epsilon-1))
    #den = kappa + math.sqrt(kappa**2 + K**2*(epsilon-1))

    num = -K**2*(epsilon - 1.)
    den = (kappa + math.sqrt(kappa**2 + K**2*(epsilon - 1.)))**2
    return num/den

@njit("float64(float64, float64, float64)", cache=True)
def rTM(K, k, epsilon):
    r"""Fresnel reflection coefficients for TM-polarized modes.

    .. math::
        r_\mathrm{TM}(iK, k) = \frac{\epsilon(iK)\kappa - \sqrt{\kappa^2 + K^2[\epsilon(iK)-1]}}{\epsilon(iK)\kappa + \sqrt{\kappa^2 + K^2[\epsilon(iK)-1]}

    Parameters
    ----------
    K : float
        positive, wavenumber in medium
    k : float
        positive, parallel wavenumber
    epsilon : float
        positive, relative permittivity

    Returns
    -------
    rTM : float
        TM fresnel reflection coefficient :math:`\in[0, 1]`

    """
    if epsilon == math.inf:
        return 1.
    kappa = math.sqrt(k**2 + K**2)
    #num = epsilon*kappa - math.sqrt(kappa**2 + K**2*(epsilon-1))
    #den = epsilon*kappa + math.sqrt(kappa**2 + K**2*(epsilon-1))
    num = k**2*(epsilon**2 - 1.) + K**2*(epsilon-1)*epsilon
    den = (epsilon*kappa + math.sqrt(kappa**2 + K**2*(epsilon - 1.)))**2
    return num/den
