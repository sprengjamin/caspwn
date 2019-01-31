r"""Fresnel coefficients.

"""
from numba import njit
import math

@njit
def rTE(K, k, epsilon):
    r"""Fresnel coefficients for TE-polarized modes.

    .. math::
        r_\mathrm{TE}(iK, k) = \frac{\kappa - \sqrt{\kappa^2 + K^2[\epsilon(iK)-1]}}{\kappa + \sqrt{\kappa^2 + K^2[\epsilon(iK)-1]}

    Parameters
    ----------
    K : float
        positive, wave number
    k : float
        positive, parallel wave number
    epsilon : float
        positive, relative permittivity

    Returns
    -------
    rTE : float
        TE fresnel coefficient :math:`\in[-1,0]`

    """
    kappa = math.sqrt(k**2 + K**2)
    #num = kappa - math.sqrt(kappa**2 + K**2*(epsilon-1))
    #den = kappa + math.sqrt(kappa**2 + K**2*(epsilon-1))

    num = -K**2*(epsilon - 1.)
    den = (kappa + math.sqrt(kappa**2 + K**2*(epsilon - 1.)))**2
    return num/den

@njit
def rTM(K, k, epsilon):
    r"""Fresnel coefficients for TM-polarized modes.

    .. math::
        r_\mathrm{TM}(iK, k) = \frac{\epsilon(iK)\kappa - \sqrt{\kappa^2 + K^2[\epsilon(iK)-1]}}{\epsilon(iK)\kappa + \sqrt{\kappa^2 + K^2[\epsilon(iK)-1]}

    Parameters
    ----------
    K : float
        positive, wave number
    k : float
        positive, parallel wave number
    epsilon : float
        positive, relative permittivity

    Returns
    -------
    rTE : float
        TE fresnel coefficient :math:`\in[-1,0]`

    """
    kappa = math.sqrt(k**2 + K**2)
    #num = epsilon*kappa - math.sqrt(kappa**2 + K**2*(epsilon-1))
    #den = epsilon*kappa + math.sqrt(kappa**2 + K**2*(epsilon-1))
    num = k**2*(epsilon**2 - 1.) + K**2*(epsilon-1)*epsilon
    den = (epsilon*kappa + math.sqrt(kappa**2 + K**2*(epsilon - 1.)))**2
    return num/den
    
