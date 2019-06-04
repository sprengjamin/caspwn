r"""Fresnel reflection coefficients.

They describe the reflection amplitudes of plane waves for a given polarization.

.. todo::
    * implement different material classes like drude, plasma

"""
from numba import njit
import math

@njit("float64(float64, float64, float64, string)", cache=True)
def rTE(K, k, epsilon, materialclass):
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
    materialclass: string
        name of materialclass

    Returns
    -------
    rTE : float
        TE fresnel reflection coefficient :math:`\in[-1,0]`

    """
    if K == 0.:
        if materialclass == "drude":
            return 0.
        elif materialclass == "PR":
            return -1.
        elif materialclass == "dielectric":
            return 0.
        else:
            # throw assert if materialclass is not supported
            assert(False)
            return 0.
    else:
        if epsilon == math.inf:
            return -1.
        kappa = math.sqrt(k**2 + K**2)
        num = -K**2*(epsilon - 1.)
        den = (kappa + math.sqrt(kappa**2 + K**2*(epsilon - 1.)))**2
        return num/den

@njit("float64(float64, float64, float64, string)", cache=True)
def rTM(K, k, epsilon, materialclass):
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
    materialclass: string
        name of materialclass

    Returns
    -------
    rTM : float
        TM fresnel reflection coefficient :math:`\in[0, 1]`

    """
    if K == 0.:
        if materialclass == "drude":
            return 1.
        elif materialclass == "PR":
            return 1.
        elif materialclass == "dielectric":
            return (epsilon-1.)/(epsilon+1.)
        else:
            # throw assert if materialclass is not supported
            assert(False)
            return 0.
    else:
        if epsilon == math.inf:
            return 1.
        kappa = math.sqrt(k**2 + K**2)
        num = k**2*(epsilon**2 - 1.) + K**2*(epsilon-1)*epsilon
        den = (epsilon*kappa + math.sqrt(kappa**2 + K**2*(epsilon - 1.)))**2
        return num/den

if __name__ == "__main__":
    K = 0.
    k = 1.
    eps = 2.
    materialclass = "dielectric"
    print(rTE(K, k, eps, materialclass))
    print(rTM(K, k, eps, materialclass))
