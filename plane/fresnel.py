r"""Fresnel reflection coefficients.

They describe the reflection amplitudes of plane waves for a given polarization.

.. todo::
    * cache=True throws LLVM ERROR! why?

"""
from numba import njit
import math

@njit("float64(float64, float64, float64)")
def rTE_Kfinite(K, k, epsilon):
    r"""Fresnel reflection coefficients for TE-polarized modes for finite wave numbers.

    .. math::
        r_\mathrm{TE}(iK, k) = \frac{\kappa - \sqrt{\kappa^2 + K^2[\epsilon(iK)-1]}}{\kappa + \sqrt{\kappa^2 + K^2[\epsilon(iK)-1]}

    Parameters
    ----------
    K : float
        positive, wavenumber in medium
    k : float
        positive, parallel wavenumber
    epsilon : float
        positive, relative permittivity between plane and medium

    Returns
    -------
    rTE : float
        TE fresnel reflection coefficient :math:`\in[-1,0]`

    """
    if epsilon == math.inf:
        return -1.
    kappa = math.sqrt(k**2 + K**2)
    num = -K**2*(epsilon - 1.)
    den = (kappa + math.sqrt(kappa**2 + K**2*(epsilon - 1.)))**2
    return num/den

@njit("float64(float64, float64, string)")
def rTE_Kzero(k, Kp, materialclass):
    r"""Fresnel reflection coefficients for TE-polarized modes at vanishing wavenumbers.

    Parameters
    ----------
    k : float
        positive, transverse wavenumber
    Kp : float
        positive, plasma wavenumber
    materialclass : string
        materialclass

    Returns
    -------
    rTE : float
        TE fresnel reflection coefficient

    """
    if materialclass == "drude":
        return 0.
    elif materialclass == "PR":
        return -1.
    elif materialclass == "dielectric":
        return 0.
    elif materialclass == "plasma":
        return -Kp**2/(k + math.sqrt(k**2 + Kp**2))**2
    else:
        # throw assert if materialclass is not supported
        assert(False)
        return 0.

@njit("float64(float64, float64, float64)")
def rTM_Kfinite(K, k, epsilon):
    r"""Fresnel reflection coefficients for TM-polarized modes for finite wavenumbers.

    .. math::
        r_\mathrm{TM}(iK, k) = \frac{\epsilon(iK)\kappa - \sqrt{\kappa^2 + K^2[\epsilon(iK)-1]}}{\epsilon(iK)\kappa + \sqrt{\kappa^2 + K^2[\epsilon(iK)-1]}

    Parameters
    ----------
    K : float
        positive, wavenumber in medium
    k : float
        positive, transverse wavenumber
    epsilon : float
        positive, relative permittivity between plane and medium

    Returns
    -------
    rTM : float
        TM fresnel reflection coefficient

    """
    if epsilon == math.inf:
        return 1.
    kappa = math.sqrt(k**2 + K**2)
    num = k**2*(epsilon**2 - 1.) + K**2*(epsilon-1)*epsilon
    den = (epsilon*kappa + math.sqrt(kappa**2 + K**2*(epsilon - 1.)))**2
    return num/den

@njit("float64(float64, float64, string)")
def rTM_Kzero(k, epsilon, materialclass):
    r"""Fresnel reflection coefficients for TM-polarized modes at vanishing wavenumbers.

    .. math::
        r_\mathrm{TM}(iK, k) = \frac{\epsilon(iK)\kappa - \sqrt{\kappa^2 + K^2[\epsilon(iK)-1]}}{\epsilon(iK)\kappa + \sqrt{\kappa^2 + K^2[\epsilon(iK)-1]}

    Parameters
    ----------
    k : float
        positive, parallel wavenumber
    epsilon : float
        positive, relative permittivity between plane and medium
    materialclass: string
        name of materialclass

    Returns
    -------
    rTM : float
        TM fresnel reflection coefficient :math:`\in[0, 1]`

    """
    if materialclass == "drude" or materialclass == "plasma":
        return 1.
    elif materialclass == "PR":
        return 1.
    elif materialclass == "dielectric":
        return (epsilon-1.)/(epsilon+1.)
    else:
        # throw assert if materialclass is not supported
        assert(False)
        return 0.

if __name__ == "__main__":
    K = 0.
    k = 1.
    eps = 2.
    materialclass = "dielectric"
    print(rTE(K, k, eps, materialclass))
    print(rTM(K, k, eps, materialclass))
