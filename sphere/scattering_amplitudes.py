r"""Exponentially scaled Mie scattering amplitudes for plane waves.

This module provides functions for computing the exponentially scaled Mie scattering amplitudes for plane waves :math:`\tilde S_1` and :math:`\tilde S_2` defined by

.. math::
    \begin{aligned}
        \tilde S_1(x, z) &= \exp\left(-2 x \sqrt{(1+z)/2}\right) S_1(x, z)\,, \\
        \tilde S_2(x, z) &= \exp\left(-2 x \sqrt{(1+z)/2}\right) S_2(x, z)
    \end{aligned}

and thus in terms of the exponentially scaled functions

.. math::
    \begin{aligned}
    \tilde S_1(x, z) &= -\sum_{\ell=1}^\infty 
        \left[ \tilde a_\ell(ix) \tilde p_\ell(z) +  \tilde b_\ell(ix) \tilde t_\ell(z)\right]e^{\chi(\ell,x,z)}\\
    \tilde S_2(x, z) &= \phantom{-}\sum_{\ell=1}^\infty \left[\tilde a_\ell(i x)\tilde t_\ell(z) +\tilde b_\ell(i x)\tilde p_\ell(z)\right]e^{\chi(\ell,x,z)}\,,
    \end{aligned}

with

.. math::
    \chi(\ell,x,z) = (\ell+1/2)\mathrm{arccosh}z + \sqrt{(2\ell+1)^2 + (2 x)^2} + (2\ell+1) \log\frac{2x}{2\ell+1+\sqrt{(2\ell+1)^2+(2x)^2}}-2x\sqrt{(1+z)/2}\,.

.. todo::
    * Write tests for real materials(mpmath for low l, perhaps also for higher ones; test vs asymptotics).
    * Understand behavior close to z=1. for large x,
      see analysis/scattering-amplitude/S1S2/plot_high.py
    * see if implementation using logarithms runs faster

"""
import math
from numba import njit
from math import sqrt
from math import lgamma
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))
from angular import pte_array
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../ufuncs"))
from bessel import fraction

@njit("float64(float64, float64)", cache=True)
def plasma_coefficient(l, z):
    nu = l+0.5
    return -l/(l+1)*(1-2*nu/z/fraction(nu-1, z))

@njit("UniTuple(float64, 2)(float64, float64, int64, unicode_type)", cache=True)
def S1S2_zero(x, alpha, lmax, materialclass):
    r"""Mie scattering amplitudes for plane waves at vanishing frequency/wavenumber.
    The implementation depends on the material class.

    Parameters
    ----------
    x : float
        positive, parameter related to scattering angle
    alpha : float
        positive, parameter depending on materialclass
    lmax : int
        positive, cut-off angular momentum
    materialclass: string
        the material class (currently supports: drude, dielectric, plasma, PR)

    Returns
    -------
    (float, float)
        (:math:`\tilde S_1`, :math:`\tilde S_2`)
    
    """
    if materialclass == "dielectric":
        e = alpha
        if x > 500.:
            PFA = 0.5*(e-1)/(e+1)
            correction1 = -2/(e+1)/x
            correction2 = -2*(e-1)/(e+1)**2/x**2
            S = PFA*(1+correction1+correction2)
            return 0., S
        else:
            err = 1.e-16
            l_init = int(0.5*x)+1
            if l_init > lmax:
                l_init = lmax
            logx = math.log(x)
            S =  (e-1)/(e+(l_init+1)/l_init)*math.exp(2*l_init*logx - lgamma(2*l_init+1)-x)
            if S == 0.:
                return 0., 0.
            
            # upward summation
            l = l_init + 1
            while l <= lmax:
                term = (e-1)/(e+(l+1)/l)*math.exp(2*l*logx - lgamma(2*l+1)-x)
                S += term
                if term/S < err:
                    break
                l += 1

            # downward summation
            l = l_init - 1
            while l > 0:
                term = (e-1)/(e+(l+1)/l)*math.exp(2*l*logx - lgamma(2*l+1)-x)
                S += term
                if term/S < err:
                    break
                l -= 1
            return 0, S

    elif materialclass == "plasma":
        if x < 0.1:
            S2 = (x**2/2 + x**4/24 + x**6/720 + x**8/40320)*math.exp(-x)
        else:
            S2 = 0.5*(1+math.exp(-2*x))-math.exp(-x)
        err = 1.e-16
        l_init = int(0.5*x)+1
        if l_init > lmax:
            l_init = lmax
        logx = math.log(x)
        S1 =  plasma_coefficient(l_init, alpha)*math.exp(2*l_init*logx - lgamma(2*l_init+1)-x)
        if S1 == 0.:
            return 0., S2
        
        # upward summation
        l = l_init + 1
        while l <= lmax:
            term = plasma_coefficient(l, alpha)*math.exp(2*l*logx - lgamma(2*l+1)-x)
            S1 += term
            if term/S1 < err:
                break
            l += 1

        # downward summation
        l = l_init - 1
        while l > 0:
            term = plasma_coefficient(l, alpha)*math.exp(2*l*logx - lgamma(2*l+1)-x)
            S1 += term
            if term/S1 < err:
                break
            l -= 1
        return S1, S2

    elif materialclass == "drude":
        if x == 0.:
            return 0., 0.
        else:
            S1 = 0.
            if x < 0.1:
                S2 = (x**2/2 + x**4/24 + x**6/720 + x**8/40320)*math.exp(-x)
            else:
                S2 = 0.5*(1+math.exp(-2*x))-math.exp(-x)
            return S1, S2

    elif materialclass == "PR":
        if x == 0.:
            return 0., 0.
        else:
            if x < 0.1:
                S1 = -(x**2/4 + x**4/36 + x**6/960 + x**8/50400)*math.exp(-x)
                S2 = (x**2/2 + x**4/24 + x**6/720 + x**8/40320)*math.exp(-x)
            else:
                S1 = -((x**2+2)*0.5*(1+math.exp(-2*x))-x*(-math.expm1(-2*x))-2*math.exp(-x))/x**2
                S2 = 0.5*(1+math.exp(-2*x))-math.exp(-x)
            return S1, S2

    else:
        assert(False)
        return 0., 0.


@njit("float64(float64, float64)", cache=True)
def chi_back(nu, x):
    return nu**2/(math.sqrt(nu**2 + x**2) + x) + nu*math.log(x/(nu + math.sqrt(nu**2 + x**2)))


@njit("float64(float64, int64, float64[:], float64[:])", cache=True)
def S_back(x, lmax, mie_a, mie_b):
    r"""Mie scattering amplitudes for plane waves in the backward scattering limit.

    Parameters
    ----------
    x : float
        positive, imaginary frequency
    lmax : int
        positive, cut-off angular momentum
    mie_a : list
        list of mie coefficients for electric polarizations
    mie_b : list
        list of mie coefficients for magnetic polarizations
    
    Returns
    -------
    float
        (:math:`\tilde S`)
    """
    err = 1.0e-16

    l = 1
    exp = math.exp(2*chi_back(l+0.5, x))
    S = (l+0.5)*(mie_a[l-1] + mie_b[l-1])*exp
    l += 1
    while(l <= lmax):
        exp = math.exp(2*chi_back(l+0.5, x))
        Sterm = (l+0.5)*(mie_a[l-1] + mie_b[l-1])*exp
        if Sterm/S < err:
            S += Sterm
            break
        S += Sterm
        l += 1
    return S


@njit("UniTuple(float64, 2)(float64, float64, float64)", cache=True)
def S1S2_asymptotics(x, z, n):
    r"""Asymptotic expansion of the scattering amplitudes for large size
    parameter :math:`x`.

    The implementation evaluates the first two terms of the expansion and is
    valid even when :math:`z=1`.

    Parameters
    ----------
    x : float
        positive, imaginary frequency
    z : float
        positive, :math:`z=-\cos \Theta`
    n : float
        positive, refractive index (may be infinite)
    
    Returns
    -------
    (float, float)
        (:math:`\tilde S_1`, :math:`\tilde S_2`)

    """
    s = sqrt((1+z)/2)
    
    s1_PR = 0.5*(1-2*s**2)/s**3
    s2_PR = -0.5/s**3

    if n == math.inf:
        S1 = -0.5*x*(1 + s1_PR/x)
        S2 = 0.5*x*(1 + s2_PR/x)
        return S1, S2
    else:
        eps = n**2
        rTE = -(eps-1.)/(s + sqrt(eps-1. + s**2))**2
        rTM = ((eps-1.)*s - (eps-1.)/(s + sqrt(eps-1. + s**2)))/(eps*s + sqrt(eps-1 + s**2))
        S1wkb = 0.5*x*rTE
        S2wkb = 0.5*x*rTM

        c2 = 1 - s**2

        s1_diel1 = 1/s/(c2+s*sqrt(n**2-c2))
        s1_diel2 = -0.5*(2*n**2-c2)/(n**2-c2)**1.5
        s1_diel = (s1_PR + s1_diel1 + s1_diel2)/x

        s2_diel1 = 1/s/(c2-s*sqrt(n**2-c2))
        s2_diel2 = 0.5*n**2/(n**2-c2)**1.5*(2*n**4-n**2*c2*(1+c2)-c2**2)/(n**2*s**2-c2)**2
        s2_diel3 = -c2/s**3*(2*n**4*s**2-n**2*c2*(c2*s**2+1)+c2**3)/(n**2-c2)/(n**2*s**2-c2)**2
        s2_diel = (s2_PR + s2_diel1 + s2_diel2 + s2_diel3)/x

        S1 = S1wkb*(1. + s1_diel)
        S2 = S2wkb*(1. + s2_diel)
        return S1, S2


@njit("float64(int64, float64, float64, float64)", cache=True)
def chi(l, x, z, acoshz):
    nu = l + 0.5
    return nu*acoshz + 2*(math.sqrt(nu*nu + x*x) - nu*math.asinh(nu/x) - x*math.sqrt((1+z)/2))


@njit("UniTuple(float64, 2)(float64, float64, float64, int64, float64[:], float64[:], boolean)", cache=True)
def S1S2_finite(x, z, n, lmax, mie_a, mie_b, use_asymptotics):
    r"""Mie scattering amplitudes for plane waves at finite frequency/wavenumber.

    Parameters
    ----------
    x : float
        positive, imaginary size parameter
    z : float
        positive, :math:`z=-\cos \Theta`
    n : float
        positive, refractive index
    lmax : int
        positive, cut-off angular momentum
    mie_a, mie_b : list
        list of mie coefficients for electric and magnetic polarization
    use_asymptotics : boolean
        when True asymptotics are used for the
        scattering amplitude when x > 5000

    Returns
    -------
    (float, float)
        (:math:`\tilde S_1`, :math:`\tilde S_2`)
    
    """
    if x > 5.e3 and use_asymptotics:
        return S1S2_asymptotics(x, z, n)

    if z <= 1.:
        S = S_back(x, lmax, mie_a, mie_b)
        return -S, S
    
    err = 1.0e-16       # convergence
    dl = 1000           # chunks-size for pte
    
    # precompute frequently used values
    acoshz = math.acosh(z)
    
    # estimated l with main contribution to the sum
    l_est = int(math.ceil(x*math.sqrt(math.fabs(z-1)/2)))
    l_init = min(lmax, l_est)
    
    if l_init < 1000:
        # only upward summation starting from l=1
        pe, te = pte_array(1, l_init+dl, acoshz)
        exp = math.exp(chi(1, x, z, acoshz))
        S1 = (mie_a[0]*pe[0] + mie_b[0]*te[0])*exp
        S2 = (mie_a[0]*te[0] + mie_b[0]*pe[0])*exp

        # upwards summation
        l = 2
        i = 1
        while l <= lmax:
            if i >= len(pe):
                pe, te = pte_array(l, l+dl, acoshz)
                i = 0
            exp = math.exp(chi(l, x, z, acoshz))
            S1term = (mie_a[l-1]*pe[i] + mie_b[l-1]*te[i])*exp
            S2term = (mie_a[l-1]*te[i] + mie_b[l-1]*pe[i])*exp
            if S1 > 0.:
                if S1term/S1 < err:
                    S1 += S1term
                    S2 += S2term
                    break
            S1 += S1term
            S2 += S2term
            l += 1
            i += 1

        return -S1, S2
    else:
        # upward and downward summation starting at l_init_int
        pe, te = pte_array(l_init, l_init+dl, acoshz)
        exp = math.exp(chi(l_init, x, z, acoshz))
        S1 = (mie_a[l_init-1]*pe[0] + mie_b[l_init-1]*te[0])*exp
        S2 = (mie_a[l_init-1]*te[0] + mie_b[l_init-1]*pe[0])*exp
        if S1 == 0.:
            return 0., 0.

        # upwards summation
        l = l_init+1
        i = 1
        while l <= lmax:
            if i >= len(pe):
                pe, te = pte_array(l, l+dl, acoshz)
                i = 0
            exp = math.exp(chi(l, x, z, acoshz))
            S1term = (mie_a[l-1]*pe[i] + mie_b[l-1]*te[i])*exp
            S2term = (mie_a[l-1]*te[i] + mie_b[l-1]*pe[i])*exp
            if S1term/S1 < err:
                S1 += S1term
                S2 += S2term
                break
            S1 += S1term
            S2 += S2term
            l += 1
            i += 1
        
        # downwards summation
        l = l_init-1
        i = -1
        while(True):
            if i < 0:
                pe, te = pte_array(max(l-dl, 1), l, acoshz)
                i = len(pe)-1
            exp = math.exp(chi(l, x, z, acoshz))
            S1term = (mie_a[l-1]*pe[i] + mie_b[l-1]*te[i])*exp
            S2term = (mie_a[l-1]*te[i] + mie_b[l-1]*pe[i])*exp
            if S1term/S1 < err:
                S1 += S1term
                S2 += S2term
                break
            S1 += S1term
            S2 += S2term
            l -= 1
            i -= 1
            if l == 0:
                break
        return -S1, S2


if __name__ == "__main__":
    x = 0.001
    z = 32.622776601683796
    n = 1.1
    lmax = 100000000
    """
    from mie import mie_e_array
    mie_a, mie_b = mie_e_array(lmax, x, n)
    S1, S2 = S1S2(x, z, n, lmax, mie_a, mie_b, False)
    sigma = math.sqrt((1+z)/2)
    S1a, S2a = S1S2_asymptotics(x, z, n)
    print("compare to asymptotics")
    print(S1)
    print(S1a)
    print((S1-S1a)/S1a)
    print(S2)
    print(S2a)
    print((S2-S2a)/S2a)
    #width = math.sqrt(x*math.sqrt((1+z)/2))
    print("width", width)
    print("6*width", 6*width)
    #jit()(S1S2).inspect_types()
    """
    import matplotlib.pyplot as plt
    print(S1S2_zero(32599.3474702326, 2280.4788413404267, 12000, "plasma"))
    #X = np.logspace(-3, 7, 50)
    #Y1 = np.array([zero_frequency(x, 2280.4788413404267, lmax, "plasma")[0] for x in X])
    #Y2 = np.array([zero_frequency(x, 2280.4788413404267, lmax, "PR")[0] for x in X])
    #plt.loglog(X, -Y1)
    #plt.loglog(X, -Y2)
    #plt.show()

    #print(zero_frequency(x, n, lmax, "drude"))
