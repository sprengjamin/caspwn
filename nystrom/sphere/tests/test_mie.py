import numpy as np
from nystrom.sphere.mie import mie_e
from nystrom.ufuncs.bessel import InuKnu_e_asymptotics
from mpmath import *
import os

dir_path = os.path.dirname(__file__)

mp.dps = 80
maxterms = 1e6

rtol = 2.0e-15

def nueta(nu, x):
    """
    nu*eta exponent of bessels
    """
    return sqrt(nu*nu+x*x) + nu*log(x/(nu+sqrt(nu*nu+x*x)))

def mp_mie_e(l, x):
    """
    Exponentially scaled Mie coefficients for perfect reflectors.

    """
    bim = besseli(l-1/2, x, maxterms=maxterms)
    bip = besseli(l+1/2, x, maxterms=maxterms)
    bkm = besselk(l-1/2, x, maxterms=maxterms)
    bkp = besselk(l+1/2, x, maxterms=maxterms)
    efu = exp(-2*nueta(l+1/2, x))
    ale = pi/2*(x*bim - l*bip)/(x*bkm + l*bkp)*efu
    ble = pi/2*bip/bkp*efu
    return ale, ble

def mp_mie_e_mat(l, x, n):
    """
    Exponentially scaled Mie coefficients for real materials.

    """
    l = mpf(l)
    x = mpf(x)
    n = mpf(n)

    i_m_x = besseli(l-1/2, x, maxterms=maxterms)
    i_p_x = besseli(l+1/2, x, maxterms=maxterms)
    i_m_nx = besseli(l-1/2, n*x, maxterms=maxterms)
    i_p_nx = besseli(l+1/2, n*x, maxterms=maxterms)
    k_m_x = besselk(l-1/2, x, maxterms=maxterms)
    k_p_x = besselk(l+1/2, x, maxterms=maxterms)

    sa = i_p_nx*(x*i_m_x - l*i_p_x)
    sb = i_p_x*(n*x*i_m_nx - l*i_p_nx)
    sc = i_p_nx*(x*k_m_x + l*k_p_x)
    sd = k_p_x*(n*x*i_m_nx - l*i_p_nx)
    
    efu = exp(-2*nueta(l+1/2, x))
    
    ale = pi/2*(n**2*sa - sb)/(n**2*sc + sd)*efu
    ble = pi/2*(sb - sa)/(sc + sd)*efu
    return float(ale), float(ble)


def test_mie_e():
    mp_data = np.loadtxt(os.path.join(dir_path,  "testdata/mie_e.dat"))
    for data in mp_data:
        l = data[0]
        x = data[1]
        inum, knum = InuKnu_e_asymptotics(l-0.5, x)
        inup, knup = InuKnu_e_asymptotics(l+0.5, x)
        num_ale, num_ble = mie_e(data[0], data[1], inum, knum, inup, knup)
        print("l", data[0], "x","%.16e"% data[1])
        print(np.abs(num_ale/data[2]-1.))
        print(np.abs(num_ble/data[3]-1.))
        np.testing.assert_allclose(num_ale, data[2], rtol=rtol)
        np.testing.assert_allclose(num_ble, data[3], rtol=rtol)


if __name__ == "__main__":
    test_mie_cache()