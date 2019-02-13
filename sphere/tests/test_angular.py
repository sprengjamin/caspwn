import numpy as np
import sys
sys.path.append(".")
import os
from angular import pte_asymptotics
from angular import _c1, _c2, _c3, _c4, _c5

from mpmath import *
mp.dps = 80
maxterms = 1e6
def mp_pte(l, x):
    z = cosh(x)
    pe = (2*l+1)/(l*(l+1))*legenp(l, 1, z, type=3, maxterms=maxterms)/sinh(x)*exp(-(l+1/2)*x) 
    te = -z*pe + (2*l+1)*legendre(l, z, maxterms=maxterms)*exp(-(l+1/2)*x)
    return pe, te

rtol=1.0e-15

def test_pte_low():
    lValues = np.floor(np.logspace(3.003, 9.08, 11))
    for l in lValues:
        xmax = np.arcsinh(25/(l+1))
        X = np.empty(3)
        # create random data between 0. and xmax
        X = np.random.random(3)*xmax
        for x in X:
            mp_pe, mp_te = mp_pte(mpf(l), mpf(x))
            num_pe, num_te = pte_asymptotics(l, x)
            print("l", l, "x", x)
            print("pe prec","%.16e"%(num_pe/mp_pe-1.))
            print("pe","%.16e"%num_pe)
            print("mp","%.16e"%mp_pe)
            print("te prec","%.16e"%(num_te/mp_te-1.))
            print("te","%.16e"%num_te)
            print("mp","%.16e"%mp_te)
            np.testing.assert_allclose(num_pe, float(mp_pe), rtol=rtol)
            np.testing.assert_allclose(num_te, float(mp_te), rtol=rtol)

rtol = 1e-15
def test_pte_high():
    InuKnue_data = np.loadtxt("tests/testdata/pte_high.dat")
    for data in InuKnue_data:
        num_pe, num_te = pte_asymptotics(data[0], data[1])
        print("l", data[0], "x", data[1])
        print(np.abs(num_pe/data[2]-1.))
        print(np.abs(num_te/data[3]-1.))
        np.testing.assert_allclose(num_pe, data[2], rtol=rtol)
        np.testing.assert_allclose(num_te, data[3], rtol=rtol)

def mp_c1(x):
    return (1 - x*coth(x))/(8.*x**2)

def mp_c2(x):
    return (21 + 8*x**2 - 18*x*coth(x) - 3*x**2*coth(x)**2)/(384.*x**4)

def mp_c3(x):
    return (99 + 40*x**2 - 81*x*coth(x) - 15*x**2*coth(x)**2 - 3*x**3*coth(x)**3)/(3072.*x**6) 

def mp_c4(x):
    return (32175 + 13200*x**2 + 64*x**4 - 25740*x*coth(x) + 30*x**2*(-165 + 8*x**2)*coth(x)**2 - 1260*x**3*coth(x)**3 - 225*x**4*coth(x)**4)/(1.47456e6*x**8) 

def mp_c5(x):
    return (62985 + 26000*x**2 + 192*x**4 - x*(49725 + 64*x**4)*coth(x) + 30*x**2*(-325 + 24*x**2)*coth(x)**2 + 10*x**3*(-273 + 16*x**2)*coth(x)**3 - 675*x**4*coth(x)**4 - 105*x**5*coth(x)**5)/(3.93216e6*x**10) 

rtol = 1.e-14
def test_c_coefficients():
    X = np. logspace(-6, 3, 50)
    for x in X:
        print(x)
        # allow less precision for higher coefficients since contribution decreases
        np.testing.assert_allclose(_c1(x), float(mp_c1(mpf(x))), rtol=1.e-15, atol=0.1)
        np.testing.assert_allclose(_c2(x), float(mp_c2(mpf(x))), rtol=1.e-13, atol=0.1)
        print(_c3(x))
        print(float(mp_c3(mpf(x))))
        np.testing.assert_allclose(_c3(x), float(mp_c3(mpf(x))), rtol=1.e-11)
        np.testing.assert_allclose(_c4(x), float(mp_c4(mpf(x))), rtol=1.e-9)
        np.testing.assert_allclose(_c5(x), float(mp_c5(mpf(x))), rtol=1.e-7)


if __name__ == "__main__":
    #test_pte_low()
    #test_pte_high()
    test_c_coefficients()
    """
    #print(mp_c1(mpf(0.001)))
    import matplotlib.pyplot as plt
    x = np.arccosh(1.000001)
    N = 40
    L = np.floor(np.logspace(1, 4, 40))
    pe = np.empty(N)
    te = np.empty(N)
    for i, l in enumerate(L):
        print(i, l)
        mp_pe, mp_te = mp_pte(mpf(l), mpf(x))
        pe[i], te[i] = pte_asymptotics(l, x)
        pe[i] = np.abs(pe[i]/mp_pe-1.)
        te[i] = np.abs(te[i]/mp_te-1.)

    f, ax = plt.subplots()
    ax.loglog(L, pe)
    ax.loglog(L, te)
    plt.show()
    """
