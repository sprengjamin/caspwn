import numpy as np
from caspwn.sphere.angular import pte_asymptotics, pte_low, pte_next, pte_array
from caspwn.sphere.angular import _c1, _c2, _c3, _c4, _c5
from mpmath import *
import os

dir_path = os.path.dirname(__file__)

mp.dps = 80
maxterms = 1e5
def mp_pte(l, x):
    z = cosh(x)
    pe = (2*l+1)/(l*(l+1))*legenp(l, 1, z, type=3, maxterms=maxterms)/sinh(x)*exp(-(l+1/2)*x) 
    te = -z*pe + (2*l+1)*legendre(l, z, maxterms=maxterms)*exp(-(l+1/2)*x)
    return pe, te

rtol=1.0e-15

def test_pte_lowx():
    # test pte for high l, but low x values
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
def test_pte_highx():
    # test pte for high l and high x values
    InuKnue_data = np.loadtxt(os.path.join(dir_path,  "testdata/pte_high.dat"))
    for data in InuKnue_data:
        num_pe, num_te = pte_asymptotics(data[0], data[1])
        print("l", data[0], "x", data[1])
        print(np.abs(num_pe/data[2]-1.))
        print(np.abs(num_te/data[3]-1.))
        np.testing.assert_allclose(num_pe, data[2], rtol=rtol)
        np.testing.assert_allclose(num_te, data[3], rtol=rtol)


def test_pte_lowl():
    # test pte for low l values
    X = np.logspace(-3, 1, 10)
    for x in X:
        pe, te = pte_low(1000, x)
        L = [1, 10, 100, 1000]
        for l in L:
            rtol = 1e-13*l
            mp_pe, mp_te = mp_pte(mpf(l), mpf(x))
            num_pe, num_te = pe[l-1], te[l-1]
            print("l", l, "x", x)
            print("pe prec","%.16e"%(num_pe/mp_pe-1.))
            print("pe","%.16e"%num_pe)
            print("mp","%.16e"%mp_pe)
            print("te prec","%.16e"%(num_te/mp_te-1.))
            print("te","%.16e"%num_te)
            print("mp","%.16e"%mp_te)
            np.testing.assert_allclose(num_pe, float(mp_pe), rtol=rtol)
            np.testing.assert_allclose(num_te, float(mp_te), rtol=rtol)


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
        np.testing.assert_allclose(_c1(x)/float(mp_c1(mpf(x))), 1., rtol=1.e-15)
        np.testing.assert_allclose(_c2(x)/float(mp_c2(mpf(x))), 1., rtol=1.e-13)
        print(_c3(x))
        print(float(mp_c3(mpf(x))))
        np.testing.assert_allclose(_c3(x), float(mp_c3(mpf(x))), rtol=1.e-11)
        np.testing.assert_allclose(_c4(x), float(mp_c4(mpf(x))), rtol=1.e-9)
        np.testing.assert_allclose(_c5(x), float(mp_c5(mpf(x))), rtol=1.e-7)

def test_recurrence():
    rtol = 1.e-12
    L = [10, 100, 1000, 10000]#, 100000, 1000000]
    X = np.logspace(-3, 2, 20)
    for l in L:
        for x in X:
            mp_pe1, mp_te1 = mp_pte(mpf(l), mpf(x))
            mp_pe2, mp_te2 = mp_pte(mpf(l+1), mpf(x))
            mp_pe3, mp_te3 = mp_pte(mpf(l+2), mpf(x))
            z = np.cosh(x)
            emx = np.exp(-x)
            em2x = np.exp(-2*x)
            my_pe3, my_te3 = pte_next(l+1, x, z, emx, em2x, float(mp_pe2), float(mp_pe1))
            print("l:", l, "x:", x)
            print("pe prec:", np.abs(my_pe3/float(mp_pe3)-1.))
            print("te prec:", np.abs(my_te3/float(mp_te3)-1.))
            np.testing.assert_allclose(my_pe3, float(mp_pe3), rtol=rtol)
            np.testing.assert_allclose(my_te3, float(mp_te3), rtol=rtol)
    
def test_pte_array():
    rtol = 2.e-10
    Lmin = [10, 100, 989, 5679, 10000, 887766]
    dL = [1000, 1459, 2397]
    X = np.logspace(-3, 2, 5)
    for lmin in Lmin:
        for x in X:
            for dl in dL:
                pe, te = pte_array(lmin, lmin+dl, x)
                pe_a, te_a = pte_asymptotics(lmin+dl, x)
                print("lmin:", lmin, "dl:", dl, "x:", x)
                print("pe prec:", np.abs(pe[-1]/pe_a-1.))
                print("te prec:", np.abs(te[-1]/te_a-1.))
                np.testing.assert_allclose(pe[-1]/pe_a, 1., rtol = rtol)
                np.testing.assert_allclose(te[-1]/te_a, 1., rtol = rtol)


if __name__ == "__main__":
    test_recurrence()
    #test_pte_lowx()
    #test_pte_highx()
    #test_c_coefficients()
    #test_pte_lowl()
    #test_pte_array()