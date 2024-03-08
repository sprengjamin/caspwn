import numpy as np
from mpmath import *
from nystrom.ufuncs.legendre import Ple_high, Ple_low, gammafraction
import os

dir_path = os.path.dirname(__file__)

mp.dps = 40
maxterms = 1e6

rtol = 1.e-15
atol = 1.e-100

def mp_Ple(l, x):
    """
    exponentiall scaled legendre polynomials
    """
    return legendre(l, cosh(x), maxterms=maxterms)*exp(-(l+1/2)*x)

def test_consistency():
    """
    Ple_low and Ple_high have to agree for the overlap parameters
    """
    lValues = np.logspace(3.003, 9.08, 11).astype(int)
    for l in lValues:
        x = np.arcsinh(25/(l+1))
        #x = np.sqrt((1/(l+1))**2+1)
        print("l", l, "x","%.16e"%x)
        low = Ple_low(l, x)
        high = Ple_high(l, x)
        print("lowe", low)
        print("high", high)
        np.testing.assert_allclose(low, high, rtol=rtol)

def test_Ple_low():
    # test for x=0.
    lValues = np.logspace(3.003, 9.08, 11).astype(int)
    for l in lValues:
        low = Ple_low(l, 0.)
        #print("l", l, "result", low)
        np.testing.assert_allclose(low, 1., rtol=rtol)
    
    # arbitrary x:
    lValues = np.floor(np.logspace(3.003, 9.08, 11))
    for l in lValues:
        xmax = np.arcsinh(25/(l+1))
        # create random data between 0. and xmax
        xValues = np.random.random(3)*xmax
        for x in xValues:
            mp_ans = float(mp_Ple(mpf(l), mpf(x)))
            low = Ple_low(l, x)
            #print("mp ", mp_val)
            #print("low", low)
            np.testing.assert_allclose(low, mp_ans, rtol=rtol)

def test_Ple_high():
    lValues = np.floor(np.logspace(3.003, 9.08, 11))
    # low x values
    for l in lValues:
        xmin = np.arcsinh(25/(l+1))
        x = 10*xmin
        high = Ple_high(l, x)
        print("l", l, "x","%.16e"%x)
        print(high)
        mp_ans = float(mp_Ple(mpf(l), mpf(x)))
        print(mp_ans)
        np.testing.assert_allclose(high, mp_ans, rtol=rtol)
    # fixed x values
    print(" ")
    mp_data = np.loadtxt(os.path.join(dir_path, "testdata/Ple_high.dat"))
    for data in mp_data:
        high = Ple_high(data[0], data[1])
        print("l", data[0], "x","%.16e"%data[1])
        print("%.16e"%high)
        print("%.16e"%data[2])
        np.testing.assert_allclose(high, data[2], rtol=rtol, atol=atol)

def mp_gammafraction(l):
    return gamma(l+1)/gamma(l+3/2)

def test_gammafraction():
    lValues = np.floor(np.logspace(3.003, 9.08, 11))
    for l in lValues:
        mp_gf = float(mp_gammafraction(mpf(l)))
        num_gf = gammafraction(l)
        np.testing.assert_allclose(mp_gf, num_gf, rtol=rtol)

if __name__ == "__main__":
    test_Ple_high()
    #test_consistency()
