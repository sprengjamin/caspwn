import numpy as np
from nystrom.sphere.scattering_amplitudes import S1S2_zero
from mpmath import *
import mpmath as mp
mpf.dps = 100
mp.mp.dps = 100
mp.dps = 100

def mp_PR(x):
    S1 = - (x**2*1/2*(1 + exp(-2*x)) - x*(-expm1(-2*x)) + (1 + exp(-2*x)) - 2*exp(-x))/x**2
    S2 = (cosh(x) - 1.)*exp(-x)
    return S1, S2

def test_PR():
    X = np.logspace(-5, 5, 100)
    for x in X:
        S1ref, S2ref = mp_PR(mpf(x))
        S1, S2 = S1S2_zero(x, -1., -1., 'PEC')
        print('x', x)
        print('rel S1', abs(S1/float(S1ref)-1))
        print('rel S2', abs(S2/float(S2ref)-1))
        np.testing.assert_allclose(S1, float(S1ref), rtol=1.e-11)
        np.testing.assert_allclose(S2, float(S2ref), rtol=1.e-11)

if __name__ == '__main__':
    test_PR()



