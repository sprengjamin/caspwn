import numpy as np
from hypothesis import given
from hypothesis.strategies import floats
import sys
sys.path.append(".")
import os
#from kernel import phase

from mpmath import *
from kernels import phase
mp.dps = 80

def mp_phase(rho, xi, k1, k2, phi):
    rho = mpf(rho)
    xi = mpf(xi)
    k1 = mpf(k1)
    k2 = mpf(k2)
    phi = mpf(phi)
    kappa1 = sqrt(k1**2 + xi**2)
    kappa2 = sqrt(k2**2 + xi**2)
    z = (kappa1*kappa2 + k1*k2*cos(phi))/xi**2 
    return float(2*rho*xi*sqrt((1+z)/2) - (1+rho)*(kappa1 + kappa2))

rtol = 1.e-15
@given(rho=floats(min_value=1.0e-6, max_value=1.0e6),
       xi=floats(min_value=1.0e-8, max_value=1.0e8),
       k1=floats(min_value=1.0e-8, max_value=1.0e8),
       k2=floats(min_value=1.0e-8, max_value=1.0e8),
       phi=floats(min_value=0., max_value=3.141278494324434))
def test_phase(rho, xi, k1, k2, phi):
    result = phase(rho, 1., xi, k1, k2, phi)
    if result < -100.:
        return
    exact = mp_phase(rho, xi, k1, k2, phi)
    #print()
    #print(result)
    #print(exact)
    #print(abs(result/exact-1.))
    np.testing.assert_allclose(exact, result, rtol=rtol, atol=1.)
    
if __name__ == "__main__":
    test_phase()
