from hypothesis import given
from hypothesis.strategies import floats
import sys
sys.path.append(".")
from fresnel import rTE, rTM
import numpy as np
from mpmath import *
mp.prec = 100
mp.dps = 100

def mp_fresnel(K, k, eps):
    K = mpf(K)
    k = mpf(k)
    eps = mpf(eps)
    
    kappa = sqrt(k**2 + K**2)
    num_TE = kappa - sqrt(kappa**2 + K**2*(eps-1))
    den_TE = kappa + sqrt(kappa**2 + K**2*(eps-1))

    num_TM = eps*kappa - sqrt(kappa**2 + K**2*(eps-1))
    den_TM = eps*kappa + sqrt(kappa**2 + K**2*(eps-1))
    return float(num_TE/den_TE), float(num_TM/den_TM)

rtol = 1.e-12
@given(K=floats(min_value=1.0e-8, max_value=1.0e8),
       k=floats(min_value=1.0e-8, max_value=1.0e8),
       e=floats(min_value=1.e-02, max_value=1.0e8))
def test_fresnel(K, k, e):
    my_TE = rTE(K, k, e)
    print(my_TE)
    my_TM = rTM(K, k, e)
    mp_TE, mp_TM = mp_fresnel(K, k, e)
    print(mp_TE)
    np.testing.assert_allclose(my_TE, mp_TE, rtol=rtol)
    np.testing.assert_allclose(my_TM, mp_TM, rtol=rtol)    