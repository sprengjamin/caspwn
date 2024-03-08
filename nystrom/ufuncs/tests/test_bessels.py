import numpy as np
from nystrom.ufuncs.bessel import InuKnu_e_asymptotics, Ine
import os

dir_path = os.path.dirname(__file__)

from mpmath import *

mp.dps = 40
maxterms = 1e6
def mp_Ine(l, x):
    """
    exponentially scaled modified Bessel function of first kind
    """
    return besseli(l, x, maxterms=maxterms)*exp(-x)

rtol = 1e-15
def test_InuKnu_e_asymptotics():
    InuKnue_data = np.loadtxt(os.path.join(dir_path,  "testdata/InuKnue.dat"))
    for data in InuKnue_data:
        num_Inue, num_Knue = InuKnu_e_asymptotics(data[0], data[1])
        print("l", data[0], "x", data[1])
        print(np.abs(num_Inue/data[2]-1.))
        print(np.abs(num_Knue/data[3]-1.))
        np.testing.assert_allclose(num_Inue, data[2], rtol=rtol)
        np.testing.assert_allclose(num_Knue, data[3], rtol=rtol)

def test_Ine():
    X = [9.120108393559096371e-04, 3.076096814740708063e-01, 1.037528415818012633e+02, 3.499451670283569547e+04,
         1.180320635651729815e+07, 3.981071705534969330e+09]
    L = np.arange(13, dtype=np.int64)
    for x in X:
        ref_Ine = [float(mp_Ine(l, x)) for l in L]
        num_Ine = Ine(12, x)
        print("x", "%.16f"%x)
        print(np.abs(num_Ine/ref_Ine-1.))
        np.testing.assert_allclose(num_Ine, ref_Ine, rtol=rtol)

if __name__ == "__main__":
    print(dir_path)
    test_Ine()

