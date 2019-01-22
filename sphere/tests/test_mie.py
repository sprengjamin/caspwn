import numpy as np
import sys
sys.path.append(".")
from mie import mie_e, mie_cache
from bessel import InuKnu_e_asymptotics

rtol = 2.0e-15

def test_mie_e():
    mp_data = np.loadtxt("tests/testdata/mie_e.dat")
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

def test_mie_cache():
    mp_data = np.loadtxt("tests/testdata/mie_e.dat")
    for data in mp_data:
        l = int(data[0])
        x = data[1]
        mie = mie_cache(l, x)
        num_ale, num_ble = mie.read(l)
        #print("l", data[0], "x","%.16e"% data[1])
        #print(np.abs(num_ale/data[2]-1.))
        #print(np.abs(num_ble/data[3]-1.))
        np.testing.assert_allclose(num_ale, data[2], rtol=rtol)
        np.testing.assert_allclose(num_ble, data[3], rtol=rtol)

if __name__ == "__main__":
    test_mie_cache()
