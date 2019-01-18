import numpy as np
import sys
sys.path.append(".")
from bessel import InuKnu_e_asymptotics, Ine

rtol = 1e-15
def test_InuKnu_e_asymptotics():
    InuKnue_data = np.loadtxt("tests/testdata/InuKnue.dat")
    for data in InuKnue_data:
        num_Inue, num_Knue = InuKnu_e_asymptotics(data[0], data[1])
        print("l", data[0], "x", data[1])
        print(np.abs(num_Inue/data[2]-1.))
        print(np.abs(num_Knue/data[3]-1.))
        np.testing.assert_allclose(num_Inue, data[2], rtol=rtol)
        np.testing.assert_allclose(num_Knue, data[3], rtol=rtol)

def test_Ine():
    Ine_data = np.loadtxt("tests/testdata/Ine.dat")
    for data in Ine_data:
        num_Ine = Ine(12, data[0])
        print("x", "%.16f"%data[0])
        print(np.abs(num_Ine/data[1:]-1.))
        np.testing.assert_allclose(num_Ine, data[1:], rtol=rtol)

if __name__ == "__main__":
    test_Ine()

