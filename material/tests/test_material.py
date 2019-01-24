import numpy as np
import sys
sys.path.append(".")
import material

def test_compare_casspy():
    # test some of the permittivity with casspy implementation
    rtol = 1e-2
    materials = ["PS1", "PS2", "PTFE", "Silica1", "Water"]
    for mat in materials:
        casspy_data = np.loadtxt("tests/testdata/"+mat+".dat").T
        casspy_X = casspy_data[0]
        casspy_eps = casspy_data[1]
        print(mat)

        my_eps = [eval("material."+mat+".epsilon(x)") for x in casspy_X]
        np.testing.assert_allclose(my_eps, casspy_eps, rtol=rtol)
        
        # make an extra test for this later
        assert(np.all(np.greater(my_eps, 1.)))

if __name__ == "__main__":
    test_compare_casspy()
