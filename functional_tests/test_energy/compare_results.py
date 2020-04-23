import numpy as np
from scipy.constants import hbar, c

for i in range(1, 21):
    I = "%02d" % (i,)
    ref = np.loadtxt("reference_data/nystrom/data"+I+".out")
    new = np.loadtxt("test_data/data"+I+".out")
    try:
        new[0] = new[0]+new[1]
    except IndexError:
        pass

    np.testing.assert_allclose(new, ref, rtol=1e-06, err_msg="Problem occured in data"+I+".out")

for i, j in enumerate(range(21, 25)):
    I = "%02d" % (i+1,)
    ref = np.loadtxt("reference_data/caps/data"+I+".out", delimiter=",")
    L = ref[1]
    R = ref[2]
    E_ref = ref[5]*hbar*c/(L+R)
    new = np.loadtxt("test_data/data"+str(j)+".out")
    E_new = new[0]+new[1]
    np.testing.assert_allclose(E_new, E_ref, rtol=5e-06, err_msg="Problem occured in data"+str(j)+".out")

print("All tests successful!")

    
