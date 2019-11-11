import numpy as np
from scipy.constants import hbar, c

for i in range(1, 21):
    if i > 4 and i < 9:
        continue
    I = "%02d" % (i,)
    ref = np.loadtxt("reference_data/nystrom/data"+I+".out")
    new = np.loadtxt("test_data/data"+I+".out")
    if i > 8 and i < 13:
        ref *= hbar*c
        if i == 9:
            ref /= 1.e-06
        elif i == 10:
            ref /= 0.25e-06
        elif i == 11:
            ref /= 3.e-06
        elif i == 12:
            ref /= 0.75e-06
    np.testing.assert_allclose(new, ref, rtol=1e-06, err_msg="Problem occured in data"+I+".out")

for i, j in enumerate(range(21, 25)):
    I = "%02d" % (i+1,)
    ref = np.loadtxt("reference_data/caps/data"+I+".out", delimiter=",")
    L = ref[1]
    R = ref[2]
    E_ref = ref[5]*hbar*c/(L+R)
    new = np.loadtxt("test_data/data"+str(j)+".out")
    E_new = new[0]
    np.testing.assert_allclose(E_new, E_ref, rtol=5e-06, err_msg="Problem occured in data"+str(j)+".out")

print("All tests successful!")

    
