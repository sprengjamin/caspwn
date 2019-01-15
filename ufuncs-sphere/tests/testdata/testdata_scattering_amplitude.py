import numpy as np
import os, sys
sys.path.append("../")
sys.path.append("../../") # to prevent crash
from test_scattering_amplitude import mp_S1S2

X = np.logspace(-2, 2, 6)
Z = 1+np.logspace(-2, 2, 6)

if not os.path.isfile("scattering_amplitude.dat"):
    for x in X:
        for z in Z:
            print()
            print(x, z)
            data = np.empty((1, 5),dtype=np.float)
            data[0,0] = x
            data[0,1] = z
            data[0,2], data[0,3], data[0,4] = mp_S1S2(x, z)
            f=open("scattering_amplitude.dat", "ab")
            np.savetxt(f, data)
            f.close()
