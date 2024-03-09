"""Generate permittivity data from casspy which serves as test data.

"""


import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("/home/benjamin/wd/casppy/casppy1/casppy")
import materials

# names: 1st entry: umrath name
#        2nd entry: my name
names = [("Ptfe", "PTFE"),
         ("Silicon", "Silica1"),
         ("Altern1Ps", "PS1"),
         ("Altern2Ps", "PS2"),
         ("Water", "Water")]

X = np.logspace(10, 18, 100)
for n in names:
    umrath_name, my_name = n
    eps = eval("materials."+umrath_name+".epsilon(X/2.998e14)")
    header = "frequency in rad/s       permittivity"
    np.savetxt(my_name+".dat", np.vstack((X, eps)).T, header=header)
