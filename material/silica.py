import numpy as np
from scipy.constants import e as eV
from scipy.constants import hbar

materialclass = 'dielectric'
# (set 1)
data = np.array([[7.84e-1, 2.03e-1, 4.17e-1, 3.93e-1, 5.01e-2, 8.20e-1, 2.17e-1, 5.50e-2],
                 [4.11e-2, 1.12e-1, 1.12e-1, 1.11e-1, 1.45e+1, 1.70e+1, 8.14e+0, 9.16e+1]])

# paramters of oscillator model
C = data[0]
wi = data[1]*eV/hbar

def epsilon(xi):
    return 1 + np.sum(C/(1 + (xi/wi)**2))
