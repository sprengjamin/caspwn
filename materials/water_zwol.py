import numpy as np
from scipy.constants import e as eV
from scipy.constants import hbar

materialclass = 'dielectric'
data = np.array([[1.43e+0, 9.74e+0, 2.16e+0, 5.32e-1, 3.89e-1, 2.65e-1, 1.36e-1],
                 [2.29e-2, 8.77e-4, 4.93e-3, 1.03e-1, 9.50e+0, 2.09e+1, 2.64e+1]])
static_value = 78.7

# paramters of oscillator model
C = data[0]
wi = data[1]*eV/hbar

def epsilon(xi):
    if xi == 0.:
        return static_value
    else:
        return 1 + np.sum(C/(1 + (xi/wi)**2))
