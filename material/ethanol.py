import numpy as np
from scipy.constants import e as eV
from scipy.constants import hbar

materialclass = 'dielectric'
data = [[9.57e-1, 1.62e+0, 1.40e-1, 1.26e-1, 4.16e-1, 2.44e-1, 7.10e-2],
        [1.62e-3, 4.32e-3, 1.12e-1, 6.87e+0, 1.52e+1, 1.56e+1, 4.38e+1]]
data = np.array(data)

# paramters of oscillator model
C = data[0]
wi = data[1]*eV/hbar

def epsilon(xi):
    return 1 + np.sum(C/(1 + (xi/wi)**2))
