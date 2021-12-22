import numpy as np
from scipy.constants import e as eV
from scipy.constants import hbar

materialclass = 'dielectric'
# from Zwol
data = [[9.30e-3, 1.83e-2, 1.39e-1, 1.12e-1, 1.95e-1, 4.38e-1, 1.06e-1, 3.86e-2],
        [3.00e-4, 7.60e-3, 5.57e-2, 1.26e-1, 6.71e+0, 1.86e+1, 4.21e+1, 7.76e+1]]
data = np.array(data)

# parameters of oscillator model
C = data[0]
wi = data[1]*eV/hbar

def epsilon(xi):
    return 1 + np.sum(C/(1 + (xi/wi)**2))

