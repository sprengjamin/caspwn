import numpy as np
from scipy.constants import e as eV
from scipy.constants import hbar

materialclass = 'dielectric'
# (set 1, 2008 from Zwol)
data = np.array([[1.21e-2, 2.19e-2, 1.79e-2, 3.06e-2, 3.03e-1, 6.23e-1, 3.25e-1, 3.31e-2],
                 [1.00e-3, 1.32e-2, 3.88e+0, 1.31e-1, 5.99e+0, 1.02e+1, 1.88e+1, 5.15e+1]])

# parameters of oscillator model
C = data[0]
wi = data[1]*eV/hbar

def epsilon(xi):
    return 1 + np.sum(C/(1 + (xi/wi)**2))

