from scipy.constants import hbar
from scipy.constants import e as eV

materialclass = "drude"
wp = 9*eV/hbar # plasma frequency
gamma = 0.035*eV/hbar # damping

def epsilon(xi):
    return 1.+wp**2/xi/(xi+gamma)
