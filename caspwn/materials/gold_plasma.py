from scipy.constants import hbar
from scipy.constants import e as eV

materialclass = "plasma"
wp = 9*eV/hbar # plasma frequency

def epsilon(xi):
    return 1.+wp**2/xi**2
