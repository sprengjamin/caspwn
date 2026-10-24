from scipy.constants import hbar, c
from scipy.constants import e as eV

materialclass = "plasma"
wp = 9*eV/hbar # plasma frequency
K_plasma = wp/c # plasma wave number

def epsilon(xi):
    return 1.+wp**2/xi**2
