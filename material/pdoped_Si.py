"""
from Davids et al. PRA 82, 062111 (2010)
"""

from scipy.constants import hbar
from scipy.constants import e as eV

materialclass = "drude"
eps0 = 11.87
epsinf = 1.035
w0 = 6.6e15
wp = 3.6151e14 # plasma frequency
gamma = 7.868e13 # damping

def epsilon(xi):
    eps_Si = epsinf + (eps0-epsinf)*w0**2/(xi**2 + w0**2)
    return eps_Si + wp**2/(xi*(xi+gamma))
