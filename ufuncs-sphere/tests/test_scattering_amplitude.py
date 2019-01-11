import numpy as np
import sys
sys.path.append(".")
from mpmath import *
from scattering_amplitude import chi

x = 100.
z = 1.0001
lest = x*np.sqrt(np.abs(z-1)/2)
l = int(lest)+20
print(l, x, z, lest)
print(chi(l, x, z, lest))

def mp_chi(l, x, z):
    l = mpf(l)
    x = mpf(x)
    z = mpf(z)
    nu = l+1/2
    nueta = sqrt(nu**2 + x**2) + nu*log(x/(nu + sqrt(nu**2 + x**2)))
    return nu*acosh(z) + 2*nueta - 2*x*sqrt((1+z)/2)

print(float(mp_chi(l, x ,z)))

rtol = 1e-12



