"""
We analyze the numerical error in the function chi used for the scattering amplitudes.

"""
import numpy as np
import matplotlib.pyplot as plt
import sys
from mpmath import *
mp.dps = 50
sys.path.append('../../ufuncs-sphere')

from scattering_amplitude import chi

def mp_chi(l, x, z):
    l = mpf(l)
    x = mpf(x)
    z = mpf(z)
    nu = l+0.5
    return float(nu*acosh(z) + 2*sqrt(nu**2 + x**2) - 2*nu*asinh(nu/x) - 2*x*sqrt((1+z)/2))

def np_chi(l, x, z):
    # implementation of chi according to its definition
    nu = l + 0.5
    return nu*np.arccosh(z) + 2*np.sqrt(nu**2 + x**2) -  2*nu*np.arcsinh(nu/x) - 2*x*np.sqrt((1+z)/2)

Nx = 100
Nz = 100
X = np.logspace(-5, 5, Nx)
Z = 1. + np.logspace(-5, 5, Nz)
chi_result = np.empty((Nx, Nz))
chi_np = np.empty((Nx, Nz))
chi_expected = np.empty((Nx, Nz))

for i, x in enumerate(X):
    for j, z in enumerate(Z):
        lest = int(np.ceil(x*np.sqrt(abs(z-1)/2) - 0.5))
        print(x, z, lest)
        chi_result[i, j] = chi(lest, x, z)
        chi_np[i, j] = np_chi(lest, x, z)
        chi_expected[i, j] = mp_chi(lest, x, z)

f, (ax1, ax2, ax3) = plt.subplots(1, 3)

rel1 = np.log(np.fabs(chi_result/chi_expected - 1.))
rel2 = np.log(np.fabs(chi_np/chi_expected - 1.))

vmin = -35.#min(np.nanmin(rel1), np.nanmin(rel2))
print(vmin)
vmax = max(np.amax(rel1), np.amax(rel2))
print(vmax)

cplot1 = ax1.imshow(rel1, vmin=vmin, vmax=vmax, interpolation='None')
f.colorbar(cplot1, ax=ax1)

cplot2 = ax2.imshow(rel2, vmin=vmin, vmax=vmax, interpolation='None')
f.colorbar(cplot2, ax=ax2)

Z, X = np.meshgrid(Z, X)
Lest = np.log(np.ceil(X*np.sqrt(np.fabs(Z-1)/2)))
cplot3 = ax3.imshow(Lest)
f.colorbar(cplot3, ax=ax3)

plt.show()
    

