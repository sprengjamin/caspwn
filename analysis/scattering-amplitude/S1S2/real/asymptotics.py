"""Script to check the numerical error made if the scattering amplitudes where
replaced by their asymptotic expansion with the first two terms.
The discussion is extended to dielectrics.

Results
-------

n = 10:
for x>5e2 the error is less than 1e-06
for x>5e3 the error is less than 1e-08

n = 1.1:
slightly smaller errors than above


"""
import numpy as np
from numpy import sqrt
import matplotlib.pyplot as plt
import sys
sys.path.append("../../../../sphere/")
from mie import mie_cache
from scattering_amplitude import S1S2

n = 1.1

X = np.array([1e2, 5e2, 1e3, 5e3, 1e4])
#X = np.array([5e2, 1e3, 5e3])
Z = 1. + np. logspace(-2, 2, 30)

z, x = np.meshgrid(Z, X)

eps = n**2
s = np.sqrt((1+z)/2)
rTE = -(eps-1.)/(s + np.sqrt(eps-1. + s**2))**2
rTM = ((eps-1.)*s - (eps-1.)/(s + np.sqrt(eps-1. + s**2)))/(eps*s + np.sqrt(eps-1 + s**2))
S1wkb = 0.5*x*rTE
S2wkb = 0.5*x*rTM

c2 = 1 - s**2

s1_1 = 0.5*(1-2*s**2)/s**3
s1_2 = 1/s/(c2+s*sqrt(n**2-c2))
s1_3 = -0.5*(2*n**2-c2)/(n**2-c2)**1.5
s1 = (s1_1 + s1_2 + s1_3)/x

s2_1 = -0.5/s**3
s2_2 = 1/s/(c2-s*sqrt(n**2-c2))
s2_3 = 0.5*n**2/(n**2-c2)**1.5*(2*n**4-n**2*c2*(1+c2)-c2**2)/(n**2*s**2-c2)**2
s2_4 = -c2/s**3*(2*n**4*s**2-n**2*c2*(c2*s**2+1)+c2**3)/(n**2-c2)/(n**2*s**2-c2)**2
s2 = (s2_1 + s2_2 + s2_3 + s2_4)/x

S1a = S1wkb*(1. + s1)
S2a = S2wkb*(1. + s2)

data1 = np.empty((len(X), len(Z)))
data2 = np.empty((len(X), len(Z)))

for i, x in enumerate(X):
    for j, z in enumerate(Z):
        print(x, z)
        mie = mie_cache(1e4, x, n)
        data1[i, j], data2[i, j] = S1S2(x, z, mie, False)

f, (ax1, ax2) = plt.subplots(2)
ax1.imshow(np.log10(np.fabs((data1-S1a)/S1wkb)))
ax2.imshow(np.log10(np.fabs((data2-S2a)/S2wkb)))
plt.show()
