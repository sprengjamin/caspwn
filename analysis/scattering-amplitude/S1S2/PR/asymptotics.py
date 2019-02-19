"""Script to check the numerical error made if the scattering amplitudes where
replaced by their asymptotic expansion with the first two terms. Only PR are
considered here.

Result:
for x>5e2 the error is less than 1e-06
for x>5e3 the error is less than 1e-08

"""


import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../../../../sphere/")
from mie import mie_cache
from scattering_amplitude import S1S2

X = np.array([1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5])
X = np.array([5e2, 1e3, 5e3, 1e4])
Z = 1. + np. logspace(-3, 3, 30)

z, x = np.meshgrid(Z, X)
S1wkb = -0.5*x
S2wkb = 0.5*x
s = np.sqrt((1+z)/2)
s1 = 0.5*(1-2*s**2)/s**3/x
s2 = -0.5/s**3/x
S1a = S1wkb*(1.+s1)
S2a = S2wkb*(1.+s2)

print(x[0,:])
print(x[:,0])
print(z)
data1 = np.empty((len(X), len(Z)))
data2 = np.empty((len(X), len(Z)))

for i, x in enumerate(X):
    for j, z in enumerate(Z):
        print(x, z)
        mie = mie_cache(1e4, x, np.inf)
        data1[i, j], data2[i, j] = S1S2(x, z, mie, False)

f, (ax1, ax2) = plt.subplots(2)
ax1.imshow(np.log10(np.fabs((data1-S1a)/S1wkb)))
ax2.imshow(np.log10(np.fabs((data2-S2a)/S2wkb)))
plt.show()
