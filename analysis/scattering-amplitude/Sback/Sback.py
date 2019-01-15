"""
Plot for the scattering amplitude in the back-scattering limit.

"""
import numpy as np
import sys
sys.path.append("../../../ufuncs-sphere/")
from scattering_amplitude import S_back
from mie import mie_e_array
import matplotlib.pyplot as plt

x = 4000
z = 30
ale, ble = mie_e_array(1e5, x)
n=30
X = np.logspace(1, 3, n)
s = np.empty(n)
for i, x in enumerate(X):
    ale, ble = mie_e_array(1e5, x)
    s[i] = S_back(x, ale, ble)
f, ax = plt.subplots()
#ax.loglog(X, np.abs(s), "b.")
#ax.loglog(X, X/2, "k--")
#ax.loglog(X, 1.5*X**3, "k--")
ax.loglog(X, 0.5*X**-4, "k", label=r"$1/2 (R\xi/c)^{-4}$")
ax.loglog(X, np.abs(s/(0.5*X)-(1-0.5/X)), "b.", label=r"$S_{1/2}$")
#ax.loglog(X, X**-2, "k--")

ax.set_title(r"$\cos\Theta=-1$")
ax.set_xlabel(r"$R\xi/c$")
ax.set_ylabel(r"$S_p/S_{p,\mathrm{WKB}}-(1+s_p/R)$")

ax.legend()
#plt.savefig("fig5.pdf")
plt.show()
