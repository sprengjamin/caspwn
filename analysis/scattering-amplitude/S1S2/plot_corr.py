import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../../ufuncs-sphere')

from scattering_amplitude import S1S2
from mie import mie_e_array

N = 50
X = np.logspace(-2,3,N)
z = 1e6
#z = 1.01

S1 = np.empty(N)
S2 = np.empty(N)

sigma = np.sqrt((1+z)/2)
S1a = 0.5*X*(1+((1-2*sigma**2)/(2*sigma**3))/X)
S2a = 0.5*X*(1+(-1/(2*sigma**3))/X)

for i, x in enumerate(X):
    print(x)
    ale, ble = mie_e_array(int(x*np.sqrt(z))+1e2, x)
    S1[i], S2[i] = S1S2(x, z, ale, ble)

f, ax = plt.subplots()
ax.loglog(X, 1/z*X**-2, "k", label=r"$0.01(R\xi/c)^{-2}$") 
#ax.loglog(X, 0.5*X**-4, "k", label=r"$0.01(R\xi/c)^{-2}$") 
ax.loglog(X, np.abs(S1-S1a)/(0.5*X), "r.", label=r"$S_1$")
ax.loglog(X, np.abs(S2-S2a)/(0.5*X), "b.", label=r"$S_2$")

ax.set_title(r"$\cos\Theta=-$"+str(z))
ax.set_xlabel(r"$R\xi/c$")
ax.set_ylabel(r"$S_p/S_{p,\mathrm{WKB}}-(1+s_p/R)$")
ax.legend()
plt.show()



