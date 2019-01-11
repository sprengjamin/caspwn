import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../../ufuncs-sphere')

from scattering_amplitude import S1S2
from mie import mie_e_array

N = 50
X = np.logspace(-3,3,N)
z = 1.001
#z = 1.01

S1 = np.empty(N)
S2 = np.empty(N)

sigma = np.sqrt((1+z)/2)
#S1a = 0.5*X*(1+((1-2*sigma**2)/(2*sigma**3))/X)
#S2a = 0.5*X*(1+(-1/(2*sigma**3))/X)
S1a = 0.5*X*(1)
S2a = 0.5*X*(1)

for i, x in enumerate(X):
    print(x)
    ale, ble = mie_e_array(1e5, x)
    S1[i], S2[i] = S1S2(x, z, ale, ble)

f, ax = plt.subplots()
#ax.loglog(X, np.abs(S1-S1a)/(0.5*X), "ro", label=r"$S_1$")
#ax.loglog(X, np.abs(S2-S2a)/(0.5*X), "bo", label=r"$S_2$")
ax.loglog(X, (2+z)/2*X**3, "k--") 
ax.loglog(X, (1+2*z)/2*X**3, "k--", label=r"$\sim(R\xi/c)^3$") 
ax.loglog(X, 0.5*X**1, "k", label=r"$\sim(R\xi/c)$") 
ax.loglog(X, S1, "r.", label=r"$S_1$")
ax.loglog(X, S2, "b.", label=r"$S_2$")
#ax.loglog(X, 1/(2*sigma**3*X), "k--", label=r"$(R\xi/c)^{-2}$") 
#ax.loglog(X, 300*X**-0.5, "k:") 

ax.set_title(r"$\cos\Theta=-$"+str(z))
ax.set_xlabel(r"$R\xi/c$")
ax.set_ylabel(r"$\left\vert S_p \right\vert/\exp(2(\xi R/c)\sin(\Theta/2))$")
ax.set_ylim((0.9*min(S1[0], S2[0]), 1.1*max(S1[-1], S2[-1])))
ax.legend(loc=4)
plt.savefig("dipol-WKB2.pdf")
