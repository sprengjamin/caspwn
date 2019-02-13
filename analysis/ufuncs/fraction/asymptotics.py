import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../../../ufuncs")
from bessel import fraction

x = 1000.

N = 40
L = np.asarray(np.ceil(np.logspace(1, 8, N)), dtype=np.int)
frac = np.empty(N)

for i, l in enumerate(L):
    frac[i] = fraction(l+0.5, x)

plt.loglog(L+0.5, np.fabs(frac/(2*L/x)-1.), "b.")
plt.loglog(L+0.5, 1/L, "k-")
#plt.loglog(L+0.5, 2*L/x, "k-")
plt.show()



