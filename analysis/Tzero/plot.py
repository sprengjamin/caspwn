import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("full_energy_PR_Vacuum_PR.dat").T
L = data[0]
en = data[1]

rho = 1/L
teo = 1 + (1/3 - 20/np.pi**2)*L
plt.loglog(L, np.fabs(en/(-np.pi**3*rho/720.)-1.))
plt.loglog(L, np.fabs((1/3 - 20/np.pi**2)*L))
plt.show()

