import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("energy3_PR_Vacuum_PR_v2.dat").T
L = data[0]
en = data[1]
R1 = 1
R2 = 3
Reff = R1*R2/(R1+R2)

corr = -L/(R1+R2) + (1/3 - 20/np.pi**2)*(L/R1+L/R2)
teo = 1 + corr
plt.loglog(L, np.fabs(en/(-Reff*np.pi**3/L/720.)-teo), label="data")
plt.loglog(L, 4.*L**1.5, label=r"$4L^{3/2}$")
plt.loglog(L, 50.*L**2, label=r"$50 L^2$")
plt.title(r"$R_1/R_2 = 3$")
plt.xlabel(r"$L$")
plt.ylabel(r"$(E-E_\mathrm{Teo})/E_\mathrm{PFA}$")
plt.legend()
plt.savefig("ratio3.pdf")
plt.show()

