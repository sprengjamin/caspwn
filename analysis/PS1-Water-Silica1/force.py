import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../../sphere-sphere")
from PFA import PFA
data0 = np.loadtxt("energy_PS1_Water_Silica1_v3.dat").T
data1 = np.loadtxt("energy_PS1_Water_Silica1_v4.dat").T
#data2 = np.loadtxt("energy_PS1_Water_Silica1.dat").T
pfa2 = np.loadtxt("pfa.dat")

pfa = np.empty((2, len(data1[0])))
# compute PFA
for i, L in enumerate(data1[0]):
    R1 = 8.e-06
    R2 = 16.5e-06
    T = 293.015
    materials = ("PS1", "Water", "Silica1")
    pfa[0, i], pfa[1, i] = PFA(R1, R2, L, T, materials)

np.savetxt("my_pfa.dat", pfa)

pfa = np.loadtxt("my_pfa.dat")

R1 = 8.e-06
R2 = 16.5e-06
L = data1[0, 1:-1]
#L2 = data2[0, 1:-1]

#teo = -0.5*L/(R1+R2) + (1/6 - 10/np.pi**2)*L*(1/R1 + 1/R2)

full_force = np.gradient(data1[1], 1.e-09)
full_force2 = np.gradient(data1[2], 1.e-09)
rel = np.fabs(pfa[0]/full_force-1.)
rel2 = np.fabs(pfa[1]/full_force2-1.)

f, ax = plt.subplots()
ax.plot(data0[0], np.gradient(data0[2], 1.e-09))
ax.plot(data1[0], np.gradient(data1[2], 1.e-09))
#ax.loglog(L*1e6, rel[1:-1], "b-",label="with n=0")
#ax.loglog(L*1e6, rel2[1:-1], "g-",label="without")
#ax.loglog(L*1e6, 0.1*L*1e6, "k-",label="power")
#ax.loglog(L*1e6, np.fabs(teo), "r-",label="teo")
ax.set_title(r"Polystyrene ($8\mu m$) - Water - Silica ($16.5 \mu m$)")
ax.set_xlabel(r"$L$ $[\mu m]$")
ax.set_ylabel(r"$\vert F/F_\mathrm{PFA} - 1\vert$")
#plt.plot(data[0], pfa, label="pfa")
#plt.plot(data[0], pfa2, label="pfa2")
plt.legend()
plt.savefig("PFA-correction.pdf")
plt.show()

