import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("roundtrips.dat").T

L = data[0,:]
trM1 = data[1,:]/1
trM2 = data[2,:]/2
trM3 = data[3,:]/3

pfa1 = 1/L/(16*np.pi)
pfa2 = 1/L/(16*np.pi)/2**4
pfa3 = 1/L/(16*np.pi)/3**4

#plt.loglog(L, np.abs(trM1/pfa1-1.), label="1")
plt.loglog(L, np.abs(trM2/pfa2-1.+L), label="2")
plt.loglog(L, np.abs(trM3/pfa3-1.+8/3*L), label="3")
plt.loglog(L, L**2, "k--",label="L**2")
plt.loglog(L, 17/3*L**2, "k:",label="17/3L**2")

plt.legend()
plt.show()
