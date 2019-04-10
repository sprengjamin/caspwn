import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("all_roundtrips.dat").T

L = data[0,:]
for i in range(10):
    if i == 0:
        continue
    r = i+1
    trM = data[i+1,:]/r
    pfa = 1/L/(16*np.pi)/r**4
    corr = (r**2-1)/3
    ## 1st corr
    #plt.loglog(L, np.abs(trM/pfa-1.), label=str(r))
    #plt.loglog(L, (r**2-1)/3*L, "k--")

    ## 2nd corr
    plt.loglog(L, np.abs(trM/pfa-1.+(r**2-1)/3*L), label=str(r))

plt.loglog(L, L**2, "k--")
plt.loglog(L, 17/3*L**2, "k--")
plt.loglog(L, 20*L**2, "k--")
plt.loglog(L, 52*L**2, "k--")
plt.loglog(L, 110*L**2, "k--")
plt.loglog(L, 200*L**2, "k--")
plt.loglog(L, 10*L**1.5, "k-")

plt.legend()
plt.savefig("ten_roundtrips.pdf")
plt.show()
