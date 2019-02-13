"""We compare the asymptotics of the Mie coefficients
with the analytical expressions.

When n is chosen large, the Mie coefficients
go over to the expressions known from the perfect reflector limit.

To see this for the Mie coefficient b, n needs to be large compared
to some function of l and x.

"""
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("/home/benjamin/wd/nystrom/sphere")
sys.path.append("/home/benjamin/wd/nystrom/ufuncs")
from mie import mie_e_array_mat, mie_e_array_PR

lmax = 1e5
x = 1000.
n = 100.

# numerical
ale, ble = mie_e_array_mat(lmax, x, n)
ale_PR, ble_PR = mie_e_array_PR(lmax, x)

# asymptotics
l = np.arange(1, lmax+1)
L = (l+0.5)/x   # L = Lambda

a = 0.5*(n**2*np.sqrt(1+L**2) - np.sqrt(n**2+L**2))/(n**2*np.sqrt(1+L**2) + np.sqrt(n**2+L**2))
b = 0.5*(np.sqrt(n**2+L**2) - np.sqrt(1+L**2))/(np.sqrt(n**2+L**2) + np.sqrt(1+L**2))
#b = 0.5*(n**2-1)/(np.sqrt(n**2+L**2) + np.sqrt(1+L**2))**2

# a correction
ac_t1 = 0.25/np.sqrt(1+L**2)
ac_t2 = 7/12.*L**2/(1+L**2)**1.5 

ac_t3 = L**2/((n**2-1)*(n**2+(n**2+1)*L**2))*((n**2+L**2)/(1+L**2)**1.5 - (n**2*np.sqrt(1+L**2))/(n**2+L**2))
ac = (ac_t1 + ac_t2 + ac_t3)/x

# b correction
bc_t1 = 0.25/np.sqrt(1+L**2)
bc_t2 = -5/12.*L**2/(1+L**2)**1.5 
bc_t3 = L**2/(n**2-1)*(np.sqrt(1+L**2)/(n**2+L**2) - 1/np.sqrt(1+L**2))

bc = (bc_t1 + bc_t2 + bc_t3)/x
plt.loglog(l, np.fabs(ale/a-1.), "r-", label="a")
#plt.loglog(l, np.fabs(ale_PR/0.5-1.))
#plt.loglog(l, np.fabs(ble_PR/0.5-1.))
plt.loglog(l, np.fabs(ac), "k--", label="a corr")
plt.loglog(l, np.fabs(ble/b-1.), "b-", label="b")
plt.loglog(l, np.fabs(bc), "k:", label="b corr")

plt.legend()
plt.show()
