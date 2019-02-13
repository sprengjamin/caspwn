import numpy as np
from numpy import sqrt
import matplotlib.pyplot as plt
import sys
sys.path.append('../../../../sphere')
sys.path.append('../../../../ufuncs')

from scattering_amplitude import S1S2
from mie import mie_cache

N = 50
X = np.logspace(1,4,N)
z = 3.4
n = 1.4
#z = 1.01

# problematic values:
#X = np.logspace(1,5,N)
#z = 1.000001

S1 = np.empty(N)
S2 = np.empty(N)

sigma = np.sqrt((1+z)/2)
eps = n**2
rTE = -(eps-1.)/(sigma + np.sqrt(eps-1. + sigma**2))**2
rTM = ((eps-1.)*sigma - (eps-1.)/(sigma + np.sqrt(eps-1. + sigma**2)))/(eps*sigma + np.sqrt(eps-1 + sigma**2))
S1a = 0.5*X*rTE
S2a = 0.5*X*rTM

s = sigma
c2 = 1 - s**2

s1_1 = 0.5*(1-2*s**2)/s**3
s1_2 = 1/s/(c2+s*sqrt(n**2-c2))
s1_3 = -0.5*(2*n**2-c2)/(n**2-c2)**1.5
s1 = s1_1 + s1_2 + s1_3

s2_1 = -0.5/s**3
s2_2 = 1/s/(c2-s*sqrt(n**2-c2))
s2_3 = 0.5*n**2/(n**2-c2)**1.5*(2*n**4-n**2*c2*(1+c2)-c2**2)/(n**2*s**2-c2)**2
s2_4 = -c2/s**3*(2*n**4*s**2-n**2*c2*(c2*s**2+1)+c2**3)/(n**2-c2)/(n**2*s**2-c2)**2
s2 = s2_1 + s2_2 + s2_3 + s2_4


corr22 = -2*sqrt(2)/(sqrt(z+1)*(sqrt(z+1)*sqrt(2*n**2+z-1)+z-1))

corr23 = -sqrt(2)*n**2*(-8*n**4 - 2*n**2*(z - 1) + (n**2 + 1)*(z - 1)**2)/((2*n**2 + z - 1)**(3/2)*(n**2*(z + 1) + z - 1)**2) 
corr24 = -sqrt(2)*(z - 1)*(-8*n**4*(z + 1) + n**2*(z - 1)*(z**2 - 5) + (z - 1)**3)/((z + 1)**(3/2)*(2*n**2 + z - 1)*(n**2*(z + 1) + z - 1)**2)


#corr22 = -sqrt(2)*(4*n**2+z-1)/(2*(2*n**2+z-1)**1.5) + 2*sqrt(2)/(sqrt(z+1)*(sqrt(z+1)*sqrt(2*n**2+z-1)+1-z))
 
#corr2 = (corr21 + corr22+corr23+corr24)/X
for i, x in enumerate(X):
    print(x)
    mie = mie_cache(int(x)+100, x, n)
    S1[i], S2[i] = S1S2(x, z, mie)

f, ax = plt.subplots()
#ax.loglog(X, 0.01*X**-2, "k--", label=r"$0.01(R\xi/c)^{-2}$") 
#ax.loglog(X, X**(-1), "k", label=r"$0.01(R\xi/c)^{1}$") 
ax.loglog(X, np.fabs((S1-S1a*(1+s1/X))/(0.5*X*rTE)), "r.", label=r"$S_1$")
#ax.loglog(X, np.fabs(s1/X), "k--", label=r"$s_1$")
ax.loglog(X, np.fabs((S2-S2a*(1+s2/X))/(0.5*X*rTM)), "b.", label=r"$S_2$")
#ax.loglog(X, np.fabs(s2/X), "k:", label=r"$s_2$")
ax.loglog(X, 0.5/X**2, "k-", label=r"$0.01(R\xi/c)^{1}$") 

ax.set_title(r"$\cos\Theta=-$"+str(z))
ax.set_xlabel(r"$R\xi/c$")
ax.set_ylabel(r"$S_p/S_{p,\mathrm{WKB}}-(1+s_p/R)$")
ax.legend()
plt.show()
#plt.savefig("fig4.pdf")
print(((z-1)/2)**(-1/4))



