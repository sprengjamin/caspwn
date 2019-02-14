import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../../../plane-sphere")
import energy
sys.path.append("../../../sphere")
from kernel import kernel_polar
from mie import mie_cache
sys.path.append("../../../")
from integration import quadrature
sys.path.append("../../../ufuncs")
from integration import quadrature

energy.phiSequence = energy.make_phiSequence(kernel_polar)

rho = 10
K = 1.
M = 10
N = 50
k, w = quadrature(N)

mie = mie_cache(1e4, K, np.inf)
w = np.sqrt(w*2*np.pi/M)

phiarr = np.empty((4, M, N, N))
for i in range(N):
    for j in range(N):
        phiarr[..., i, j] = energy.phiSequence(rho, K, M, k[i], k[j], w[i], w[j], mie)

f, ax = plt.subplots(2,2)
ax[0, 0].imshow(phiarr[0, 1, ...])
ax[0, 1].imshow(phiarr[2, 1, ...])
ax[1, 0].imshow(phiarr[3, 1, ...])
ax[1, 1].imshow(phiarr[1, 1, ...])
plt.show()

marr = np.fft.rfft(phiarr, axis=1)

f, ax = plt.subplots(2,2)
ax[0, 0].imshow(marr[0, 1, ...].real)
ax[0, 1].imshow(marr[2, 1, ...].imag)
ax[1, 0].imshow(marr[3, 1, ...].imag)
ax[1, 1].imshow(marr[1, 1, ...].real)
plt.show()


