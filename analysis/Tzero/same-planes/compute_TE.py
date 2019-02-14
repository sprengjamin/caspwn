"""Calculate TE contribution when A=1, B=C=D=0

"""
import numpy as np
import sys, os
import time
from numba import njit
sys.path.append("../../../plane-sphere")
sys.path.append("../../../sphere")
sys.path.append("../../../material")
from scattering_amplitude import S1S2
from energy import energy_zero, make_phiSequence
import energy
from kernel import kernel_polar
import kernel

@njit
def S1S2_TE_only(x, z, mie):
    S1, S2 = S1S2(x, z, mie)
    return S1, 0.

@njit
def ABCD_same_plane(xi, k1, k2, phi):
    return 1., 0., 0., 0. 

# set A=1, B=C=D=0, only consider TE
kernel.S1S2 = S1S2_TE_only
kernel.ABCD = ABCD_same_plane
kernel_polar.recompile()

energy.phiSequence = make_phiSequence(kernel.kernel_polar)

R = 1.
Lvals = np.logspace(-1, -4, 61)
materials = ("PR", "Vacuum", "PR")

eta = 10.
nproc = 4

filename = "TE_energy_"+materials[0]+"_"+materials[1]+"_"+materials[2]+"_v2.dat"

if not os.path.isfile(filename):
    f=open(filename, "a")
    for L in Lvals:
        print("computing L=",L)
        rho = R/L
        N = int(eta*np.sqrt(rho))
        M = N
        start = time.time()
        en = energy_zero(R, L, materials, N, M, nproc)
        end = time.time()
        t = end-start
        f=open(filename, "ab")
        np.savetxt(f, [[L, en, t]])
        f.close()
else:
    print("File already exists!")




