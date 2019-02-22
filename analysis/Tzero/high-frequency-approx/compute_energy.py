"""Calculate TM contribution when A=1, B=C=D=0

"""
import numpy as np
import sys, os
import time
from numba import njit
sys.path.append("../../../plane-sphere")
sys.path.append("../../../sphere")
sys.path.append("../../../material")
from energy import energy_zero
import energy
from kernel import kernel_polar
import kernel

@njit
def S1S2_high_frequency(x, z, mie):
    return -0.5*x, 0.5*x

# replace S1S2 by the high frequency asmpytotics
kernel.S1S2 = S1S2_high_frequency
kernel_polar.recompile()
energy.phi_array.recompile()

R = 1.
Lvals = np.logspace(-1, -4, 61)
materials = ("PR", "Vacuum", "PR")

eta = 10.
nproc = 4

filename = "approx_energy_"+materials[0]+"_"+materials[1]+"_"+materials[2]+"_v3.dat"

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




