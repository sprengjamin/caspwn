import numpy as np
import sys, os
import time
sys.path.append("../../sphere-sphere")
sys.path.append("../../ufuncs")
sys.path.append("../../sphere")
sys.path.append("../../material")
from energy import energy_faster, make_phiSequence
from kernel import phiKernel
import energy
energy.phiSequence = make_phiSequence(phiKernel)

R1 = 8.e-06
R2 = 16.5e-06
Lvals = np.arange(0.008, 0.801, 0.001)[::-1]*1.e-06
T = 293.015
materials = ("PS1", "Water", "Silica1")

eta = 10.
nproc = 4

filename = "energy_"+materials[0]+"_"+materials[1]+"_"+materials[2]+"_v2.dat"

if not os.path.isfile(filename):
    f=open(filename, "a")
    for L in Lvals:
        print("computing L=",L)
        rho1 = R1/L
        rho2 = R2/L
        N = int(eta*np.sqrt(max(rho1, rho2)))
        M = N
        start = time.time()
        en = energy_faster(R1, R2, L, T, materials, N, M, nproc)
        end = time.time()
        t = end-start
        f=open(filename, "ab")
        np.savetxt(f, [[L, en, t]])
        f.close()
else:
    print("File already exists!")




