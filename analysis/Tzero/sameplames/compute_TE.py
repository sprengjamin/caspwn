"""Calculate TM contribution when A=1, B=C=D=0

"""
import numpy as np
import sys, os
import time
sys.path.append("../../../plane-sphere")
sys.path.append("../../../ufuncs")
sys.path.append("../../../sphere")
sys.path.append("../../../material")
from energy import energy_zero, make_phiSequence
import energy
from test_kernel import phiKernel_TE
energy.phiSequence = make_phiSequence(phiKernel_TE)

R = 1.
Lvals = np.logspace(-1, -4, 61)
materials = ("PR", "Vacuum", "PR")

eta = 10.

from multiprocessing import cpu_count
nproc = cpu_count()
print("Computing on "+str(nproc)+" CPUs!")

filename = "TE_energy_"+materials[0]+"_"+materials[1]+"_"+materials[2]+".dat"

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




