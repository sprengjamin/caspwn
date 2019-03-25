import numpy as np
import sys, os
import time
sys.path.append("../../../plane-sphere")
sys.path.append("../../../material")
from energy import energy_zero

R = 1.
Lvals = np.logspace(-1, -4, 61)
materials = ("PR", "Vacuum", "PR")

eta = 10.
nproc = 4

filename = "full_energy_"+materials[0]+"_"+materials[1]+"_"+materials[2]+"_v4.dat"

# read number of lines
f=open(filename, "r+")
num_lines = 0
for line in f:
    num_lines += 1
f.close()

for i, L in enumerate(Lvals):
    print("computing L =",L)
    if i < num_lines:
        print("Point already computed!")
    else:
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
