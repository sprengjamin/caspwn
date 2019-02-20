import numpy as np
import sys, os
import time
sys.path.append("../../sphere-sphere")
from energy import energy_faster
import energy

R1 = 8.e-06
R2 = 16.5e-06
Lvals = np.arange(0.008, 0.801, 0.001)[::-1]*1.e-06
T = 293.015
materials = ("PS1", "Water", "Silica1")

eta = 15.
nproc = 10

filename = "energy_"+materials[0]+"_"+materials[1]+"_"+materials[2]+"_v7.dat"

if not os.path.isfile(filename):
    f=open(filename, "a")
    for L in Lvals:
        print("computing L=",L)
        rho1 = R1/L
        rho2 = R2/L
        rhoeff = rho1*rho2/(rho1+rho2)
        Nout = int(eta*np.sqrt(rhoeff))
        Nin = int(eta*np.sqrt(rho1+rho2))
        M = Nin
        start = time.time()
        en0, en = energy_faster(R1, R2, L, T, materials, Nin, Nout, M, nproc)
        end = time.time()
        t = end-start
        f=open(filename, "ab")
        np.savetxt(f, [[L, en0, en, t]])
        f.close()
else:
    print("File already exists!")




