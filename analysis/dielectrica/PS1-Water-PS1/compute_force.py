import numpy as np
import sys, os
import time
sys.path.insert(0, "../../../sphere-sphere")
from force import force_faster

R1 = 2.e-06
R2 = 2e-06
Lvals = np.logspace(np.log10(0.002), np.log10(0.2), 30)[::-1]*1.e-06
T = 293.015
materials = ("PS1", "Water", "PS1")

eta = 10.
nproc = 4

filename = "force_"+materials[0]+"_"+materials[1]+"_"+materials[2]+".dat"

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
        fo0, fo = force_faster(R1, R2, L, T, materials, Nin, Nout, M, nproc)
        end = time.time()
        t = end-start
        f=open(filename, "ab")
        np.savetxt(f, [[L, fo0, fo, t]])
        f.close()
else:
    print("File already exists!")





