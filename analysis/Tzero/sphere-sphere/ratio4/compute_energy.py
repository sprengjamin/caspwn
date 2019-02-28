import numpy as np
import sys, os
import time
sys.path.append("../../../../sphere-sphere")
from energy import energy_zero

R1 = 1.
R2 = 4.
R_eff = R1*R2/(R1+R2)
Eps_eff = np.logspace(-1, -4, 61)

materials = ("PR", "Vacuum", "PR")

eta = 10.
nproc = 4

filename = "energy4_"+materials[0]+"_"+materials[1]+"_"+materials[2]+"_v3.dat"

if not os.path.isfile(filename):
    f=open(filename, "a")
    for eps_eff in Eps_eff:
        print("computing eps_eff=", eps_eff)
        L = eps_eff*R_eff
        rho1 = R1/L
        rho2 = R2/L
        rhoeff = rho1*rho2/(rho1+rho2)
        
        Nin = int(eta*np.sqrt(rho1+rho2))
        Nout = int(eta*np.sqrt(rhoeff))
        M = Nin
        X = 200
        
        start = time.time()
        en = energy_zero(R1, R2, L, materials, Nin, Nout, M, X, nproc)
        end = time.time()
        t = end-start
        f=open(filename, "ab")
        np.savetxt(f, [[L, en, t]])
        f.close()
else:
    print("File already exists!")
