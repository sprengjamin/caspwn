import numpy as np
import sys
sys.path.append("..")
from compute import compute_to_file

R1 = 1.
R2 = 3.
R_eff = R1*R2/(R1+R2)
Eps_eff = np.logspace(-1, -4, 61)
L = Eps_eff*R_eff

materials = ("PR", "Vacuum", "PR")

eta = 10.
X = 200
nproc = 4

filename = "energy3_"+materials[0]+"_"+materials[1]+"_"+materials[2]+".dat"

compute_to_file(filename, L, R1, R2, materials, eta, X, nproc)
