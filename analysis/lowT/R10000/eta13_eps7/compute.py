import numpy as np
import time
from scipy.constants import hbar, c, k
import sys
sys.path.append("../../../../plane-sphere")
from energy import energy_finite_nozero

L = 1.e-4
R = 1.
materials = ("PR", "Vacuum", "PR")

lambda_T = np.logspace(-3., -1., 30)

Tvals = hbar*c/k/lambda_T

eta = 13
nproc = 4
epsrel_T = 1.e-07

filename = "energy_lowT_R10000_eta13_eps7.dat"

# read number of lines
f=open(filename, "r+")
num_lines = 0
for line in f:
    num_lines += 1
f.close()

for i, T in enumerate(Tvals):
    print("computing T =",T)
    if i < num_lines:
        print("Point already computed!")
    else:
        rho = R/L
        N = int(eta*np.sqrt(rho))
        M = N
        start = time.time()
        en = energy_finite_nozero(R, L, T, materials, N, M, epsrel_T, nproc)
        end = time.time()
        t = end-start
        f=open(filename, "ab")
        np.savetxt(f, [[T, en, t]])
        f.close()
