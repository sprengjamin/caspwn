import numpy as np
import time
from scipy.constants import hbar, c, k
import sys
sys.path.append("/home/benjamin/wd/nystrom/plane-sphere")
from energy import energy_finite_nozero

L = 1.e-3
R = 1.
materials = ("PR", "Vacuum", "PR")

lambda_T = np.logspace(-2.,-1.,10)

Tvals = hbar*c/k/lambda_T

eta = 10.
nproc = 4

filename = "energy_lowT_eta10.dat"

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
        en = energy_finite_nozero(R, L, T, materials, N, M, nproc)
        end = time.time()
        t = end-start
        f=open(filename, "ab")
        np.savetxt(f, [[T, en, t]])
        f.close()
