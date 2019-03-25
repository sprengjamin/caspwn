# script for computing an array of data points and saving to a file
import numpy as np
import time
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../sphere-sphere"))
from energy import energy_zero

def read_num_lines(filename):
    num_lines = 0
    if os.path.isfile(filename):
        f=open(filename, "r+")
        for line in f:
            num_lines += 1
        f.close()
    return num_lines

def compute_to_file(filename, L_array, R1, R2, materials, eta, X, nproc):
    num_lines = read_num_lines(filename)

    for i, L in enumerate(L_array):
        print("computing L =", L)
        if i < num_lines:
            print("Point already computed!")
        else:
            rho1 = R1/L
            rho2 = R2/L
            rhoeff = rho1*rho2/(rho1+rho2)
            
            Nin = int(eta*np.sqrt(rho1+rho2))
            Nout = int(eta*np.sqrt(rhoeff))
            M = Nin
            
            start = time.time()
            en = energy_zero(R1, R2, L, materials, Nin, Nout, M, X, nproc)
            end = time.time()
            t = end-start
            f=open(filename, "ab")
            np.savetxt(f, [[L, en, t]])
            f.close()
