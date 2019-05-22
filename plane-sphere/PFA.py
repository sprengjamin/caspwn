import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../plane-plane/"))
from energy import energy_faster


def PFA(R, L, T, materials):
    energy0, energy = energy_faster(L, T, materials)
    return 2*np.pi*R*energy0, 2*np.pi*R*energy


if __name__ == "__main__":
    R = 1.
    L = 1.e-03
    T = 0.2289885278703586
    materials = ("PR", "Vacuum", "PR")
    print(PFA(R, L, T, materials))
