import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../plane-plane/"))
from energy import energy_finite


def PFA(R1, R2, L, T, materials):
    Reff = R1*R2/(R1+R2)
    return 2*np.pi*Reff*energy_finite(L, T, materials)/L**2


if __name__ == "__main__":
    R1 = 1.e-6
    R2 = 1.e-6
    L = 0.1e-6
    T = 293.015
    materials = ("PS1", "Water", "Silica1")
    print(PFA(R1, R2, L, T, materials))
