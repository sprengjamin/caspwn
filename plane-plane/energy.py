import numpy as np
#from numba import njit
from scipy.integrate import quad
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../plane/"))
from fresnel import rTE, rTM
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../material/"))
import material
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../ufuncs/"))
from psd import psd

from scipy.constants import Boltzmann, hbar, c

def integrand(K, k, eps1, matclass1, eps2, matclass2):
    kappa = np.sqrt(K**2 + k**2)
    ans = np.log1p(-rTE(K, k, eps1, matclass1)*rTE(K, k, eps2, matclass2)*np.exp(-2*kappa))
    ans += np.log1p(-rTM(K, k, eps1, matclass1)*rTM(K, k, eps2, matclass2)*np.exp(-2*kappa))
    return k/(2*np.pi)*ans
     

def energy_finite(L, T, materials, mode, epsrel=1.e-08, N=None):
    K_matsubara = Boltzmann*T/(hbar*c)
    eps_plane1 = eval("material."+materials[0]+".epsilon(0.)")
    matclass1 = eval("material."+materials[0]+".materialclass")
    eps_medium = eval("material."+materials[1]+".epsilon(0.)")
    eps_plane2 = eval("material."+materials[2]+".epsilon(0.)")
    matclass2 = eval("material."+materials[2]+".materialclass")
    eps1 = eps_plane1/eps_medium
    eps2 = eps_plane2/eps_medium
    f = lambda k: integrand(0., k, eps1, matclass1, eps2, matclass2)
    energy0 = quad(f, 0, np.inf)[0]/L**2
    energy = 0.
    if mode == "psd":
        Teff = 4*np.pi*Boltzmann/hbar/c*T*L
        if N == None:
            N = int(max(np.ceil((1-1.5*np.log10(np.abs(epsrel)))/np.sqrt(Teff)), 5))
        xi, eta = psd(N)    
        energy = 0.
        for n in range(N):
            eps_plane1 = eval("material."+materials[0]+".epsilon(K_matsubara*xi[n])")
            eps_medium = eval("material."+materials[1]+".epsilon(K_matsubara*xi[n])")
            eps_plane2 = eval("material."+materials[2]+".epsilon(K_matsubara*xi[n])")
            eps1 = eps_plane1/eps_medium
            eps2 = eps_plane2/eps_medium
            f = lambda k: integrand(K_matsubara*xi[n]*L*np.sqrt(eps_medium), k, eps1, matclass1, eps2, matclass2)
            term = quad(f, 0, np.inf)[0]/L**2
            energy += 2*eta[n]*term
    elif mode == "msd":
        energy = 0.
        n = 1
        if N == None:
            N = np.inf
        while n < N:
            eps_plane1 = eval("material."+materials[0]+".epsilon(2*np.pi*K_matsubara*n)")
            eps_medium = eval("material."+materials[1]+".epsilon(2*np.pi*K_matsubara*n)")
            eps_plane2 = eval("material."+materials[2]+".epsilon(2*np.pi*K_matsubara*n)")
            eps1 = eps_plane1/eps_medium
            eps2 = eps_plane2/eps_medium
            f = lambda k: integrand(2*np.pi*K_matsubara*n*L*np.sqrt(eps_medium), k, eps1, matclass1, eps2, matclass2)
            term = quad(f, 0, np.inf)[0]/L**2
            print(n, term)
            energy += 2*term
            if abs(term/energy) < epsrel:
                break
            n += 1
    else:
        assert(True)
    return 0.5*T*Boltzmann*(energy0+energy), 0.5*T*Boltzmann*energy
    


if __name__ == "__main__":
    L = 0.01e-06
    T = 293.015
    materials = ("PS1", "Water", "PS1")
    materials = ("PR", "Vacuum", "PR")
    print(energy_finite(L, T, materials, "msd"))
    print(energy_finite(L, T, materials, "psd"))

