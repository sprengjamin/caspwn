import numpy as np
#from numba import njit
from scipy.integrate import quad
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../plane/"))
from fresnel import rTE_zero, rTE_finite, rTM_zero, rTM_finite
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../material/"))
import material
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../ufuncs/"))
from psd import psd

from scipy.constants import Boltzmann, hbar, c

def integrand_zero(k, eps1, matclass1, eps2, matclass2):
    ans = np.log1p(-rTE_zero(k, eps1, matclass1)*rTE_zero(k, eps2, matclass2)*np.exp(-2*k))
    ans += np.log1p(-rTM_zero(k, eps1, matclass1)*rTM_zero(k, eps2, matclass2)*np.exp(-2*k))
    return k/(2*np.pi)*ans

def integrand_finite(K, k, eps1, eps2):
    kappa = np.sqrt(K**2 + k**2)
    ans = np.log1p(-rTE_finite(K, k, eps1)*rTE_finite(K, k, eps2)*np.exp(-2*kappa))
    ans += np.log1p(-rTM_finite(K, k, eps1)*rTM_finite(K, k, eps2)*np.exp(-2*kappa))
    return k/(2*np.pi)*ans
     

def energy_finite(L, T, materials, mode, epsrel=1.e-08, N=None):
    K_matsubara = Boltzmann*T/(hbar*c)
    eps_medium = eval("material."+materials[1]+".epsilon(0.)")
    
    matclass1 = eval("material."+materials[0]+".materialclass")
    if matclass1 == "dielectric":
        eps_plane1 = eval("material."+materials[0]+".epsilon(0.)")
        eps1 = eps_plane1/eps_medium
    elif matclass1 == "PR":
        eps1 = 1.
    else:
        eps1 = eval("material."+materials[0]+".K_plasma")*L

    matclass2 = eval("material."+materials[2]+".materialclass")
    if matclass2 == "dielectric":
        eps_plane2 = eval("material."+materials[2]+".epsilon(0.)")
        eps2 = eps_plane2/eps_medium
    elif matclass2 == "PR":
        eps2 = 1.
    else:
        eps2 = eval("material."+materials[2]+".K_plasma")*L
    
    f = lambda k: integrand_zero(k, eps1, matclass1, eps2, matclass2)
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
            f = lambda k: integrand_finite(K_matsubara*xi[n]*L*np.sqrt(eps_medium), k, eps1, eps2)
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
            f = lambda k: integrand_finite(2*np.pi*K_matsubara*n*L*np.sqrt(eps_medium), k, eps1, eps2)
            term = quad(f, 0, np.inf)[0]/L**2
            #print(n, term)
            energy += 2*term
            if N == np.inf:
                if abs(term/energy) < epsrel:
                    print("msd nmax", n)
                    break
            n += 1
    else:
        assert(True)
    return 0.5*T*Boltzmann*energy0, 0.5*T*Boltzmann*energy
    


if __name__ == "__main__":
    L = 0.01e-06
    T = 298.015
    materials = ("PS1", "WaterRT", "PS1")
    #materials = ("PR", "Vacuum", "PR")
    print(energy_finite(L, T, materials, "msd", N=700))
    print(energy_finite(L, T, materials, "psd"))

