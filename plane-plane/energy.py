import numpy as np
from numba import njit
from scipy.integrate import quad
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../plane/"))
from fresnel import rTE, rTM
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../material/"))
import material
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../ufuncs/"))
from psd import psd

from scipy.constants import Boltzmann, hbar, c

def integrand(K, k, eps1, eps2):
    kappa = np.sqrt(K**2 + k**2)
    ans = np.log1p(-rTE(K, k, eps1)*rTE(K, k, eps2)*np.exp(-2*kappa))
    ans += np.log1p(-rTM(K, k, eps1)*rTM(K, k, eps2)*np.exp(-2*kappa))
    return k/(2*np.pi)*ans
     
def energy_finite(L, T, materials):
    K_matsubara = 2*np.pi*Boltzmann*T/(hbar*c)
    eps_plane1 = eval("material."+materials[0]+".epsilon(0.)")
    eps_medium = eval("material."+materials[1]+".epsilon(0.)")
    eps_plane2 = eval("material."+materials[2]+".epsilon(0.)")
    eps1 = eps_plane1/eps_medium
    eps2 = eps_plane2/eps_medium
    f = lambda k: integrand(0., k, eps1, eps2)
    energy0 = quad(f, 0, np.inf)[0]
    energy = 0.
    n = 1
    while(True):    
        eps_plane1 = eval("material."+materials[0]+".epsilon(K_matsubara*n)")
        eps_medium = eval("material."+materials[1]+".epsilon(K_matsubara*n)")
        eps_plane2 = eval("material."+materials[2]+".epsilon(K_matsubara*n)")
        eps1 = eps_plane1/eps_medium
        eps2 = eps_plane2/eps_medium
        f = lambda k: integrand(K_matsubara*n*L*np.sqrt(eps_medium), k, eps1, eps2)
        term = 2*quad(f, 0, np.inf)[0]
        energy += term
        if abs(term/energy) < 1.e-12:
            break
        n += 1
    return 0.5*T*Boltzmann(energy0+energy), 0.5*T*Boltzmann*energy


def energy_faster(L, T, materials):
    K_matsubara = Boltzmann*T/(hbar*c)
    eps_plane1 = eval("material."+materials[0]+".epsilon(0.)")
    eps_medium = eval("material."+materials[1]+".epsilon(0.)")
    eps_plane2 = eval("material."+materials[2]+".epsilon(0.)")
    eps1 = eps_plane1/eps_medium
    eps2 = eps_plane2/eps_medium
    f = lambda k: integrand(0., k, eps1, eps2)
    energy0 = quad(f, 0, np.inf)[0]/L**2
    Teff = 4*np.pi*Boltzmann/hbar/c*T*L
    N = int(max(np.ceil((1-1.5*np.log10(np.abs(epsrel)))/np.sqrt(Teff)), 5))
    xi, eta = psd(N)    
    energy = 0.
    for n in range(N):
        eps_plane1 = eval("material."+materials[0]+".epsilon(K_matsubara*xi[n])")
        eps_medium = eval("material."+materials[1]+".epsilon(K_matsubara*xi[n])")
        eps_plane2 = eval("material."+materials[2]+".epsilon(K_matsubara*xi[n])")
        eps1 = eps_plane1/eps_medium
        eps2 = eps_plane2/eps_medium
        f = lambda k: integrand(K_matsubara*xi[n]*L*np.sqrt(eps_medium), k, eps1, eps2)
        term = quad(f, 0, np.inf)[0]/L**2
        energy += 2*eta[n]*term
    return 0.5*T*Boltzmann*(energy0+energy), 0.5*T*Boltzmann*energy
    


if __name__ == "__main__":
    L = 0.001e-06
    T = 293.015
    materials = ("PS1", "Water", "Silica1")
    print(energy_finite(L, T, materials))
    print(energy_faster(L, T, materials))

