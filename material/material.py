"""Materials
"""
import numpy as np
from scipy.constants import c, e, hbar
import os
dirname = os.path.dirname(os.path.abspath(__file__))

class material:
    def __init__(self, name):
        self.name = name

class lorentz_oscillator(material):
    def __init__(self, name, data, dformat="zwol", static_value=None):
        r"""Lorentz oscillator model.
        
        .. math::
            \epsilon(i\xi) = 1 + \sum_{i=1}^N \frac{\xi_\mathrm{P}^2}{\xi_\mathrm{R}^2 + \xi^2 + \gamma\xi}


        
        Parameters
        ----------
        name : string
            name of material
        data : array
            array constaining parameters for Lorentz model
        static_value : float
            (optional) static value of permittivity

        Returns
        -------
        instance of lorentz_oscillator class

        """
        self.name = name
        if dformat == "zwol":
            self.data = convert_zwol_to_lorentz(data)
        elif dformat == "lorentz":
            self.data = data
        else:
            raise ValueError("Data format unknown!")
        self.static_value = static_value
        self.materialclass = "dielectric"

    def epsilon(self, K):
        r"""Dielectric function.

        Parameters
        ----------
        K : flaot
            vacuum wavenumber rad/m

        Returns
        -------
        eps : flaot
            permittivity, dimensionless number

        """
        if K == 0. and self.static_value != None:
            return self.static_value
        else:
            eps = 1.
            for params in self.data:
                K_P, K_R, gamma = params
                eps += K_P**2/(K_R**2 + K**2 + gamma*K)
            return eps
    
    def n(self, K):
        r"""Refractive index.

        Note that the implementation assumes non-magnetic materials.

        Parameters
        ----------
        K : flaot
            vacuum wavenumber in rad/m

        Returns
        -------
        n : flaot
            refractive index, dimensionless number

        """
        return np.sqrt(self.epsilon(K))
 
        
class optical_data(material):
    def __init__(self, name, data, materialclass, f_extra=None):
        r"""Permittivity is given by inter- and extrapolating optical data.
        
        Parameters
        ----------
        name : string
            name of material
        data : numpy.ndarray
            optical data in the format (K [rad/m], eps(i K))
        materialclass : string
            material class, e.g. "drude", "dielectric", "PR"
        f_extra : function
            (optional) function defining the extrapolation towards :math:`K=0`.
            Default is constant extrapolation.

        Returns
        -------
        instance of optical_data class

        """
        self.name = name
        self.data = data
        self.materialclass = materialclass
        self.f_extra = f_extra

    def epsilon(self, K):
        r"""Dielectric function.

        Parameters
        ----------
        K : flaot
            vacuum wavenumber rad/m

        Returns
        -------
        eps : flaot
            permittivity, dimensionless number

        """
        if K == 0.:
            if self.f_extra == None:
                return self.data[0][1]
            else:
                raise NotImplementedError
        else:
            xi = K*c
            i = np.searchsorted(self.data[:,0], xi)
            if i == 0:
                # for too small xi extrapolate with the first point
                return self.data[0][1]
            elif i >= len(self.data):
                # for too large xi extrapolate with the last point
                return self.data[-1][1]
            else:
                xi1 = self.data[i-1, 0]
                eps1 = self.data[i-1,1]
                xi2 = self.data[i, 0]
                eps2 = self.data[i,1]
                return eps1 + (eps2-eps1)*(xi-xi1)/(xi2-xi1)

    def n(self, K):
        r"""Refractive index.

        Note that the implementation assumes non-magnetic materials.

        Parameters
        ----------
        K : flaot
            vacuum wavenumber in rad/m

        Returns
        -------
        n : flaot
            refractive index, dimensionless number

        """
        return np.sqrt(self.epsilon(K))

class drude(material):
    def __init__(self, name, plasma, gamma):
        """ plasma in eV
            gamma in eV
        """
        self.name = name
        self.materialclass = "drude"
        self.K_plasma = plasma*e/hbar/c
        self.K_gamma = gamma*e/hbar/c
    
    def epsilon(self, K):
        if K == 0.:
            # dummy number to avoid crash
            return 1.
        else:
            return 1 + self.K_plasma**2/K/(K+self.K_gamma)

    def n(self, K):
        if K == 0.:
            # dummy number to avoid crash
            return 1.
        else:
            return np.sqrt(self.epsilon(K))
        


class perfect_reflector(material):
    def __init__(self):
        self.name = "PR"
        self.materialclass = "PR"
    
    def epsilon(self, xi):
        return np.inf        
    
    def n(self, xi):
        return np.inf


class vacuum(material):
    def __init__(self):
        self.name = "Vacuum"
    
    def epsilon(self, xi):
        return 1.       
    
    def n(self, xi):
        return 1.


def convert_zwol_to_lorentz(data):
    xiR = np.array(data[1])*e/hbar
    xiP = np.sqrt(np.array(data[0]))*xiR
    gamma = np.zeros(len(data[0]))
    return np.vstack((xiP, xiR, gamma)).T/c

Gold = drude("Gold", 9., 0.035)

Electrolyte = lorentz_oscillator("Electrolyte", [[],[]], static_value = 0)

filename = os.path.join(os.path.dirname(__file__), "./optical_data/FUSED_SILICA_EPS-iw.dat")
fused_silica = optical_data("Fused Silica", np.loadtxt(filename), "dielectric")

PR = perfect_reflector() 
           
PTFE_data = [[9.30e-3, 1.83e-2, 1.39e-1, 1.12e-1, 1.95e-1, 4.38e-1, 1.06e-1, 3.86e-2],
             [3.00e-4, 7.60e-3, 5.57e-2, 1.26e-1, 6.71e+0, 1.86e+1, 4.21e+1, 7.76e+1]]
PTFE = lorentz_oscillator("PTFE", PTFE_data)

Silica1_data = [[7.84e-1, 2.03e-1, 4.17e-1, 3.93e-1, 5.01e-2, 8.20e-1, 2.17e-1, 5.50e-2],
                [4.11e-2, 1.12e-1, 1.12e-1, 1.11e-1, 1.45e+1, 1.70e+1, 8.14e+0, 9.16e+1]]
Silica1 = lorentz_oscillator("Silica1", Silica1_data)

Silica2_data = [[1.19e+0, 6.98e-2, 1.35e-2, 7.10e-1, 1.80e-1, 5.95e-1, 2.27e-1, 5.58e-2],
                [5.47e-2, 1.23e-2, 5.74e-4, 1.29e-1, 9.10e+0, 1.43e+1, 2.31e+1, 7.90e+1]]
Silica2 = lorentz_oscillator("Silica2", Silica2_data)

PS1_data = [[1.21e-2, 2.19e-2, 1.79e-2, 3.06e-2, 3.03e-1, 6.23e-1, 3.25e-1, 3.31e-2],
            [1.00e-3, 1.32e-2, 3.88e+0, 1.31e-1, 5.99e+0, 1.02e+1, 1.88e+1, 5.15e+1]]
PS1 = lorentz_oscillator("PS1", PS1_data)

PS2_data = [[3.12e-2, 1.17e-2, 2.17e-2, 9.20e-3, 2.93e-1, 6.54e-1, 4.17e-1, 2.13e-2],
            [1.18e-1, 9.00e-4, 1.19e-2, 1.56e+0, 6.12e+0, 1.01e+1, 2.02e+1, 6.86e+1]]
PS2 = lorentz_oscillator("PS2", PS2_data)

Water_data = [[1.43e+0, 9.74e+0, 2.16e+0, 5.32e-1, 3.89e-1, 2.65e-1, 1.36e-1],
              [2.29e-2, 8.77e-4, 4.93e-3, 1.03e-1, 9.50e+0, 2.09e+1, 2.64e+1]]
Water_static = 78.7
Water = lorentz_oscillator("Water", Water_data, static_value = Water_static)

modifiedWater = lorentz_oscillator("modifiedWater", Water_data, static_value = 0.)

Vacuum = vacuum()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    X = np.logspace(2,10,100)
    n = [Silica1.n(x) for x in X]
    eps = [Silica1.epsilon(x) for x in X]
    eps2 = [fused_silica.epsilon(x) for x in X]
    plt.semilogx(X, eps)
    plt.semilogx(X, eps2)
    plt.show()
