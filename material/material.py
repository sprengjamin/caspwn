"""Materials

.. todo::
    * explain lorentz_oscillator data format
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
    def __init__(self, name, data, materialclass, wp_low=None, gamma_low=None, wp_high=None, gamma_high=None):
        r"""Permittivity is given by inter- and extrapolating optical data.
        
        Parameters
        ----------
        name : string
            name of material
        data : numpy.ndarray
            optical data in the format (K [rad/m], eps(i K))
        materialclass : string
            material class, e.g. "drude", "dielectric", "PR"
        wp_low, gamma_low : float
            plasma frequency and damping constant, for extrapolation at low frequencies in eV
        wp_high, gamma_high : float
            plasma frequency and damping constant, for extrapolation at high frequencies in eV

        Returns
        -------
        instance of optical_data class

        """
        self.name = name
        self.data = data
        self.materialclass = materialclass
        if wp_low != None:
            self.Kp_low = wp_low*e/hbar/c
        else:
            self.Kp_low = None
        if gamma_low != None:
            self.Kgamma_low = gamma_low*e/hbar/c
        else:
            self.Kgamma_low = None
        if wp_high != None:
            self.Kp_high = wp_high*e/hbar/c
        else:
            self.Kp_high = None
        if gamma_high != None:
            self.Kgamma_high = gamma_high*e/hbar/c
        else:    
            self.Kgamma_high = None
    

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
                if self.Kp_low != None:
                    return 1 + self.Kp_low**2/K/(K+self.Kgamma_low)
                else:
                    # for too small xi extrapolate with the first point
                    return self.data[0][1]
            elif i >= len(self.data):
                if self.Kp_high != None:
                    return 1 + self.Kp_high**2/K/(K+self.Kgamma_high)
                else:
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


class plasma(material):
    def __init__(self, name, plasma):
        """ plasma in eV
            gamma in eV
        """
        self.name = name
        self.materialclass = "plasma"
        self.K_plasma = plasma * e / hbar / c

    def epsilon(self, K):
        if K == 0.:
            # dummy number to avoid crash
            return 1.
        else:
            return 1 + self.K_plasma ** 2 / K**2

    def n(self, K):
        if K == 0.:
            # dummy number to avoid crash
            return 1.
        else:
            return np.sqrt(self.epsilon(K))



class drude_smith(material):
    def __init__(self, name, omega_p, gamma, c1):
        r"""Drude-Smith model.
        
        .. math::
            \epsilon(i\xi) = 1 + \frac{\omega_\mathrm{P}^2}{+ \xi^2 + \gamma\xi}\left(1+\frac{\gamma c_1}{\xi + \gamma}\right)

        In the zero-frequency limit the behavior corresponds to a Drude metal.
        
        Parameters
        ----------
        name : string
            name of material
        omega_p : float
            plasma frequency in rad/s
        gamma : float
            dissipation frequency in rad/s
        c1 : float
            Drude-Smith coefficient

        Returns
        -------
        instance of drude_smith class

        """
        self.name = name
        self.K_p = omega_p/c
        self.gamma = gamma/c
        self.c1 = c1
        self.materialclass = "drude"

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
            # dummy number to avoid crash
            return 1.
        else:
            return 1 + self.K_p**2/(K**2 + K*self.gamma)*(1 + self.gamma*self.c1/(K + self.gamma))
    
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

glass_data = [[np.sqrt(1.282)*1.911*1.e16/c, 1.911*1.e16/c, 0.]] 
Glass =  lorentz_oscillator("Glass", glass_data, dformat="lorentz")

Gold_drude = drude("Gold_drude", 9., 0.035)

Gold_drude_MH = drude("Gold_drude", 9., 0.030) # according to MH phd thesis

Gold_plasma = plasma("Gold_plasma", 9.)

Mercury = drude_smith("Mercury", 1.975e16, 1.647e15, -0.49)

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

Water_Renan = lorentz_oscillator("Water_Renan", Water_data, static_value = 78.)
PS_Renan = lorentz_oscillator("PS1", PS1_data, static_value = 2.5)

filename = os.path.join(os.path.dirname(__file__), "./optical_data/GoldDalvit.dat")
Gold_Decca = optical_data("Gold_Decca", np.loadtxt(filename), "drude", wp_low=9, gamma_low=0.03, wp_high=54.475279, gamma_high=211.48855)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    X = np.logspace(10,22,100)/c
    eps = [Gold_Decca.epsilon(x) for x in X]

    plt.loglog(X*c, eps)
    plt.show()
