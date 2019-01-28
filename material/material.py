"""Materials
"""
import numpy as np
from scipy.constants import e, hbar

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

    def epsilon(self, xi):
        r"""Dielectric function.

        Parameters
        ----------
        xi : flaot
            frequency in rad/s

        Returns
        -------
        eps : flaot
            permittivity, dimensionless number

        """
        if xi == 0. and self.static_value != None:
            return self.static_value
        else:
            eps = 1.
            for params in self.data:
                xiP, xiR, gamma = params
                eps += xiP**2/(xiR**2 + xi**2 + gamma*xi)
            return eps
    
    def n(self, xi):
        r"""Refractive index.

        Note that the implementation assumes non-magnetic materials.

        Parameters
        ----------
        xi : flaot
            frequency in rad/s

        Returns
        -------
        n : flaot
            refractive index, dimensionless number

        """
        return np.sqrt(self.epsilon(xi))
        

class perfect_reflector(material):
    def __init__(self):
        self.name = "perfect_reflector"
    
    def epsilon(self, xi):
        return np.inf        
    
    def n(self, xi):
        return np.inf

def convert_zwol_to_lorentz(data):
    xiR = np.array(data[1])*e/hbar
    xiP = np.sqrt(np.array(data[0]))*xiR
    gamma = np.zeros(len(data[0]))
    return np.vstack((xiP, xiR, gamma)).T
    
           
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

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    X = np.logspace(10,18,100)
    n = [Silica1.n(x) for x in X]
    eps = [Silica1.epsilon(x) for x in X]
    plt.semilogx(X, n)
    plt.semilogx(X, eps)
    plt.show()