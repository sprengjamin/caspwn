import numpy as np
from scipy.constants import e as eV
from scipy.constants import hbar

materialclass = 'dielectric'
# from Fiedler et al.
#             c, 1/tau [eV]  
debye =     [[0.47, 6.84e-6],
             [72.62, 7.98e-5]]
debye = np.array(debye)
#     omega [eV], c, gamma [eV]
ir = [[8.46e-4, 2.59e-1, 3.92e-4],
      [4.19e-3, 1.04,    7.43e-3],  
      [2.12e-2, 1.62,    2.60e-2],  
      [6.25e-2, 5.55e-1, 3.98e-2],  
      [8.49e-2, 2.38e-1, 2.99e-2],  
      [2.04e-1, 1.34e-2, 8.43e-3],  
      [4.18e-1, 7.17e-2, 3.41e-2]]
uv = [[8.34, 4.47e-2, 0.75],
      [9.50, 3.27e-2, 1.12],  
      [10.41, 4.66e-2, 1.26],  
      [11.67, 6.67e-2, 1.58],  
      [12.95, 7.42e-2, 1.65],  
      [14.13, 9.30e-2, 1.86],  
      [15.50, 7.79e-2, 2.22],  
      [17.17, 7.9e-2,  2.7],  
      [18.89, 4.18e-2, 2.82],  
      [21.45, 1.07e-1, 6.87],  
      [30.06, 1.33e-1, 18.28],  
      [49.45, 5.66e-2, 36.28]]
lorentz = np.vstack((np.array(ir), np.array(uv)))

# model parameters
ci = debye[:,0]
invtaui = debye[:,1]*eV/hbar
omegaj = lorentz[:,0]*eV/hbar
cj = lorentz[:,1]
gammaj = lorentz[:,2]*eV/hbar

def epsilon(xi):
    eps = 1.
    eps += np.sum(ci/(1 + xi/invtaui))
    eps += np.sum(cj*omegaj**2/(omegaj**2 + xi*gammaj + xi**2))
    return eps
