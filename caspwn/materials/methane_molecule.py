import numpy as np
from scipy.constants import epsilon_0, pi

materialclass = "dielectric"

wi = np.array([1.75, 2.56, 4.42, 10.0, 48.3])*1.e16
ci = np.array([89.3, 137.5, 41.2, 2.78, 0.18])*1.e-42

r = 1.655e-10

def alpha(xi):
    return np.sum(ci/(1+(xi/wi)**2))

def epsilon(xi):
    a = alpha(xi)
    return (1 + 2*a/4/pi/epsilon_0/r**3)/(1 - a/4/pi/epsilon_0/r**3)
