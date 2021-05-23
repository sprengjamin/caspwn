import numpy as np
from scipy.constants import epsilon_0, pi


wi = np.array([1.75, 2.56, 4.42, 10.0, 48.3])*1.e16
ci = np.array([89.3, 137.5, 41.2, 2.78, 0.18])*1.e-42

r = 1.655e-10

def alpha(xi):
    return np.sum(ci/(1+(xi/wi)**2))

def epsilon(xi):
    a = alpha(xi)
    return (1 + 2*a/4/pi/epsilon_0/r**3)/(1 - a/4/pi/epsilon_0/r**3)



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    xi, e = np.loadtxt("optical_data/dielectric_CH4.txt", unpack=True)
    E = np.array([epsilon(x) for x in xi])

    plt.semilogx(xi, E, label="mine")
    plt.semilogx(xi, e, label="his")
    plt.legend()
    plt.show()

