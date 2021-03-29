import numpy as np
import matplotlib.pyplot as plt
import importlib

def plot_dielfunc(XI, mat):
    epsilon = importlib.import_module(mat).epsilon
    Y = [epsilon(xi) for xi in XI]
    plt.semilogx(XI, Y, label=mat)

xi = np.logspace(10, 18, 100)
plot_dielfunc(xi, "water_zwol")
plot_dielfunc(xi, "water_fiedler")
plot_dielfunc(xi, "polystyrene")
plot_dielfunc(xi, "silica")
plot_dielfunc(xi, "vacuum")
plot_dielfunc(xi, "ethanol")

plt.legend()
plt.show()
