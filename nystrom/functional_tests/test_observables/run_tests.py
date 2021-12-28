import numpy as np

h = [5.e-09, 5.e-09, 1.e-09, 1.e-09, 5.e-09, 5.e-09, 1.e-09, 1.e-09]
for i in range(8):
    lo = np.loadtxt("data/out"+str(i+1)+"m.dat")
    mi = np.loadtxt("data/out"+str(i+1)+".dat")
    hi = np.loadtxt("data/out"+str(i+1)+"p.dat")
    num_force = -(hi[0]-lo[0])/2/h[i]
    np.testing.assert_allclose(num_force[0], mi[1][0], rtol=7.e-4)
    np.testing.assert_allclose(num_force[1], mi[1][1], rtol=6.e-3)
    np.testing.assert_allclose(num_force[2], mi[1][2], rtol=6.e-3)
    num_forcegradient = (hi[1] - lo[1]) / 2 / h[i]
    np.testing.assert_allclose(num_forcegradient[0], mi[2][0], rtol=8.e-4)
    np.testing.assert_allclose(num_forcegradient[1], mi[2][1], rtol=6.e-3)
    np.testing.assert_allclose(num_forcegradient[2], mi[2][2], rtol=6.e-3)

print("All tests successful!")