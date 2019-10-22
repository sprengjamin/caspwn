import numpy as np

for i in range(1, 21):
    I = "%02d" % (i,)
    ref = np.loadtxt("reference_data/data"+I+".out")
    new = np.loadtxt("test_data/data"+I+".out")
    np.testing.assert_allclose(new, ref, rtol=1e-06)

print("All tests successful!")

    
